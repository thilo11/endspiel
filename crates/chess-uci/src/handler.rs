use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use chess_common::{Board, Move, Score};
use chess_engine::{Engine, InfoCallback, SearchInfo, SearchParams};

use crate::protocol::{GoParams, UciCommand, UciInfo, UciOptionDef, UciOptionType, UciResponse};

/// WDL normalization parameters for P(win | score) = sigmoid((score − a) / b).
/// Re-run `wdl-fit --input data/archive/archive_shuf.bin` after each net promotion
/// and update these two constants with the printed values.
const WDL_A: f64 = 9.6;
const WDL_B: f64 = 262.0;

const ENGINE_NAME: &str = "Endspiel";

/// Convert a centipawn score to WDL in millipawns (0–1000, summing to 1000).
fn score_to_wdl(cp: i32) -> (u32, u32, u32) {
    let s = cp as f64;
    let p_win  = 1.0 / (1.0 + (-(s - WDL_A) / WDL_B).exp());
    let p_loss = 1.0 / (1.0 + ( (s + WDL_A) / WDL_B).exp());
    let win  = (p_win  * 1000.0).round() as u32;
    let loss = (p_loss * 1000.0).round() as u32;
    let draw = 1000u32.saturating_sub(win + loss);
    (win, draw, loss)
}

fn normalize_display_score(score: Score) -> Score {
    if score.is_mate() {
        return score;
    }

    if let Some(display_score) = chess_engine::syzygy::wdl_display_score(score.centipawns()) {
        return Score(display_score);
    }

    Score((score.centipawns() as f64 * 100.0 / WDL_B).round() as i32)
}

const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");
const ENGINE_BUILD_ID: Option<&str> = option_env!("ENDSPIEL_BUILD_ID");
const ENGINE_AUTHOR: &str = "Thilo Horstmann";

fn engine_name_string() -> String {
    match ENGINE_BUILD_ID {
        Some(build_id) if !build_id.is_empty() => {
            format!("{ENGINE_NAME} {ENGINE_VERSION} [{build_id}]")
        }
        _ => format!("{ENGINE_NAME} {ENGINE_VERSION}"),
    }
}

/// The main UCI protocol handler.
pub struct UciHandler {
    board: Board,
    engine: Engine,
    /// Handle to signal the search thread to stop.
    stop_handle: Arc<AtomicBool>,
    /// Join handle for the currently running search thread, if any.
    search_thread: Option<thread::JoinHandle<()>>,
    /// Whether the GUI has explicitly set the Hash size via setoption.
    /// If false, `handle_isready` will apply the default 4096 MB.
    hash_explicitly_set: bool,
    /// Whether to emit `wdl <win> <draw> <loss>` on each info line.
    show_wdl: bool,
    /// Number of best lines to search (MultiPV). 1 = normal.
    multi_pv: usize,
    /// True while a `go ponder` search is running and `ponderhit` has not
    /// yet arrived. While set, the search runs in `infinite` mode and the
    /// resulting move is held back until `ponderhit`/`stop`.
    ponder_active: bool,
    /// Move-time budget (ms) to apply once `ponderhit` converts the ongoing
    /// ponder search into our real search.
    ponder_alloc_ms: u64,
}

impl Default for UciHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl UciHandler {
    pub fn new() -> Self {
        let engine = Engine::new();
        let stop_handle = engine.stop_handle();
        Self {
            board: Board::starting_position(),
            engine,
            stop_handle,
            search_thread: None,
            hash_explicitly_set: false,
            show_wdl: false,
            multi_pv: 1,
            ponder_active: false,
            ponder_alloc_ms: 0,
        }
    }

    /// Pre-configure Syzygy tablebases (called before the UCI loop when
    /// `--syzygy <path>` is passed on the command line).
    pub fn set_syzygy(&mut self, path: &str) {
        match self.engine.set_syzygy_path(path) {
            Ok(()) => log::info!("Syzygy tablebases loaded from '{path}'"),
            Err(e) => log::error!("Failed to load Syzygy tablebases from '{path}': {e}"),
        }
    }

    /// Run the main UCI input loop reading from stdin.
    pub fn run(&mut self) {
        let stdin = io::stdin();
        let reader = stdin.lock();

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    log::error!("Error reading stdin: {e}");
                    break;
                }
            };

            log::debug!(">> {}", line);

            let Some(cmd) = UciCommand::parse(&line) else {
                continue;
            };

            match cmd {
                UciCommand::Quit => {
                    self.handle_stop();
                    break;
                }
                _ => self.handle_command(cmd),
            }
        }
    }

    fn handle_command(&mut self, cmd: UciCommand) {
        match cmd {
            UciCommand::Uci => self.handle_uci(),
            UciCommand::IsReady => self.handle_isready(),
            UciCommand::UciNewGame => self.handle_ucinewgame(),
            UciCommand::Position { fen, moves } => self.handle_position(fen, moves),
            UciCommand::Go(params) => self.handle_go(params),
            UciCommand::Stop => self.handle_stop(),
            UciCommand::SetOption { name, value } => self.handle_setoption(name, value),
            UciCommand::Debug(_) => { /* acknowledged, no action needed */ }
            UciCommand::Register => { /* no registration needed */ }
            UciCommand::PonderHit => self.handle_ponderhit(),
            UciCommand::Quit => unreachable!(),
        }
    }

    fn handle_uci(&self) {
        send_response(&UciResponse::Id {
            name: engine_name_string(),
            author: ENGINE_AUTHOR.to_string(),
        });
        // Advertise configurable options
        send_response(&UciResponse::Option(UciOptionDef {
            name: "Hash".to_string(),
            opt_type: UciOptionType::Spin {
                default: 256,
                min: 1,
                max: 131072,
            },
        }));
        send_response(&UciResponse::Option(UciOptionDef {
            name: "Threads".to_string(),
            opt_type: UciOptionType::Spin {
                default: self.engine.num_threads() as i64,
                min: 1,
                max: 256,
            },
        }));
        send_response(&UciResponse::Option(UciOptionDef {
            name: "Move Overhead".to_string(),
            opt_type: UciOptionType::Spin {
                default: self.engine.move_overhead_ms() as i64,
                min: 0,
                max: 5000,
            },
        }));
        send_response(&UciResponse::Option(UciOptionDef {
            name: "Ponder".to_string(),
            opt_type: UciOptionType::Check { default: false },
        }));
        send_response(&UciResponse::Option(UciOptionDef {
            name: "Slow Mover".to_string(),
            opt_type: UciOptionType::Spin {
                default: self.engine.slow_mover() as i64,
                min: 10,
                max: 300,
            },
        }));
        send_response(&UciResponse::Option(UciOptionDef {
            name: "Contempt".to_string(),
            opt_type: UciOptionType::Spin {
                default: self.engine.contempt() as i64,
                min: 0,
                max: 100,
            },
        }));
        send_response(&UciResponse::Option(UciOptionDef {
            name: "SingularExt".to_string(),
            opt_type: UciOptionType::Spin {
                default: self.engine.singular_ext_mode() as i64,
                min: 0,
                max: 2,
            },
        }));
        send_response(&UciResponse::Option(UciOptionDef {
            name: "BookFile".to_string(),
            opt_type: UciOptionType::String {
                default: String::new(), // empty = no book
            },
        }));
        send_response(&UciResponse::Option(UciOptionDef {
            name: "UseNNUE".to_string(),
            opt_type: UciOptionType::Check {
                default: true,
            },
        }));
        send_response(&UciResponse::Option(UciOptionDef {
            name: "EvalFile".to_string(),
            opt_type: UciOptionType::String {
                default: String::new(), // empty = use embedded net
            },
        }));
        send_response(&UciResponse::Option(UciOptionDef {
            name: "SyzygyPath".to_string(),
            opt_type: UciOptionType::String {
                default: String::new(), // empty = no tablebases
            },
        }));
        send_response(&UciResponse::Option(UciOptionDef {
            name: "MultiPV".to_string(),
            opt_type: UciOptionType::Spin {
                default: 1,
                min: 1,
                max: 256,
            },
        }));
        send_response(&UciResponse::Option(UciOptionDef {
            name: "UCI_ShowWDL".to_string(),
            opt_type: UciOptionType::Check { default: false },
        }));
        // ── SPSA-tunable search parameters ────────────────────────────────
        let t = self.engine.tune();
        for (name, default, min, max) in [
            ("LmrBase",        t.lmr_base,          10,  200),
            ("LmrDiv",         t.lmr_div,           100, 400),
            ("HistLmrDiv",     t.hist_lmr_div,      500, 20000),
            ("RfpMarginImp",   t.rfp_margin_imp,    10,  300),
            ("RfpMarginNoImp", t.rfp_margin_noimp,  10,  300),
            ("FutMarginImp",   t.fut_margin_imp,    10,  300),
            ("FutMarginNoImp", t.fut_margin_noimp,  10,  300),
            ("SeeQuietMargin", t.see_quiet_margin,  10,  200),
        ] {
            send_response(&UciResponse::Option(UciOptionDef {
                name: name.to_string(),
                opt_type: UciOptionType::Spin {
                    default: default as i64,
                    min: min as i64,
                    max: max as i64,
                },
            }));
        }
        let eval_mode = if self.engine.use_nnue() {
            "NNUE (HalfKP 704\u{00d7}32\u{2192}768\u{00d7}2\u{2192}1 SCReLU)".to_string()
        } else {
            "HCE (no trained NNUE net)".to_string()
        };
        send_response(&UciResponse::Info(UciInfo {
            string: Some(format!("Eval: {eval_mode}")),
            ..UciInfo::default()
        }));
        send_response(&UciResponse::UciOk);
    }

    fn handle_isready(&mut self) {
        // Wait for any pending search to finish before responding.
        self.wait_for_search();
        // Allocate the TT at the configured size now that all setoptions
        // have been processed.  If the GUI never sent setoption Hash, apply
        // the default 4096 MB here (deferred from Engine::new to avoid
        // allocating 4 GB × concurrency at startup).
        if !self.hash_explicitly_set {
            self.engine.set_hash_mb(256);
            log::info!("Hash defaulting to 256 MB (no setoption received)");
        }
        send_response(&UciResponse::ReadyOk);
    }

    fn handle_ucinewgame(&mut self) {
        self.handle_stop();
        self.board = Board::starting_position();
        self.engine.clear_tt();
    }

    fn handle_position(&mut self, fen: Option<String>, moves: Vec<String>) {
        // Set up the base position
        let mut board = if let Some(fen) = fen {
            match Board::from_fen(&fen) {
                Ok(b) => b,
                Err(e) => {
                    log::error!("Invalid FEN '{}': {}", fen, e);
                    return;
                }
            }
        } else {
            Board::starting_position()
        };

        // Apply each move in sequence
        for (i, move_str) in moves.iter().enumerate() {
            let Some(m) = find_legal_move(&board, move_str) else {
                log::error!(
                    "handle_position: could not apply move {} (index {}) — board NOT updated",
                    move_str, i
                );
                return;
            };
            board.make_move(m);
        }

        self.board = board;
    }

    fn handle_go(&mut self, params: GoParams) {
        // Stop any existing search first
        self.handle_stop();

        let mut search_params = SearchParams {
            max_depth: params.depth.unwrap_or(64),
            max_nodes: params.nodes,
            move_time_ms: params.movetime,
            white_time_ms: params.wtime,
            black_time_ms: params.btime,
            white_inc_ms: params.winc,
            black_inc_ms: params.binc,
            moves_to_go: params.movestogo,
            infinite: params.infinite,
            use_nnue: self.engine.use_nnue(),
            move_overhead_ms: self.engine.move_overhead_ms(),
            slow_mover: self.engine.slow_mover(),
            contempt: self.engine.contempt(),
            singular_ext_mode: self.engine.singular_ext_mode(),
            multi_pv: self.multi_pv,
            tune: self.engine.tune().clone(),
        };

        // Pondering: the board is the predicted position (opponent's expected
        // reply already applied). Compute the move-time budget we *would* spend
        // on it, then run the search in `infinite` mode so it never returns on
        // its own — the move is held back until `ponderhit` (which starts the
        // clock, see handle_ponderhit) or `stop` (ponder-miss / quit).
        let is_ponder = params.ponder;
        if is_ponder {
            self.ponder_alloc_ms =
                chess_engine::search::allocated_move_time_ms(&search_params, &self.board)
                    .unwrap_or(0);
            search_params.infinite = true;
            self.ponder_active = true;
        } else {
            self.ponder_active = false;
        }

        // Clone what we need for the search thread
        let board = self.board.clone();

        // Share the TT from the main engine so entries persist across searches
        // (important for analysis mode where many positions are evaluated
        // sequentially). Create a fresh stop handle for this search.
        let tt = self.engine.shared_tt();
        let num_threads = self.engine.num_threads();
        let net = Arc::clone(self.engine.nnue_net());
        let root_tb_solution = if let Some(tb) = self.engine.take_syzygy_tb() {
            let solution = chess_engine::syzygy::solve_root_position(&tb, &board, 128);
            self.engine.set_syzygy_tb(Some(tb));
            solution
        } else {
            None
        };
        let syzygy_tb = self.engine.syzygy_tb().cloned();
        let book = self.engine.book();
        let show_wdl = self.show_wdl;
        let multi_pv = self.multi_pv;
        let stop = Arc::new(AtomicBool::new(false));
        self.stop_handle = Arc::clone(&stop);

        let handle = thread::Builder::new()
            .stack_size(4 * 1024 * 1024) // 4 MB – match helper thread stack size
            .spawn(move || {
            let search_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let info_cb: InfoCallback = Box::new(move |info: &SearchInfo| {
                    let nps = (info.nodes * 1000).checked_div(info.time_ms);
                    // WDL uses the raw score (sigmoid formula is calibrated against it).
                    let wdl = if show_wdl && !info.score.is_mate() {
                        Some(score_to_wdl(info.score.centipawns()))
                    } else {
                        None
                    };
                    // Normalize displayed cp score: divide raw score by WDL_B/100 so that
                    // 100 displayed cp ≈ 1 "WDL pawn" (consistent with Stockfish's convention).
                    // Mate scores are passed through unchanged.
                    let displayed_score = normalize_display_score(info.score);
                    let uci_info = UciInfo {
                        depth: Some(info.depth),
                        seldepth: Some(info.seldepth),
                        multipv: if multi_pv > 1 { Some(info.multipv_line) } else { None },
                        score: Some(displayed_score),
                        nodes: Some(info.nodes),
                        time: Some(info.time_ms),
                        pv: info.pv.clone(),
                        hashfull: Some(info.hashfull),
                        nps,
                        wdl,
                        string: None,
                    };
                    send_response(&UciResponse::Info(uci_info));
                });

                let pool = chess_engine::threads::ThreadPool::new(num_threads);
                pool.search(&board, &search_params, &stop, &tt, Some(info_cb), &net, syzygy_tb, root_tb_solution, book)
            }));

            let result = match search_result {
                Ok(r) => r,
                Err(e) => {
                    log::error!("Search panicked: {:?}", e);
                    chess_engine::SearchResult {
                        best_move: Move::NULL,
                        score: chess_common::Score(0),
                        depth: 0,
                        nodes: 0,
                        pv: vec![],
                    }
                }
            };

            // Pondering: do not surrender the move until we are told to play
            // it. `ponderhit` converts this into a timed search that trips
            // `stop` after the allocated budget; a `stop` (ponder-miss or quit)
            // trips it immediately. This guarantees we never emit a move
            // mid-ponder, even if the search resolves the position early.
            if is_ponder {
                while !stop.load(Ordering::SeqCst) {
                    thread::sleep(Duration::from_millis(2));
                }
            }

            let ponder_move = if result.pv.len() > 1 {
                Some(result.pv[1])
            } else {
                None
            };

            send_response(&UciResponse::BestMove {
                best: result.best_move,
                ponder: ponder_move,
            });
        }).expect("failed to spawn search thread");

        self.search_thread = Some(handle);
    }

    fn handle_stop(&mut self) {
        // A `stop` ends any pondering (ponder-miss or quit): the held-back
        // move is released as soon as the stop flag is observed.
        self.ponder_active = false;
        // Signal the search to stop
        self.stop_handle.store(true, Ordering::SeqCst);
        // Wait for the search thread to finish
        self.wait_for_search();
    }

    /// `ponderhit`: the opponent played the move we were pondering on, so the
    /// ongoing (infinite) ponder search becomes our real search. Start the
    /// move clock now by tripping `stop` after the allocated budget; the
    /// search thread then releases its best move. With no budget known
    /// (e.g. `go ponder infinite`), play the pondered result immediately.
    fn handle_ponderhit(&mut self) {
        if !self.ponder_active {
            return;
        }
        self.ponder_active = false;
        let stop = Arc::clone(&self.stop_handle);
        let alloc = self.ponder_alloc_ms;
        if alloc == 0 {
            stop.store(true, Ordering::SeqCst);
            return;
        }
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(alloc));
            stop.store(true, Ordering::SeqCst);
        });
    }

    fn wait_for_search(&mut self) {
        if let Some(handle) = self.search_thread.take()
            && let Err(e) = handle.join()
        {
            log::error!("Search thread panicked: {:?}", e);
        }
    }

    fn handle_setoption(&mut self, name: String, value: Option<String>) {
        log::info!("setoption name={} value={:?}", name, value);
        match name.to_lowercase().as_str() {
            "hash" => {
                if let Some(v) = value
                    && let Ok(mb) = v.trim().parse::<usize>()
                {
                    self.hash_explicitly_set = true;
                    self.engine.set_hash_mb(mb);
                    log::info!("Hash set to {} MB", self.engine.hash_mb());
                }
            }
            "threads" => {
                if let Some(v) = value
                    && let Ok(t) = v.trim().parse::<usize>()
                {
                    self.engine.set_threads(t);
                    log::info!("Threads set to {}", self.engine.num_threads());
                }
            }
            "ponder" => {
                // Capability flag only. Pondering is driven by the
                // `go ponder` / `ponderhit` commands, so there is no
                // engine state to set here.
            }
            "move overhead" => {
                if let Some(v) = value
                    && let Ok(ms) = v.trim().parse::<u64>()
                {
                    self.engine.set_move_overhead(ms);
                    log::info!("Move Overhead set to {} ms", self.engine.move_overhead_ms());
                }
            }
            "slow mover" => {
                if let Some(v) = value
                    && let Ok(pct) = v.trim().parse::<u64>()
                {
                    self.engine.set_slow_mover(pct);
                    log::info!("Slow Mover set to {}%", self.engine.slow_mover());
                }
            }
            "contempt" => {
                if let Some(v) = value
                    && let Ok(cp) = v.trim().parse::<i32>()
                {
                    self.engine.set_contempt(cp);
                    log::info!("Contempt set to {} cp", self.engine.contempt());
                }
            }
            "singularext" => {
                if let Some(v) = value
                    && let Ok(mode) = v.trim().parse::<u8>()
                {
                    self.engine.set_singular_ext_mode(mode);
                    log::info!("SingularExt set to {}", self.engine.singular_ext_mode());
                }
            }
            "bookfile" => {
                if let Some(path) = value {
                    match self.engine.set_book_file(path.trim()) {
                        Ok(()) => {
                            log::info!("BookFile set to '{}'", path.trim());
                            send_response(&UciResponse::Info(UciInfo {
                                string: Some(format!("BookFile loaded from '{}'", path.trim())),
                                ..UciInfo::default()
                            }));
                        }
                        Err(e) => {
                            log::error!("{e}");
                            send_response(&UciResponse::Info(UciInfo {
                                string: Some(format!("BookFile ERROR: {e}")),
                                ..UciInfo::default()
                            }));
                        }
                    }
                }
            }
            "usennue" => {
                if let Some(v) = value {
                    let enabled = v.trim().eq_ignore_ascii_case("true");
                    self.engine.set_use_nnue(enabled);
                    log::info!("UseNNUE set to {}", self.engine.use_nnue());
                }
            }
            "evalfile" => {
                if let Some(path) = value {
                    let path = path.trim();
                    match self.engine.set_nnue_file(path) {
                        Ok(()) => log::info!("EvalFile set to '{path}'"),
                        Err(e) => log::error!("Failed to load EvalFile '{path}': {e}"),
                    }
                }
            }
            "syzygypath" => {
                if let Some(path) = value {
                    match self.engine.set_syzygy_path(path.trim()) {
                        Ok(()) => log::info!(
                            "SyzygyPath set to '{}' (max {} pieces)",
                            path.trim(),
                            self.engine.syzygy_tb().map_or(0, |tb| tb.max_pieces())
                        ),
                        Err(e) => log::error!("Failed to load SyzygyPath '{}': {e}", path.trim()),
                    }
                }
            }
            "multipv" => {
                if let Some(v) = value
                    && let Ok(n) = v.trim().parse::<usize>()
                {
                    self.multi_pv = n.clamp(1, 256);
                    log::info!("MultiPV set to {}", self.multi_pv);
                }
            }
            "uci_showwdl" => {
                if let Some(v) = value {
                    self.show_wdl = v.trim().eq_ignore_ascii_case("true");
                    log::info!("UCI_ShowWDL set to {}", self.show_wdl);
                }
            }
            // SPSA-tunable search parameters
            "lmrbase" => {
                if let Some(v) = value && let Ok(n) = v.trim().parse::<i32>() {
                    self.engine.set_tune_param("lmr_base", n);
                }
            }
            "lmrdiv" => {
                if let Some(v) = value && let Ok(n) = v.trim().parse::<i32>() {
                    self.engine.set_tune_param("lmr_div", n);
                }
            }
            "histlmrdiv" => {
                if let Some(v) = value && let Ok(n) = v.trim().parse::<i32>() {
                    self.engine.set_tune_param("hist_lmr_div", n);
                }
            }
            "rfpmargimp" => {
                if let Some(v) = value && let Ok(n) = v.trim().parse::<i32>() {
                    self.engine.set_tune_param("rfp_margin_imp", n);
                }
            }
            "rfpmarginnoimp" => {
                if let Some(v) = value && let Ok(n) = v.trim().parse::<i32>() {
                    self.engine.set_tune_param("rfp_margin_noimp", n);
                }
            }
            "futmargimp" => {
                if let Some(v) = value && let Ok(n) = v.trim().parse::<i32>() {
                    self.engine.set_tune_param("fut_margin_imp", n);
                }
            }
            "futmarginnoimp" => {
                if let Some(v) = value && let Ok(n) = v.trim().parse::<i32>() {
                    self.engine.set_tune_param("fut_margin_noimp", n);
                }
            }
            "seequietmargin" => {
                if let Some(v) = value && let Ok(n) = v.trim().parse::<i32>() {
                    self.engine.set_tune_param("see_quiet_margin", n);
                }
            }
            _ => {
                log::debug!("Unknown option: {}", name);
            }
        }
    }
}

/// Find the legal move matching a UCI string in the current position.
///
/// We generate all legal moves and find the one matching the from/to squares
/// and promotion piece. This ensures the move flag (capture, en passant,
/// castling, etc.) is set correctly.
fn find_legal_move(board: &Board, uci_str: &str) -> Option<Move> {
    let parsed = Move::from_uci(uci_str)?;
    let legal_moves = chess_core::generate_legal_moves(board);

    legal_moves.as_slice().iter().find(|&&legal| {
        legal.from_sq() == parsed.from_sq()
            && legal.to_sq() == parsed.to_sq()
            && legal.flag().promotion_piece() == parsed.flag().promotion_piece()
    }).copied()
}

/// Send a UCI response to stdout.
///
/// Explicitly flushes after each response so GUIs receive output immediately,
/// even when stdout is piped (block-buffered rather than line-buffered).
fn send_response(response: &UciResponse) {
    use std::io::Write;
    let text = response.to_string();
    for line in text.lines() {
        println!("{line}");
        log::debug!("<< {line}");
    }
    let _ = std::io::stdout().flush();
}

#[cfg(test)]
mod tests {
    use super::normalize_display_score;
    use chess_common::Score;
    use chess_engine::syzygy::{TB_LOSS_SCORE, TB_WIN_SCORE};

    #[test]
    fn normal_scores_are_normalized_for_display() {
        assert_eq!(normalize_display_score(Score(262)), Score(100));
        assert_eq!(normalize_display_score(Score(-131)), Score(-50));
    }

    #[test]
    fn syzygy_scores_are_mapped_to_human_facing_display_values() {
        assert_eq!(normalize_display_score(Score(TB_WIN_SCORE)), Score(3_000));
        assert_eq!(normalize_display_score(Score(TB_LOSS_SCORE)), Score(-3_000));
        assert_eq!(normalize_display_score(Score(2)), Score(20));
        assert_eq!(normalize_display_score(Score(-2)), Score(-20));
        assert_eq!(normalize_display_score(Score(0)), Score(0));
    }
}
