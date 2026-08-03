use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use chess_common::{Board, Move, Score};
use chess_engine::search::PersistentHistory;
use chess_engine::{Engine, InfoCallback, SearchInfo, SearchParams};

use crate::protocol::{GoParams, UciCommand, UciInfo, UciOptionDef, UciOptionType, UciResponse};

/// WDL normalization parameters for P(win | score) = sigmoid((score − a) / b).
/// Re-fit against training data after each net promotion and update these two
/// constants with the printed values.
const WDL_A: f64 = 9.6;
const WDL_B: f64 = 262.0;

const ENGINE_NAME: &str = "Endspiel";

/// Convert a centipawn score to WDL in millipawns (0–1000, summing to 1000).
fn score_to_wdl(cp: i32) -> (u32, u32, u32) {
    let s = cp as f64;
    let p_win = 1.0 / (1.0 + (-(s - WDL_A) / WDL_B).exp());
    let p_loss = 1.0 / (1.0 + ((s + WDL_A) / WDL_B).exp());
    let win = (p_win * 1000.0).round() as u32;
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

/// Final-depth MultiPV lines captured from the info stream for the opening
/// variety draw: (depth, [(raw score, pv)]). Reset whenever a deeper
/// iteration starts reporting.
type VarietyLines = Arc<Mutex<(u8, Vec<(i32, Vec<Move>)>)>>;

/// The main UCI protocol handler.
pub struct UciHandler {
    board: Board,
    engine: Engine,
    /// Handle to signal the search thread to stop.
    stop_handle: Arc<AtomicBool>,
    /// Join handle for the currently running search thread, if any.
    search_thread: Option<thread::JoinHandle<()>>,
    /// Whether the GUI has explicitly set the Hash size via setoption.
    /// If false, `handle_isready` will apply the device-adaptive default
    /// (see `chess_common::platform::default_hash_mb`).
    hash_explicitly_set: bool,
    /// Whether to emit `wdl <win> <draw> <loss>` on each info line.
    show_wdl: bool,
    /// Number of best lines to search (MultiPV). 1 = normal.
    multi_pv: usize,
    /// Opening variety window in centipawns (0 = off). See
    /// [`SearchParams::opening_variety`].
    opening_variety: i32,
    /// True while a `go ponder` search is running and `ponderhit` has not
    /// yet arrived. While set, the search runs in `infinite` mode and the
    /// resulting move is held back until `ponderhit`/`stop`.
    ponder_active: bool,
    /// Shared with the search thread: true from `go ponder` until
    /// `ponderhit`/`stop`. The thread's hold-back loop watches this (not just
    /// the stop flag), so a search that finished while pondering releases its
    /// move the moment the hit arrives instead of sleeping out the remaining
    /// budget — after `ponderhit`, a finished search has nothing left to buy
    /// with the clock (mate-break and TB-restricted roots finish in
    /// milliseconds). Fresh per `go`, like `stop_handle`.
    ponder_pending: Arc<AtomicBool>,
    /// Soft budget (ms) the current ponder search computed from the clocks at
    /// `go ponder`. Only consulted at `ponderhit` to detect the no-budget case
    /// (0 = no clock info → play the pondered result at once); a nonzero
    /// budget is enforced by the search's own time manager, which activates
    /// when `ponder_pending` clears and guarantees a bounded amount of fresh
    /// thinking on a healthy classical clock.
    ponder_alloc_ms: u64,
    /// Game-level history/correction tables, persisted across moves and shared
    /// into the per-`go` search thread. Reset on `ucinewgame`.
    history: Arc<Mutex<PersistentHistory>>,
    /// Why the most recent `position` command was rejected. While set, `go`
    /// returns `bestmove 0000` instead of searching the previous (stale) board.
    position_error: Option<String>,
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
            opening_variety: 0,
            ponder_active: false,
            ponder_pending: Arc::new(AtomicBool::new(false)),
            ponder_alloc_ms: 0,
            history: Arc::new(Mutex::new(PersistentHistory::new())),
            position_error: None,
        }
    }

    /// Pre-configure Syzygy tablebases (called before the UCI loop when
    /// `--syzygy <path>` is passed on the command line).
    pub fn set_syzygy(&mut self, path: &str) {
        match self.engine.set_syzygy_path(path) {
            Ok(()) => log::info!(
                "Syzygy tablebases loaded from '{path}' (max {} pieces)",
                self.engine.syzygy_tb().map_or(0, |tb| tb.max_pieces())
            ),
            Err(e) => log::error!("{e}"),
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
                default: chess_common::platform::default_hash_mb(self.engine.num_threads()) as i64,
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
            opt_type: UciOptionType::Check { default: true },
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
            name: "OpeningVariety".to_string(),
            opt_type: UciOptionType::Spin {
                default: 0,
                min: 0,
                max: 200,
            },
        }));
        send_response(&UciResponse::Option(UciOptionDef {
            name: "UCI_ShowWDL".to_string(),
            opt_type: UciOptionType::Check { default: false },
        }));
        // ── SPSA-tunable search parameters ────────────────────────────────
        let t = self.engine.tune();
        for (name, default, min, max) in [
            ("LmrBase", t.lmr_base, 10, 200),
            ("LmrDiv", t.lmr_div, 100, 400),
            ("HistLmrDiv", t.hist_lmr_div, 500, 20000),
            ("RfpMarginImp", t.rfp_margin_imp, 10, 300),
            ("RfpMarginNoImp", t.rfp_margin_noimp, 10, 300),
            ("FutMarginImp", t.fut_margin_imp, 10, 300),
            ("FutMarginNoImp", t.fut_margin_noimp, 10, 300),
            ("SeeQuietMargin", t.see_quiet_margin, 10, 200),
            ("CorrHistMult", t.corrhist_mult, 0, 300),
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
            "NNUE (state-aware HalfKP 785\u{00d7}32\u{2192}(1024 pairwise 512)\u{00d7}2\u{2192}16\u{2192}32\u{2192}1)".to_string()
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
        // `isready` may arrive mid-search or mid-ponder and must answer
        // `readyok` without stopping anything (UCI spec) — and a pondering
        // thread parks until `ponderhit`/`stop`, so joining it here would
        // deadlock. Only reap a thread that has already finished; while one
        // is running, skip straight to the ping reply and leave the deferred
        // TT sizing for a quiet moment (resizing under a live search is not
        // safe).
        let search_running = self
            .search_thread
            .as_ref()
            .is_some_and(|h| !h.is_finished());
        if !search_running {
            self.wait_for_search();
            // Allocate the TT at the configured size now that all setoptions
            // have been processed.  If the GUI never sent setoption Hash, apply
            // the device-adaptive default here (deferred from Engine::new to avoid
            // allocating it × concurrency at startup).
            if !self.hash_explicitly_set {
                let mb = chess_common::platform::default_hash_mb(self.engine.num_threads());
                self.engine.set_hash_mb(mb);
                log::info!(
                    "Hash defaulting to {mb} MB for {} threads (no setoption received)",
                    self.engine.num_threads()
                );
            }
        }
        send_response(&UciResponse::ReadyOk);
    }

    fn handle_ucinewgame(&mut self) {
        self.handle_stop();
        self.board = Board::starting_position();
        self.position_error = None;
        self.engine.clear_tt();
        // Fresh game: drop accumulated history/corrections. handle_stop() above
        // has joined any search thread, so the lock is free.
        *self.history.lock().unwrap_or_else(|p| p.into_inner()) = PersistentHistory::new();
    }

    fn handle_position(&mut self, fen: Option<String>, moves: Vec<String>) {
        match build_position(fen.as_deref(), &moves) {
            Ok(board) => {
                self.board = board;
                self.position_error = None;
            }
            Err(error) => {
                log::error!("Rejected UCI position: {error}");
                send_response(&UciResponse::Info(UciInfo {
                    string: Some(format!("Invalid position: {error}")),
                    ..UciInfo::default()
                }));
                self.position_error = Some(error);
            }
        }
    }

    fn handle_go(&mut self, params: GoParams) {
        // Stop any existing search first
        self.handle_stop();

        // Never search a stale board after rejecting the latest `position`
        // command. A null best move is the UCI-safe response and keeps the
        // process alive for a later valid position or `ucinewgame`.
        if let Some(error) = &self.position_error {
            send_response(&UciResponse::Info(UciInfo {
                string: Some(format!("Cannot search invalid position: {error}")),
                ..UciInfo::default()
            }));
            send_response(&UciResponse::BestMove {
                best: Move::NULL,
                ponder: None,
            });
            return;
        }

        // Opening variety: bookless play is deterministic (same eval => same
        // game). When enabled, opening searches run with a widened MultiPV
        // (the existing machinery) and the final move is drawn at random from
        // the lines scoring within the window of the best. The men >= 24
        // guard keeps this out of endgames and TB positions even for bare-FEN
        // probes, where game_ply reads 0.
        let opening_variety = self.opening_variety;
        let variety_active = opening_variety > 0
            && self.board.position_history.len() < 8
            && self.board.all_occupancy().count() >= 24;
        let effective_multi_pv = if variety_active {
            self.multi_pv.max(4)
        } else {
            self.multi_pv
        };

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
            multi_pv: effective_multi_pv,
            tune: self.engine.tune().clone(),
            ponder: None,
        };

        // Pondering: the board is the predicted position (opponent's expected
        // reply already applied). The search runs with its normal soft/hard
        // budgets computed from the clocks, but the shared `ponder_pending`
        // flag suspends every time-based stop while it is set — the search
        // deepens freely on the opponent's clock and the move is held back
        // until `ponderhit` (which clears the flag and hands control to the
        // regular time manager, difficulty adaptation included) or `stop`
        // (ponder-miss / quit). Pondered time is credited against the budget,
        // but a hit still receives a small phase-aware amount of fresh thought.
        let is_ponder = params.ponder;
        self.ponder_pending = Arc::new(AtomicBool::new(is_ponder));
        let ponder_pending = Arc::clone(&self.ponder_pending);
        if is_ponder {
            // Kept only to decide at `ponderhit` whether the search has any
            // budget to manage (0 = no clock info -> play the move at once).
            self.ponder_alloc_ms =
                chess_engine::search::allocated_move_time_ms(&search_params, &self.board)
                    .unwrap_or(0);
            search_params.ponder = Some(Arc::clone(&self.ponder_pending));
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
        let root_tb_ranking = if let Some(tb) = self.engine.take_syzygy_tb() {
            let ranking = chess_engine::syzygy::rank_root_moves(&tb, &board);
            self.engine.set_syzygy_tb(Some(tb));
            ranking
        } else {
            None
        };
        let syzygy_tb = self.engine.syzygy_tb().cloned();
        let book = self.engine.book();
        let show_wdl = self.show_wdl;
        let multi_pv = effective_multi_pv;
        let variety_lines: VarietyLines = Arc::new(Mutex::new((0, Vec::new())));
        let variety_lines_cb = Arc::clone(&variety_lines);
        let stop = Arc::new(AtomicBool::new(false));
        self.stop_handle = Arc::clone(&stop);
        let history = Arc::clone(&self.history);

        let handle = thread::Builder::new()
            .stack_size(4 * 1024 * 1024) // 4 MB – match helper thread stack size
            .spawn(move || {
                let search_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let info_cb: InfoCallback = Box::new(move |info: &SearchInfo| {
                        {
                            let mut vl = variety_lines_cb
                                .lock()
                                .unwrap_or_else(|p| p.into_inner());
                            if info.depth > vl.0 {
                                *vl = (info.depth, Vec::new());
                            }
                            if info.depth == vl.0 && !info.pv.is_empty() {
                                vl.1.push((info.score.0, info.pv.clone()));
                            }
                        }
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
                            multipv: if multi_pv > 1 {
                                Some(info.multipv_line)
                            } else {
                                None
                            },
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
                    // Hold the game-level history across this search so corrections
                    // accumulate over the game. Recover from a poisoned lock (a prior
                    // search panic) rather than propagating the panic — the tables are
                    // just stats, never unsafe. The guard is released here, before the
                    // ponder-wait below.
                    let mut guard = history.lock().unwrap_or_else(|p| p.into_inner());
                    pool.search(
                        &board,
                        &search_params,
                        &stop,
                        &tt,
                        Some(info_cb),
                        &net,
                        syzygy_tb,
                        root_tb_ranking,
                        book,
                        Some(&mut guard),
                    )
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

                // Opening variety draw: uniform pick among the final-depth
                // MultiPV lines scoring within the window of the best. Mate
                // scores are never randomized (neither giving nor defending).
                let mut result = result;
                if variety_active && !result.score.is_mate() && !result.best_move.is_null() {
                    let vl = variety_lines.lock().unwrap_or_else(|p| p.into_inner());
                    let top = vl.1.iter().map(|(s, _)| *s).max().unwrap_or(result.score.0);
                    let eligible: Vec<&(i32, Vec<Move>)> = vl
                        .1
                        .iter()
                        .filter(|(s, pv)| {
                            !pv.is_empty()
                                && !chess_common::Score(*s).is_mate()
                                && *s >= top - opening_variety
                        })
                        .collect();
                    if eligible.len() > 1 {
                        let seed = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| (u64::from(d.subsec_nanos()) << 20) ^ d.as_secs())
                            .unwrap_or(0x9E37_79B9_7F4A_7C15);
                        let mut x = seed | 1;
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        let pick = eligible[(x as usize) % eligible.len()];
                        if pick.1[0] != result.best_move {
                            log::info!(
                                "opening variety: playing {} ({}cp) over {} ({}cp), {} candidates in {}cp window",
                                pick.1[0].to_uci(),
                                pick.0,
                                result.best_move.to_uci(),
                                result.score.0,
                                eligible.len(),
                                opening_variety
                            );
                        }
                        result.score = chess_common::Score(pick.0);
                        result.pv = pick.1.clone();
                        result.best_move = pick.1[0];
                    }
                }

                // Pondering: do not surrender the move until we are told to play
                // it (`ponderhit` or `stop`, which clear `ponder_pending`). This
                // guarantees we never emit a move mid-ponder, even if the search
                // resolves the position early.
                //
                // The hold watches `ponder_pending` and deliberately NOT the
                // shared stop flag: pool.search stores `stop = true` itself to
                // terminate its helper threads whenever the main thread finishes
                // early (mate break, TB-restricted root, depth cap), and a hold
                // keyed on `stop` then leaks the bestmove mid-ponder — a UCI
                // protocol violation the GUI answers with a desynced ponder
                // state. Conversely, holding until `stop` alone would sit on a
                // finished search for the whole remaining budget after
                // `ponderhit`. Watching the handler-owned flag fixes both: the
                // move plays the instant the GUI asks for it and never before.
                if is_ponder {
                    while ponder_pending.load(Ordering::SeqCst) {
                        thread::sleep(Duration::from_millis(2));
                    }
                }

                // Advertise the expected reply (PV[1]) as the ponder move, but only
                // if it is actually legal after best_move. PV reconstruction can
                // yield an illegal continuation (e.g. a TT hash collision); sending
                // an illegal `ponder` move makes a GUI reject it ("invalid ponder
                // move") and can desync/hang its ponder state machine.
                let ponder_move = validated_ponder_move(&board, result.best_move, &result.pv);

                send_response(&UciResponse::BestMove {
                    best: result.best_move,
                    ponder: ponder_move,
                });
            })
            .expect("failed to spawn search thread");

        self.search_thread = Some(handle);
    }

    fn handle_stop(&mut self) {
        // A `stop` ends any pondering (ponder-miss or quit): the held-back
        // move is released as soon as the stop flag is observed.
        self.ponder_active = false;
        self.ponder_pending.store(false, Ordering::SeqCst);
        // Signal the search to stop
        self.stop_handle.store(true, Ordering::SeqCst);
        // Wait for the search thread to finish
        self.wait_for_search();
    }

    /// `ponderhit`: the opponent played the move we were pondering on, so the
    /// ongoing ponder search becomes our real search. Clearing `ponder_pending`
    /// does two things at once: the hold-back loop releases a search that
    /// already finished while pondering (mate break, TB-restricted root,
    /// forced move) so it plays immediately, and a still-running search
    /// switches to its normal soft/hard time management — difficulty
    /// adaptation included, which the old flat `alloc - elapsed` timer
    /// disconnected (a pondered move could never think longer on a collapsing
    /// eval, nor shorter on a trivial one). Pondered time remains credited, but
    /// on a healthy classical clock the search also guarantees a small bounded
    /// amount of fresh thinking after the hit.
    ///
    /// With no budget known (e.g. `go ponder infinite` or no clock info),
    /// there is nothing for the time manager to enforce — play the pondered
    /// result immediately, as before.
    fn handle_ponderhit(&mut self) {
        if !self.ponder_active {
            return;
        }
        self.ponder_active = false;
        self.ponder_pending.store(false, Ordering::SeqCst);
        if self.ponder_alloc_ms == 0 {
            self.stop_handle.store(true, Ordering::SeqCst);
        }
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
                    let path = path.trim();
                    // Always emit a UCI `info string` for the load result: it is visible
                    // in the GUI / bot log regardless of RUST_LOG, and a silent failure
                    // here (no tablebases) leaves the engine playing endgames on eval
                    // alone — exactly the failure that drew a won KQP-vs-KP game.
                    match self.engine.set_syzygy_path(path) {
                        Ok(()) => {
                            let max_pieces =
                                self.engine.syzygy_tb().map_or(0, |tb| tb.max_pieces());
                            log::info!("SyzygyPath set to '{path}' (max {max_pieces} pieces)");
                            send_response(&UciResponse::Info(UciInfo {
                                string: Some(format!(
                                    "Syzygy tablebases loaded from '{path}' (max {max_pieces} pieces)"
                                )),
                                ..UciInfo::default()
                            }));
                        }
                        Err(e) => {
                            log::error!("Failed to load SyzygyPath '{path}': {e}");
                            send_response(&UciResponse::Info(UciInfo {
                                string: Some(format!("SyzygyPath ERROR: {e}")),
                                ..UciInfo::default()
                            }));
                        }
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
            "openingvariety" => {
                if let Some(v) = value
                    && let Ok(n) = v.trim().parse::<i32>()
                {
                    self.opening_variety = n.clamp(0, 200);
                    log::info!("OpeningVariety set to {}", self.opening_variety);
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
                if let Some(v) = value
                    && let Ok(n) = v.trim().parse::<i32>()
                {
                    self.engine.set_tune_param("lmr_base", n);
                }
            }
            "lmrdiv" => {
                if let Some(v) = value
                    && let Ok(n) = v.trim().parse::<i32>()
                {
                    self.engine.set_tune_param("lmr_div", n);
                }
            }
            "histlmrdiv" => {
                if let Some(v) = value
                    && let Ok(n) = v.trim().parse::<i32>()
                {
                    self.engine.set_tune_param("hist_lmr_div", n);
                }
            }
            "rfpmarginimp" => {
                if let Some(v) = value
                    && let Ok(n) = v.trim().parse::<i32>()
                {
                    self.engine.set_tune_param("rfp_margin_imp", n);
                }
            }
            "rfpmarginnoimp" => {
                if let Some(v) = value
                    && let Ok(n) = v.trim().parse::<i32>()
                {
                    self.engine.set_tune_param("rfp_margin_noimp", n);
                }
            }
            "futmarginimp" => {
                if let Some(v) = value
                    && let Ok(n) = v.trim().parse::<i32>()
                {
                    self.engine.set_tune_param("fut_margin_imp", n);
                }
            }
            "futmarginnoimp" => {
                if let Some(v) = value
                    && let Ok(n) = v.trim().parse::<i32>()
                {
                    self.engine.set_tune_param("fut_margin_noimp", n);
                }
            }
            "seequietmargin" => {
                if let Some(v) = value
                    && let Ok(n) = v.trim().parse::<i32>()
                {
                    self.engine.set_tune_param("see_quiet_margin", n);
                }
            }
            "corrhistmult" => {
                if let Some(v) = value
                    && let Ok(n) = v.trim().parse::<i32>()
                {
                    self.engine.set_tune_param("corrhist_mult", n);
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
/// Derive the ponder move (the expected opponent reply) to advertise alongside
/// `best`, returning it only when it is genuinely playable.
///
/// The ponder move is the second PV element. PV reconstruction can produce an
/// illegal continuation (e.g. a transposition-table hash collision), and an
/// illegal `ponder` move corrupts a GUI's ponder state machine — lichess-bot
/// logs "Engine sent invalid ponder move" and can desync/hang pondering,
/// occasionally losing on time. Returns `None` unless `best` is real, the PV
/// starts with `best`, and `pv[1]` is legal in the position after `best`.
fn validated_ponder_move(board: &Board, best: Move, pv: &[Move]) -> Option<Move> {
    if best == Move::NULL || pv.len() < 2 || pv[0] != best {
        return None;
    }
    let reply = pv[1];
    let mut after = board.clone();
    after.make_move(best);
    chess_core::is_legal_move(&after, reply).then_some(reply)
}

fn find_legal_move(board: &Board, uci_str: &str) -> Option<Move> {
    let parsed = Move::from_uci(uci_str)?;
    let legal_moves = chess_core::generate_legal_moves(board);

    legal_moves
        .as_slice()
        .iter()
        .find(|&&legal| {
            legal.from_sq() == parsed.from_sq()
                && legal.to_sq() == parsed.to_sq()
                && legal.flag().promotion_piece() == parsed.flag().promotion_piece()
        })
        .copied()
}

/// Build and validate a complete UCI position without mutating handler state.
/// The side not to move must not be in check: otherwise the supplied history
/// claims that the previous move illegally left its own king attacked.
fn build_position(fen: Option<&str>, moves: &[String]) -> Result<Board, String> {
    let mut board = match fen {
        Some(fen) => Board::from_fen(fen).map_err(|e| format!("{e}"))?,
        None => Board::starting_position(),
    };
    chess_core::validate_position(&board).map_err(|e| e.to_string())?;

    for (index, move_str) in moves.iter().enumerate() {
        let m = find_legal_move(&board, move_str)
            .ok_or_else(|| format!("illegal move {move_str} at index {index}"))?;
        board.make_move(m);
        chess_core::validate_position(&board)
            .map_err(|e| format!("invalid board after move {move_str} at index {index}: {e}"))?;
    }
    Ok(board)
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
    use super::{build_position, normalize_display_score, validated_ponder_move};
    use chess_common::{Board, Move, Score};
    use chess_engine::syzygy::{TB_LOSS_SCORE, TB_WIN_SCORE};

    fn mv(uci: &str) -> Move {
        Move::from_uci(uci).unwrap()
    }

    #[test]
    fn ponder_move_returned_when_pv_is_consistent() {
        // 1. e4 e5 — PV starts with best_move and the reply is legal.
        let board = Board::starting_position();
        let best = mv("e2e4");
        let reply = mv("e7e5");
        assert_eq!(
            validated_ponder_move(&board, best, &[best, reply]),
            Some(reply)
        );
    }

    #[test]
    fn ponder_move_dropped_when_reply_is_illegal() {
        // After 1. e4, "e2e4" again is illegal (e2 is empty) — a corrupt PV
        // continuation must not be advertised as a ponder move.
        let board = Board::starting_position();
        let best = mv("e2e4");
        assert_eq!(
            validated_ponder_move(&board, best, &[best, mv("e2e4")]),
            None
        );
    }

    #[test]
    fn ponder_move_dropped_when_pv_head_differs_from_best() {
        let board = Board::starting_position();
        assert_eq!(
            validated_ponder_move(&board, mv("e2e4"), &[mv("d2d4"), mv("e7e5")]),
            None
        );
    }

    #[test]
    fn ponder_move_dropped_when_pv_too_short_or_best_null() {
        let board = Board::starting_position();
        let best = mv("e2e4");
        assert_eq!(validated_ponder_move(&board, best, &[best]), None);
        assert_eq!(validated_ponder_move(&board, best, &[]), None);
        assert_eq!(
            validated_ponder_move(&board, Move::NULL, &[Move::NULL, mv("e7e5")]),
            None
        );
    }

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

    #[test]
    fn invalid_positions_are_rejected_before_search() {
        assert!(build_position(Some("8/8/8/8/8/8/8/8 w - - 0 1"), &[]).is_err());
        assert!(build_position(Some("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1"), &[]).is_err());
        assert!(build_position(None, &["e2e5".to_string()]).is_err());
    }

    #[test]
    fn valid_position_history_is_accepted() {
        let moves = ["e2e4".to_string(), "e7e5".to_string()];
        let board = build_position(None, &moves).expect("legal position history");
        assert_eq!(board.side_to_move, chess_common::Color::White);
    }
}
