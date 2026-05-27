use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use chess_common::{Board, Color, Move, PieceKind, Score, Square};
use chess_common::moves::MoveFlag;

use chess_nnue::{Accumulator, NnueNetwork, nnue_evaluate};

use crate::eval::evaluate;
use crate::polyglot::OpeningBook;
use crate::see;
use crate::syzygy::{self, SyzygyTB};
use crate::tt::{SharedTT, TTFlag};
use crate::{InfoCallback, SearchInfo, SearchParams, SearchResult, TuneParams};

// ---------------------------------------------------------------------------
// LMR reduction table (built at search start from TuneParams)
// ---------------------------------------------------------------------------

fn build_lmr_table(base_x100: i32, div_x100: i32) -> [[u8; 64]; 64] {
    let base = base_x100 as f64 / 100.0;
    let div  = div_x100  as f64 / 100.0;
    let mut table = [[0u8; 64]; 64];
    #[allow(clippy::needless_range_loop)]
    for depth in 1..64 {
        for moves in 1..64 {
            table[depth][moves] =
                (base + (depth as f64).ln() * (moves as f64).ln() / div) as u8;
        }
    }
    table
}

// ---------------------------------------------------------------------------
// Killer moves, history & search state
// ---------------------------------------------------------------------------

const MAX_PLY: usize = 128;
const MAX_KILLERS: usize = 2;
const MATE_THRESHOLD: i32 = 29_000;

/// Adjust a mate score from root-relative to position-relative before TT store.
/// Non-mate scores pass through unchanged.
#[inline]
fn score_to_tt(score: i32, ply: u8) -> i32 {
    if score > MATE_THRESHOLD {
        score + ply as i32
    } else if score < -MATE_THRESHOLD {
        score - ply as i32
    } else {
        score
    }
}

/// Adjust a mate score from position-relative to root-relative after TT probe.
/// Non-mate scores pass through unchanged.
#[inline]
fn score_from_tt(score: i32, ply: u8) -> i32 {
    if score > MATE_THRESHOLD {
        score - ply as i32
    } else if score < -MATE_THRESHOLD {
        score + ply as i32
    } else {
        score
    }
}

/// Continuation history: indexed by [piece_kind][to_sq][cur_piece_kind][cur_to_sq].
/// This tracks how good a move is in the context of the previous move.
type ContHistory = Box<[[[[i32; 64]; 6]; 64]; 6]>;

fn new_cont_history() -> ContHistory {
    Box::new([[[[0i32; 64]; 6]; 64]; 6])
}

/// Per-ply move context for continuation history lookback.
#[derive(Clone, Copy)]
struct PlyContext {
    piece_kind: usize,  // PieceKind index of the moved piece
    to_sq: usize,       // destination square index
}

const NULL_PLY_CONTEXT: PlyContext = PlyContext { piece_kind: 0, to_sq: 0 };

/// Capture history: indexed by [moving_piece][to_sq][captured_piece].
type CaptureHistory = Box<[[[i32; 6]; 64]; 6]>;

fn new_capture_history() -> CaptureHistory {
    Box::new([[[0i32; 6]; 64]; 6])
}

/// Correction history: learns the systematic error of the static eval, keyed
/// by various facets of the position (pawn structure, non-pawn placement,
/// minor/major placement, and the previous move). Each facet has its own
/// table indexed by [stm][key]; per-facet values are stored in fixed-point
/// with scale `CORRHIST_GRAIN` and clamped to `CORRHIST_LIMIT` (±32cp). The
/// applied correction sums the facets and is capped at `CORR_TOTAL_CAP`.
const CORRHIST_SIZE: usize = 16384; // power of two
const CORR_MASK: usize = CORRHIST_SIZE - 1;
const CORRHIST_GRAIN: i32 = 256; // fixed-point scale
const CORRHIST_LIMIT: i32 = CORRHIST_GRAIN * 32; // per-facet cap: ±32cp
const CORR_TOTAL_CAP: i32 = 64; // cap on the summed correction (cp)

/// A hash-keyed correction table, indexed by [stm][key & CORR_MASK].
type CorrTable = Box<[[i32; CORRHIST_SIZE]; 2]>;
fn new_corr_table() -> CorrTable {
    Box::new([[0i32; CORRHIST_SIZE]; 2])
}

/// Continuation correction table, keyed by [stm][prev_piece_kind][prev_to].
type ContCorrTable = Box<[[[i32; 64]; 6]; 2]>;
fn new_cont_corr_table() -> ContCorrTable {
    Box::new([[[0i32; 64]; 6]; 2])
}

/// Game-level learning tables that persist across moves (cleared on
/// `ucinewgame`). They are swapped into the per-search [`SearchState`] for the
/// duration of a search so history and corrections accumulate over the whole
/// game instead of resetting every move. Ply-local scratch (killers,
/// `ply_context`, `static_evals`, accumulators) stays per-search and is *not*
/// carried here.
pub struct PersistentHistory {
    history: [[i32; 64]; 64],
    capture_history: CaptureHistory,
    counter_moves: [[Move; 64]; 64],
    cont_history: ContHistory,
    pawn_corrhist: CorrTable,
    white_corrhist: CorrTable,
    black_corrhist: CorrTable,
    minor_corrhist: CorrTable,
    major_corrhist: CorrTable,
    cont_corrhist: ContCorrTable,
}

impl PersistentHistory {
    pub fn new() -> Self {
        Self {
            history: [[0; 64]; 64],
            capture_history: new_capture_history(),
            counter_moves: [[Move::NULL; 64]; 64],
            cont_history: new_cont_history(),
            pawn_corrhist: new_corr_table(),
            white_corrhist: new_corr_table(),
            black_corrhist: new_corr_table(),
            minor_corrhist: new_corr_table(),
            major_corrhist: new_corr_table(),
            cont_corrhist: new_cont_corr_table(),
        }
    }
}

impl Default for PersistentHistory {
    fn default() -> Self {
        Self::new()
    }
}

struct SearchState {
    tt: Arc<SharedTT>,
    killers: [[Move; MAX_KILLERS]; MAX_PLY],
    history: [[i32; 64]; 64],
    capture_history: CaptureHistory,
    counter_moves: [[Move; 64]; 64],
    cont_history: ContHistory,
    pawn_corrhist: CorrTable,
    white_corrhist: CorrTable,
    black_corrhist: CorrTable,
    minor_corrhist: CorrTable,
    major_corrhist: CorrTable,
    cont_corrhist: ContCorrTable,
    ply_context: [PlyContext; MAX_PLY],
    static_evals: [i32; MAX_PLY],
    accumulators: Box<[Accumulator; MAX_PLY]>,
    use_nnue: bool,
    net: Arc<NnueNetwork>,
    lmr_table: [[u8; 64]; 64],
    tune: TuneParams,
    nodes: u64,
    seldepth: u8,
    start_time: Instant,
    time_limit_ms: Option<u64>,
    use_soft_limit: bool,
    inc_ms: u64,
    time_remaining_ms: u64,
    max_nodes: Option<u64>,
    game_ply: usize,
    contempt: i32,
    singular_ext_mode: u8,
    stop: bool,
    /// Syzygy tablebase handle for WDL probing (None if disabled or not loaded).
    syzygy_tb: Option<SyzygyTB>,
}

impl SearchState {
    #[allow(clippy::too_many_arguments)]
    fn new(time_limit_ms: Option<u64>, use_soft_limit: bool, inc_ms: u64, time_remaining_ms: u64, max_nodes: Option<u64>, game_ply: usize, contempt: i32, singular_ext_mode: u8, tt: &Arc<SharedTT>, use_nnue: bool, net: Arc<NnueNetwork>, syzygy_tb: Option<SyzygyTB>, tune: TuneParams) -> Self {
        // Use a Vec to avoid stack allocation of the large accumulator array
        let mut acc_vec: Vec<Accumulator> = Vec::with_capacity(MAX_PLY);
        for _ in 0..MAX_PLY {
            acc_vec.push(Accumulator::new());
        }
        let accumulators: Box<[Accumulator; MAX_PLY]> = acc_vec.into_boxed_slice().try_into().ok().unwrap();

        let lmr_table = build_lmr_table(tune.lmr_base, tune.lmr_div);

        Self {
            tt: Arc::clone(tt),
            killers: [[Move::NULL; MAX_KILLERS]; MAX_PLY],
            history: [[0; 64]; 64],
            capture_history: new_capture_history(),
            counter_moves: [[Move::NULL; 64]; 64],
            cont_history: new_cont_history(),
            pawn_corrhist: new_corr_table(),
            white_corrhist: new_corr_table(),
            black_corrhist: new_corr_table(),
            minor_corrhist: new_corr_table(),
            major_corrhist: new_corr_table(),
            cont_corrhist: new_cont_corr_table(),
            ply_context: [NULL_PLY_CONTEXT; MAX_PLY],
            static_evals: [0; MAX_PLY],
            accumulators,
            use_nnue,
            net,
            lmr_table,
            tune,
            nodes: 0,
            seldepth: 0,
            start_time: Instant::now(),
            time_limit_ms,
            use_soft_limit,
            inc_ms,
            time_remaining_ms,
            max_nodes,
            game_ply,
            contempt,
            singular_ext_mode,
            stop: false,
            syzygy_tb,
        }
    }

    fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    fn should_stop(&self, stop_flag: &AtomicBool) -> bool {
        if self.stop || stop_flag.load(Ordering::Relaxed) {
            return true;
        }
        if let Some(limit) = self.time_limit_ms
            && self.elapsed_ms() >= limit
        {
            return true;
        }
        if let Some(max) = self.max_nodes
            && self.nodes >= max
        {
            return true;
        }
        false
    }

    fn store_killer(&mut self, ply: usize, m: Move) {
        if ply < MAX_PLY && self.killers[ply][0] != m {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = m;
        }
    }

    fn update_history(&mut self, m: Move, depth: u8, board: &Board, ply: u8) {
        let bonus = depth as i32 * depth as i32;
        let from = m.from_sq().index();
        let to = m.to_sq().index();

        // Regular from-to history (gravity formula)
        let max_val = 16384;
        let entry = &mut self.history[from][to];
        *entry += bonus - *entry * bonus.abs() / max_val;
        *entry = (*entry).clamp(-max_val, max_val);

        // Continuation history (1-ply and 2-ply lookback)
        if let Some(piece) = board.piece_at(m.from_sq()) {
            let cur_kind = piece.kind.index();
            let cur_to = to;
            // 1-ply lookback
            if ply >= 1 && (ply as usize - 1) < MAX_PLY {
                let ctx = self.ply_context[ply as usize - 1];
                let entry = &mut self.cont_history[ctx.piece_kind][ctx.to_sq][cur_kind][cur_to];
                *entry += bonus - *entry * bonus.abs() / max_val;
                *entry = (*entry).clamp(-max_val, max_val);
            }
            // 2-ply lookback
            if ply >= 2 && (ply as usize - 2) < MAX_PLY {
                let ctx = self.ply_context[ply as usize - 2];
                let entry = &mut self.cont_history[ctx.piece_kind][ctx.to_sq][cur_kind][cur_to];
                *entry += bonus - *entry * bonus.abs() / max_val;
                *entry = (*entry).clamp(-max_val, max_val);
            }
        }
    }

    fn update_history_malus(&mut self, m: Move, depth: u8, board: &Board, ply: u8) {
        let malus = depth as i32 * depth as i32;
        let from = m.from_sq().index();
        let to = m.to_sq().index();

        // Regular history malus (gravity formula)
        let max_val = 16384;
        let entry = &mut self.history[from][to];
        *entry += -malus - *entry * malus.abs() / max_val;
        *entry = (*entry).clamp(-max_val, max_val);

        // Continuation history malus (1-ply and 2-ply lookback)
        if let Some(piece) = board.piece_at(m.from_sq()) {
            let cur_kind = piece.kind.index();
            let cur_to = to;
            // 1-ply lookback
            if ply >= 1 && (ply as usize - 1) < MAX_PLY {
                let ctx = self.ply_context[ply as usize - 1];
                let entry = &mut self.cont_history[ctx.piece_kind][ctx.to_sq][cur_kind][cur_to];
                *entry += -malus - *entry * malus.abs() / max_val;
                *entry = (*entry).clamp(-max_val, max_val);
            }
            // 2-ply lookback
            if ply >= 2 && (ply as usize - 2) < MAX_PLY {
                let ctx = self.ply_context[ply as usize - 2];
                let entry = &mut self.cont_history[ctx.piece_kind][ctx.to_sq][cur_kind][cur_to];
                *entry += -malus - *entry * malus.abs() / max_val;
                *entry = (*entry).clamp(-max_val, max_val);
            }
        }
    }

    /// Learned static-eval correction for the current position, in centipawns
    /// (side-to-move perspective). Sums all corrhist facets, caps the total,
    /// then scales by the tunable `corrhist_mult` (×100).
    fn correction(&self, board: &Board, ply: u8) -> i32 {
        let stm = if board.side_to_move == Color::White { 0 } else { 1 };
        let k = crate::eval::corr_keys(board);
        let mut sum = self.pawn_corrhist[stm][k.pawn as usize & CORR_MASK]
            + self.white_corrhist[stm][k.white as usize & CORR_MASK]
            + self.black_corrhist[stm][k.black as usize & CORR_MASK]
            + self.minor_corrhist[stm][k.minor as usize & CORR_MASK]
            + self.major_corrhist[stm][k.major as usize & CORR_MASK];
        if ply >= 1 && (ply as usize - 1) < MAX_PLY {
            let ctx = self.ply_context[ply as usize - 1];
            sum += self.cont_corrhist[stm][ctx.piece_kind][ctx.to_sq];
        }
        let cp = (sum / CORRHIST_GRAIN).clamp(-CORR_TOTAL_CAP, CORR_TOTAL_CAP);
        cp * self.tune.corrhist_mult / 100
    }

    /// Update every corrhist facet from the residual `diff = search_score -
    /// static_eval`. Deeper searches are trusted more (larger weight).
    fn update_corrhist(&mut self, board: &Board, depth: u8, diff: i32, ply: u8) {
        let stm = if board.side_to_move == Color::White { 0 } else { 1 };
        let k = crate::eval::corr_keys(board);
        let scaled = (diff * CORRHIST_GRAIN).clamp(-CORRHIST_LIMIT, CORRHIST_LIMIT);
        let weight = (depth as i32 + 1).min(16);
        let blend = |e: &mut i32| {
            *e = (*e * (256 - weight) + scaled * weight) / 256;
            *e = (*e).clamp(-CORRHIST_LIMIT, CORRHIST_LIMIT);
        };
        blend(&mut self.pawn_corrhist[stm][k.pawn as usize & CORR_MASK]);
        blend(&mut self.white_corrhist[stm][k.white as usize & CORR_MASK]);
        blend(&mut self.black_corrhist[stm][k.black as usize & CORR_MASK]);
        blend(&mut self.minor_corrhist[stm][k.minor as usize & CORR_MASK]);
        blend(&mut self.major_corrhist[stm][k.major as usize & CORR_MASK]);
        if ply >= 1 && (ply as usize - 1) < MAX_PLY {
            let ctx = self.ply_context[ply as usize - 1];
            blend(&mut self.cont_corrhist[stm][ctx.piece_kind][ctx.to_sq]);
        }
    }

    /// Swap the game-level learning tables between this search state and a
    /// persistent holder. Called once before and once after a search so the
    /// tables accumulate across moves within a game.
    fn swap_history(&mut self, h: &mut PersistentHistory) {
        std::mem::swap(&mut self.history, &mut h.history);
        std::mem::swap(&mut self.capture_history, &mut h.capture_history);
        std::mem::swap(&mut self.counter_moves, &mut h.counter_moves);
        std::mem::swap(&mut self.cont_history, &mut h.cont_history);
        std::mem::swap(&mut self.pawn_corrhist, &mut h.pawn_corrhist);
        std::mem::swap(&mut self.white_corrhist, &mut h.white_corrhist);
        std::mem::swap(&mut self.black_corrhist, &mut h.black_corrhist);
        std::mem::swap(&mut self.minor_corrhist, &mut h.minor_corrhist);
        std::mem::swap(&mut self.major_corrhist, &mut h.major_corrhist);
        std::mem::swap(&mut self.cont_corrhist, &mut h.cont_corrhist);
    }

    fn store_counter_move(&mut self, prev_move: Move, counter: Move) {
        if !prev_move.is_null() {
            self.counter_moves[prev_move.from_sq().index()][prev_move.to_sq().index()] = counter;
        }
    }

    fn get_counter_move(&self, prev_move: Move) -> Move {
        if prev_move.is_null() {
            Move::NULL
        } else {
            self.counter_moves[prev_move.from_sq().index()][prev_move.to_sq().index()]
        }
    }

    fn get_cont_history_bonus(&self, m: Move, board: &Board, ply: u8) -> i32 {
        let piece = match board.piece_at(m.from_sq()) {
            Some(p) => p,
            None => return 0,
        };
        let cur_kind = piece.kind.index();
        let cur_to = m.to_sq().index();

        let mut bonus = 0i32;
        // 1-ply lookback
        if ply >= 1 && (ply as usize - 1) < MAX_PLY {
            let ctx = self.ply_context[ply as usize - 1];
            bonus += self.cont_history[ctx.piece_kind][ctx.to_sq][cur_kind][cur_to];
        }
        // 2-ply lookback (follow-up history)
        if ply >= 2 && (ply as usize - 2) < MAX_PLY {
            let ctx = self.ply_context[ply as usize - 2];
            bonus += self.cont_history[ctx.piece_kind][ctx.to_sq][cur_kind][cur_to] / 2;
        }
        bonus
    }

    fn get_capture_history(&self, m: Move, board: &Board) -> i32 {
        let piece = match board.piece_at(m.from_sq()) {
            Some(p) => p,
            None => return 0,
        };
        let captured = match board.piece_at(m.to_sq()) {
            Some(p) => p,
            None => return 0,
        };
        self.capture_history[piece.kind.index()][m.to_sq().index()][captured.kind.index()]
    }

    fn update_capture_history(&mut self, m: Move, depth: u8, board: &Board) {
        let piece = match board.piece_at(m.from_sq()) {
            Some(p) => p,
            None => return,
        };
        let captured = match board.piece_at(m.to_sq()) {
            Some(p) => p,
            None => return,
        };
        let bonus = depth as i32 * depth as i32;
        let max_val = 16384;
        let entry = &mut self.capture_history[piece.kind.index()][m.to_sq().index()][captured.kind.index()];
        *entry += bonus - *entry * bonus.abs() / max_val;
        *entry = (*entry).clamp(-max_val, max_val);
    }

    fn update_capture_history_malus(&mut self, m: Move, depth: u8, board: &Board) {
        let piece = match board.piece_at(m.from_sq()) {
            Some(p) => p,
            None => return,
        };
        let captured = match board.piece_at(m.to_sq()) {
            Some(p) => p,
            None => return,
        };
        let malus = depth as i32 * depth as i32;
        let max_val = 16384;
        let entry = &mut self.capture_history[piece.kind.index()][m.to_sq().index()][captured.kind.index()];
        *entry += -malus - *entry * malus.abs() / max_val;
        *entry = (*entry).clamp(-max_val, max_val);
    }

    fn store_ply_context(&mut self, ply: u8, m: Move, board: &Board) {
        if (ply as usize) < MAX_PLY
            && let Some(piece) = board.piece_at(m.from_sq())
        {
            self.ply_context[ply as usize] = PlyContext {
                piece_kind: piece.kind.index(),
                to_sq: m.to_sq().index(),
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Time management
// ---------------------------------------------------------------------------

/// Per-move time cap (ms) for sudden-death (no-increment) time controls.
///
/// With no increment the clock must last the whole game with no refund, yet the
/// piece-count `moves_left` estimate in `compute_time_limit` spends *faster* as
/// pieces come off — so a long endgame conversion can flag even from a winning
/// position (observed: a 180+0 game flagged ~move 90 while mating). Bound the
/// slice to `time/N` (large `N`) so enough clock stays in reserve for a long
/// game. Only pure sudden death is affected: any increment refunds the clock
/// and is left to the standard tuning, so increment controls — including the
/// `10+0.1` bench/SF-test path — are unchanged.
fn sudden_death_time_cap(target_ms: u64, inc_ms: u64, time_ms: u64) -> u64 {
    if inc_ms != 0 {
        return target_ms;
    }
    let n: u64 = if time_ms > 300_000 { 38 } else { 34 };
    target_ms.min(time_ms / n)
}

/// Returns (time_limit_ms, use_soft_limit, inc_ms, time_remaining_ms).
/// `use_soft_limit` is true for clock-based time controls (wtime/btime)
/// where we must save time for future moves, false for fixed movetime.
///
/// Supports typical tournament time controls:
///   - FIDE/DSB Classical: 40 moves / 90 min + 30s inc, then 15-30 min + 30s
///   - FIDE/DSB Rapid:     15 min + 10s inc
///   - FIDE/DSB Blitz:     3-5 min + 2-3s inc
///   - Bullet:             1 min + 0-1s inc
fn compute_time_limit(params: &SearchParams, board: &Board) -> (Option<u64>, bool, u64, u64) {
    if params.infinite {
        return (None, false, 0, 0);
    }

    let overhead = params.move_overhead_ms;

    if let Some(mt) = params.move_time_ms {
        return (Some(mt.saturating_sub(overhead)), false, 0, 0);
    }

    let side = board.side_to_move;
    let (time_ms, inc_ms) = match side {
        Color::White => (params.white_time_ms, params.white_inc_ms),
        Color::Black => (params.black_time_ms, params.black_inc_ms),
    };

    if let Some(time) = time_ms {
        let inc = inc_ms.unwrap_or(0);
        let game_ply = board.position_history.len() as u64;

        // Estimate moves remaining in this time control period
        let mut moves_left = if let Some(mtg) = params.moves_to_go {
            // Known time control (e.g., "40 moves in 90 min")
            (mtg as u64).max(1)
        } else {
            // Game phase-aware estimate based on piece count
            let total_pieces = (board.occupancy[0] | board.occupancy[1]).count();
            let base_moves = match total_pieces {
                28.. => 32, // opening
                22..=27 => 26, // middlegame
                16..=21 => 20, // late middlegame
                _ => 16,  // endgame
            };
            // In fast time controls, use time more aggressively since
            // games tend to end sooner and deep calculation matters more
            // per move. Scale down moves_left for blitz/bullet.
            if time <= 15_000 {
                // Ultra-blitz / bullet: still reduce moves_left, but keep a
                // safer reserve to avoid practical flags.
                (base_moves * 65 / 100).max(1)
            } else if time < 120_000 {
                // Bullet (< 2 min): moderately aggressive
                (base_moves * 3 / 4).max(1)
            } else if time < 300_000 {
                // Blitz (< 5 min): moderately aggressive
                (base_moves * 3 / 4).max(1)
            } else {
                base_moves
            }
        };

        // In increment games, we can safely plan for fewer "must-save" moves,
        // so spend more time per move in the middlegame.
        let moves_left_scale_permille = if inc >= 5_000 {
            650
        } else if inc >= 2_000 {
            740
        } else if inc >= 1_000 {
            840
        } else if inc > 0 {
            930
        } else {
            1000
        };
        moves_left = (moves_left * moves_left_scale_permille / 1000).max(1);

        // Base time per move
        let base = time / moves_left;

        // Add an increment share. Keep it conservative in severe time trouble,
        // but in normal increment controls we can spend close to the increment.
        let inc_share_permille = if time <= 10_000 {
            500
        } else if time <= 30_000 {
            650
        } else if time <= 120_000 {
            800
        } else {
            900
        };
        let mut inc_term = inc * inc_share_permille / 1000;

        // If increment is large relative to the base slice, safely use a bit
        // more of it (up to +20%). This helps typical increment games avoid
        // under-spending while preserving low-time safety.
        if base > 0 && inc > 0 {
            let inc_ratio_permille = (inc.saturating_mul(1000) / base).min(1000);
            inc_term += inc * inc_ratio_permille / 5000;
        }
        // Use a stronger share of excess bank and steer toward a phase-aware
        // target reserve. This reduces ending the game with large unused time.
        let target_bank_ms = if inc > 0 {
            if game_ply < 20 {
                8_000u64.saturating_add(inc.saturating_mul(16))
            } else if game_ply < 40 {
                6_000u64.saturating_add(inc.saturating_mul(12))
            } else if game_ply < 60 {
                5_000u64.saturating_add(inc.saturating_mul(9))
            } else {
                4_000u64.saturating_add(inc.saturating_mul(7))
            }
        } else if game_ply < 40 {
            5_000
        } else {
            3_000
        };
        let usable_bank = time.saturating_sub(target_bank_ms);
        let burn_horizon = if let Some(mtg) = params.moves_to_go {
            (mtg as u64).clamp(4, 16)
        } else if game_ply < 20 {
            18
        } else if game_ply < 40 {
            14
        } else {
            10
        };
        let bank_term_raw = usable_bank / burn_horizon.saturating_mul(3).max(1);
        // Opening throttle: avoid spending too much bank time in the first
        // few moves when there is no opening book. Apply to all time controls.
        let opening_bank_scale_permille = if game_ply < 6 {
            if time <= 15_000 { 600u64 }
            else { 350u64 }
        } else if game_ply < 12 {
            if time <= 15_000 { 700u64 }
            else if time <= 300_000 { 550u64 }
            else { 600u64 }
        } else if game_ply < 20 {
            if time <= 300_000 { 800u64 } else { 900u64 }
        } else {
            1000u64
        };
        let bank_term_raw = bank_term_raw * opening_bank_scale_permille / 1000;
        let bank_term_cap = base.saturating_mul(3) / 2 + inc;
        let bank_term = bank_term_raw.min(bank_term_cap);

        let target = base + inc_term + bank_term;

        // Apply slow mover scaling factor
        let target = target * params.slow_mover / 100;

        // Sudden-death (no-increment) safety: keep enough clock in reserve for
        // a long game (incl. long endgame conversions). See fn docs.
        let target = sudden_death_time_cap(target, inc, time);

        // Hard maximum with stronger low-time safety to reduce flags.
        let mut max = if time <= 30_000 {
            time / 8 + inc * 2
        } else if time < 120_000 {
            time / 6 + inc
        } else {
            time / 4 + inc / 2
        };

        // In healthy increment controls, allow a little more burst usage.
        if base > 0 && inc.saturating_mul(2) >= base {
            max = max.saturating_add(inc / 2);
        }

        // Additional early-opening cap to prevent very long thinks in the
        // first moves without a book. Apply to all time controls.
        if game_ply < 6 {
            let frac = if time <= 300_000 { 16u64 } else { 30u64 };
            max = max.min(time / frac + inc.saturating_mul(2));
        } else if game_ply < 12 {
            let frac = if time <= 300_000 { 13u64 } else { 22u64 };
            max = max.min(time / frac + inc.saturating_mul(2));
        }

        // Minimum think time: always use at least the increment when
        // there's meaningful clock time remaining. This guarantees the
        // engine never plays faster than its increment allows.
        let min = if time > 1000 {
            (time / 60).max(inc.saturating_sub(overhead)).max(50)
        } else {
            // Ultra-low time (bullet flag): use increment if available,
            // otherwise a tiny amount to avoid instant moves.
            if inc > overhead { inc.saturating_sub(overhead) } else { 10 }
        };

        let allocated = target.min(max).max(min);
        (Some(allocated.saturating_sub(overhead)), true, inc, time)
    } else {
        (None, false, 0, 0)
    }
}

/// Move-time budget (ms) the engine would allocate for `params` at `board`,
/// or `None` for an infinite/untimed search. Used by the UCI layer to convert
/// a pondering (infinite) search into a timed one when `ponderhit` arrives.
pub fn allocated_move_time_ms(params: &SearchParams, board: &Board) -> Option<u64> {
    compute_time_limit(params, board).0
}

// ---------------------------------------------------------------------------
// Move ordering helpers
// ---------------------------------------------------------------------------

/// Pick the highest-scored move from `start..len`, swap it to position `start`.
#[inline]
fn pick_next_move(moves: &mut [Move], scores: &mut [i32], start: usize) {
    let mut best_idx = start;
    let mut best_score = scores[start];
    for (i, &score) in scores.iter().enumerate().take(moves.len()).skip(start + 1) {
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    if best_idx != start {
        moves.swap(start, best_idx);
        scores.swap(start, best_idx);
    }
}

// ---------------------------------------------------------------------------
// Staged move picker — generates moves lazily in priority order
// ---------------------------------------------------------------------------
//
// Stages (main search):
//   1. TT move
//   2. Good captures  (SEE ≥ 0, sorted by MVV-LVA)
//   3. Quiet moves    (killers / counter / history-sorted)
//   4. Bad captures   (SEE < 0)
//
// This avoids generating quiet moves entirely when a beta cutoff happens
// on a capture, and eliminates the board-clone legality check from movegen.
// Legality is verified inline in the search after make_move.

const MAX_PICKER_MOVES: usize = 256;
const MAX_BAD_CAPTURES: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum MpStage {
    TtMove,
    InitCaptures,
    GoodCaptures,
    InitQuiets,
    Quiets,
    BadCaptures,
    Done,
}

struct MovePicker {
    stage: MpStage,
    tt_move: Move,

    moves:  [Move; MAX_PICKER_MOVES],
    scores: [i32; MAX_PICKER_MOVES],
    count:  usize,
    idx:    usize,

    bad_captures: [Move; MAX_BAD_CAPTURES],
    num_bad:      usize,
    bad_idx:      usize,
}

impl MovePicker {
    fn new(tt_move: Move) -> Self {
        Self {
            stage: if !tt_move.is_null() { MpStage::TtMove } else { MpStage::InitCaptures },
            tt_move,
            moves:  [Move::NULL; MAX_PICKER_MOVES],
            scores: [0; MAX_PICKER_MOVES],
            count:  0,
            idx:    0,
            bad_captures: [Move::NULL; MAX_BAD_CAPTURES],
            num_bad: 0,
            bad_idx: 0,
        }
    }

    /// Return the next pseudo-legal move in priority order.
    /// The caller must check legality after `make_move`.
    fn next(
        &mut self,
        board: &Board,
        state: &SearchState,
        ply: u8,
        killers: &[Move; MAX_KILLERS],
        counter_move: Move,
    ) -> Option<Move> {
        loop {
            match self.stage {
                // ── 1. Hash move ────────────────────────────────────
                MpStage::TtMove => {
                    self.stage = MpStage::InitCaptures;
                    if !self.tt_move.is_null() && is_move_safe(board, self.tt_move) {
                        return Some(self.tt_move);
                    }
                }
                // ── 2a. Generate captures ───────────────────────────
                MpStage::InitCaptures => {
                    let captures = chess_core::generate_pseudo_legal_captures(board);
                    self.count = 0;
                    for &m in captures.iter() {
                        if m == self.tt_move { continue; }
                        let mvv = mvv_lva_score(board, m);
                        let cap_hist = state.get_capture_history(m, board);
                        let score = mvv * 16 + cap_hist / 64;
                        self.moves[self.count] = m;
                        self.scores[self.count] = score;
                        self.count += 1;
                    }
                    self.idx = 0;
                    self.stage = MpStage::GoodCaptures;
                }
                // ── 2b. Yield good captures (SEE ≥ 0) ──────────────
                MpStage::GoodCaptures => {
                    while self.idx < self.count {
                        pick_next_move(
                            &mut self.moves[..self.count],
                            &mut self.scores[..self.count],
                            self.idx,
                        );
                        let m = self.moves[self.idx];
                        self.idx += 1;
                        if see::see_ge(board, m, 0) {
                            return Some(m);
                        } else if self.num_bad < MAX_BAD_CAPTURES {
                            self.bad_captures[self.num_bad] = m;
                            self.num_bad += 1;
                        }
                    }
                    self.stage = MpStage::InitQuiets;
                }
                // ── 3a. Generate quiet moves ────────────────────────
                MpStage::InitQuiets => {
                    let quiets = chess_core::generate_pseudo_legal_quiets(board);
                    self.count = 0;
                    for &m in quiets.iter() {
                        if m == self.tt_move { continue; }
                        let hist = state.history[m.from_sq().index()][m.to_sq().index()];
                        let cont = state.get_cont_history_bonus(m, board, ply);
                        let score = if m == killers[0] {
                            900_000
                        } else if m == killers[1] {
                            800_000
                        } else if m == counter_move && !counter_move.is_null() {
                            700_000
                        } else {
                            hist + cont / 2
                        };
                        self.moves[self.count] = m;
                        self.scores[self.count] = score;
                        self.count += 1;
                    }
                    self.idx = 0;
                    self.stage = MpStage::Quiets;
                }
                // ── 3b. Yield quiet moves sorted by score ───────────
                MpStage::Quiets => {
                    if self.idx < self.count {
                        pick_next_move(
                            &mut self.moves[..self.count],
                            &mut self.scores[..self.count],
                            self.idx,
                        );
                        let m = self.moves[self.idx];
                        self.idx += 1;
                        return Some(m);
                    }
                    self.stage = MpStage::BadCaptures;
                    self.bad_idx = 0;
                }
                // ── 4. Bad captures (SEE < 0) ───────────────────────
                MpStage::BadCaptures => {
                    if self.bad_idx < self.num_bad {
                        let m = self.bad_captures[self.bad_idx];
                        self.bad_idx += 1;
                        return Some(m);
                    }
                    self.stage = MpStage::Done;
                }
                MpStage::Done => return None,
            }
        }
    }
}

/// Validate that a TT move is safe to pass to `make_move` (won't corrupt
/// the board). Hash collisions can produce arbitrary 16-bit move values,
/// so we check piece presence, color, self-capture, and flag sanity.
#[inline]
fn is_move_safe(board: &Board, m: Move) -> bool {
    if m.is_null() { return false; }
    let from = m.from_sq();
    let to = m.to_sq();
    let flag = m.flag();

    // Must have a piece of our color on the source square.
    let piece = match board.piece_at(from) {
        Some(p) if p.color == board.side_to_move => p,
        _ => return false,
    };

    // Must not capture own piece (castling excluded — king "captures" rook square).
    if flag != MoveFlag::KingsideCastle && flag != MoveFlag::QueensideCastle
        && let Some(p) = board.piece_at(to)
        && p.color == board.side_to_move
    {
        return false;
    }

    // Validate capture flag vs. board state: TT hash collisions can produce
    // moves whose flag doesn't match the current position.  Making a quiet
    // move onto an occupied square (or a capture onto an empty one) corrupts
    // the bitboards.
    if flag != MoveFlag::KingsideCastle && flag != MoveFlag::QueensideCastle {
        let to_occupied = board.piece_at(to).is_some();
        if flag == MoveFlag::EnPassant {
            // EP destination square is always empty.
            if to_occupied { return false; }
        } else if flag.is_capture() {
            // A capture (non-EP) must land on an enemy piece.
            if !to_occupied { return false; }
        } else {
            // A quiet move must not land on an occupied square.
            if to_occupied { return false; }
        }
    }

    // Castling: piece must be a king on the correct square.
    if flag == MoveFlag::KingsideCastle || flag == MoveFlag::QueensideCastle {
        if piece.kind != PieceKind::King { return false; }
        let expected_from = match board.side_to_move {
            Color::White => Square::E1,
            Color::Black => Square::E8,
        };
        if from != expected_from { return false; }
    }

    // Promotion: piece must be a pawn on the correct rank.
    if flag.is_promotion() {
        if piece.kind != PieceKind::Pawn { return false; }
        let promo_rank = match board.side_to_move {
            Color::White => 6, // rank 7 (0-indexed)
            Color::Black => 1, // rank 2 (0-indexed)
        };
        if from.rank() != promo_rank { return false; }
    }

    // En passant: board must have an EP square set, and it must match `to`,
    // and piece must be a pawn.
    if flag == MoveFlag::EnPassant {
        if piece.kind != PieceKind::Pawn { return false; }
        match board.en_passant {
            Some(ep) if ep == to => {},
            _ => return false,
        }
    }

    // Double pawn push: piece must be a pawn on the starting rank.
    if flag == MoveFlag::DoublePawnPush {
        if piece.kind != PieceKind::Pawn { return false; }
        let start_rank = match board.side_to_move {
            Color::White => 1, // rank 2 (0-indexed)
            Color::Black => 6, // rank 7 (0-indexed)
        };
        if from.rank() != start_rank { return false; }
    }

    true
}

/// MVV-LVA score for capture ordering (higher = better).
#[inline]
fn mvv_lva_score(board: &Board, m: Move) -> i32 {
    let flag = m.flag();
    let victim = if flag == MoveFlag::EnPassant {
        PieceKind::Pawn.value()
    } else {
        board.piece_at(m.to_sq()).map(|p| p.kind.value()).unwrap_or(0)
    };
    let attacker = board.piece_at(m.from_sq()).map(|p| p.kind.value()).unwrap_or(0);
    let promo = if flag.is_promotion() { 800 } else { 0 };
    victim * 10 - attacker + promo
}

/// After `board.make_move(m)`, returns true when the side that just moved
/// left its own king in check (i.e. the move was illegal).
#[inline]
fn move_is_illegal(board: &Board, mover: Color) -> bool {
    let king_sq = board.king_square(mover);
    chess_core::attacks::is_square_attacked(board, king_sq, board.side_to_move)
}

// ---------------------------------------------------------------------------
// Opening book
// ---------------------------------------------------------------------------

fn probe_opening_book(board: &Board, book: &OpeningBook) -> Option<Move> {
    book.pick_move(board)
}

// ---------------------------------------------------------------------------
// Iterative deepening
// ---------------------------------------------------------------------------

/// Depth offsets for Lazy SMP helper threads (Stockfish-style).
/// Thread 0 gets no offset. Helpers alternate between +/-1, +/-2, etc.
/// This creates depth diversity so different threads explore different depths.
const DEPTH_OFFSETS: [i8; 16] = [0, 1, -1, 2, -2, 1, -1, 3, -3, 2, -2, 1, -1, 4, -4, 3];

fn depth_offset(thread_id: usize) -> i8 {
    if thread_id == 0 {
        0
    } else {
        DEPTH_OFFSETS[thread_id % DEPTH_OFFSETS.len()]
    }
}

#[allow(clippy::too_many_arguments)]
pub fn iterative_deepening(
    board: &Board,
    params: &SearchParams,
    stop: &Arc<AtomicBool>,
    tt: &Arc<SharedTT>,
    info_callback: Option<InfoCallback>,
    thread_id: usize,
    net: &Arc<NnueNetwork>,
    node_counter: Option<&AtomicU64>,
    syzygy_tb: Option<SyzygyTB>,
    root_tb_solution: Option<(Score, Vec<Move>)>,
    external_book: Option<Arc<OpeningBook>>,
    mut persistent: Option<&mut PersistentHistory>,
) -> SearchResult {
    // Try opening book first (only in the opening — limit to 30 half-moves
    // to avoid polyglot hash collisions returning garbage moves in endgames)
    let game_ply = board.position_history.len();
    if let Some(book) = external_book.as_deref()
        && game_ply <= 30
        && let Some(book_move) = probe_opening_book(board, book)
    {
        // Report a stable static eval of the current position (before the
        // book move).  Evaluating *after* the move can be misleading for
        // tactical captures.  For book UI reporting, clamp aggressively so
        // GUIs never display absurd outlier values.
        let raw_book_eval = evaluate(board).0.clamp(-MATE_THRESHOLD + 1, MATE_THRESHOLD - 1);
        let stm_eval = match board.side_to_move {
            Color::White => raw_book_eval,
            Color::Black => raw_book_eval.saturating_neg(),
        };
        let score = Score(stm_eval.clamp(-40, 40));
        if let Some(ref cb) = info_callback {
            cb(&SearchInfo {
                depth: 1,
                seldepth: 1,
                score,
                nodes: 1,
                time_ms: 0,
                pv: vec![book_move],
                hashfull: 0,
                multipv_line: 1,
            });
        }
        return SearchResult {
            best_move: book_move,
            score,
            depth: 1,
            nodes: 1,
            pv: vec![book_move],
        };
    }

    let (time_limit, use_soft_limit, inc, time_remaining) = compute_time_limit(params, board);
    let mut state = SearchState::new(time_limit, use_soft_limit, inc, time_remaining, params.max_nodes, game_ply, params.contempt, params.singular_ext_mode, tt, params.use_nnue, Arc::clone(net), syzygy_tb, params.tune.clone());

    // Initialize root accumulator for NNUE
    if state.use_nnue {
        state.accumulators[0].refresh(board, &state.net);
    }
    let offset = depth_offset(thread_id);

    // Book-exit time boost: allocate extra time for the first few moves
    // of original play after exiting the opening book.
    let book_exit_bonus: u64 = if external_book.is_some() && (8..=33).contains(&game_ply) {
        if game_ply <= 20 { 200 } else { 200 * (33 - game_ply) as u64 / 13 }
    } else {
        0
    };

    let mut best_move = Move::NULL;
    let mut best_score = Score(0);
    let mut best_pv = Vec::new();
    let mut best_depth = 0u8;
    let root_tb_solution = root_tb_solution.map(|(score, pv)| (score, sanitize_pv(board, &pv)));

    let root_moves = chess_core::generate_legal_moves(board);
    if root_moves.is_empty() {
        let score = if chess_core::is_in_check(board) {
            Score::NEG_MATE
        } else {
            Score::DRAW
        };
        return SearchResult {
            best_move: Move::NULL,
            score,
            depth: 0,
            nodes: 0,
            pv: Vec::new(),
        };
    }

    if root_moves.len() == 1 && !params.infinite {
        // Only one legal move — do a quick search after making it to get
        // an accurate score (important for mate detection and analysis).
        let only_move = *root_moves.iter().next().unwrap();
        let mut board_copy = board.clone();
        let prev_castling = board_copy.castling;
        let prev_ep = board_copy.en_passant;
        let prev_halfmove = board_copy.halfmove_clock;
        let captured = board_copy.make_move(only_move);
        // Quick fixed-depth search to evaluate the resulting position
        let quick_depth = 10u8;
        let mut quick_state = SearchState::new(Some(2000), false, 0, 0, None, game_ply, params.contempt, params.singular_ext_mode, tt, params.use_nnue, Arc::clone(net), state.syzygy_tb.clone(), params.tune.clone());
        if quick_state.use_nnue {
            quick_state.accumulators[0].refresh(board, &quick_state.net);
            update_accumulator_for_move(&mut quick_state, &board_copy, only_move, captured, 0, 1);
        }
        let mut quick_pv = Vec::new();
        let child_score = -alpha_beta(
            &mut board_copy, quick_depth, 1, Score::NEG_INF.0, Score::INF.0,
            &mut quick_pv, &mut quick_state, stop, only_move, Move::NULL,
        );
        board_copy.unmake_move(only_move, captured, prev_castling, prev_ep, prev_halfmove);
        let score = Score(child_score);
        let mut pv = vec![only_move];
        pv.extend_from_slice(&quick_pv);
        let clean_pv = sanitize_pv(board, &pv);
        if let Some(ref cb) = info_callback {
            cb(&SearchInfo {
                depth: quick_depth,
                seldepth: quick_state.seldepth,
                score,
                nodes: quick_state.nodes,
                time_ms: quick_state.elapsed_ms(),
                pv: clean_pv.clone(),
                hashfull: tt.hashfull(),
                multipv_line: 1,
            });
        }
        return SearchResult {
            best_move: only_move,
            score,
            depth: quick_depth,
            nodes: quick_state.nodes,
            pv: clean_pv,
        };
    }

    // Swap in the game-level learning tables so history/corrhist accumulate
    // across moves. Swapped back out before returning.
    if let Some(h) = persistent.as_deref_mut() {
        state.swap_history(h);
    }

    let max_depth = params.max_depth.min(MAX_PLY as u8);
    let mut prev_score = 0i32;

    // Root move ordering: track scores from previous iteration
    let mut root_move_scores: Vec<(Move, i32)> = root_moves.iter().map(|&m| (m, 0i32)).collect();

    // Pre-order root moves using TT so the 0-nodes fallback (stop before any
    // evaluation) picks the TT move rather than an arbitrary first generated move.
    {
        let tt_best = tt.probe(board.hash).map(|e| e.best_move).unwrap_or(Move::NULL);
        if !tt_best.is_null()
            && let Some(pos) = root_move_scores.iter().position(|(m, _)| *m == tt_best)
            && pos > 0
        {
            root_move_scores.swap(0, pos);
        }
    }

    // EWMA (exponentially weighted moving average) of each root move's score
    // across all completed depth iterations. Used for best-move persistence:
    // if a move was consistently best for many depths but gets displaced at
    // the final depth due to search instability, the EWMA will still reflect
    // its historical strength.
    let mut root_move_ewma: Vec<(Move, f64)> = root_moves.iter().map(|&m| (m, 0.0f64)).collect();
    let mut ewma_initialized = false;

    // PV instability: track best move changes for time management
    let mut pv_changes = 0u32;
    // Best move stability: consecutive iterations with same best move
    let mut best_move_stability = 0u32;
    // Score volatility: max absolute swing between consecutive iterations
    let mut max_score_swing = 0i32;
    // Track last iteration duration for next-iteration prediction
    let mut last_iter_ms: u64;
    // Best-move persistence: remember the last highly stable move
    let mut stable_move = Move::NULL;
    let mut stable_move_stability = 0u32;


    for base_depth in 1..=max_depth {
        // Apply depth offset for helper threads (Lazy SMP depth diversity)
        let depth = (base_depth as i16 + offset as i16).clamp(1, max_depth as i16) as u8;
        if depth == 0 {
            continue;
        }

        // Sort root moves by previous iteration score (best first)
        if base_depth > 1 {
            root_move_scores.sort_by_key(|b| std::cmp::Reverse(b.1));
        }
        // If the best move has been highly stable, ensure it is searched first
        // at root regardless of TT or score ordering. In PVS, the first move
        // gets a full window while others get null windows — this ensures a
        // long-stable move isn't displaced by SMP TT noise.
        if best_move_stability >= 4 && !best_move.is_null() {
            for i in 1..root_move_scores.len() {
                if root_move_scores[i].0 == best_move {
                    root_move_scores.swap(0, i);
                    break;
                }
            }
        }

        let iter_start_ms = state.elapsed_ms();
        let mut best_node_fraction = 1.0f32;
        let multi_pv_count = params.multi_pv.max(1).min(root_move_scores.len());
        // Per-line results: (score, sanitized_pv, seldepth)
        let mut line_results: Vec<(i32, Vec<Move>, u8)> = Vec::with_capacity(multi_pv_count);
        let mut excluded_moves: Vec<Move> = Vec::with_capacity(multi_pv_count);
        let mut aspiration_failures = 0u32;
        let mut depth_completed = true;

        for line_idx in 0..multi_pv_count {
            // Build working_scores: excluded moves are pushed to the back with i32::MIN
            let mut working_scores: Vec<(Move, i32)> = root_move_scores
                .iter()
                .map(|&(m, s)| if excluded_moves.contains(&m) { (m, i32::MIN) } else { (m, s) })
                .collect();
            working_scores.sort_by_key(|b| std::cmp::Reverse(b.1));

            let mut pv = Vec::new();
            let score;

            if line_idx == 0 {
                // Aspiration windows: narrow window with progressive widening
                let (mut alpha, mut beta) = if depth >= 4 {
                    (prev_score - 30, prev_score + 30)
                } else {
                    (Score::NEG_INF.0, Score::INF.0)
                };
                let mut delta = 60i32;
                let mut s;
                loop {
                    pv.clear();
                    s = alpha_beta_root(
                        &mut board.clone(),
                        depth,
                        alpha,
                        beta,
                        &mut pv,
                        &mut state,
                        stop,
                        &mut working_scores,
                        &mut best_node_fraction,
                    );

                    if state.should_stop(stop) && depth > 1 {
                        // Don't abort during aspiration fail-low if we have time safety margin
                        let in_fail_low = s <= alpha;
                        let can_extend = state.time_limit_ms.is_some_and(|limit| {
                            state.elapsed_ms() < limit * 6 / 5
                                && state.time_remaining_ms >= 10_000
                                && state.time_remaining_ms > state.inc_ms * 8
                        });
                        if !(in_fail_low && can_extend) {
                            break;
                        }
                    }

                    if s <= alpha {
                        alpha = (s - delta).max(Score::NEG_INF.0);
                        delta += delta / 2;
                        aspiration_failures += 1;
                        continue;
                    }
                    if s >= beta {
                        beta = s + delta;
                        delta += delta / 2;
                        aspiration_failures += 1;
                        continue;
                    }
                    break;
                }
                score = s;
                // Propagate line 0 score ordering back to master root_move_scores
                for (m, ws) in &working_scores {
                    if let Some(entry) = root_move_scores.iter_mut().find(|(rm, _)| rm == m) {
                        entry.1 = *ws;
                    }
                }
            } else {
                // Full window for lines 2+
                score = alpha_beta_root(
                    &mut board.clone(),
                    depth,
                    Score::NEG_INF.0,
                    Score::INF.0,
                    &mut pv,
                    &mut state,
                    stop,
                    &mut working_scores,
                    &mut best_node_fraction,
                );
            }

            if state.should_stop(stop) && depth > 1 {
                depth_completed = false;
                break;
            }

            let line_best = if !pv.is_empty() { pv[0] } else { working_scores[0].0 };
            excluded_moves.push(line_best);
            line_results.push((score, sanitize_pv(board, &pv), state.seldepth));
        }

        last_iter_ms = state.elapsed_ms().saturating_sub(iter_start_ms);

        if !depth_completed && depth > 1 {
            break;
        }

        if line_results.is_empty() {
            break;
        }

        if let Some((tb_score, tb_pv)) = root_tb_solution.as_ref()
            && !tb_pv.is_empty()
        {
            let seldepth = line_results[0].2.max(tb_pv.len().min(u8::MAX as usize) as u8);
            line_results[0] = (tb_score.0, tb_pv.clone(), seldepth);
        }

        let score = line_results[0].0;
        let score_drop = if base_depth > 1 { (prev_score - score).max(0) } else { 0 };
        if base_depth > 1 {
            max_score_swing = max_score_swing.max((prev_score - score).abs());
        }
        prev_score = score;
        let line0_pv = &line_results[0].1;
        let new_best = if !line0_pv.is_empty() {
            line0_pv[0]
        } else {
            let tt_best = tt.probe(board.hash).map(|e| e.best_move).unwrap_or(Move::NULL);
            if !tt_best.is_null() && root_move_scores.iter().any(|(m, _)| *m == tt_best) {
                tt_best
            } else {
                root_move_scores[0].0
            }
        };
        if base_depth > 1 && new_best != best_move {
            pv_changes += 1;
            // Save the stable move before resetting
            if best_move_stability >= 6 {
                stable_move = best_move;
                stable_move_stability = best_move_stability;
            }
            best_move_stability = 0;
        } else if base_depth > 1 {
            best_move_stability += 1;
        }
        best_move = new_best;
        best_score = Score(score);
        best_pv = line_results[0].1.clone();
        best_depth = depth;

        // Update EWMA for all root moves after each completed depth
        {
            let alpha_coeff = if ewma_initialized { 0.2 } else { 1.0 };
            for (ewma_move, ewma_val) in root_move_ewma.iter_mut() {
                if let Some(&(_, score_val)) = root_move_scores.iter().find(|(m, _)| m == ewma_move) {
                    *ewma_val = alpha_coeff * score_val as f64 + (1.0 - alpha_coeff) * *ewma_val;
                }
            }
            ewma_initialized = true;
        }

        if let Some(ref cb) = info_callback {
            let elapsed = state.elapsed_ms();
            for (line_idx, (line_score, line_pv, line_seldepth)) in line_results.iter().enumerate() {
                cb(&SearchInfo {
                    depth,
                    seldepth: *line_seldepth,
                    score: Score(*line_score),
                    nodes: state.nodes,
                    time_ms: elapsed,
                    pv: line_pv.clone(),
                    hashfull: tt.hashfull(),
                    multipv_line: line_idx + 1,
                });
            }
        }

        if let Some(counter) = node_counter {
            counter.store(state.nodes, Ordering::Relaxed);
        }

        log::info!(
            "depth {} score {} nodes {} time {}ms pv {}",
            depth, best_score, state.nodes, state.elapsed_ms(),
            best_pv.iter().map(|m| m.to_uci()).collect::<Vec<_>>().join(" ")
        );

        // When we find a forced mate for us, keep searching for shorter mates
        // but stop if the mate is very short (within 3 moves) since
        // further improvement is unlikely, or if we've spent > 20% of time.
        // Only applies to winning mates (best_score > 0): when we are being
        // mated we must keep searching to find the best defence.
        if best_score.is_mate() && best_score.0 > 0 {
            let mate_distance = (Score::MATE.0 - best_score.0) as u32;
            if mate_distance <= 6 { // mate in 3 or fewer full moves
                break;
            }
            if let Some(limit) = state.time_limit_ms
                && state.elapsed_ms() > limit / 5
            {
                break;
            }
        }

        if state.should_stop(stop) {
            break;
        }

        if state.use_soft_limit
            && let Some(limit) = state.time_limit_ms
        {
            // Dynamic soft time limit based on multiple factors:
            // Base: increase in increment controls to avoid under-spending.
            let inc_ratio_permille = state.inc_ms
                .saturating_mul(1000)
                .checked_div(limit)
                .unwrap_or(0)
                .min(2000);
            let base_frac = if inc_ratio_permille >= 600 {
                860u64
            } else if inc_ratio_permille >= 350 {
                800u64
            } else if inc_ratio_permille >= 200 {
                740u64
            } else {
                650u64
            };

            // PV instability: +8% per change, up to +40%
            let instability_bonus = (pv_changes as u64).min(5) * 80;

            // Score drop: +5% per 25cp drop, up to +40%
            let drop_bonus = ((score_drop as u64) / 25).min(8) * 50;

            // Score volatility: +6% per 30cp swing, up to +30%
            let volatility_bonus = ((max_score_swing as u64) / 30).min(5) * 60;

            // Aspiration failures: +8% per failure, up to +32%
            let aspiration_bonus = (aspiration_failures as u64).min(4) * 80;

            // Root move complexity: +1% per move above 20, up to +20%
            let root_moves_count = root_move_scores.len() as u64;
            let complexity_bonus = root_moves_count.saturating_sub(20).min(20) * 10;

            // Increment safety net: when increment is significant relative
            // to allocated time, we can afford to spend more since it
            // replenishes. +0-15% based on inc/allocated ratio.
            let inc_bonus = state.inc_ms
                .saturating_mul(150)
                .checked_div(limit)
                .unwrap_or(0)
                .min(150);

            // Clock-surplus bonus: if we have a healthy bank beyond a
            // phase-aware reserve, bias more strongly toward spending now.
            let desired_bank = if state.inc_ms > 0 {
                if game_ply < 20 {
                    8_000u64.saturating_add(state.inc_ms.saturating_mul(16))
                } else if game_ply < 40 {
                    6_000u64.saturating_add(state.inc_ms.saturating_mul(12))
                } else if game_ply < 60 {
                    5_000u64.saturating_add(state.inc_ms.saturating_mul(9))
                } else {
                    4_000u64.saturating_add(state.inc_ms.saturating_mul(7))
                }
            } else if game_ply < 40 {
                5_000
            } else {
                3_000
            };
            let surplus = state.time_remaining_ms.saturating_sub(desired_bank);
            let surplus_bonus = if limit > 0 {
                (surplus.saturating_mul(320) / (limit.saturating_mul(8).max(1))).min(320)
            } else {
                0
            };

            // Blitz safety: in low time, bias toward spending less of the
            // allocated budget to reduce practical timeout risk.
            let blitz_adjust: i64 = if state.time_remaining_ms <= 15_000 {
                -180 // -18%
            } else if state.time_remaining_ms <= 30_000 {
                -80 // -8%
            } else {
                0
            };

            // Best move stability: -3% per stable iteration, up to -15%
            let stability_discount = (best_move_stability as u64).min(5) * 30;

            // Best-move node fraction: if >80% nodes go to best move,
            // position is decided → stop sooner. If <50%, unclear → longer.
            let node_frac_adj: i64 = if best_node_fraction > 0.80 {
                -100 // stop 10% sooner
            } else if best_node_fraction < 0.50 {
                200 // search 20% longer
            } else {
                0
            };

            // Score-aware adjustment:
            // - Spend less in clearly decided and stable positions (won/lost)
            // - Spend more in unclear or volatile positions.
            let eval_abs = best_score.0.saturating_abs() as u64;
            let eval_adjust: i64 = if best_score.is_mate() {
                // Once mate is visible, prioritize conversion speed.
                -220
            } else if eval_abs >= 500 && best_move_stability >= 2 && max_score_swing <= 40 {
                -220
            } else if eval_abs >= 300 && best_move_stability >= 2 && max_score_swing <= 60 {
                -140
            } else if eval_abs <= 50 && max_score_swing >= 35 {
                180
            } else if eval_abs <= 90 && max_score_swing >= 25 {
                100
            } else {
                0
            };

            // Early opening brake: keep first moves snappier and save heavy
            // spending for richer middlegame positions. Apply to all time controls.
            let opening_adjust: i64 = if game_ply < 6 {
                if state.time_remaining_ms <= 300_000 { -240 } else { -160 }
            } else if game_ply < 12 {
                if state.time_remaining_ms <= 300_000 { -140 } else { -90 }
            } else if game_ply < 20 {
                if state.time_remaining_ms <= 300_000 { -60 } else { -30 }
            } else {
                0
            };

            let soft_frac = (base_frac + instability_bonus + drop_bonus
                + volatility_bonus + aspiration_bonus + complexity_bonus
                + inc_bonus + surplus_bonus + book_exit_bonus)
                .saturating_sub(stability_discount);
            let soft_frac = (soft_frac as i64 + node_frac_adj + eval_adjust + opening_adjust + blitz_adjust).max(0) as u64;
            // Safer range: 35%-180% of allocated time
            let soft_frac = soft_frac.clamp(350, 1800);
            let soft_limit = limit * soft_frac / 1000;
            let elapsed = state.elapsed_ms();
            if elapsed > soft_limit {
                break;
            }
            // Predict whether the next iteration can complete in time.
            // Each deeper iteration typically takes 2-3x the previous one.
            // Estimate next iteration cost as ~3x the last one (conservative
            // branching factor). Stop if it would exceed the soft limit.
            // In unstable/tactical positions, the next iteration growth is
            // often closer to ~2x than ~3x; using 3x can stop too early.
            // Keep the conservative 3x in quiet/stable nodes.
            let unstable = pv_changes > 0 || max_score_swing >= 35 || best_node_fraction < 0.55;
            let growth = if unstable { 2 } else { 3 };
            let predicted_next = last_iter_ms * growth;
            let prediction_margin_permille = if surplus > limit {
                980u64
            } else if inc_ratio_permille >= 350 {
                950u64
            } else {
                900u64
            };
            if depth >= 6
                && elapsed + predicted_next > soft_limit.saturating_mul(prediction_margin_permille) / 1000
            {
                break;
            }
        }
    }

    // Best-move persistence using EWMA: if a highly stable move was
    // displaced and its EWMA (score averaged across all depths) is
    // significantly higher than the current best move's EWMA, revert.
    // This handles search instability where a tactically complex move
    // scores well for many depths but then drops at the final depth.
    if !stable_move.is_null() && best_move != stable_move && stable_move_stability >= 6 {
        let stable_ewma = root_move_ewma.iter().find(|(m, _)| *m == stable_move).map(|(_, e)| *e);
        let best_ewma = root_move_ewma.iter().find(|(m, _)| *m == best_move).map(|(_, e)| *e);
        if let (Some(se), Some(be)) = (stable_ewma, best_ewma)
            && se > be + 50.0
        {
            log::info!(
                "EWMA persistence: {} (ewma {:.0}) over {} (ewma {:.0}), stability was {}",
                stable_move.to_uci(), se, best_move.to_uci(), be, stable_move_stability
            );
            best_move = stable_move;
            // Report the EWMA as the score (better reflects the move's value)
            best_score = Score(se as i32);
        }
    }

    if best_move.is_null() {
        best_move = root_move_scores[0].0;
    }

    if best_pv.is_empty()
        && let Some((tb_score, tb_pv)) = root_tb_solution
        && !tb_pv.is_empty()
    {
        best_move = tb_pv[0];
        best_score = tb_score;
        best_pv = tb_pv;
        best_depth = best_depth.max(best_pv.len().min(u8::MAX as usize) as u8);
    }

    // Swap the (now-updated) learning tables back into the persistent holder.
    if let Some(h) = persistent {
        state.swap_history(h);
    }

    SearchResult {
        best_move,
        score: best_score,
        depth: best_depth,
        nodes: state.nodes,
        pv: sanitize_pv(board, &best_pv),
    }
}

// ---------------------------------------------------------------------------
// Root-specific alpha-beta (uses pre-ordered root move list)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn alpha_beta_root(
    board: &mut Board,
    depth: u8,
    mut alpha: i32,
    beta: i32,
    pv: &mut Vec<Move>,
    state: &mut SearchState,
    stop: &AtomicBool,
    root_move_scores: &mut [(Move, i32)],
    best_node_fraction: &mut f32,
) -> i32 {
    pv.clear();

    if state.nodes & 1023 == 0 && state.should_stop(stop) {
        state.stop = true;
        return 0;
    }

    state.nodes += 1;
    let in_check = chess_core::is_in_check(board);
    let effective_depth = if in_check { depth + 1 } else { depth };

    // TT probe (for move ordering at root, not cutoff)
    let tt_entry = state.tt.probe(board.hash);
    let tt_move = tt_entry.map(|e| e.best_move).unwrap_or(Move::NULL);

    // Ensure TT move is first in root moves if present, but don't promote excluded moves (score == i32::MIN)
    if !tt_move.is_null() {
        for i in 0..root_move_scores.len() {
            if root_move_scores[i].0 == tt_move && i > 0 && root_move_scores[i].1 != i32::MIN {
                root_move_scores.swap(0, i);
                break;
            }
        }
    }

    let mut best_score = Score::NEG_INF.0;
    let mut best_move = root_move_scores[0].0;
    let mut child_pv = Vec::new();
    let mut tmp_pv: Vec<Move> = Vec::new();
    let mut tt_store_flag = TTFlag::UpperBound;
    let total_nodes_start = state.nodes;
    let mut best_move_nodes: u64 = 0;
    for (moves_searched, (m, move_score)) in root_move_scores.iter_mut().enumerate() {
        if *move_score == i32::MIN {
            continue; // excluded by multi-PV logic
        }
        let m = *m;
        let move_nodes_start = state.nodes;

        let prev_castling = board.castling;
        let prev_ep = board.en_passant;
        let prev_halfmove = board.halfmove_clock;

        // Store ply context BEFORE make_move (piece is still on from_sq)
        state.store_ply_context(0, m, board);

        let captured = board.make_move(m);

        // Incremental accumulator update for root (ply 0 → ply 1)
        if state.use_nnue {
            update_accumulator_for_move(state, board, m, captured, 0, 1);
        }

        let gives_check = chess_core::is_in_check(board);

        let search_depth = effective_depth - 1;
        let moves_searched = moves_searched as u32;

        let score = if moves_searched == 0 {
            -alpha_beta(
                board, search_depth, 1, -beta, -alpha,
                &mut child_pv, state, stop, m, Move::NULL,
            )
        } else {
            let mut do_full_search = true;
            let mut reduced_score = 0;

            // LMR at root for late moves
            if moves_searched >= 3
                && depth >= 3
                && !in_check
                && !gives_check
                && !m.is_capture()
                && !m.is_promotion()
            {
                let d = (depth as usize).min(63);
                let ms = (moves_searched as usize).min(63);
                let reduction = (state.lmr_table[d][ms] as i8).clamp(0, (search_depth as i8 - 1).max(0));

                let reduced_depth = search_depth.saturating_sub(reduction as u8);
                tmp_pv.clear();
                reduced_score = -alpha_beta(
                    board, reduced_depth, 1, -alpha - 1, -alpha,
                    &mut tmp_pv, state, stop, m, Move::NULL,
                );

                if reduced_score <= alpha {
                    do_full_search = false;
                    std::mem::swap(&mut child_pv, &mut tmp_pv);
                }
            }

            if do_full_search {
                tmp_pv.clear();
                let nw_score = -alpha_beta(
                    board, search_depth, 1, -alpha - 1, -alpha,
                    &mut tmp_pv, state, stop, m, Move::NULL,
                );

                if nw_score > alpha && nw_score < beta {
                    -alpha_beta(
                        board, search_depth, 1, -beta, -alpha,
                        &mut child_pv, state, stop, m, Move::NULL,
                    )
                } else {
                    std::mem::swap(&mut child_pv, &mut tmp_pv);
                    nw_score
                }
            } else {
                reduced_score
            }
        };

        board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);

        // Store score for root move ordering in next iteration
        *move_score = score;

        if state.stop {
            return 0;
        }

        let move_nodes = state.nodes - move_nodes_start;

        if score > best_score {
            best_score = score;
            best_move = m;
            best_move_nodes = move_nodes;

            if score > alpha {
                alpha = score;
                tt_store_flag = TTFlag::Exact;

                pv.clear();
                pv.push(m);
                pv.extend_from_slice(&child_pv);

                if score >= beta {
                    tt_store_flag = TTFlag::LowerBound;
                    break;
                }
            }
        }
    }

    // Compute best-move node fraction for time management
    let total_nodes = state.nodes - total_nodes_start;
    if total_nodes > 0 {
        *best_node_fraction = best_move_nodes as f32 / total_nodes as f32;
    }

    state.tt.store(board.hash, effective_depth, score_to_tt(best_score, 0), tt_store_flag, best_move);

    best_score
}

// ---------------------------------------------------------------------------
// Alpha-beta search
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn alpha_beta(
    board: &mut Board,
    depth: u8,
    ply: u8,
    alpha: i32,
    beta: i32,
    pv: &mut Vec<Move>,
    state: &mut SearchState,
    stop: &AtomicBool,
    prev_move: Move,
    excluded_move: Move,
) -> i32 {
    pv.clear();

    if state.nodes & 1023 == 0 && state.should_stop(stop) {
        state.stop = true;
        return 0;
    }

    // Ply limit guard: prevent stack overflow and out-of-bounds accumulator access.
    // With non-zero contempt the engine avoids draws, potentially driving ply past
    // MAX_PLY (and eventually wrapping the u8 ply counter).  Cap it here.
    if ply as usize >= MAX_PLY {
        let eval = evaluate(board).0;
        return match board.side_to_move {
            Color::White => eval,
            Color::Black => -eval,
        };
    }

    // Draw detection
    if ply > 0 {
        if board.halfmove_clock >= 100 {
            return -state.contempt;
        }
        // Twofold in the current search path: standard practice since the
        // opponent can always force a third repetition.
        if board.is_twofold_in_search(state.game_ply) {
            return -state.contempt;
        }
        // Threefold across the full history (catches game-history positions).
        if board.is_repetition() {
            return -state.contempt;
        }
        // Twofold in recent move history: discourage shuffling
        if board.has_repeated(state.game_ply) {
            return -state.contempt;
        }
    }

    // -------------------------------------------------------------------
    // Mate distance pruning (MDP)
    // -------------------------------------------------------------------
    // Tighten alpha/beta based on the shortest possible mate from this
    // node. If we already know a mate-in-N, any subtree that can't
    // beat it is pruned immediately, dramatically speeding up mate
    // searches.
    let (mut alpha, mut beta) = (alpha, beta);
    if ply > 0 {
        let mating_value = Score::MATE.0 - ply as i32;
        if mating_value < beta {
            beta = mating_value;
            if alpha >= beta {
                return alpha;
            }
        }
        let mated_value = -Score::MATE.0 + ply as i32;
        if mated_value > alpha {
            alpha = mated_value;
            if alpha >= beta {
                return beta;
            }
        }
    }

    state.nodes += 1;
    let in_check = chess_core::is_in_check(board);

    // Leaf node: only enter qsearch if NOT in check.
    // Positions in check must get at least 1 ply of full search to avoid
    // missing checkmates and quiet evasion moves.
    if depth == 0 && !in_check {
        return quiescence(board, ply, alpha, beta, pv, state, stop, true);
    }

    let is_root = ply == 0;
    let is_pv = beta - alpha > 1;

    // Check extension: ensure at least depth 1 when in check
    let effective_depth = if in_check { depth.max(1) } else { depth };

    // Syzygy WDL probe: probe tablebases for positions with no castling rights
    // and a piece count within the loaded tablebase range.  Returns an exact
    // game-theoretic score that is used to bound (or immediately return from)
    // this node.  We skip probing during singular-extension searches to avoid
    // interfering with the null-window test.
    if let Some(ref tb) = state.syzygy_tb {
        let piece_count = board.all_occupancy().count();
        if piece_count <= tb.max_pieces()
            && board.castling.0 == 0
            && excluded_move.is_null()
            && let Some(wdl) = syzygy::probe_wdl(tb, board)
        {
            use pyrrhic_rs::WdlProbeResult;
            let tb_score = syzygy::wdl_to_score(wdl);
            let flag = match wdl {
                WdlProbeResult::Win | WdlProbeResult::CursedWin => TTFlag::LowerBound,
                WdlProbeResult::Loss | WdlProbeResult::BlessedLoss => TTFlag::UpperBound,
                WdlProbeResult::Draw => TTFlag::Exact,
            };
            state.tt.store(board.hash, effective_depth, score_to_tt(tb_score, ply), flag, Move::NULL);
            match flag {
                TTFlag::LowerBound => {
                    if tb_score >= beta {
                        return tb_score;
                    }
                    alpha = alpha.max(tb_score);
                }
                TTFlag::UpperBound => {
                    if tb_score <= alpha {
                        return tb_score;
                    }
                    beta = beta.min(tb_score);
                }
                TTFlag::Exact => return tb_score,
            }
        }
    }

    // TT probe
    let tt_entry = state.tt.probe(board.hash);
    let (tt_move, tt_score, tt_depth, tt_flag) =
        if let Some(entry) = tt_entry {
            let adj_score = score_from_tt(entry.score, ply);
            if !is_pv && !is_root && entry.depth >= effective_depth && excluded_move.is_null() {
                match entry.flag {
                    TTFlag::Exact => return adj_score,
                    TTFlag::LowerBound => {
                        if adj_score >= beta { return adj_score; }
                    }
                    TTFlag::UpperBound => {
                        if adj_score <= alpha { return adj_score; }
                    }
                }
            }
            (entry.best_move, adj_score, entry.depth, entry.flag)
        } else {
            (Move::NULL, 0, 0, TTFlag::UpperBound)
        };

    // In check: static eval is unreliable and all eval-based pruning is
    // skipped. Use a sentinel so `improving` is true after escaping check
    // (erring on the side of searching more — the safe direction).
    let static_eval = if in_check {
        i32::MIN / 2
    } else {
        let raw = evaluate_for_side(board, state, ply);
        // Refine with TT bound: the search result at this position is
        // often a more accurate estimate than the static eval.
        if tt_entry.is_some() && tt_score.abs() < MATE_THRESHOLD {
            match tt_flag {
                TTFlag::Exact => tt_score,
                TTFlag::LowerBound if tt_score > raw => tt_score,
                TTFlag::UpperBound if tt_score < raw => tt_score,
                _ => raw,
            }
        } else {
            raw
        }
    };

    if (ply as usize) < MAX_PLY {
        state.static_evals[ply as usize] = static_eval;
    }

    let improving = !in_check
        && ply >= 2
        && (ply as usize) < MAX_PLY
        && static_eval > state.static_evals[(ply as usize) - 2];

    // -----------------------------------------------------------------------
    // Pre-move pruning
    // -----------------------------------------------------------------------
    // Skip all pre-move pruning when the search window contains mate
    // scores. These heuristics return centipawn values and would discard
    // the forced-mate proof we are trying to build.
    let searching_for_mate = !(-MATE_THRESHOLD..=MATE_THRESHOLD).contains(&alpha)
        || !(-MATE_THRESHOLD..=MATE_THRESHOLD).contains(&beta);

    if !is_pv && !in_check && !is_root && excluded_move.is_null() && !searching_for_mate {
        // Reverse futility pruning
        if depth <= 6 {
            let margin = (if improving { state.tune.rfp_margin_imp } else { state.tune.rfp_margin_noimp }) * depth as i32;
            if static_eval - margin >= beta {
                return static_eval - margin;
            }
        }

        // Null move pruning
        if depth >= 3
            && static_eval >= beta
            && has_non_pawn_material(board, board.side_to_move)
        {
            let null_r = 3 + depth as u32 / 4 + ((static_eval - beta) as u32 / 200).min(3);
            let null_depth = depth.saturating_sub(1 + null_r as u8);

            let prev_ep = board.en_passant;
            let prev_hash = board.hash;
            // Incremental hash update: XOR out old ep file (if any) + flip side.
            // Must happen before en_passant is cleared.
            board.update_hash_for_null_move();
            board.side_to_move = board.side_to_move.opposite();
            board.en_passant = None;

            // Null move doesn't change pieces — just copy the accumulator
            if state.use_nnue && ply as usize + 1 < MAX_PLY {
                state.accumulators[ply as usize + 1] = state.accumulators[ply as usize].clone();
            }

            let mut null_pv = Vec::new();
            let null_score = -alpha_beta(
                board, null_depth, ply + 1, -beta, -beta + 1,
                &mut null_pv, state, stop, Move::NULL, Move::NULL,
            );

            board.side_to_move = board.side_to_move.opposite();
            board.en_passant = prev_ep;
            board.hash = prev_hash;

            if null_score >= beta {
                return beta;
            }
        }

        // Probcut: if a shallow search on captures beats beta by a margin,
        // the full-depth search is very likely to beat beta too.
        if depth >= 5 && !searching_for_mate {
            let probcut_beta = beta + 200;
            if static_eval >= probcut_beta - 100 {
                let probcut_depth = depth - 4;
                let pc_us = board.side_to_move;
                let captures = chess_core::generate_pseudo_legal_captures(board);

                for &m in captures.iter() {
                    if !see::see_ge(board, m, probcut_beta - static_eval) {
                        continue;
                    }

                    let prev_castling = board.castling;
                    let prev_ep = board.en_passant;
                    let prev_halfmove = board.halfmove_clock;

                    state.store_ply_context(ply, m, board);
                    if board.piece_at(m.from_sq()).is_none() {
                        continue;
                    }
                    let captured = board.make_move(m);

                    if move_is_illegal(board, pc_us) {
                        board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);
                        continue;
                    }

                    // Incremental accumulator update for probcut
                    if state.use_nnue {
                        update_accumulator_for_move(state, board, m, captured, ply as usize, ply as usize + 1);
                    }

                    state.nodes += 1;
                    let mut pc_pv = Vec::new();
                    let score = -quiescence(
                        board, ply + 1, -probcut_beta, -probcut_beta + 1,
                        &mut pc_pv, state, stop, true,
                    );

                    let score = if score >= probcut_beta {
                        -alpha_beta(
                            board, probcut_depth, ply + 1, -probcut_beta, -probcut_beta + 1,
                            &mut pc_pv, state, stop, m, Move::NULL,
                        )
                    } else {
                        score
                    };

                    board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);

                    if score >= probcut_beta {
                        state.tt.store(
                            board.hash, depth, score_to_tt(score, ply),
                            TTFlag::LowerBound, m,
                        );
                        return score;
                    }
                }
            }
        }

        // Razoring
        if depth <= 3 {
            let razor_margin = 300 + 200 * depth as i32;
            if static_eval + razor_margin <= alpha {
                let qscore = quiescence(board, ply, alpha, beta, pv, state, stop, true);
                if qscore <= alpha {
                    return qscore;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // IIR: Internal Iterative Reduction
    // -----------------------------------------------------------------------
    // When no TT move exists, reduce depth by 1 instead of doing a full
    // recursive IID search. Much cheaper and nearly as effective.
    let iir_reduction = if tt_move.is_null() && depth >= 3 {
        if !is_pv && depth >= 8 { 2i8 } else { 1 }
    } else {
        0
    };

    // -----------------------------------------------------------------------
    // Staged move generation via MovePicker
    // -----------------------------------------------------------------------
    // Moves are generated lazily: TT move → captures → quiets → bad captures.
    // Legality is checked inline after make_move (no board cloning).

    let killers = if (ply as usize) < MAX_PLY {
        state.killers[ply as usize]
    } else {
        [Move::NULL; MAX_KILLERS]
    };
    let counter_move = state.get_counter_move(prev_move);
    let us = board.side_to_move;

    let mut picker = MovePicker::new(tt_move);

    let mut best_score = Score::NEG_INF.0;
    let mut best_move = Move::NULL;
    let mut child_pv = Vec::new();
    let mut tt_store_flag = TTFlag::UpperBound;
    let mut moves_searched = 0u32;
    // Fixed-size stack arrays avoid a heap allocation per node.
    // 128 slots is far more than any real search node will fill before a cutoff.
    let mut quiet_moves_searched = [Move::NULL; 128];
    let mut quiet_count: usize = 0;
    let mut captures_searched = [Move::NULL; 128];
    let mut capture_count: usize = 0;
    // Single reusable buffer for LMR/null-window PV — cleared before each use.
    let mut tmp_pv: Vec<Move> = Vec::new();

    while let Some(m) = picker.next(board, state, ply, &killers, counter_move) {
        if m == excluded_move && !excluded_move.is_null() {
            continue;
        }

        // -------------------------------------------------------------------
        // Singular extension (BEFORE making the move)
        // -------------------------------------------------------------------
        let mut extension: i8 = 0;
        let singular_mode = state.singular_ext_mode;
        let singular_candidate = !is_root
            && !in_check
            && moves_searched == 0
            && m == tt_move
            && !tt_move.is_null()
            && excluded_move.is_null()
            && tt_score.abs() < MATE_THRESHOLD;
        let conservative_gate = singular_mode == 1
            && !is_pv
            && depth >= 8
            && tt_depth >= depth - 2
            && tt_flag == TTFlag::LowerBound
            && tt_score >= alpha + 50;
        let aggressive_gate = singular_mode >= 2
            && depth >= 7
            && tt_depth >= depth - 3
            && tt_flag != TTFlag::UpperBound
            && tt_score >= alpha + 30;
        if singular_mode > 0 && singular_candidate && (conservative_gate || aggressive_gate) {
            let se_beta = tt_score - 2 * depth as i32;
            let se_depth = (depth - 1) / 2;

            let mut se_pv = Vec::new();
            let se_score = alpha_beta(
                board, se_depth, ply, se_beta - 1, se_beta,
                &mut se_pv, state, stop, prev_move, m,
            );

            if se_score < se_beta {
                extension = 1;
            }
        }

        // -------------------------------------------------------------------
        // Passed pawn push extension: pawn to 7th rank or promotion
        // These moves create immediate promotion threats that drastically
        // change the position.  Extending ensures the search sees the
        // consequences (promotion, new queen activity, mating attacks).
        // -------------------------------------------------------------------
        if extension == 0 {
            if let Some(piece) = board.piece_at(m.from_sq())
                && piece.kind == PieceKind::Pawn
            {
                let to_rank = m.to_sq().rank();
                let is_7th = match piece.color {
                    Color::White => to_rank == 6,
                    Color::Black => to_rank == 1,
                };
                if is_7th {
                    extension = 1;
                }
            }
            // Also extend promotion moves (queen promotions are critical)
            if m.is_promotion() {
                extension = 1;
            }
        }

        let prev_castling = board.castling;
        let prev_ep = board.en_passant;
        let prev_halfmove = board.halfmove_clock;

        // -------------------------------------------------------------------
        // Pre-make pruning (SEE on PRE-move board)
        // -------------------------------------------------------------------
        if moves_searched > 0 && !is_pv && !in_check {
            // SEE pruning for bad captures (extended to depth 7)
            // Must be done BEFORE make_move — SEE reads the pre-move board.
            if depth <= 7 && m.is_capture() {
                let see_threshold = -30 * depth as i32;
                if !see::see_ge(board, m, see_threshold) {
                    continue;
                }
            }

            // SEE pruning for quiet moves: prune moves that lose material
            // (e.g., moving a piece to a square attacked by a pawn)
            if depth <= 5 && !m.is_capture() && !m.is_promotion()
                && !see::see_ge(board, m, -state.tune.see_quiet_margin * depth as i32)
            {
                continue;
            }
        }

        // Store ply context BEFORE make_move (piece is still on from_sq)
        state.store_ply_context(ply, m, board);

        // Safety guard: skip move if from-square is empty (board corruption
        // from a hash-collision TT move that slipped past is_move_safe).
        if board.piece_at(m.from_sq()).is_none() {
            continue;
        }

        let captured = board.make_move(m);

        // ── Inline legality check ──────────────────────────────────────
        if move_is_illegal(board, us) {
            board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);
            continue;
        }

        let gives_check = chess_core::is_in_check(board);

        // -------------------------------------------------------------------
        // Move pruning (non-first moves, non-PV)
        // -------------------------------------------------------------------
        if moves_searched > 0 && !is_pv && !in_check {
            // Futility pruning for quiet moves (extended to depth 5)
            if !gives_check && depth <= 5 && !m.is_capture() && !m.is_promotion()
                && !searching_for_mate
            {
                let futility_margin = (if improving { state.tune.fut_margin_imp } else { state.tune.fut_margin_noimp }) * depth as i32 + 50;
                if static_eval + futility_margin <= alpha {
                    board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);
                    continue;
                }
            }

            // Late move pruning (extended to depth 6)
            if !gives_check && depth <= 6 && !m.is_capture() && !m.is_promotion()
                && !searching_for_mate
            {
                let lmp_threshold = if improving {
                    3 + depth as usize * depth as usize
                } else {
                    (3 + depth as usize * depth as usize) / 2
                };
                if moves_searched as usize >= lmp_threshold {
                    board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);
                    continue;
                }
            }

            // History pruning: prune quiet moves with consistently terrible history
            if !gives_check && depth <= 4 && !m.is_capture() && !m.is_promotion()
                && !searching_for_mate
            {
                let hist = state.history[m.from_sq().index()][m.to_sq().index()];
                let cont = state.get_cont_history_bonus(m, board, ply);
                if hist + cont / 2 < -3000 * depth as i32 {
                    board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);
                    continue;
                }
            }
        }

        // Incremental accumulator update — deferred until after pruning so
        // moves that get pruned away never pay this cost.
        if state.use_nnue && ply as usize + 1 < MAX_PLY {
            update_accumulator_for_move(state, board, m, captured, ply as usize, ply as usize + 1);
        }

        // -------------------------------------------------------------------
        // Search this move
        // -------------------------------------------------------------------
        let search_depth = (effective_depth as i8 - 1 + extension - iir_reduction).max(0) as u8;

        let score = if moves_searched == 0 {
            // First move: full window
            -alpha_beta(
                board, search_depth, ply + 1, -beta, -alpha,
                &mut child_pv, state, stop, m, Move::NULL,
            )
        } else {
            let mut do_full_search = true;
            let mut reduced_score = 0;

            // Late move reductions (logarithmic table)
            if moves_searched >= 2
                && depth >= 3
                && !in_check
                && !gives_check
                && !m.is_capture()
                && !m.is_promotion()
            {
                let d = (depth as usize).min(63);
                let ms = (moves_searched as usize).min(63);
                let mut reduction = state.lmr_table[d][ms] as i8;

                // Reduce less at PV nodes
                if is_pv { reduction -= 1; }
                // Reduce more at non-PV (likely cut) nodes
                if !is_pv { reduction += 1; }
                // Reduce less for killers/counter-moves
                if m == killers[0] || m == killers[1] || m == counter_move {
                    reduction -= 1;
                }
                // Reduce more when not improving
                if !improving { reduction += 1; }
                // Continuous history-based reduction with continuation history:
                // good history → less reduction, bad history → more reduction.
                let hist = state.history[m.from_sq().index()][m.to_sq().index()];
                let cont = state.get_cont_history_bonus(m, board, ply);
                reduction -= ((hist + cont / 2) / state.tune.hist_lmr_div) as i8;
                // Extra reduction for very negative history
                if hist < -4000 { reduction += 1; }
                // Reduce more when eval is below alpha (position looks bad)
                if !is_pv && static_eval + 150 < alpha { reduction += 1; }

                reduction = reduction.clamp(0, (search_depth as i8 - 1).max(0));

                let reduced_depth = search_depth.saturating_sub(reduction as u8);
                tmp_pv.clear();
                reduced_score = -alpha_beta(
                    board, reduced_depth, ply + 1, -alpha - 1, -alpha,
                    &mut tmp_pv, state, stop, m, Move::NULL,
                );

                if reduced_score <= alpha {
                    do_full_search = false;
                    std::mem::swap(&mut child_pv, &mut tmp_pv);
                }
            }

            if do_full_search {
                // Null window search
                tmp_pv.clear();
                let nw_score = -alpha_beta(
                    board, search_depth, ply + 1, -alpha - 1, -alpha,
                    &mut tmp_pv, state, stop, m, Move::NULL,
                );

                if nw_score > alpha && nw_score < beta {
                    -alpha_beta(
                        board, search_depth, ply + 1, -beta, -alpha,
                        &mut child_pv, state, stop, m, Move::NULL,
                    )
                } else {
                    std::mem::swap(&mut child_pv, &mut tmp_pv);
                    nw_score
                }
            } else {
                reduced_score
            }
        };

        board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);

        if !m.is_capture() && !m.is_promotion() && quiet_count < 128 {
            quiet_moves_searched[quiet_count] = m;
            quiet_count += 1;
        } else if m.is_capture() && capture_count < 128 {
            captures_searched[capture_count] = m;
            capture_count += 1;
        }
        moves_searched += 1;

        if state.stop {
            return 0;
        }

        if score > best_score {
            best_score = score;
            best_move = m;

            if score > alpha {
                alpha = score;
                tt_store_flag = TTFlag::Exact;

                pv.clear();
                pv.push(m);
                pv.extend_from_slice(&child_pv);

                if score >= beta {
                    tt_store_flag = TTFlag::LowerBound;

                    if !m.is_capture() {
                        state.store_killer(ply as usize, m);
                        state.update_history(m, depth, board, ply);
                        state.store_counter_move(prev_move, m);

                        // History malus: penalize all previously searched quiet moves
                        for &qm in &quiet_moves_searched[..quiet_count] {
                            if qm != m {
                                state.update_history_malus(qm, depth, board, ply);
                            }
                        }
                    } else {
                        // Capture history: bonus for the cutoff capture
                        state.update_capture_history(m, depth, board);
                        // Malus for previously searched captures
                        for &cm in &captures_searched[..capture_count] {
                            if cm != m {
                                state.update_capture_history_malus(cm, depth, board);
                            }
                        }
                    }

                    break;
                }
            }
        }
    }

    // No legal move found → checkmate or stalemate.
    // When inside a singular-extension search, the TT move is excluded
    // and may be the only legal move, so return alpha instead.
    if moves_searched == 0 {
        if !excluded_move.is_null() {
            return alpha;
        }
        return if in_check {
            -Score::MATE.0 + ply as i32
        } else {
            Score::DRAW.0
        };
    }

    if excluded_move.is_null() {
        // Update pawn corrhist from the residual between the search result and
        // the static eval — but only from trustworthy nodes: not in check, a
        // quiet best move, non-mate score, and a bound that agrees with the sign
        // of the residual.
        if !in_check
            && !best_move.is_null()
            && !best_move.is_capture()
            && best_score < MATE_THRESHOLD
            && best_score > -MATE_THRESHOLD
            && match tt_store_flag {
                TTFlag::Exact => true,
                TTFlag::LowerBound => best_score >= static_eval,
                TTFlag::UpperBound => best_score <= static_eval,
            }
        {
            state.update_corrhist(board, depth, best_score - static_eval, ply);
        }
        state.tt.store(board.hash, effective_depth, score_to_tt(best_score, ply), tt_store_flag, best_move);
    }

    best_score
}

// ---------------------------------------------------------------------------
// Quiescence search
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn quiescence(
    board: &mut Board,
    ply: u8,
    mut alpha: i32,
    beta: i32,
    pv: &mut Vec<Move>,
    state: &mut SearchState,
    stop: &AtomicBool,
    generate_checks: bool,
) -> i32 {
    pv.clear();

    if state.nodes & 1023 == 0 && state.should_stop(stop) {
        state.stop = true;
        return 0;
    }

    // Ply limit guard (mirrors the alpha_beta guard).
    if ply as usize >= MAX_PLY {
        let eval = evaluate(board).0;
        let stand_pat = match board.side_to_move {
            Color::White => eval,
            Color::Black => -eval,
        };
        return stand_pat.clamp(alpha, beta);
    }

    // Draw detection (mirrors alpha_beta, needed when depth=0 during a
    // repetition cycle and quiescence is entered without going back through
    // alpha_beta's draw check).
    if ply > 0 {
        if board.halfmove_clock >= 100 {
            return -state.contempt;
        }
        if board.is_twofold_in_search(state.game_ply) {
            return -state.contempt;
        }
        if board.is_repetition() {
            return -state.contempt;
        }
        if board.has_repeated(state.game_ply) {
            return -state.contempt;
        }
    }

    state.nodes += 1;

    if ply > state.seldepth {
        state.seldepth = ply;
    }

    let in_check = chess_core::is_in_check(board);

    // TT probe in qsearch
    let tt_entry = state.tt.probe(board.hash);
    if let Some(entry) = tt_entry
        && !in_check
    {
        let adj_score = score_from_tt(entry.score, ply);
        match entry.flag {
            TTFlag::Exact => return adj_score,
            TTFlag::LowerBound => {
                if adj_score >= beta { return adj_score; }
            }
            TTFlag::UpperBound => {
                if adj_score <= alpha { return adj_score; }
            }
        }
    }
    let tt_move = tt_entry.map(|e| e.best_move).unwrap_or(Move::NULL);

    // When in check: search ALL moves (not just captures), skip stand-pat.
    // Uses pseudo-legal generation with inline legality instead of cloning.
    if in_check {
        let pseudo = chess_core::generate_pseudo_legal_moves(board);
        let us = board.side_to_move;

        let mut best_score = Score::NEG_INF.0;
        let mut best_move = Move::NULL;
        let mut child_pv = Vec::new();
        let mut any_legal = false;

        for &m in pseudo.iter() {
            // Safety guard: skip if from-square is empty
            if board.piece_at(m.from_sq()).is_none() {
                continue;
            }
            let prev_castling = board.castling;
            let prev_ep = board.en_passant;
            let prev_halfmove = board.halfmove_clock;
            let captured = board.make_move(m);

            // Inline legality check
            if move_is_illegal(board, us) {
                board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);
                continue;
            }
            any_legal = true;

            if state.use_nnue && ply as usize + 1 < MAX_PLY {
                update_accumulator_for_move(state, board, m, captured, ply as usize, ply as usize + 1);
            }

            let score = -quiescence(board, ply + 1, -beta, -alpha, &mut child_pv, state, stop, false);

            board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);

            if state.stop {
                return 0;
            }

            if score > best_score {
                best_score = score;
                best_move = m;

                if score > alpha {
                    alpha = score;
                    pv.clear();
                    pv.push(m);
                    pv.extend_from_slice(&child_pv);

                    if score >= beta {
                        state.tt.store(board.hash, 0, score_to_tt(best_score, ply), TTFlag::LowerBound, best_move);
                        return beta;
                    }
                }
            }
        }

        if !any_legal {
            // Checkmate
            return -Score::MATE.0 + ply as i32;
        }

        let flag = if best_score > alpha { TTFlag::Exact } else { TTFlag::UpperBound };
        state.tt.store(board.hash, 0, score_to_tt(best_score, ply), flag, best_move);
        return alpha;
    }

    // Not in check: normal qsearch with stand-pat
    let stand_pat = evaluate_for_side(board, state, ply);

    if stand_pat >= beta {
        state.tt.store(board.hash, 0, score_to_tt(stand_pat, ply), TTFlag::LowerBound, Move::NULL);
        return beta;
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    // Delta pruning
    let delta = PieceKind::Queen.value() + 200;
    if stand_pat + delta < alpha {
        return alpha;
    }

    // Generate only captures+promotions (no quiet moves at all)
    let captures_list = chess_core::generate_pseudo_legal_captures(board);
    let us = board.side_to_move;
    let num_caps = captures_list.len();

    if num_caps == 0 {
        return alpha;
    }

    // Score and order captures via selection sort
    let mut cap_moves = [Move::NULL; 128];
    let mut cap_scores = [0i32; 128];
    let mut cap_count = 0usize;
    for &m in captures_list.iter() {
        let score = if m == tt_move && !tt_move.is_null() { 10_000_000 } else { capture_value(board, m) };
        cap_moves[cap_count] = m;
        cap_scores[cap_count] = score;
        cap_count += 1;
        if cap_count >= 128 { break; }
    }

    let mut best_score = stand_pat;
    let mut best_move = Move::NULL;
    let mut child_pv = Vec::new();

    let mut idx = 0;
    while idx < cap_count {
        pick_next_move(&mut cap_moves[..cap_count], &mut cap_scores[..cap_count], idx);
        let m = cap_moves[idx];
        idx += 1;

        if !see::see_ge(board, m, 0) {
            continue;
        }

        let prev_castling = board.castling;
        let prev_ep = board.en_passant;
        let prev_halfmove = board.halfmove_clock;

        // Safety guard: skip if from-square is empty
        if board.piece_at(m.from_sq()).is_none() {
            continue;
        }
        let captured = board.make_move(m);

        // Inline legality check
        if move_is_illegal(board, us) {
            board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);
            continue;
        }

        if state.use_nnue && ply as usize + 1 < MAX_PLY {
            update_accumulator_for_move(state, board, m, captured, ply as usize, ply as usize + 1);
        }

        let score = -quiescence(board, ply + 1, -beta, -alpha, &mut child_pv, state, stop, false);

        board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);

        if state.stop {
            return 0;
        }

        if score > best_score {
            best_score = score;
            best_move = m;
        }

        if score >= beta {
            state.tt.store(board.hash, 0, score_to_tt(best_score, ply), TTFlag::LowerBound, best_move);
            return beta;
        }
        if score > alpha {
            alpha = score;
            pv.clear();
            pv.push(m);
            pv.extend_from_slice(&child_pv);
        }
    }

    // Quiet moves that give check — only at the first qsearch ply to control
    // branching factor. SEE >= 0 guards against searching moves that hang the piece.
    if generate_checks {
        let quiets = chess_core::generate_pseudo_legal_quiets(board);
        for &m in quiets.iter() {
            if board.piece_at(m.from_sq()).is_none() {
                continue;
            }
            if !see::see_ge(board, m, 0) {
                continue;
            }

            let prev_castling = board.castling;
            let prev_ep = board.en_passant;
            let prev_halfmove = board.halfmove_clock;
            let captured = board.make_move(m);

            if move_is_illegal(board, us) {
                board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);
                continue;
            }

            if !chess_core::is_in_check(board) {
                board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);
                continue;
            }

            if state.use_nnue && ply as usize + 1 < MAX_PLY {
                update_accumulator_for_move(state, board, m, captured, ply as usize, ply as usize + 1);
            }

            let score = -quiescence(board, ply + 1, -beta, -alpha, &mut child_pv, state, stop, false);

            board.unmake_move(m, captured, prev_castling, prev_ep, prev_halfmove);

            if state.stop {
                return 0;
            }

            if score > best_score {
                best_score = score;
                best_move = m;
            }

            if score >= beta {
                state.tt.store(board.hash, 0, score_to_tt(best_score, ply), TTFlag::LowerBound, best_move);
                return beta;
            }
            if score > alpha {
                alpha = score;
                pv.clear();
                pv.push(m);
                pv.extend_from_slice(&child_pv);
            }
        }
    }

    if !best_move.is_null() {
        let flag = if best_score > stand_pat { TTFlag::Exact } else { TTFlag::UpperBound };
        state.tt.store(board.hash, 0, score_to_tt(best_score, ply), flag, best_move);
    }

    alpha
}

fn capture_value(board: &Board, m: Move) -> i32 {
    let flag = m.flag();
    let victim = if flag == MoveFlag::EnPassant {
        PieceKind::Pawn.value()
    } else {
        board.piece_at(m.to_sq()).map(|p| p.kind.value()).unwrap_or(0)
    };
    let attacker = board.piece_at(m.from_sq()).map(|p| p.kind.value()).unwrap_or(0);
    victim * 10 - attacker
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn evaluate_for_side(board: &Board, state: &mut SearchState, ply: u8) -> i32 {
    let eval = if state.use_nnue && (ply as usize) < MAX_PLY {
        // Lazy refresh: king moves defer the accumulator recompute until here.
        // The board is at position P_ply so refresh is always valid at this call site.
        if state.accumulators[ply as usize].needs_refresh {
            let net = Arc::clone(&state.net);
            state.accumulators[ply as usize].refresh(board, &net);
        }
        // nnue_evaluate returns score from side-to-move perspective.
        // scale_for_endgame expects White's perspective, so flip for Black then flip back.
        let raw = nnue_evaluate(&state.accumulators[ply as usize], board.side_to_move, &state.net);
        let sign = if board.side_to_move == Color::White { 1 } else { -1 };
        crate::eval::scale_for_endgame(board, raw * sign) * sign
    } else {
        let e = evaluate(board).0;
        match board.side_to_move {
            Color::White => e,
            Color::Black => -e,
        }
    };
    // Nudge the static eval by the learned correction for this position.
    eval + state.correction(board, ply)
}

/// Incrementally update the accumulator from `src_ply` to `dst_ply` for a move.
/// Must be called AFTER `make_move` (board reflects the new position).
/// The `captured` result from `make_move` is used to detect captures.
///
/// After calling this, `state.accumulators[dst_ply]` is correctly updated.
fn update_accumulator_for_move(
    state: &mut SearchState,
    board: &Board,
    m: Move,
    captured: Option<chess_common::Piece>,
    src_ply: usize,
    dst_ply: usize,
) {
    let flag = m.flag();
    let from = m.from_sq();
    let to = m.to_sq();
    // The moving piece (side that already moved; board.side_to_move is now the opponent)
    let us = board.side_to_move.opposite();

    // Determine the piece kind (needed to detect king moves before branching)
    let moving_kind = if flag.is_promotion() {
        PieceKind::Pawn
    } else if flag == MoveFlag::KingsideCastle || flag == MoveFlag::QueensideCastle {
        PieceKind::King
    } else {
        board.piece_at(to).map(|p| p.kind).unwrap_or(PieceKind::Pawn)
    };

    // King moves change the king bucket and require a full refresh.
    // Non-king moves from a dirty parent also cannot be updated incrementally.
    // In both cases: mark the destination accumulator as dirty and return —
    // the refresh is deferred to evaluate_for_side so pruned nodes pay no cost.
    if moving_kind == PieceKind::King || state.accumulators[src_ply].needs_refresh {
        state.accumulators[dst_ply].needs_refresh = true;
        return;
    }

    // Non-king move from a clean parent: incremental update.
    state.accumulators[dst_ply] = state.accumulators[src_ply].clone();
    let white_king = board.king_square(Color::White);
    let black_king = board.king_square(Color::Black);
    let net = &state.net;
    let acc = &mut state.accumulators[dst_ply];

    // Remove piece from source square
    acc.sub_piece(net, white_king, black_king, us, moving_kind, from);

    // Handle capture: remove captured piece
    if let Some(cap) = captured {
        if flag == MoveFlag::EnPassant {
            // En passant: captured pawn is on the same file as `to`, same rank as `from`
            let cap_sq = Square::new(to.file(), from.rank());
            acc.sub_piece(net, white_king, black_king, cap.color, cap.kind, cap_sq);
        } else {
            acc.sub_piece(net, white_king, black_king, cap.color, cap.kind, to);
        }
    }

    // Add piece to destination square (promoted piece if promotion, else moving piece)
    if let Some(promo) = flag.promotion_piece() {
        acc.add_piece(net, white_king, black_king, us, promo, to);
    } else {
        acc.add_piece(net, white_king, black_king, us, moving_kind, to);
    }
}

/// Truncate a PV at the first move that is not legal in the given position.
/// Stale TT entries can cause the deeper moves in a PV to be illegal.
fn sanitize_pv(board: &Board, pv: &[Move]) -> Vec<Move> {
    let mut b = board.clone();
    let mut result = Vec::with_capacity(pv.len());
    for &m in pv {
        let legal = chess_core::generate_legal_moves(&b);
        if !legal.as_slice().contains(&m) {
            break;
        }
        b.make_move(m);
        // Stop at a forced draw: the game ends here regardless of what the
        // rest of the PV claims. Drop the draw-creating move as well — GUIs
        // (fastchess) warn when the PV includes the move that triggers the
        // threefold or 50-move rule, because nothing after it can legally
        // be played.
        if b.halfmove_clock >= 100 || b.is_repetition() {
            break;
        }
        result.push(m);
    }
    result
}

fn has_non_pawn_material(board: &Board, color: Color) -> bool {
    let ci = color.index();
    !board.pieces[ci][PieceKind::Knight.index()].is_empty()
        || !board.pieces[ci][PieceKind::Bishop.index()].is_empty()
        || !board.pieces[ci][PieceKind::Rook.index()].is_empty()
        || !board.pieces[ci][PieceKind::Queen.index()].is_empty()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::syzygy::SyzygyTB;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::path::PathBuf;
    use crate::tt::SharedTT;

    #[test]
    fn sudden_death_cap_bounds_blitz_and_rapid_no_increment() {
        // 180+0: an over-large target is capped to time/34.
        assert_eq!(sudden_death_time_cap(15_000, 0, 180_000), 180_000 / 34);
        // 10+0 rapid uses the gentler time/38 divisor.
        assert_eq!(sudden_death_time_cap(40_000, 0, 600_000), 600_000 / 38);
    }

    #[test]
    fn sudden_death_cap_noop_when_small_target_or_any_increment() {
        // Target already under the cap is returned unchanged.
        assert_eq!(sudden_death_time_cap(1_000, 0, 180_000), 1_000);
        // ANY increment disables the cap (only pure sudden death is at risk),
        // so the 10+0.1 (inc=100) bench/SF-test path is left unchanged.
        assert_eq!(sudden_death_time_cap(15_000, 100, 180_000), 15_000);
        assert_eq!(sudden_death_time_cap(15_000, 2_000, 180_000), 15_000);
    }

    #[test]
    fn allocated_time_reserves_clock_in_no_increment_blitz() {
        // 180+0 opening: the sudden-death cap keeps the per-move slice well
        // below the old piece-count allocation (~time/24 + bank ≈ 9.4s),
        // leaving enough clock for a long game.
        let board = Board::starting_position();
        let params = SearchParams {
            white_time_ms: Some(180_000),
            black_time_ms: Some(180_000),
            white_inc_ms: Some(0),
            black_inc_ms: Some(0),
            ..Default::default()
        };
        let alloc = allocated_move_time_ms(&params, &board).expect("timed search");
        assert!(alloc > 0);
        assert!(alloc <= 180_000 / 30, "allocated {alloc} too high for 180+0");
    }

    /// Helper: run a fixed-depth search and return the result.
    ///
    /// Spawns a thread with a large stack so that debug builds (which have much
    /// bigger stack frames than release) don't overflow on deep searches.
    /// Each `alpha_beta` frame carries ~2 KB of fixed arrays (MovePicker buffers
    /// + history move lists); 128 levels × large debug frames exceeds the
    ///   ~1 MB Windows default stack.
    fn search_position(fen: &str, max_depth: u8) -> SearchResult {
        let fen = fen.to_owned();
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let board = Board::from_fen(&fen).expect("valid FEN");
                let stop = Arc::new(AtomicBool::new(false));
                let tt = Arc::new(SharedTT::new(16)); // 16 MB
                let params = SearchParams {
                    max_depth,
                    use_nnue: false, // use HCE for tests (zeroed net is useless)
                    ..Default::default()
                };
                iterative_deepening(&board, &params, &stop, &tt, None, 0, &NnueNetwork::embedded(), None, None, None, None, None)
            })
            .expect("failed to spawn search thread")
            .join()
            .expect("search thread panicked")
    }

    /// Helper: run a search with a specific contempt value.
    fn search_with_contempt(fen: &str, max_depth: u8, contempt: i32) -> SearchResult {
        let fen = fen.to_owned();
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let board = Board::from_fen(&fen).expect("valid FEN");
                let stop = Arc::new(AtomicBool::new(false));
                let tt = Arc::new(SharedTT::new(16));
                let params = SearchParams {
                    max_depth,
                    use_nnue: false,
                    contempt,
                    ..Default::default()
                };
                iterative_deepening(&board, &params, &stop, &tt, None, 0, &NnueNetwork::embedded(), None, None, None, None, None)
            })
            .expect("failed to spawn search thread")
            .join()
            .expect("search thread panicked")
    }

    fn syzygy_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/syzygy")
    }

    fn search_position_with_syzygy(fen: &str, max_depth: u8, syzygy_tb: Option<SyzygyTB>) -> SearchResult {
        let fen = fen.to_owned();
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let board = Board::from_fen(&fen).expect("valid FEN");
                let stop = Arc::new(AtomicBool::new(false));
                let tt = Arc::new(SharedTT::new(16));
                let params = SearchParams {
                    max_depth,
                    use_nnue: false,
                    ..Default::default()
                };
                let root_tb_solution = syzygy_tb.as_ref().and_then(|tb| crate::syzygy::solve_root_position(tb, &board, 128));
                iterative_deepening(&board, &params, &stop, &tt, None, 0, &NnueNetwork::embedded(), None, syzygy_tb, root_tb_solution, None, None)
            })
            .expect("failed to spawn search thread")
            .join()
            .expect("search thread panicked")
    }

    #[test]
    fn mate_in_one() {
        // White: Kg6, Re1. Black: Kg8. Re8# is mate in 1.
        let result = search_position("6k1/8/6K1/8/8/8/8/4R3 w - - 0 1", 4);
        assert!(result.score.is_mate(), "should detect mate, got {}", result.score);
        assert_eq!(result.best_move.to_uci(), "e1e8");
    }

    #[test]
    fn mate_in_two() {
        // White: Kf5, Ra1. Black: Kh8. Mate in 2: Kg6 Kg8 Ra8#
        let result = search_position("7k/8/8/5K2/8/8/8/R7 w - - 0 1", 8);
        assert!(result.score.is_mate(), "should detect mate, got {}", result.score);
        // Optimal first move is Kg6 (approaching the king)
        assert_eq!(result.best_move.to_uci(), "f5g6");
    }

    #[test]
    fn mate_distance_pruning_efficiency() {
        // Back-rank mate: 1k6/8/1K6/8/8/8/8/7R w - - 0 1
        // White: Kb6, Rh1. Black: Kb8. Rh8# is mate in 1.
        // With MDP, the search should break after finding mate at depth 2
        // and reach high depths rapidly because all non-mating branches
        // are pruned once the short mate is known.
        let result = search_position("1k6/8/1K6/8/8/8/8/7R w - - 0 1", 12);
        assert!(result.score.is_mate(), "should detect mate, got {}", result.score);
        assert_eq!(result.best_move.to_uci(), "h1h8");
        // With MDP, the search should have broken early (iterative deepening
        // exits on finding mate). Depth should be low (≤ 3).
        assert!(result.depth <= 3, "should break early on mate, searched to depth {}", result.depth);
    }

    #[test]
    fn game_position_move35() {
        // Position from the game after 35.Ke2 Kd7 — Black has Q+pawns vs 2R+pawns
        // with an advanced c-pawn. Should be clearly winning for Black.
        // Key: c7 pawn blocks the Rb7 from attacking Kd7.
        let fen = "8/1Rpk4/5p2/p6p/P1pP1p1P/2q2P1R/4K1P1/8 w - - 2 36";
        let board = Board::from_fen(fen).expect("valid FEN");
        // Verify board is correct (e2 = file 4, rank 1; d7 = file 3, rank 6)
        assert_eq!(board.king_square(Color::White), Square::new(4, 1));
        assert_eq!(board.king_square(Color::Black), Square::new(3, 6));
        // Search should find a strong advantage for Black (negative score = Black winning)
        let result = search_position(fen, 10);
        eprintln!("move35: score={} best={} depth={} nodes={}",
            result.score, result.best_move.to_uci(), result.depth, result.nodes);
        assert!(result.score.0 < -100,
            "Black should be clearly winning, got score {}", result.score);
    }

    /// Contempt: draws at ply > 0 should be scored as -contempt from the side-to-move's
    /// perspective.  K vs K with halfmove_clock=99 — every White king move increments the
    /// clock to 100, triggering the 50-move-rule check at ply=1 and returning -contempt.
    /// The parent (White, ply=0) negates that: score = +contempt.
    #[test]
    fn contempt_draw_score_fifty_move() {
        // K vs K, kings far apart, halfmove_clock = 99.
        // Any White king move is non-capture and non-pawn → clock becomes 100 at ply=1.
        let fen = "8/8/8/8/8/8/1K6/7k w - - 99 1";

        let with_contempt = search_with_contempt(fen, 1, 20);
        let no_contempt   = search_with_contempt(fen, 1, 0);

        // All moves lead to a 50-move draw at ply=1.  Each draw scores -contempt from
        // Black's perspective; the root (White) negates → +contempt.
        assert_eq!(
            with_contempt.score.0, 20,
            "draw with contempt=20 should score +20 from White's perspective, got {}",
            with_contempt.score
        );
        assert_eq!(
            no_contempt.score.0, 0,
            "draw with contempt=0 should score 0, got {}",
            no_contempt.score
        );
        // Either way a legal move must be returned.
        assert!(!with_contempt.best_move.is_null());
        assert!(!no_contempt.best_move.is_null());
    }

    /// Contempt + ply limit: an engine with non-zero contempt must not overflow the ply
    /// counter or write past the end of the accumulator array, even in positions with
    /// many extensions (perpetual checks, forced repetitions).  This test ensures the
    /// search completes without panicking.
    #[test]
    fn contempt_no_crash_near_ply_limit() {
        // A complex middlegame position that can produce long check sequences.
        // Searched at depth 20 so extensions can push ply well past 64.
        let fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5";
        // Must complete without panic; score and move are irrelevant.
        let result = search_with_contempt(fen, 20, 50);
        assert!(!result.best_move.is_null(), "engine must return a legal move");
    }

    /// Performance test: engine must find Nf6+ (g4f6) in a complex tactical position.
    /// After 1.Nf6+! Kh8 (forced) 2.d8=Q# it is mate in 2.
    #[test]
    fn find_best_move_nf6_mate_in_two() {
        let result = search_position("8/3P3k/n2K3p/2p3n1/1b4N1/2p1p1P1/8/3B4 w - - 0 1", 6);
        assert_eq!(
            result.best_move.to_uci(),
            "g4f6",
            "expected Nf6+ (g4f6), got {}",
            result.best_move.to_uci()
        );
    }

    #[test]
    fn syzygy_guides_precapture_transition_to_bxc6() {
        let _guard = crate::syzygy::syzygy_test_lock().lock().expect("lock syzygy test mutex");
        let path = syzygy_path();
        if !path.exists() {
            return;
        }

        let tb = SyzygyTB::new(path.to_string_lossy().as_ref()).expect("load syzygy tables");
        let result = search_position_with_syzygy("b7/8/P1P5/6p1/3K1k2/8/8/8 b - - 0 53", 8, Some(tb));

        assert_eq!(result.best_move.to_uci(), "a8c6");
        assert!(result.score.0 >= crate::syzygy::TB_WIN_SCORE, "expected immediate TB win score, got {}", result.score.0);
    }
}
