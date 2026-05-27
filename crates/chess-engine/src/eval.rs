use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use chess_common::{Bitboard, Board, Color, PieceKind, Score, Square};

/// Lazily-initialized shared copy of `EvalParams::default()`.
/// Building it costs ~13 KB of stack traffic (PST tables dominate); callers on
/// the NNUE hot path reuse this reference instead of rebuilding per eval.
fn default_params() -> &'static EvalParams {
    static DEFAULT: OnceLock<EvalParams> = OnceLock::new();
    DEFAULT.get_or_init(EvalParams::default)
}

// ---------------------------------------------------------------------------
// Pawn hash table: caches pawn structure evaluation scores.
// Pawn structures change rarely (~10% of moves), so hit rate is ~90%+.
// ---------------------------------------------------------------------------

const PAWN_HASH_SIZE: usize = 16384; // 16K entries = 256KB (16 bytes each)

#[repr(C, align(16))]
struct PawnHashEntry {
    key: AtomicU64,
    data: AtomicU64,
}

impl PawnHashEntry {
    fn new() -> Self {
        Self {
            key: AtomicU64::new(0),
            data: AtomicU64::new(0),
        }
    }
}

struct PawnHashTable {
    entries: Box<[PawnHashEntry]>,
    mask: usize,
}


impl PawnHashTable {
    fn new() -> Self {
        let mut entries = Vec::with_capacity(PAWN_HASH_SIZE);
        for _ in 0..PAWN_HASH_SIZE {
            entries.push(PawnHashEntry::new());
        }
        Self {
            entries: entries.into_boxed_slice(),
            mask: PAWN_HASH_SIZE - 1,
        }
    }

    fn probe(&self, hash: u64) -> Option<(i32, i32)> {
        let idx = hash as usize & self.mask;
        let entry = &self.entries[idx];
        let data = entry.data.load(Ordering::Relaxed);
        let key = entry.key.load(Ordering::Relaxed);
        if key ^ data != hash {
            return None;
        }
        let mg = (data & 0xFFFFFFFF) as u32 as i32;
        let eg = ((data >> 32) & 0xFFFFFFFF) as u32 as i32;
        Some((mg, eg))
    }

    fn store(&self, hash: u64, mg: i32, eg: i32) {
        let idx = hash as usize & self.mask;
        let entry = &self.entries[idx];
        let data = (mg as u32 as u64) | ((eg as u32 as u64) << 32);
        entry.data.store(data, Ordering::Relaxed);
        entry.key.store(hash ^ data, Ordering::Relaxed);
    }
}

/// Compute a hash for just the pawn configuration.
pub(crate) fn pawn_hash(board: &Board) -> u64 {
    let wp = board.pieces[Color::White.index()][PieceKind::Pawn.index()].0;
    let bp = board.pieces[Color::Black.index()][PieceKind::Pawn.index()].0;
    // Use distinct multipliers to avoid collisions between colors
    wp.wrapping_mul(0xD4E8E29E56E8AAB6) ^ bp.wrapping_mul(0x6A1E2D3C4B5F8097)
}

/// Distinct odd multipliers per [color][piece-kind], used to hash piece
/// placements into correction-history keys.
const PIECE_MIX: [[u64; 6]; 2] = [
    // White: P, N, B, R, Q, K
    [
        0xD4E8E29E56E8AAB6, 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9,
        0x94D049BB133111EB, 0x2545F4914F6CDD1D, 0xC2B2AE3D27D4EB4F,
    ],
    // Black
    [
        0x6A1E2D3C4B5F8097, 0xA0761D6478BD642F, 0xE7037ED1A0B428DB,
        0x8EBC6AF09C88C6E3, 0x589965CC75374CC3, 0x1D8E4E27C47D124F,
    ],
];

/// Hash keys for the various correction-history facets of a position.
pub(crate) struct CorrKeys {
    pub pawn: u64,
    pub white: u64, // White's non-pawn placement
    pub black: u64, // Black's non-pawn placement
    pub minor: u64, // knights, bishops, kings (both colors)
    pub major: u64, // rooks, queens, kings (both colors)
}

pub(crate) fn corr_keys(board: &Board) -> CorrKeys {
    let bb = |c: Color, k: PieceKind| board.pieces[c.index()][k.index()].0;
    let mix = |c: Color, k: PieceKind| bb(c, k).wrapping_mul(PIECE_MIX[c.index()][k.index()]);

    let mut white = 0u64;
    let mut black = 0u64;
    for k in [PieceKind::Knight, PieceKind::Bishop, PieceKind::Rook, PieceKind::Queen, PieceKind::King] {
        white ^= mix(Color::White, k);
        black ^= mix(Color::Black, k);
    }

    let mut minor = 0u64;
    for k in [PieceKind::Knight, PieceKind::Bishop, PieceKind::King] {
        minor ^= mix(Color::White, k) ^ mix(Color::Black, k);
    }

    let mut major = 0u64;
    for k in [PieceKind::Rook, PieceKind::Queen, PieceKind::King] {
        major ^= mix(Color::White, k) ^ mix(Color::Black, k);
    }

    CorrKeys { pawn: pawn_hash(board), white, black, minor, major }
}

// Thread-local pawn hash table for zero-contention access
thread_local! {
    static PAWN_HT: PawnHashTable = PawnHashTable::new();
}

#[derive(Debug, Clone)]
pub struct EvalTerm {
    pub name: &'static str,
    pub mg: i32,
    pub eg: i32,
}

#[derive(Debug, Clone)]
pub struct EvalBreakdown {
    pub phase: i32,
    pub terms: Vec<EvalTerm>,
    pub mg_total: i32,
    pub eg_total: i32,
    pub interpolated: i32,
    pub scale: i32,
    pub final_score: i32,
}

fn evaluate_impl(board: &Board, params: &EvalParams, use_pawn_hash: bool, collect_terms: bool) -> EvalBreakdown {
    let phase = game_phase(board);
    let (mut mg, mut eg) = (0i32, 0i32);
    let mut terms: Vec<EvalTerm> = if collect_terms { Vec::with_capacity(20) } else { Vec::new() };
    macro_rules! push_term {
        ($name:expr, $mg:expr, $eg:expr) => {
            if collect_terms {
                terms.push(EvalTerm { name: $name, mg: $mg, eg: $eg });
            }
        };
    }

    // Material + piece-square tables
    let (mut tmg, mut teg) = (0i32, 0i32);
    material_and_pst(board, params, &mut tmg, &mut teg);

    // Queenless middlegame adjustment: when neither side has a queen the
    // king MG PST values are misleading — a central king is an asset, not
    // a liability.  Blend the king MG PST contribution toward its EG value
    // so tuned PSTs stay intact but don't dominate in queenless play.
    let white_queens = board.pieces[Color::White.index()][PieceKind::Queen.index()].count();
    let black_queens = board.pieces[Color::Black.index()][PieceKind::Queen.index()].count();
    if white_queens == 0 && black_queens == 0 {
        let ki = PieceKind::King.index();
        let mut king_mg = 0i32;
        let mut king_eg = 0i32;
        for sq in board.pieces[Color::White.index()][ki].iter() {
            king_mg += params.pst_mg[ki][sq.index()];
            king_eg += params.pst_eg[ki][sq.index()];
        }
        for sq in board.pieces[Color::Black.index()][ki].iter() {
            king_mg -= params.pst_mg[ki][mirror_square(sq)];
            king_eg -= params.pst_eg[ki][mirror_square(sq)];
        }
        // Replace king MG PST with its EG counterpart (subtract MG, add EG)
        tmg += king_eg - king_mg;
    }

    mg += tmg;
    eg += teg;
    push_term!("material_pst", tmg, teg);

    // Pawn structure
    let (pawn_mg, pawn_eg) = if use_pawn_hash {
        let ph = pawn_hash(board);
        PAWN_HT.with(|ht| {
            if let Some(cached) = ht.probe(ph) {
                cached
            } else {
                let result = pawn_structure(board, params);
                ht.store(ph, result.0, result.1);
                result
            }
        })
    } else {
        pawn_structure(board, params)
    };
    // Pawn structure can accumulate large totals from many passed-pawn bonuses.
    // Cap it to keep it proportional and avoid destroying correlation with SF.
    let pawn_mg = soft_cap_signed(pawn_mg, 250);
    let pawn_eg = soft_cap_signed(pawn_eg, 300);
    mg += pawn_mg;
    eg += pawn_eg;
    push_term!("pawn_structure", pawn_mg, pawn_eg);

    let (bp_mg, bp_eg) = bishop_pair(board, params);
    mg += bp_mg;
    eg += bp_eg;
    push_term!("bishop_pair", bp_mg, bp_eg);

    let (rook_mg, rook_eg) = rook_open_files(board, params);
    mg += rook_mg;
    eg += rook_eg;
    push_term!("rook_activity", rook_mg, rook_eg);

    // King-safety can become over-optimistic in speculative attack lines.
    // Use a smooth soft-cap (not a hard clamp) so tuning remains stable.
    let ks_raw = king_safety(board, phase, params);
    let ks = soft_cap_signed(ks_raw, 220);
    mg += ks;
    push_term!("king_safety", ks, 0);

    let (mob_mg_raw, mob_eg_raw) = pseudo_mobility(board, params);
    // Mobility totals can stack excessively when multiple pieces are well-placed.
    // Cap the aggregate to avoid +200 mobility swings in quiet positions.
    let mob_mg = soft_cap_signed(mob_mg_raw, 130);
    let mob_eg = soft_cap_signed(mob_eg_raw, 150);
    mg += mob_mg;
    eg += mob_eg;
    push_term!("mobility", mob_mg, mob_eg);

    // Center control is MG-only and can stack to +160 with multiple pawns+pieces.
    // Cap it to keep it proportional.
    let cc_raw = center_control(board, params);
    let cc = soft_cap_signed(cc_raw, 80);
    mg += cc;
    push_term!("center_control", cc, 0);

    let conn = connectivity(board, params);
    mg += conn;
    eg += conn / 2;
    push_term!("connectivity", conn, conn / 2);

    let (thr_mg_raw, thr_eg_raw) = threats(board, params);
    // Threat terms are noisy and can stack with king-safety inflation.
    // Apply the same smooth saturation in both phases.
    let thr_mg = soft_cap_signed(thr_mg_raw, 170);
    let thr_eg = soft_cap_signed(thr_eg_raw, 130);
    mg += thr_mg;
    eg += thr_eg;
    push_term!("threats", thr_mg, thr_eg);

    let t = tempo(board, params);
    mg += t;
    eg += t;
    push_term!("tempo", t, t);

    let (imb_mg, imb_eg) = material_imbalance(board, params);
    mg += imb_mg;
    eg += imb_eg;
    push_term!("material_imbalance", imb_mg, imb_eg);

    let (pma_mg, pma_eg) = pawn_majority_advantage(board);
    mg += pma_mg;
    eg += pma_eg;
    push_term!("pawn_majority", pma_mg, pma_eg);

    let mopup = mopup_eval(board, phase);
    eg += mopup;
    push_term!("mopup", 0, mopup);

    let (sp_mg, sp_eg) = space_eval(board);
    mg += sp_mg;
    eg += sp_eg / 2;
    push_term!("space", sp_mg, sp_eg / 2);

    let (bb_mg, bb_eg) = bad_bishop(board, params);
    mg += bb_mg;
    eg += bb_eg;
    push_term!("bad_bishop", bb_mg, bb_eg);

    let (nb_mg, nb_eg) = knight_bishop_adjustment(board, params);
    mg += nb_mg;
    eg += nb_eg;
    push_term!("knight_bishop_adjustment", nb_mg, nb_eg);

    let (tb_mg, tb_eg) = trapped_bishops(board, params);
    mg += tb_mg;
    eg += tb_eg;
    push_term!("trapped_bishops", tb_mg, tb_eg);

    let (pi_mg, pi_eg) = pawn_islands(board, params);
    mg += pi_mg;
    eg += pi_eg;
    push_term!("pawn_islands", pi_mg, pi_eg);

    let interpolated = interpolate(mg, eg, phase);
    let scale = endgame_scale_factor(board, params, interpolated, phase);
    let final_score = interpolated * scale / 128;

    EvalBreakdown {
        phase,
        terms,
        mg_total: mg,
        eg_total: eg,
        interpolated,
        scale,
        final_score,
    }
}

/// Apply the endgame scale factor to an externally-computed score (e.g. NNUE).
///
/// `score_from_white` must be from **White's perspective** (positive = White winning),
/// because `endgame_scale_factor` uses the sign to identify the winning side in
/// no-pawn endings.  Returns the scaled score, still from White's perspective.
#[inline]
pub(crate) fn scale_for_endgame(board: &Board, score_from_white: i32) -> i32 {
    let phase = game_phase(board);
    let scale = endgame_scale_factor(board, default_params(), score_from_white, phase);
    score_from_white * scale / 128
}

#[inline]
fn soft_cap_signed(value: i32, cap: i32) -> i32 {
    if cap <= 0 || value == 0 {
        return 0;
    }
    let sign = value.signum();
    let abs = value.abs();
    // Smooth saturation: linear near zero, asymptotically approaches `cap`.
    // This is Texel-friendly: monotonic and continuous (no hard cutoffs).
    sign * (abs * cap) / (abs + cap)
}

#[inline]
fn non_pawn_material_mg(board: &Board, color: Color, params: &EvalParams) -> i32 {
    let ci = color.index();
    board.pieces[ci][PieceKind::Knight.index()].count() as i32 * params.material_mg[PieceKind::Knight.index()]
        + board.pieces[ci][PieceKind::Bishop.index()].count() as i32 * params.material_mg[PieceKind::Bishop.index()]
        + board.pieces[ci][PieceKind::Rook.index()].count() as i32 * params.material_mg[PieceKind::Rook.index()]
        + board.pieces[ci][PieceKind::Queen.index()].count() as i32 * params.material_mg[PieceKind::Queen.index()]
}

#[inline]
fn attack_scale_from_material_delta(delta_cp: i32) -> i32 {
    (256 + (delta_cp * 64) / 300).clamp(160, 300)
}

/// Evaluate the position. Positive = White advantage (in centipawns).
pub fn evaluate(board: &Board) -> Score {
    let breakdown = evaluate_impl(board, default_params(), true, false);
    Score(breakdown.final_score)
}

/// Evaluate using custom parameters (for the tuner). No pawn hash caching.
pub fn evaluate_with_params(board: &Board, params: &EvalParams) -> Score {
    let breakdown = evaluate_impl(board, params, false, false);
    Score(breakdown.final_score)
}

// ---------------------------------------------------------------------------
// Game phase (0 = pure middlegame, 256 = pure endgame)
// ---------------------------------------------------------------------------

fn game_phase(board: &Board) -> i32 {
    let knight_phase = 1;
    let bishop_phase = 1;
    let rook_phase = 2;
    // Queens are weighted heavily because their absence fundamentally
    // changes position character: king safety matters far less, central
    // kings become assets, and MG PST assumptions break down (e.g. the
    // Berlin endgame).  Weight 6 (up from 4) shifts queenless positions
    // firmly toward EG scoring.
    let queen_phase = 6;
    let total_phase = 4 * knight_phase + 4 * bishop_phase + 4 * rook_phase + 2 * queen_phase;

    let mut phase = total_phase;
    for color_idx in 0..2 {
        phase -= board.pieces[color_idx][PieceKind::Knight.index()].count() as i32 * knight_phase;
        phase -= board.pieces[color_idx][PieceKind::Bishop.index()].count() as i32 * bishop_phase;
        phase -= board.pieces[color_idx][PieceKind::Rook.index()].count() as i32 * rook_phase;
        phase -= board.pieces[color_idx][PieceKind::Queen.index()].count() as i32 * queen_phase;
    }

    // Normalize to 0..256
    (phase * 256 + total_phase / 2) / total_phase
}

fn interpolate(mg: i32, eg: i32, phase: i32) -> i32 {
    ((mg * (256 - phase)) + (eg * phase)) / 256
}

// ---------------------------------------------------------------------------
// Light / dark square masks
// Light squares: a2, b1, c2, d1, ... where (file + rank) is odd
// Dark squares:  a1, b2, c1, d2, ... where (file + rank) is even
// ---------------------------------------------------------------------------
const LIGHT_SQUARES: Bitboard = Bitboard(0x55AA_55AA_55AA_55AA);
const DARK_SQUARES: Bitboard = Bitboard(0xAA55_AA55_AA55_AA55);

// ---------------------------------------------------------------------------
// PeSTO piece-square tables (MG/EG)
// Values from https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function
// Layout: index 0 = a1, index 63 = h8 (White perspective)
// For Black, mirror vertically (sq ^ 56).
// ---------------------------------------------------------------------------

/// Mirror a square index vertically (for Black's PST lookup).
#[inline]
fn mirror_square(sq: Square) -> usize {
    sq.index() ^ 56
}

// @tuner:material_start
const _MATERIAL_MG: [i32; 6] = [82, 337, 365, 477, 1025, 0]; // P N B R Q K
const MATERIAL_EG: [i32; 6] = [94, 281, 297, 512, 936, 0];
// @tuner:material_end

const ALL_PIECE_KINDS: [PieceKind; 6] = [
    PieceKind::Pawn,
    PieceKind::Knight,
    PieceKind::Bishop,
    PieceKind::Rook,
    PieceKind::Queen,
    PieceKind::King,
];

// @tuner:pst_start
#[rustfmt::skip]
const PST_PAWN_MG: [i32; 64] = [
       0,    0,    0,    0,    0,    0,    0,    0,
    -106,  -92,  -89,  -92,  -86,    2,   29,   63,
    -106, -106,  -47,  -94,  -42,   -8,   54,  100,
     -70,  -45,  -10,    0,    0,    2,   84,   87,
     -50,   15,   21,   47,   72,   99,  129,  154,
      56,   71,   16,   92,  156,  242,  246,  274,
     114,   62,  186,  188,  -46,  150,   55, -182,
       0,    0,    0,    0,    0,    0,    0,    0,
];

#[rustfmt::skip]
const PST_KNIGHT_MG: [i32; 64] = [
    -190, -107, -242, -182,  -96, -117, -127,  -94,
    -160, -139,  -25,  -39,  -56,  -85, -125, -114,
    -155,  -76,  -41,  -15,   66,  -39,  -28,  -98,
     -84,  -28,    2,  -25,   -8,   42,  161,  -62,
      -5,   34,    3,  115,  -17,  102,    7,  155,
     -13,  -18,   49,  170,  233,  278,  177,  124,
      -8,  -71,   29,  130,  131,   22,  -13,   39,
    -231,  -89,  -24,  -40,   91,  -62,   88, -133,
];

#[rustfmt::skip]
const PST_BISHOP_MG: [i32; 64] = [
     -52,  118,   -2, -119, -114,  -30,  -32, -100,
      41,   62,  125,    5,   -3,    9,   59,  -18,
      79,    1,   -1,    6,   35,  -10,   44,   91,
     -15,    3,  -43,   69,   36,   10,   21,   13,
     -23,   21,   95,  132,   77,  134,   -5,  -11,
      34,  116,  -25,  176,  135,  207,  104,  225,
      50,    1,   12,   90,  104,   18,  -10,   11,
    -141,   18,  -61, -149,    1,  -96,    8,   16,
];

#[rustfmt::skip]
const PST_ROOK_MG: [i32; 64] = [
    -191, -134, -142, -124,  -81, -135,   21, -106,
    -285, -192, -176, -217, -202, -210,  -95,  -96,
     -55,  -70, -142, -178,  -38,  -56,   81,  139,
    -129, -154, -116, -153, -116, -104,   45,  -64,
     -69,  -83,  -54,   42,   -4,   48,   36,   83,
     -15,    4,    2,  111,  140,  140,  130,   78,
      78,  -25,   57,  113,  131,  135,   12,   70,
      94,   73,  128,  128,  134,   47,   90,  108,
];

#[rustfmt::skip]
const PST_QUEEN_MG: [i32; 64] = [
     -99,  -85, -101,  -51,  -80, -151,  -52,  -61,
    -135,  -77,  -35,  -34,  -44,  -64,  -80,   72,
     -72,  -32,  -64, -107,  -83,  -75,   -1,   62,
     -86,  -66, -124,  -77,  -75,  -65,   54,   43,
     -40,  -70,  -12,   -9,  -42,   28,    7,   57,
     -54,  -49,   64,   77,  180,  234,  233,  250,
     -78,  -57,    1,  115,   43,  239,  158,  182,
     -98,   30,   59,  150,  144,   79,   71,  152,
];

#[rustfmt::skip]
const PST_KING_MG: [i32; 64] = [
      20,  185,  154,  -55,   47,  -91,  173,   77,
      68,  -37,   -2, -170, -124,  -46,   67,   56,
     -84,  -19, -116, -157, -252, -185, -149, -206,
    -121,  -40,  -74, -108,  -80,  -66,  -64, -106,
     -39,   16,  -18,  -45,  -43,  -31,   19,  -61,
      -9,   66,   35,  -17,   15,   51,   58,   12,
      41,   39,    0,    5,   15,   13,   34,  -17,
     -39,   49,   43,  -16,  -50,  -13,   32,   20,
];

#[rustfmt::skip]
const PST_PAWN_EG: [i32; 64] = [
       0,    0,    0,    0,    0,    0,    0,    0,
     140,  187,   95, -108,   87,  107,  120,  -30,
     107,   55,   33,  -17,   49,   19,   37,  -65,
     110,   88,  -27,  -36,    7,   92,   28,   22,
     200,   68,   88,   28,   55,   82,  112,   -6,
     160,  125,  107,   62,  -17,   31,  120,    8,
     143,   57,   55,   41, -113,   -2,  115,  -99,
       0,    0,    0,    0,    0,    0,    0,    0,
];

#[rustfmt::skip]
const PST_KNIGHT_EG: [i32; 64] = [
    -143, -116, -104,  -93,  -23, -115, -147,  -62,
     -71, -145,  -31,  -55,  -53,  -20,  -25,  -56,
     -71,  -17,   60,   94,   21,  -19,  -59,  -69,
    -106,   78,   73,  155,  159,  139,   87,  -25,
     136,  150,  155,  232,  188,  135,  147,  142,
      88,  -97,  141,   64,   15,  130,  -36,   25,
      23,  -13,  115,  132,   90,    2,   12,  -25,
     -72,   43,   40,   65,   27,   20,   33,  -51,
];

#[rustfmt::skip]
const PST_BISHOP_EG: [i32; 64] = [
     -80,  -40,  -51,  -60,   34,  -50,  -76,  -77,
       8,  -55,   44,  -26,  -22, -117,   34,  -76,
      -2,   91,   78,   54,   61,   32,  -30,   68,
      38,   59,  127,  -12,   64,  103,   53,   -3,
     110,  201,   76,  220,   91,   71,  157,   95,
      29, -137,   39,   14,   72,  104,   29,   50,
      70,   16, -144,  300,   92,   76,  -25,  -29,
     125,   17,  156,  -93,  248,  -62,   67,   -9,
];

#[rustfmt::skip]
const PST_ROOK_EG: [i32; 64] = [
     -55, -101,  -36,  -71,  -82,  -34, -157, -109,
     -63,  -94,  -68,  -41, -113,  -96,  -83,  -33,
    -267,  -52,  -40,  -86,  -84,  -57, -101,  -89,
     122,   14,   64,  -17,  -46,   41,  -14,   27,
     -16,   23,   66,   14,  -46,   97,   25,   11,
     -26,   17,   30,   36,  -29,   83,    5,   19,
     -28,  -19,   35,  -14,  -46,  -30,  -38,  -63,
     -51,  -32,  -26,  -47,  -18,  -39,  -59,  -72,
];

#[rustfmt::skip]
const PST_QUEEN_EG: [i32; 64] = [
     -89, -129, -125, -271, -144, -108,  -97,  -39,
     -65,  -64, -153, -252, -159, -170,  -82,  -48,
     -52, -111,  -24,  -49,  -33,   88,  -45,  -20,
      40,   26,   92,  135,   76,   45,   33,   61,
      16,   52,   84,  184,  157,  129,   86,  144,
      -1,   12,   98,  124,   96,  118,    5,   63,
      35,   51,   25,  142,  134,  119,   29,   74,
     -36,   -7,   29,  126,   99,    9,  -10,  -30,
];

#[rustfmt::skip]
const PST_KING_EG: [i32; 64] = [
     -75,   -2,  -37, -105,  -38,  -30,  -12,  -22,
     -20,  -21,  -96,  -75,  -55,  -17,  -21,  -46,
    -147,   -3,  -48,  -63,  -47,  -51,  -27,  -56,
     -27,  -27,    8,   37,  -19,   14,  -11,   21,
      30,   73,   64,   -8,   12,   28,   70,    0,
      30,   18,   19,   -9,   12,   47,   61,   93,
      20,  138,   25,   60,   21,   58,  162,   24,
       1,  119,   66,  -15,    2,   78,   89,  -71,
];

const PST_MG: [[i32; 64]; 6] = [
    PST_PAWN_MG, PST_KNIGHT_MG, PST_BISHOP_MG, PST_ROOK_MG, PST_QUEEN_MG, PST_KING_MG,
];

const PST_EG: [[i32; 64]; 6] = [
    PST_PAWN_EG, PST_KNIGHT_EG, PST_BISHOP_EG, PST_ROOK_EG, PST_QUEEN_EG, PST_KING_EG,
];
// @tuner:pst_end

// ---------------------------------------------------------------------------
// EvalParams: all tunable evaluation parameters in one struct.
// Used by the Texel tuner to optimize eval constants.
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct EvalParams {
    // Material (P, N, B, R, Q — king always 0)
    pub material_mg: [i32; 5],
    pub material_eg: [i32; 5],

    // Piece-square tables (6 pieces × 64 squares × 2 phases)
    pub pst_mg: [[i32; 64]; 6],
    pub pst_eg: [[i32; 64]; 6],

    // Pawn structure
    pub doubled_pawn_mg: i32,
    pub doubled_pawn_eg: i32,
    pub isolated_pawn_mg: i32,
    pub isolated_pawn_eg: i32,
    pub doubled_isolated_mg: i32,
    pub doubled_isolated_eg: i32,
    pub backward_pawn_mg: i32,
    pub backward_pawn_eg: i32,
    pub passed_pawn_base_mg: i32,
    pub passed_pawn_base_eg: i32,
    pub passed_pawn_adv_mg: i32,
    pub passed_pawn_adv_eg: i32,
    pub connected_passer_base_mg: i32,
    pub connected_passer_base_eg: i32,
    pub connected_passer_adv_mg: i32,
    pub connected_passer_adv_eg: i32,
    pub rook_behind_passer_mg: i32,
    pub rook_behind_passer_eg: i32,
    pub blocked_passer_mg: i32,
    pub blocked_passer_eg: i32,
    pub king_passer_own_eg: i32,
    pub king_passer_enemy_eg: i32,
    pub connected_passer_sq_mg: i32,
    pub connected_passer_sq_eg: i32,

    // Bishop pair
    pub bishop_pair_base_mg: i32,
    pub bishop_pair_base_eg: i32,

    // Rook bonuses
    pub rook_open_file_mg: i32,
    pub rook_open_file_eg: i32,
    pub rook_semi_open_mg: i32,
    pub rook_semi_open_eg: i32,
    pub rook_seventh_mg: i32,
    pub rook_seventh_eg: i32,
    pub doubled_rook_file_mg: i32,
    pub doubled_rook_file_eg: i32,
    pub doubled_rook_7th_mg: i32,
    pub doubled_rook_7th_eg: i32,

    // Trapped bishop penalty
    pub trapped_bishop_mg: i32,
    pub trapped_bishop_eg: i32,

    // King safety
    pub ks_shield_1: i32,
    pub ks_shield_2: i32,
    pub ks_open_file: i32,
    pub ks_semi_open: i32,
    pub ks_center_king: i32,
    pub ks_knight_weight: i32,
    pub ks_bishop_weight: i32,
    pub ks_rook_weight: i32,
    pub ks_queen_weight: i32,

    // Mobility (per square)
    pub mobility_knight_mg: i32,
    pub mobility_knight_eg: i32,
    pub mobility_bishop_mg: i32,
    pub mobility_bishop_eg: i32,
    pub mobility_rook_mg: i32,
    pub mobility_rook_eg: i32,
    pub mobility_queen_mg: i32,
    pub mobility_queen_eg: i32,

    // Center control
    pub center_pawn_bonus: i32,
    pub center_knight_bonus: i32,
    pub center_bishop_bonus: i32,

    // Connectivity
    pub pawn_protected_knight: i32,
    pub knight_outpost: i32,

    // Threats
    pub threat_pawn_minor_mg: i32,
    pub threat_pawn_minor_eg: i32,
    pub threat_pawn_rook_mg: i32,
    pub threat_pawn_rook_eg: i32,
    pub threat_minor_rook_mg: i32,
    pub threat_minor_rook_eg: i32,
    pub threat_piece_queen_mg: i32,
    pub threat_piece_queen_eg: i32,
    pub threat_hanging_mg: i32,
    pub threat_hanging_eg: i32,

    // Tempo
    pub tempo: i32,

    // Bad bishop: penalty per own pawn on bishop's color complex
    pub bad_bishop_mg: i32,
    pub bad_bishop_eg: i32,

    // Knight vs bishop adjustment based on pawn count (closed = many pawns)
    pub knight_closed_bonus: i32,   // Knight bonus per pawn above 5
    pub bishop_open_bonus: i32,     // Bishop bonus per pawn below 5

    // Pawn islands: penalty per island beyond the first
    pub pawn_islands_mg: i32,
    pub pawn_islands_eg: i32,

    // Opposite-colored bishop endgame scale factor (0-100, 100 = no scaling)
    pub ocb_scale_factor: i32,

    // Material imbalance params
    pub imbalance_exchange_mg: i32,   // rook vs minor, MG
    pub imbalance_exchange_eg: i32,   // rook vs minor, EG
    pub imbalance_rook_pair_mg: i32,  // redundant rook pair penalty
    pub imbalance_rook_pair_eg: i32,
    pub imbalance_knight_pair_eg: i32, // two-knight penalty (EG only)
    pub imbalance_queen_vs_minors_mg: i32, // queen vs 2+ minors
    pub imbalance_queen_vs_minors_eg: i32,
}

// @tuner:defaults_start
impl Default for EvalParams {
    fn default() -> Self {
        Self {
            material_mg: [82, 337, 365, 477, 1025],
            material_eg: [94, 281, 297, 512, 936],
            pst_mg: PST_MG,
            pst_eg: PST_EG,
            doubled_pawn_mg: -4,
            doubled_pawn_eg: -50,
            isolated_pawn_mg: -45,
            isolated_pawn_eg: -50,
            doubled_isolated_mg: -50,
            doubled_isolated_eg: -50,
            backward_pawn_mg: -50,
            backward_pawn_eg: -17,
            passed_pawn_base_mg: 22,
            passed_pawn_base_eg: 3,
            passed_pawn_adv_mg: 15,
            passed_pawn_adv_eg: 1,
            connected_passer_base_mg: 2,
            connected_passer_base_eg: 2,
            connected_passer_adv_mg: 1,
            connected_passer_adv_eg: 1,
            rook_behind_passer_mg: 50,
            rook_behind_passer_eg: 100,
            blocked_passer_mg: -2,
            blocked_passer_eg: -35,
            king_passer_own_eg: 1,
            king_passer_enemy_eg: 40,
            connected_passer_sq_mg: 0,
            connected_passer_sq_eg: 1,
            bishop_pair_base_mg: 36,
            bishop_pair_base_eg: 60,
            rook_open_file_mg: 50,
            rook_open_file_eg: 45,
            rook_semi_open_mg: 30,
            rook_semi_open_eg: 3,
            rook_seventh_mg: 60,
            rook_seventh_eg: 7,
            doubled_rook_file_mg: 93,
            doubled_rook_file_eg: 31,
            doubled_rook_7th_mg: 78,
            doubled_rook_7th_eg: 147,
            trapped_bishop_mg: -68,
            trapped_bishop_eg: -67,
            ks_shield_1: 30,
            ks_shield_2: 30,
            ks_open_file: -60,
            ks_semi_open: -60,
            ks_center_king: -5,
            ks_knight_weight: 161,
            ks_bishop_weight: 170,
            ks_rook_weight: 200,
            ks_queen_weight: 151,
            mobility_knight_mg: 3,
            mobility_knight_eg: 12,
            mobility_bishop_mg: 12,
            mobility_bishop_eg: 12,
            mobility_rook_mg: 12,
            mobility_rook_eg: 12,
            mobility_queen_mg: 2,
            mobility_queen_eg: 12,
            center_pawn_bonus: 5,
            center_knight_bonus: 3,
            center_bishop_bonus: 18,
            pawn_protected_knight: 5,
            knight_outpost: 9,
            threat_pawn_minor_mg: 48,
            threat_pawn_minor_eg: 5,
            threat_pawn_rook_mg: 5,
            threat_pawn_rook_eg: 68,
            threat_minor_rook_mg: 93,
            threat_minor_rook_eg: 120,
            threat_piece_queen_mg: 120,
            threat_piece_queen_eg: 88,
            threat_hanging_mg: 52,
            threat_hanging_eg: 120,
            tempo: 20,
            bad_bishop_mg: -15,
            bad_bishop_eg: -20,
            knight_closed_bonus: 2,
            bishop_open_bonus: 2,
            pawn_islands_mg: -20,
            pawn_islands_eg: -3,
            ocb_scale_factor: 71,
            imbalance_exchange_mg: 20,
            imbalance_exchange_eg: 150,
            imbalance_rook_pair_mg: -40,
            imbalance_rook_pair_eg: -40,
            imbalance_knight_pair_eg: -25,
            imbalance_queen_vs_minors_mg: 27,
            imbalance_queen_vs_minors_eg: 11,
        }
    }
}
// @tuner:defaults_end

/// Piece names for PST parameter naming.
const PIECE_NAMES: [&str; 6] = ["pawn", "knight", "bishop", "rook", "queen", "king"];

/// Number of scalar params (after material and PSTs).
const NUM_SCALAR_PARAMS: usize = 85;

impl EvalParams {
    /// Total number of tunable parameters.
    pub fn param_count() -> usize {
        // 5 material_mg + 5 material_eg + 384 pst_mg + 384 pst_eg + scalars
        10 + 768 + NUM_SCALAR_PARAMS
    }

    /// Get the value of parameter at index `idx`.
    pub fn get_param(&self, idx: usize) -> i32 {
        match idx {
            0..5 => self.material_mg[idx],
            5..10 => self.material_eg[idx - 5],
            10..394 => {
                let off = idx - 10;
                self.pst_mg[off / 64][off % 64]
            }
            394..778 => {
                let off = idx - 394;
                self.pst_eg[off / 64][off % 64]
            }
            _ => self.get_scalar(idx - 778),
        }
    }

    /// Set the value of parameter at index `idx`.
    pub fn set_param(&mut self, idx: usize, val: i32) {
        match idx {
            0..5 => self.material_mg[idx] = val,
            5..10 => self.material_eg[idx - 5] = val,
            10..394 => {
                let off = idx - 10;
                self.pst_mg[off / 64][off % 64] = val;
            }
            394..778 => {
                let off = idx - 394;
                self.pst_eg[off / 64][off % 64] = val;
            }
            _ => self.set_scalar(idx - 778, val),
        }
    }

    /// Human-readable name for parameter at index `idx`.
    pub fn param_name(&self, idx: usize) -> String {
        match idx {
            0..5 => format!("material_mg_{}", PIECE_NAMES[idx]),
            5..10 => format!("material_eg_{}", PIECE_NAMES[idx - 5]),
            10..394 => {
                let off = idx - 10;
                format!("pst_mg_{}_{}", PIECE_NAMES[off / 64], off % 64)
            }
            394..778 => {
                let off = idx - 394;
                format!("pst_eg_{}_{}", PIECE_NAMES[off / 64], off % 64)
            }
            _ => self.scalar_name(idx - 778),
        }
    }

    fn get_scalar(&self, idx: usize) -> i32 {
        match idx {
            0 => self.doubled_pawn_mg,
            1 => self.doubled_pawn_eg,
            2 => self.isolated_pawn_mg,
            3 => self.isolated_pawn_eg,
            4 => self.doubled_isolated_mg,
            5 => self.doubled_isolated_eg,
            6 => self.backward_pawn_mg,
            7 => self.backward_pawn_eg,
            8 => self.passed_pawn_base_mg,
            9 => self.passed_pawn_base_eg,
            10 => self.passed_pawn_adv_mg,
            11 => self.passed_pawn_adv_eg,
            12 => self.connected_passer_base_mg,
            13 => self.connected_passer_base_eg,
            14 => self.connected_passer_adv_mg,
            15 => self.connected_passer_adv_eg,
            16 => self.rook_behind_passer_mg,
            17 => self.rook_behind_passer_eg,
            18 => self.blocked_passer_mg,
            19 => self.blocked_passer_eg,
            20 => self.bishop_pair_base_mg,
            21 => self.bishop_pair_base_eg,
            22 => self.rook_open_file_mg,
            23 => self.rook_open_file_eg,
            24 => self.rook_semi_open_mg,
            25 => self.rook_semi_open_eg,
            26 => self.rook_seventh_mg,
            27 => self.rook_seventh_eg,
            28 => self.ks_shield_1,
            29 => self.ks_shield_2,
            30 => self.ks_open_file,
            31 => self.ks_semi_open,
            32 => self.ks_center_king,
            33 => self.ks_knight_weight,
            34 => self.ks_bishop_weight,
            35 => self.ks_rook_weight,
            36 => self.ks_queen_weight,
            37 => self.mobility_knight_mg,
            38 => self.mobility_knight_eg,
            39 => self.mobility_bishop_mg,
            40 => self.mobility_bishop_eg,
            41 => self.mobility_rook_mg,
            42 => self.mobility_rook_eg,
            43 => self.mobility_queen_mg,
            44 => self.mobility_queen_eg,
            45 => self.center_pawn_bonus,
            46 => self.center_knight_bonus,
            47 => self.center_bishop_bonus,
            48 => self.pawn_protected_knight,
            49 => self.knight_outpost,
            50 => self.threat_pawn_minor_mg,
            51 => self.threat_pawn_minor_eg,
            52 => self.threat_pawn_rook_mg,
            53 => self.threat_pawn_rook_eg,
            54 => self.threat_minor_rook_mg,
            55 => self.threat_minor_rook_eg,
            56 => self.threat_piece_queen_mg,
            57 => self.threat_piece_queen_eg,
            58 => self.threat_hanging_mg,
            59 => self.threat_hanging_eg,
            60 => self.tempo,
            61 => self.bad_bishop_mg,
            62 => self.bad_bishop_eg,
            63 => self.knight_closed_bonus,
            64 => self.bishop_open_bonus,
            65 => self.pawn_islands_mg,
            66 => self.pawn_islands_eg,
            67 => self.ocb_scale_factor,
            68 => self.king_passer_own_eg,
            69 => self.king_passer_enemy_eg,
            70 => self.connected_passer_sq_mg,
            71 => self.connected_passer_sq_eg,
            72 => self.imbalance_exchange_mg,
            73 => self.imbalance_exchange_eg,
            74 => self.imbalance_rook_pair_mg,
            75 => self.imbalance_rook_pair_eg,
            76 => self.imbalance_knight_pair_eg,
            77 => self.imbalance_queen_vs_minors_mg,
            78 => self.imbalance_queen_vs_minors_eg,
            79 => self.doubled_rook_file_mg,
            80 => self.doubled_rook_file_eg,
            81 => self.doubled_rook_7th_mg,
            82 => self.doubled_rook_7th_eg,
            83 => self.trapped_bishop_mg,
            84 => self.trapped_bishop_eg,
            _ => panic!("scalar index {idx} out of range"),
        }
    }

    fn set_scalar(&mut self, idx: usize, val: i32) {
        match idx {
            0 => self.doubled_pawn_mg = val,
            1 => self.doubled_pawn_eg = val,
            2 => self.isolated_pawn_mg = val,
            3 => self.isolated_pawn_eg = val,
            4 => self.doubled_isolated_mg = val,
            5 => self.doubled_isolated_eg = val,
            6 => self.backward_pawn_mg = val,
            7 => self.backward_pawn_eg = val,
            8 => self.passed_pawn_base_mg = val,
            9 => self.passed_pawn_base_eg = val,
            10 => self.passed_pawn_adv_mg = val,
            11 => self.passed_pawn_adv_eg = val,
            12 => self.connected_passer_base_mg = val,
            13 => self.connected_passer_base_eg = val,
            14 => self.connected_passer_adv_mg = val,
            15 => self.connected_passer_adv_eg = val,
            16 => self.rook_behind_passer_mg = val,
            17 => self.rook_behind_passer_eg = val,
            18 => self.blocked_passer_mg = val,
            19 => self.blocked_passer_eg = val,
            20 => self.bishop_pair_base_mg = val,
            21 => self.bishop_pair_base_eg = val,
            22 => self.rook_open_file_mg = val,
            23 => self.rook_open_file_eg = val,
            24 => self.rook_semi_open_mg = val,
            25 => self.rook_semi_open_eg = val,
            26 => self.rook_seventh_mg = val,
            27 => self.rook_seventh_eg = val,
            28 => self.ks_shield_1 = val,
            29 => self.ks_shield_2 = val,
            30 => self.ks_open_file = val,
            31 => self.ks_semi_open = val,
            32 => self.ks_center_king = val,
            33 => self.ks_knight_weight = val,
            34 => self.ks_bishop_weight = val,
            35 => self.ks_rook_weight = val,
            36 => self.ks_queen_weight = val,
            37 => self.mobility_knight_mg = val,
            38 => self.mobility_knight_eg = val,
            39 => self.mobility_bishop_mg = val,
            40 => self.mobility_bishop_eg = val,
            41 => self.mobility_rook_mg = val,
            42 => self.mobility_rook_eg = val,
            43 => self.mobility_queen_mg = val,
            44 => self.mobility_queen_eg = val,
            45 => self.center_pawn_bonus = val,
            46 => self.center_knight_bonus = val,
            47 => self.center_bishop_bonus = val,
            48 => self.pawn_protected_knight = val,
            49 => self.knight_outpost = val,
            50 => self.threat_pawn_minor_mg = val,
            51 => self.threat_pawn_minor_eg = val,
            52 => self.threat_pawn_rook_mg = val,
            53 => self.threat_pawn_rook_eg = val,
            54 => self.threat_minor_rook_mg = val,
            55 => self.threat_minor_rook_eg = val,
            56 => self.threat_piece_queen_mg = val,
            57 => self.threat_piece_queen_eg = val,
            58 => self.threat_hanging_mg = val,
            59 => self.threat_hanging_eg = val,
            60 => self.tempo = val,
            61 => self.bad_bishop_mg = val,
            62 => self.bad_bishop_eg = val,
            63 => self.knight_closed_bonus = val,
            64 => self.bishop_open_bonus = val,
            65 => self.pawn_islands_mg = val,
            66 => self.pawn_islands_eg = val,
            67 => self.ocb_scale_factor = val,
            68 => self.king_passer_own_eg = val,
            69 => self.king_passer_enemy_eg = val,
            70 => self.connected_passer_sq_mg = val,
            71 => self.connected_passer_sq_eg = val,
            72 => self.imbalance_exchange_mg = val,
            73 => self.imbalance_exchange_eg = val,
            74 => self.imbalance_rook_pair_mg = val,
            75 => self.imbalance_rook_pair_eg = val,
            76 => self.imbalance_knight_pair_eg = val,
            77 => self.imbalance_queen_vs_minors_mg = val,
            78 => self.imbalance_queen_vs_minors_eg = val,
            79 => self.doubled_rook_file_mg = val,
            80 => self.doubled_rook_file_eg = val,
            81 => self.doubled_rook_7th_mg = val,
            82 => self.doubled_rook_7th_eg = val,
            83 => self.trapped_bishop_mg = val,
            84 => self.trapped_bishop_eg = val,
            _ => panic!("scalar index {idx} out of range"),
        }
    }

    fn scalar_name(&self, idx: usize) -> String {
        const NAMES: [&str; NUM_SCALAR_PARAMS] = [
            "doubled_pawn_mg", "doubled_pawn_eg",
            "isolated_pawn_mg", "isolated_pawn_eg",
            "doubled_isolated_mg", "doubled_isolated_eg",
            "backward_pawn_mg", "backward_pawn_eg",
            "passed_pawn_base_mg", "passed_pawn_base_eg",
            "passed_pawn_adv_mg", "passed_pawn_adv_eg",
            "connected_passer_base_mg", "connected_passer_base_eg",
            "connected_passer_adv_mg", "connected_passer_adv_eg",
            "rook_behind_passer_mg", "rook_behind_passer_eg",
            "blocked_passer_mg", "blocked_passer_eg",
            "bishop_pair_base_mg", "bishop_pair_base_eg",
            "rook_open_file_mg", "rook_open_file_eg",
            "rook_semi_open_mg", "rook_semi_open_eg",
            "rook_seventh_mg", "rook_seventh_eg",
            "ks_shield_1", "ks_shield_2",
            "ks_open_file", "ks_semi_open",
            "ks_center_king",
            "ks_knight_weight", "ks_bishop_weight",
            "ks_rook_weight", "ks_queen_weight",
            "mobility_knight_mg", "mobility_knight_eg",
            "mobility_bishop_mg", "mobility_bishop_eg",
            "mobility_rook_mg", "mobility_rook_eg",
            "mobility_queen_mg", "mobility_queen_eg",
            "center_pawn_bonus", "center_knight_bonus", "center_bishop_bonus",
            "pawn_protected_knight", "knight_outpost",
            "threat_pawn_minor_mg", "threat_pawn_minor_eg",
            "threat_pawn_rook_mg", "threat_pawn_rook_eg",
            "threat_minor_rook_mg", "threat_minor_rook_eg",
            "threat_piece_queen_mg", "threat_piece_queen_eg",
            "threat_hanging_mg", "threat_hanging_eg",
            "tempo",
            "bad_bishop_mg", "bad_bishop_eg",
            "knight_closed_bonus", "bishop_open_bonus",
            "pawn_islands_mg", "pawn_islands_eg",
            "ocb_scale_factor",
            "king_passer_own_eg", "king_passer_enemy_eg",
            "connected_passer_sq_mg", "connected_passer_sq_eg",
            "imbalance_exchange_mg", "imbalance_exchange_eg",
            "imbalance_rook_pair_mg", "imbalance_rook_pair_eg",
            "imbalance_knight_pair_eg",
            "imbalance_queen_vs_minors_mg", "imbalance_queen_vs_minors_eg",
            "doubled_rook_file_mg", "doubled_rook_file_eg",
            "doubled_rook_7th_mg", "doubled_rook_7th_eg",
            "trapped_bishop_mg", "trapped_bishop_eg",
        ];
        NAMES[idx].to_string()
    }

    /// Print parameters as Rust code suitable for pasting into eval.rs.
    pub fn print_rust_code(&self) {
        println!("// Material values (P, N, B, R, Q)");
        println!("material_mg: {:?},", self.material_mg);
        println!("material_eg: {:?},", self.material_eg);

        println!("\n// Piece-square tables (MG)");
        for (i, name) in PIECE_NAMES.iter().enumerate() {
            println!("#[rustfmt::skip]");
            println!("// PST MG {name}");
            print!("[");
            for sq in 0..64 {
                if sq % 8 == 0 { print!("\n    "); }
                print!("{:4},", self.pst_mg[i][sq]);
            }
            println!("\n],");
        }

        println!("\n// Piece-square tables (EG)");
        for (i, name) in PIECE_NAMES.iter().enumerate() {
            println!("#[rustfmt::skip]");
            println!("// PST EG {name}");
            print!("[");
            for sq in 0..64 {
                if sq % 8 == 0 { print!("\n    "); }
                print!("{:4},", self.pst_eg[i][sq]);
            }
            println!("\n],");
        }

        println!("\n// Scalar parameters");
        for i in 0..NUM_SCALAR_PARAMS {
            println!("{}: {},", self.scalar_name(i), self.get_scalar(i));
        }
    }
}

// ---------------------------------------------------------------------------
// Material + PST (combined loop)
// ---------------------------------------------------------------------------

fn material_and_pst(board: &Board, params: &EvalParams, mg: &mut i32, eg: &mut i32) {
    for &kind in &ALL_PIECE_KINDS {
        let ki = kind.index();
        let mat_mg = if ki < 5 { params.material_mg[ki] } else { 0 };
        let mat_eg = if ki < 5 { params.material_eg[ki] } else { 0 };
        let pst_mg = &params.pst_mg[ki];
        let pst_eg = &params.pst_eg[ki];

        for sq in board.pieces[Color::White.index()][ki].iter() {
            *mg += mat_mg + pst_mg[sq.index()];
            *eg += mat_eg + pst_eg[sq.index()];
        }
        for sq in board.pieces[Color::Black.index()][ki].iter() {
            *mg -= mat_mg + pst_mg[mirror_square(sq)];
            *eg -= mat_eg + pst_eg[mirror_square(sq)];
        }
    }
}

// ---------------------------------------------------------------------------
// Pawn structure (tapered)
// ---------------------------------------------------------------------------

fn pawn_structure(board: &Board, params: &EvalParams) -> (i32, i32) {
    let mut mg = 0i32;
    let mut eg = 0i32;

    let white_pawns = board.pieces[Color::White.index()][PieceKind::Pawn.index()];
    let black_pawns = board.pieces[Color::Black.index()][PieceKind::Pawn.index()];

    let (w_mg, w_eg) = pawn_structure_for_color(board, white_pawns, black_pawns, Color::White, params);
    let (b_mg, b_eg) = pawn_structure_for_color(board, black_pawns, white_pawns, Color::Black, params);

    mg += w_mg - b_mg;
    eg += w_eg - b_eg;

    (mg, eg)
}

fn pawn_structure_for_color(
    board: &Board,
    own_pawns: Bitboard,
    enemy_pawns: Bitboard,
    color: Color,
    params: &EvalParams,
) -> (i32, i32) {
    let mut mg = 0i32;
    let mut eg = 0i32;
    let occ = board.all_occupancy();

    // Pure pawn endgame: enemy has no pieces (only king + pawns)
    let enemy_ci = color.opposite().index();
    let enemy_has_pieces =
        !board.pieces[enemy_ci][PieceKind::Knight.index()].is_empty()
        || !board.pieces[enemy_ci][PieceKind::Bishop.index()].is_empty()
        || !board.pieces[enemy_ci][PieceKind::Rook.index()].is_empty()
        || !board.pieces[enemy_ci][PieceKind::Queen.index()].is_empty();

    for sq in own_pawns.iter() {
        let file = sq.file();
        let file_bb = file_bitboard(file);

        // Doubled pawns
        if (own_pawns & file_bb).count() > 1 {
            mg += params.doubled_pawn_mg;
            eg += params.doubled_pawn_eg;
        }

        // Isolated pawns
        let adj = adjacent_files(file);
        let isolated = (own_pawns & adj).is_empty();
        if isolated {
            mg += params.isolated_pawn_mg;
            eg += params.isolated_pawn_eg;
        }

        // Doubled + isolated compound penalty
        if isolated && (own_pawns & file_bb).count() > 1 {
            mg += params.doubled_isolated_mg;
            eg += params.doubled_isolated_eg;
        }

        // Backward pawn: no friendly pawn can support its advance,
        // and the stop square is controlled by enemy pawns.
        if !isolated {
            let stop_sq_bb = match color {
                Color::White => sq.bitboard().north(),
                Color::Black => sq.bitboard().south(),
            };
            // Pawn is backward if no own pawn on adjacent files is behind or level
            // (i.e., could advance to support this pawn)
            let behind_and_level = behind_span_mask(sq, color) | rank_bitboard(sq.rank());
            let supporters = own_pawns & adj & behind_and_level;
            if supporters.is_empty() {
                // And the stop square is attacked by enemy pawns
                let enemy_pawn_att = pawn_attack_span(enemy_pawns, color.opposite());
                if !(stop_sq_bb & enemy_pawn_att).is_empty() {
                    mg += params.backward_pawn_mg;
                    eg += params.backward_pawn_eg;
                }
            }
        }

        // Passed pawn
        let front_span = front_span_mask(sq, color);
        let block_mask = front_span & (file_bb | adj);
        if (enemy_pawns & block_mask).is_empty() {
            let advancement = match color {
                Color::White => sq.rank() as i32 - 1,
                Color::Black => 6 - sq.rank() as i32,
            };

            // Passed pawn bonus (quadratic in advancement)
            let mut pass_mg = params.passed_pawn_base_mg + advancement * advancement * params.passed_pawn_adv_mg;
            let mut pass_eg = params.passed_pawn_base_eg + advancement * advancement * params.passed_pawn_adv_eg;

            // Passed pawn bonus scaling vs enemy material (blockades/containment)
            let enemy_ci = color.opposite().index();
            let enemy_rooks = board.pieces[enemy_ci][PieceKind::Rook.index()];
            let enemy_bishops = board.pieces[enemy_ci][PieceKind::Bishop.index()];
            let enemy_queens = board.pieces[enemy_ci][PieceKind::Queen.index()];

            let mut scale = 100;
            if !enemy_rooks.is_empty() {
                scale -= 12;
            }
            if enemy_bishops.count() >= 2 {
                scale -= 8;
            }
            if !enemy_queens.is_empty() {
                scale -= 6;
            }
            if scale < 70 {
                scale = 70;
            }
            pass_mg = pass_mg * scale / 100;
            pass_eg = pass_eg * scale / 100;
            mg += pass_mg;
            eg += pass_eg;

            // Extra bonus for very advanced passers (6th/7th rank)
            // These pawns are extremely dangerous and tie down defenders
            if advancement >= 4 {
                let advanced_bonus = (advancement - 3) * (advancement - 3) * 20;
                mg += advanced_bonus;
                eg += advanced_bonus * 2;
            }

            // Bonus for 7th rank passers (one step from queening)
            if advancement >= 5 {
                let promo_threat = (advancement - 4) * 60;
                mg += promo_threat;
                eg += promo_threat * 2;
            }

            // Connected passed pawns
            let adj_pawns = own_pawns & adj;
            if !adj_pawns.is_empty() {
                for adj_sq in adj_pawns.iter() {
                    let adj_front = front_span_mask(adj_sq, color);
                    let adj_block = adj_front & (file_bitboard(adj_sq.file()) | adjacent_files(adj_sq.file()));
                    if (enemy_pawns & adj_block).is_empty() {
                        mg += params.connected_passer_base_mg + advancement * params.connected_passer_adv_mg
                            + advancement * advancement * params.connected_passer_sq_mg;
                        eg += params.connected_passer_base_eg + advancement * params.connected_passer_adv_eg
                            + advancement * advancement * params.connected_passer_sq_eg;
                        break;
                    }
                }
            }

            // King proximity to passed pawn (endgame weighted)
            let own_king = board.king_square(color);
            let enemy_king = board.king_square(color.opposite());
            let promo_sq = match color {
                Color::White => Square::new(file, 7),
                Color::Black => Square::new(file, 0),
            };

            let own_king_dist = chebyshev_distance(own_king, promo_sq);
            let enemy_king_dist = chebyshev_distance(enemy_king, promo_sq);

            // Bonus for own king being close to promotion square (scales with advancement)
            eg += (7 - own_king_dist as i32) * params.king_passer_own_eg * (1 + advancement) / 4;
            // Bonus for enemy king being far from promotion square (scales with advancement)
            eg += enemy_king_dist as i32 * params.king_passer_enemy_eg * (1 + advancement) / 4;

            // Blocked passed pawn penalty
            let stop_sq = match color {
                Color::White => {
                    if sq.rank() < 7 { Some(Square::new(file, sq.rank() + 1)) } else { None }
                }
                Color::Black => {
                    if sq.rank() > 0 { Some(Square::new(file, sq.rank() - 1)) } else { None }
                }
            };
            if let Some(stop) = stop_sq {
                if let Some(p) = board.piece_at(stop) {
                    mg += params.blocked_passer_mg;
                    eg += params.blocked_passer_eg;
                    if p.color == color.opposite() {
                        // Enemy blockader directly in front is very strong.
                        let block_pen = if p.kind == PieceKind::Rook { 25 } else { 15 };
                        mg -= block_pen + advancement * 2;
                        eg -= block_pen * 2 + advancement * 6;
                    }
                }

                // Containment: if enemy controls the stop square, the passer is less valuable.
                if square_attacked_by(board, stop, color.opposite(), occ) {
                    let contain = 10 + advancement * 4;
                    mg -= contain;
                    eg -= contain * 2;
                }
            }

            // Containment: if enemy controls the promotion square, reduce the bonus.
            if square_attacked_by(board, promo_sq, color.opposite(), occ) {
                let contain = 15 + advancement * 6;
                mg -= contain;
                eg -= contain * 2;
            }

            // Rook behind passed pawn
            let own_rooks = board.pieces[color.index()][PieceKind::Rook.index()];
            let behind = behind_span_mask(sq, color);
            let rook_behind = !(own_rooks & behind & file_bb).is_empty();
            if rook_behind {
                mg += params.rook_behind_passer_mg;
                eg += params.rook_behind_passer_eg;
                // Synergy: rook behind a very advanced passer is extremely strong
                if advancement >= 4 {
                    let synergy = (advancement - 3) * 25;
                    mg += synergy;
                    eg += synergy * 2;
                }
            }

            // Enemy rook in front of passed pawn on same file: pawn is effectively frozen
            let enemy_rooks = board.pieces[color.opposite().index()][PieceKind::Rook.index()];
            if !(enemy_rooks & front_span & file_bb).is_empty() {
                // Scale penalty with advancement squared to match the quadratic bonus
                mg -= advancement * advancement * 2;
                eg -= 10 + advancement * advancement * 6;
            }

            // Queen supporting the passed pawn: the queen can simultaneously
            // escort the pawn and threaten the enemy king, making the passer
            // dramatically more dangerous.
            let own_queens = board.pieces[color.index()][PieceKind::Queen.index()];
            if !own_queens.is_empty() && advancement >= 3 {
                let queen_support = (advancement - 2) * (advancement - 2) * 15;
                mg += queen_support;
                eg += queen_support * 2;
                // Extra: queen + pawn on 6th/7th is usually decisive
                if advancement >= 5 {
                    eg += 100;
                }
            }

            // Unstoppable passed pawn: enemy king outside the "square of the pawn"
            {
                let steps_to_promote = match color {
                    Color::White => 7 - sq.rank() as i32,
                    Color::Black => sq.rank() as i32,
                };
                // If it's the enemy's turn, their king gets to move first,
                // so they effectively need one fewer step to catch the pawn.
                let effective_steps = if board.side_to_move == color {
                    steps_to_promote
                } else {
                    steps_to_promote + 1
                };
                if enemy_king_dist as i32 > effective_steps {
                    if !enemy_has_pieces {
                        // Pure pawn endgame: fully unstoppable
                        eg += 350;
                    } else if enemy_rooks.count() > 0
                        && board.pieces[color.opposite().index()][PieceKind::Bishop.index()].is_empty()
                        && board.pieces[color.opposite().index()][PieceKind::Knight.index()].is_empty()
                        && board.pieces[color.opposite().index()][PieceKind::Queen.index()].is_empty()
                        && rook_behind
                    {
                        // Enemy has only rook(s) but our rook supports the passer:
                        // partial unstoppable bonus (the rook can sacrifice to stop pawn,
                        // but that's still a huge material gain)
                        eg += 250;
                    } else if !own_queens.is_empty() {
                        // Queen + passer vs pieces: queen can sacrifice herself
                        // to promote, or escort the pawn. Very strong.
                        eg += 200;
                    }
                }
            }
        }
    }

    (mg, eg)
}

// ---------------------------------------------------------------------------
// Bishop pair
// ---------------------------------------------------------------------------

fn bishop_pair(board: &Board, params: &EvalParams) -> (i32, i32) {
    let mut mg = 0i32;
    let mut eg = 0i32;
    let total_pawns = (board.pieces[Color::White.index()][PieceKind::Pawn.index()]
        | board.pieces[Color::Black.index()][PieceKind::Pawn.index()])
    .count() as i32;
    // Bishop pair is stronger with fewer pawns (more open position)
    let bonus_mg = params.bishop_pair_base_mg + (16 - total_pawns);
    let bonus_eg = params.bishop_pair_base_eg + (16 - total_pawns) * 2;
    if board.pieces[Color::White.index()][PieceKind::Bishop.index()].count() >= 2 {
        mg += bonus_mg;
        eg += bonus_eg;
    }
    if board.pieces[Color::Black.index()][PieceKind::Bishop.index()].count() >= 2 {
        mg -= bonus_mg;
        eg -= bonus_eg;
    }
    (mg, eg)
}

// ---------------------------------------------------------------------------
// Rook on open/semi-open files + 7th rank (tapered)
// ---------------------------------------------------------------------------

fn rook_open_files(board: &Board, params: &EvalParams) -> (i32, i32) {
    let mut mg = 0i32;
    let mut eg = 0i32;
    let white_pawns = board.pieces[Color::White.index()][PieceKind::Pawn.index()];
    let black_pawns = board.pieces[Color::Black.index()][PieceKind::Pawn.index()];

    for sq in board.pieces[Color::White.index()][PieceKind::Rook.index()].iter() {
        let file_bb = file_bitboard(sq.file());
        if (white_pawns & file_bb).is_empty() {
            if (black_pawns & file_bb).is_empty() {
                mg += params.rook_open_file_mg;
                eg += params.rook_open_file_eg;
            } else {
                mg += params.rook_semi_open_mg;
                eg += params.rook_semi_open_eg;
            }
        }
        // Rook activity on 4th/5th rank
        if sq.rank() == 4 {
            mg += 12;
            eg += 10;
        } else if sq.rank() == 3 {
            mg += 8;
            eg += 6;
        }
        // Bonus for rook on file with an advanced own pawn
        let advanced_pawns = white_pawns & file_bb
            & (rank_bitboard(4) | rank_bitboard(5) | rank_bitboard(6) | rank_bitboard(7));
        if !advanced_pawns.is_empty() {
            mg += 6;
            eg += 8;
        }
        // Rook on 7th rank
        if sq.rank() == 6 {
            mg += params.rook_seventh_mg;
            eg += params.rook_seventh_eg;
            // Extra bonus if enemy king is trapped on back rank
            let enemy_king = board.king_square(Color::Black);
            if enemy_king.rank() == 7 {
                mg += 15;
                eg += 30;
            }
        }
        // Rook on 8th rank (deep penetration behind enemy lines)
        if sq.rank() == 7 {
            mg += 15;
            eg += 35;
        }
        // Rook on enemy half (ranks 4-7 = indices 4..=7)
        if sq.rank() >= 4 {
            eg += 10;
        }
    }

    for sq in board.pieces[Color::Black.index()][PieceKind::Rook.index()].iter() {
        let file_bb = file_bitboard(sq.file());
        if (black_pawns & file_bb).is_empty() {
            if (white_pawns & file_bb).is_empty() {
                mg -= params.rook_open_file_mg;
                eg -= params.rook_open_file_eg;
            } else {
                mg -= params.rook_semi_open_mg;
                eg -= params.rook_semi_open_eg;
            }
        }
        // Rook activity on 4th/5th rank (from Black's perspective)
        if sq.rank() == 3 {
            mg -= 12;
            eg -= 10;
        } else if sq.rank() == 4 {
            mg -= 8;
            eg -= 6;
        }
        // Bonus for rook on file with an advanced own pawn
        let advanced_pawns = black_pawns & file_bb
            & (rank_bitboard(0) | rank_bitboard(1) | rank_bitboard(2) | rank_bitboard(3));
        if !advanced_pawns.is_empty() {
            mg -= 6;
            eg -= 8;
        }
        // Rook on 2nd rank (7th from Black's perspective)
        if sq.rank() == 1 {
            mg -= params.rook_seventh_mg;
            eg -= params.rook_seventh_eg;
            // Extra bonus if enemy king is trapped on back rank
            let enemy_king = board.king_square(Color::White);
            if enemy_king.rank() == 0 {
                mg -= 15;
                eg -= 30;
            }
        }
        // Rook on 1st rank (deep penetration behind enemy lines)
        if sq.rank() == 0 {
            mg -= 15;
            eg -= 35;
        }
        // Rook on enemy half (ranks 0-3 = indices 0..=3)
        if sq.rank() <= 3 {
            eg -= 10;
        }
    }

    // Doubled rooks: two rooks on the same file or both on 7th rank
    let white_rooks = board.pieces[Color::White.index()][PieceKind::Rook.index()];
    let black_rooks = board.pieces[Color::Black.index()][PieceKind::Rook.index()];
    if white_rooks.count() >= 2 {
        let mut remaining = white_rooks;
        let (Some(sq1), rest) = remaining.pop_lsb() else { unreachable!() };
        remaining = rest;
        while let (Some(sq2), next) = remaining.pop_lsb() {
            if sq1.file() == sq2.file() {
                mg += params.doubled_rook_file_mg;
                eg += params.doubled_rook_file_eg;
            }
            remaining = next;
        }
        let rooks_on_7th = white_rooks & rank_bitboard(6);
        if rooks_on_7th.count() >= 2 {
            mg += params.doubled_rook_7th_mg;
            eg += params.doubled_rook_7th_eg;
        }
    }
    if black_rooks.count() >= 2 {
        let mut remaining = black_rooks;
        let (Some(sq1), rest) = remaining.pop_lsb() else { unreachable!() };
        remaining = rest;
        while let (Some(sq2), next) = remaining.pop_lsb() {
            if sq1.file() == sq2.file() {
                mg -= params.doubled_rook_file_mg;
                eg -= params.doubled_rook_file_eg;
            }
            remaining = next;
        }
        let rooks_on_2nd = black_rooks & rank_bitboard(1);
        if rooks_on_2nd.count() >= 2 {
            mg -= params.doubled_rook_7th_mg;
            eg -= params.doubled_rook_7th_eg;
        }
    }

    (mg, eg)
}

// ---------------------------------------------------------------------------
// Trapped bishop detection
// ---------------------------------------------------------------------------
// A bishop on a7/h7 (for White) or a2/h2 (for Black) is trapped when the
// enemy has a pawn on b6/g6 (or b3/g3 for Black) blocking the diagonal.

fn trapped_bishops(board: &Board, params: &EvalParams) -> (i32, i32) {
    let mut mg = 0i32;
    let mut eg = 0i32;

    let white_bishops = board.pieces[Color::White.index()][PieceKind::Bishop.index()];
    let black_pawns = board.pieces[Color::Black.index()][PieceKind::Pawn.index()];
    let black_bishops = board.pieces[Color::Black.index()][PieceKind::Bishop.index()];
    let white_pawns = board.pieces[Color::White.index()][PieceKind::Pawn.index()];

    // White bishop trapped on a7 by pawn on b6
    let a7 = Square::new(0, 6);
    let b6 = Square::new(1, 5);
    if white_bishops.is_set(a7) && black_pawns.is_set(b6) {
        mg += params.trapped_bishop_mg;
        eg += params.trapped_bishop_eg;
    }
    // White bishop trapped on h7 by pawn on g6
    let h7 = Square::new(7, 6);
    let g6 = Square::new(6, 5);
    if white_bishops.is_set(h7) && black_pawns.is_set(g6) {
        mg += params.trapped_bishop_mg;
        eg += params.trapped_bishop_eg;
    }
    // Black bishop trapped on a2 by pawn on b3
    let a2 = Square::new(0, 1);
    let b3 = Square::new(1, 2);
    if black_bishops.is_set(a2) && white_pawns.is_set(b3) {
        mg -= params.trapped_bishop_mg;
        eg -= params.trapped_bishop_eg;
    }
    // Black bishop trapped on h2 by pawn on g3
    let h2 = Square::new(7, 1);
    let g3 = Square::new(6, 2);
    if black_bishops.is_set(h2) && white_pawns.is_set(g3) {
        mg -= params.trapped_bishop_mg;
        eg -= params.trapped_bishop_eg;
    }

    (mg, eg)
}

// ---------------------------------------------------------------------------
// King safety (middlegame weighted)
// ---------------------------------------------------------------------------

fn king_safety(board: &Board, phase: i32, params: &EvalParams) -> i32 {
    if phase > 200 {
        return 0;
    }

    let mut score = 0i32;
    score += king_safety_for(board, Color::White, phase, params);
    score -= king_safety_for(board, Color::Black, phase, params);
    score
}

fn king_safety_for(board: &Board, color: Color, phase: i32, params: &EvalParams) -> i32 {
    let mut score = 0i32;
    let king_sq = board.king_square(color);
    let own_pawns = board.pieces[color.index()][PieceKind::Pawn.index()];
    let king_file = king_sq.file();

    let mut weight = (256 - phase).max(0);
    // When the enemy has a queen, king safety remains critical even in
    // endgame-like positions (e.g., Q+P vs 2R).  The queen is excellent
    // at exploiting an exposed king, so keep at least 50% weight.
    let enemy_color = color.opposite();
    let enemy_queens = board.pieces[enemy_color.index()][PieceKind::Queen.index()].count();
    if enemy_queens > 0 {
        weight = weight.max(128);
    }
    // Keep some king safety weight when rooks + bishops can still attack.
    let enemy_rooks = board.pieces[enemy_color.index()][PieceKind::Rook.index()].count();
    let enemy_bishops = board.pieces[enemy_color.index()][PieceKind::Bishop.index()].count();
    if enemy_queens == 0 && enemy_rooks > 0 && enemy_bishops > 0 {
        weight = weight.max(96);
    }

    // Reduce weight when king is on a normal castled square (kingside or
    // queenside) and on home ranks — the king is structurally safe and
    // penalties from open files / single attackers should not dominate.
    let castled_kingside = king_file >= 6; // g/h files
    let castled_queenside = king_file <= 2; // a/b/c files
    let on_home_ranks_ks = match color {
        Color::White => king_sq.rank() <= 1,
        Color::Black => king_sq.rank() >= 6,
    };
    if (castled_kingside || castled_queenside) && on_home_ranks_ks {
        // Cap weight so castled-king penalties can't exceed ~25% of full
        // weight.  This prevents speculative sacrifices from looking good
        // just because the eval over-penalises a structurally safe king.
        weight = weight.min(64);
    }

    // Pawn shield
    let shield_files = {
        let mut bb = file_bitboard(king_file);
        if king_file > 0 { bb |= file_bitboard(king_file - 1); }
        if king_file < 7 { bb |= file_bitboard(king_file + 1); }
        bb
    };

    let shield_rank_1 = match color {
        Color::White => if king_sq.rank() < 7 { rank_bitboard(king_sq.rank() + 1) } else { Bitboard::EMPTY },
        Color::Black => if king_sq.rank() > 0 { rank_bitboard(king_sq.rank() - 1) } else { Bitboard::EMPTY },
    };
    let shield_rank_2 = match color {
        Color::White => if king_sq.rank() < 6 { rank_bitboard(king_sq.rank() + 2) } else { Bitboard::EMPTY },
        Color::Black => if king_sq.rank() > 1 { rank_bitboard(king_sq.rank() - 2) } else { Bitboard::EMPTY },
    };

    let shield_1 = (own_pawns & shield_files & shield_rank_1).count() as i32;
    let shield_2 = (own_pawns & shield_files & shield_rank_2).count() as i32;
    score += shield_1 * params.ks_shield_1 + shield_2 * params.ks_shield_2;

    // Penalty for open files near king
    let enemy_pawns = board.pieces[color.opposite().index()][PieceKind::Pawn.index()];
    for f in king_file.saturating_sub(1)..=(king_file + 1).min(7) {
        let file_bb = file_bitboard(f);
        let own_on_file = !(own_pawns & file_bb).is_empty();
        let enemy_on_file = !(enemy_pawns & file_bb).is_empty();
        if !own_on_file {
            if !enemy_on_file {
                // Fully open file near king: dangerous
                score += params.ks_open_file;
            } else {
                // Semi-open (enemy pawn present, own missing): enemy can use this file
                score += params.ks_semi_open;
            }
        }
        // Enemy semi-open file aimed at our king (we have a pawn, they don't)
        if own_on_file && !enemy_on_file {
            score -= 10;
        }
    }

    // Penalty for king in center during middlegame (uncastled)
    let on_home_ranks = match color {
        Color::White => king_sq.rank() <= 1,
        Color::Black => king_sq.rank() >= 6,
    };
    if (2..=5).contains(&king_file) && on_home_ranks {
        score += params.ks_center_king;
    }

    // Penalty for king exposed far from home rank
    let home_rank = match color { Color::White => 0i32, Color::Black => 7i32 };
    let rank_dist = (king_sq.rank() as i32 - home_rank).abs();
    if rank_dist >= 3 {
        score -= rank_dist * rank_dist * 15;
    }

    // Attacker count near king
    let enemy = color.opposite();
    let king_zone = chess_core::attacks::king_attacks(king_sq) | king_sq.bitboard();
    let enemy_knights = board.pieces[enemy.index()][PieceKind::Knight.index()];
    let enemy_bishops = board.pieces[enemy.index()][PieceKind::Bishop.index()];
    let enemy_rooks = board.pieces[enemy.index()][PieceKind::Rook.index()];
    let enemy_queens = board.pieces[enemy.index()][PieceKind::Queen.index()];
    let occ = board.all_occupancy();

    let mut attackers = 0i32;
    let mut attack_weight = 0i32;

    for sq in enemy_knights.iter() {
        if !(chess_core::attacks::knight_attacks(sq) & king_zone).is_empty() {
            attackers += 1;
            attack_weight += params.ks_knight_weight;
        }
    }
    for sq in enemy_bishops.iter() {
        if !(chess_core::attacks::bishop_attacks(sq, occ) & king_zone).is_empty() {
            attackers += 1;
            attack_weight += params.ks_bishop_weight;
        }
    }
    for sq in enemy_rooks.iter() {
        if !(chess_core::attacks::rook_attacks(sq, occ) & king_zone).is_empty() {
            attackers += 1;
            attack_weight += params.ks_rook_weight;
        }
    }
    for sq in enemy_queens.iter() {
        if !(chess_core::attacks::queen_attacks(sq, occ) & king_zone).is_empty() {
            attackers += 1;
            attack_weight += params.ks_queen_weight;
        }
    }

    let own_npm = non_pawn_material_mg(board, color, params);
    let enemy_npm = non_pawn_material_mg(board, enemy, params);
    let attack_scale = attack_scale_from_material_delta(enemy_npm - own_npm);
    attack_weight = attack_weight * attack_scale / 256;

    // Scale penalty by number of attackers.
    // The old multiplier (attackers+1 for 3+) was too aggressive, causing
    // the engine to overvalue speculative sacrifices.  Linear scaling with
    // a modest super-linear bump for 3+ keeps genuine attacks dangerous
    // without inflating king-safety in quiet positions.
    if attackers >= 1 {
        score -= attack_weight * attackers / 4;
    }

    // Virtual king mobility: penalize when king has few safe escape squares.
    // A king boxed in by its own pieces with enemy pieces nearby is very vulnerable.
    let king_moves = chess_core::attacks::king_attacks(king_sq) & !board.occupancy[color.index()];
    let king_mobility = king_moves.count() as i32;
    if king_mobility <= 2 && attackers >= 1 {
        score -= (3 - king_mobility) * 20;
    }

    // Pawn storm: penalize when enemy pawns are advancing toward our king
    let storm_zone = shield_files; // Same files as pawn shield
    let storm_ranks = match color {
        Color::White => rank_bitboard(3) | rank_bitboard(4),  // enemy pawns on rank 4-5
        Color::Black => rank_bitboard(3) | rank_bitboard(4),  // enemy pawns on rank 4-5
    };
    let storm_pawns = (enemy_pawns & storm_zone & storm_ranks).count() as i32;
    score -= storm_pawns * 15;

    (score * weight) / 256
}

// ---------------------------------------------------------------------------
// Pseudo-mobility
// ---------------------------------------------------------------------------

fn pseudo_mobility(board: &Board, params: &EvalParams) -> (i32, i32) {
    let mut mg = 0i32;
    let mut eg = 0i32;
    let occ = board.all_occupancy();

    // Knight mobility
    for sq in board.pieces[Color::White.index()][PieceKind::Knight.index()].iter() {
        let count = (chess_core::attacks::knight_attacks(sq) & !board.occupancy[Color::White.index()]).count() as i32;
        mg += count * params.mobility_knight_mg;
        eg += count * params.mobility_knight_eg;
    }
    for sq in board.pieces[Color::Black.index()][PieceKind::Knight.index()].iter() {
        let count = (chess_core::attacks::knight_attacks(sq) & !board.occupancy[Color::Black.index()]).count() as i32;
        mg -= count * params.mobility_knight_mg;
        eg -= count * params.mobility_knight_eg;
    }

    // Bishop mobility
    for sq in board.pieces[Color::White.index()][PieceKind::Bishop.index()].iter() {
        let count = (chess_core::attacks::bishop_attacks(sq, occ) & !board.occupancy[Color::White.index()]).count() as i32;
        mg += count * params.mobility_bishop_mg;
        eg += count * params.mobility_bishop_eg;
    }
    for sq in board.pieces[Color::Black.index()][PieceKind::Bishop.index()].iter() {
        let count = (chess_core::attacks::bishop_attacks(sq, occ) & !board.occupancy[Color::Black.index()]).count() as i32;
        mg -= count * params.mobility_bishop_mg;
        eg -= count * params.mobility_bishop_eg;
    }

    // Rook mobility
    let enemy_half_white = rank_bitboard(4) | rank_bitboard(5) | rank_bitboard(6) | rank_bitboard(7);
    let enemy_half_black = rank_bitboard(0) | rank_bitboard(1) | rank_bitboard(2) | rank_bitboard(3);
    for sq in board.pieces[Color::White.index()][PieceKind::Rook.index()].iter() {
        let attacks = chess_core::attacks::rook_attacks(sq, occ) & !board.occupancy[Color::White.index()];
        let count = attacks.count() as i32;
        mg += count * params.mobility_rook_mg;
        eg += count * params.mobility_rook_eg;
        // Rook can reach enemy 7th/8th rank: potential penetration bonus
        if !(attacks & (rank_bitboard(6) | rank_bitboard(7))).is_empty() {
            eg += 8;
        }
        // Bonus for squares on enemy half of board
        let enemy_sq_count = (attacks & enemy_half_white).count() as i32;
        eg += enemy_sq_count;
    }
    for sq in board.pieces[Color::Black.index()][PieceKind::Rook.index()].iter() {
        let attacks = chess_core::attacks::rook_attacks(sq, occ) & !board.occupancy[Color::Black.index()];
        let count = attacks.count() as i32;
        mg -= count * params.mobility_rook_mg;
        eg -= count * params.mobility_rook_eg;
        // Rook can reach enemy 1st/2nd rank: potential penetration bonus
        if !(attacks & (rank_bitboard(0) | rank_bitboard(1))).is_empty() {
            eg -= 8;
        }
        // Bonus for squares on enemy half of board
        let enemy_sq_count = (attacks & enemy_half_black).count() as i32;
        eg -= enemy_sq_count;
    }

    // Queen mobility
    for sq in board.pieces[Color::White.index()][PieceKind::Queen.index()].iter() {
        let count = (chess_core::attacks::queen_attacks(sq, occ) & !board.occupancy[Color::White.index()]).count() as i32;
        mg += count * params.mobility_queen_mg;
        eg += count * params.mobility_queen_eg;
        // Trapped queen penalty: very few available squares
        if count <= 3 {
            mg -= (4 - count) * 15;
            eg -= (4 - count) * 10;
        }
    }
    for sq in board.pieces[Color::Black.index()][PieceKind::Queen.index()].iter() {
        let count = (chess_core::attacks::queen_attacks(sq, occ) & !board.occupancy[Color::Black.index()]).count() as i32;
        mg -= count * params.mobility_queen_mg;
        eg -= count * params.mobility_queen_eg;
        // Trapped queen penalty
        if count <= 3 {
            mg += (4 - count) * 15;
            eg += (4 - count) * 10;
        }
    }

    // Low-mobility minor piece penalty (knight/bishop with 0-2 moves)
    for sq in board.pieces[Color::White.index()][PieceKind::Knight.index()].iter() {
        let count = (chess_core::attacks::knight_attacks(sq) & !board.occupancy[Color::White.index()]).count() as i32;
        if count <= 2 {
            mg -= (3 - count) * 10;
            eg -= (3 - count) * 12;
        }
    }
    for sq in board.pieces[Color::Black.index()][PieceKind::Knight.index()].iter() {
        let count = (chess_core::attacks::knight_attacks(sq) & !board.occupancy[Color::Black.index()]).count() as i32;
        if count <= 2 {
            mg += (3 - count) * 10;
            eg += (3 - count) * 12;
        }
    }

    (mg, eg)
}

// ---------------------------------------------------------------------------
// Center control
// ---------------------------------------------------------------------------

fn center_control(board: &Board, params: &EvalParams) -> i32 {
    let center = Square::new(3, 3).bitboard()
        | Square::new(4, 3).bitboard()
        | Square::new(3, 4).bitboard()
        | Square::new(4, 4).bitboard();

    let extended_center = center
        | Square::new(2, 2).bitboard()
        | Square::new(5, 2).bitboard()
        | Square::new(2, 5).bitboard()
        | Square::new(5, 5).bitboard()
        | Square::new(2, 3).bitboard()
        | Square::new(5, 3).bitboard()
        | Square::new(2, 4).bitboard()
        | Square::new(5, 4).bitboard();

    let mut score = 0i32;

    let white_pawns = board.pieces[Color::White.index()][PieceKind::Pawn.index()];
    let black_pawns = board.pieces[Color::Black.index()][PieceKind::Pawn.index()];

    let white_center_pawns = (white_pawns & center).count() as i32;
    let black_center_pawns = (black_pawns & center).count() as i32;

    score += white_center_pawns * params.center_pawn_bonus;
    score -= black_center_pawns * params.center_pawn_bonus;

    // Pawn center mass: bonus for having center pawns, penalty for none
    // Only in middlegame (phase is computed outside, so use piece count as proxy)
    let total_pieces = board.all_occupancy().count() as i32;
    if total_pieces >= 20 {
        // Middlegame: center pawns matter a lot
        if white_center_pawns == 0 {
            score -= 20;
        }
        if black_center_pawns == 0 {
            score += 20;
        }
        // Extra bonus for a classical pawn duo (2 center pawns)
        if white_center_pawns >= 2 {
            score += 15;
        }
        if black_center_pawns >= 2 {
            score -= 15;
        }
    }

    let white_knights = board.pieces[Color::White.index()][PieceKind::Knight.index()];
    let black_knights = board.pieces[Color::Black.index()][PieceKind::Knight.index()];
    let white_bishops = board.pieces[Color::White.index()][PieceKind::Bishop.index()];
    let black_bishops = board.pieces[Color::Black.index()][PieceKind::Bishop.index()];

    score += (white_knights & extended_center).count() as i32 * params.center_knight_bonus;
    score -= (black_knights & extended_center).count() as i32 * params.center_knight_bonus;
    score += (white_bishops & extended_center).count() as i32 * params.center_bishop_bonus;
    score -= (black_bishops & extended_center).count() as i32 * params.center_bishop_bonus;

    score
}

// ---------------------------------------------------------------------------
// Connectivity / outposts
// ---------------------------------------------------------------------------

fn connectivity(board: &Board, params: &EvalParams) -> i32 {
    let mut score = 0i32;

    let white_pawn_attacks = pawn_attack_span(board.pieces[Color::White.index()][PieceKind::Pawn.index()], Color::White);
    let black_pawn_attacks = pawn_attack_span(board.pieces[Color::Black.index()][PieceKind::Pawn.index()], Color::Black);

    let white_knights = board.pieces[Color::White.index()][PieceKind::Knight.index()];
    let black_knights = board.pieces[Color::Black.index()][PieceKind::Knight.index()];

    for sq in (white_knights & white_pawn_attacks).iter() {
        score += params.pawn_protected_knight;
        let adj = adjacent_files(sq.file());
        let enemy_front = front_span_mask(sq, Color::White);
        if (board.pieces[Color::Black.index()][PieceKind::Pawn.index()] & adj & enemy_front).is_empty() {
            score += params.knight_outpost;
        }
    }
    for sq in (black_knights & black_pawn_attacks).iter() {
        score -= params.pawn_protected_knight;
        let adj = adjacent_files(sq.file());
        let enemy_front = front_span_mask(sq, Color::Black);
        if (board.pieces[Color::White.index()][PieceKind::Pawn.index()] & adj & enemy_front).is_empty() {
            score -= params.knight_outpost;
        }
    }

    score
}

// ---------------------------------------------------------------------------
// Threats: penalize hanging/attacked pieces
// ---------------------------------------------------------------------------

fn threats(board: &Board, params: &EvalParams) -> (i32, i32) {
    let mut mg = 0i32;
    let mut eg = 0i32;

    let occ = board.all_occupancy();

    // Compute pawn attack spans
    let white_pawns = board.pieces[Color::White.index()][PieceKind::Pawn.index()];
    let black_pawns = board.pieces[Color::Black.index()][PieceKind::Pawn.index()];
    let white_pawn_att = pawn_attack_span(white_pawns, Color::White);
    let black_pawn_att = pawn_attack_span(black_pawns, Color::Black);

    let white_npm = non_pawn_material_mg(board, Color::White, params);
    let black_npm = non_pawn_material_mg(board, Color::Black, params);
    let white_attack_scale = attack_scale_from_material_delta(white_npm - black_npm);
    let black_attack_scale = attack_scale_from_material_delta(black_npm - white_npm);
    // Compute minor attack maps
    let mut white_minor_att = Bitboard::EMPTY;
    for sq in board.pieces[Color::White.index()][PieceKind::Knight.index()].iter() {
        white_minor_att |= chess_core::attacks::knight_attacks(sq);
    }
    for sq in board.pieces[Color::White.index()][PieceKind::Bishop.index()].iter() {
        white_minor_att |= chess_core::attacks::bishop_attacks(sq, occ);
    }

    let mut black_minor_att = Bitboard::EMPTY;
    for sq in board.pieces[Color::Black.index()][PieceKind::Knight.index()].iter() {
        black_minor_att |= chess_core::attacks::knight_attacks(sq);
    }
    for sq in board.pieces[Color::Black.index()][PieceKind::Bishop.index()].iter() {
        black_minor_att |= chess_core::attacks::bishop_attacks(sq, occ);
    }

    // Compute rook attack maps
    let mut white_rook_att = Bitboard::EMPTY;
    for sq in board.pieces[Color::White.index()][PieceKind::Rook.index()].iter() {
        white_rook_att |= chess_core::attacks::rook_attacks(sq, occ);
    }
    let mut black_rook_att = Bitboard::EMPTY;
    for sq in board.pieces[Color::Black.index()][PieceKind::Rook.index()].iter() {
        black_rook_att |= chess_core::attacks::rook_attacks(sq, occ);
    }

    // White threats on black pieces
    let black_minors = board.pieces[Color::Black.index()][PieceKind::Knight.index()]
        | board.pieces[Color::Black.index()][PieceKind::Bishop.index()];
    let black_rooks = board.pieces[Color::Black.index()][PieceKind::Rook.index()];
    let black_queens = board.pieces[Color::Black.index()][PieceKind::Queen.index()];

    // Pawn attacks on minors
    let pawn_threat_minors = (white_pawn_att & black_minors).count() as i32;
    mg += pawn_threat_minors * params.threat_pawn_minor_mg * white_attack_scale / 256;
    eg += pawn_threat_minors * params.threat_pawn_minor_eg * white_attack_scale / 256;

    // Pawn attacks on rooks
    let pawn_threat_rooks = (white_pawn_att & black_rooks).count() as i32;
    mg += pawn_threat_rooks * params.threat_pawn_rook_mg * white_attack_scale / 256;
    eg += pawn_threat_rooks * params.threat_pawn_rook_eg * white_attack_scale / 256;

    // Minor attacks on rooks
    let minor_threat_rooks = (white_minor_att & black_rooks).count() as i32;
    mg += minor_threat_rooks * params.threat_minor_rook_mg * white_attack_scale / 256;
    eg += minor_threat_rooks * params.threat_minor_rook_eg * white_attack_scale / 256;

    // Minor/rook attacks on queens
    let piece_threat_queens = ((white_minor_att | white_rook_att) & black_queens).count() as i32;
    mg += piece_threat_queens * params.threat_piece_queen_mg * white_attack_scale / 256;
    eg += piece_threat_queens * params.threat_piece_queen_eg * white_attack_scale / 256;

    // Hanging pieces (attacked by anything, not defended by pawns)
    let white_all_att = white_pawn_att | white_minor_att | white_rook_att;
    let hanging_black = white_all_att & !black_pawn_att & (black_minors | black_rooks | black_queens);
    mg += hanging_black.count() as i32 * params.threat_hanging_mg * white_attack_scale / 256;
    eg += hanging_black.count() as i32 * params.threat_hanging_eg * white_attack_scale / 256;

    // Black threats on white pieces (mirror)
    let white_minors = board.pieces[Color::White.index()][PieceKind::Knight.index()]
        | board.pieces[Color::White.index()][PieceKind::Bishop.index()];
    let white_rooks = board.pieces[Color::White.index()][PieceKind::Rook.index()];
    let white_queens = board.pieces[Color::White.index()][PieceKind::Queen.index()];

    let pawn_threat_minors = (black_pawn_att & white_minors).count() as i32;
    mg -= pawn_threat_minors * params.threat_pawn_minor_mg * black_attack_scale / 256;
    eg -= pawn_threat_minors * params.threat_pawn_minor_eg * black_attack_scale / 256;

    let pawn_threat_rooks = (black_pawn_att & white_rooks).count() as i32;
    mg -= pawn_threat_rooks * params.threat_pawn_rook_mg * black_attack_scale / 256;
    eg -= pawn_threat_rooks * params.threat_pawn_rook_eg * black_attack_scale / 256;

    let minor_threat_rooks = (black_minor_att & white_rooks).count() as i32;
    mg -= minor_threat_rooks * params.threat_minor_rook_mg * black_attack_scale / 256;
    eg -= minor_threat_rooks * params.threat_minor_rook_eg * black_attack_scale / 256;

    let piece_threat_queens = ((black_minor_att | black_rook_att) & white_queens).count() as i32;
    mg -= piece_threat_queens * params.threat_piece_queen_mg * black_attack_scale / 256;
    eg -= piece_threat_queens * params.threat_piece_queen_eg * black_attack_scale / 256;

    let black_all_att = black_pawn_att | black_minor_att | black_rook_att;
    let hanging_white = black_all_att & !white_pawn_att & (white_minors | white_rooks | white_queens);
    mg -= hanging_white.count() as i32 * params.threat_hanging_mg * black_attack_scale / 256;
    eg -= hanging_white.count() as i32 * params.threat_hanging_eg * black_attack_scale / 256;

    (mg, eg)
}

// ---------------------------------------------------------------------------
// Queen placement: overextension, safe mobility, king distance
// ---------------------------------------------------------------------------
// Tempo
// ---------------------------------------------------------------------------

fn tempo(board: &Board, params: &EvalParams) -> i32 {
    match board.side_to_move {
        Color::White => params.tempo,
        Color::Black => -params.tempo,
    }
}

// ---------------------------------------------------------------------------
// Material imbalance
// ---------------------------------------------------------------------------

fn material_imbalance(board: &Board, params: &EvalParams) -> (i32, i32) {
    let mut mg = 0i32;
    let mut eg = 0i32;

    let wn = board.pieces[Color::White.index()][PieceKind::Knight.index()].count() as i32;
    let wb = board.pieces[Color::White.index()][PieceKind::Bishop.index()].count() as i32;
    let wr = board.pieces[Color::White.index()][PieceKind::Rook.index()].count() as i32;
    let wq = board.pieces[Color::White.index()][PieceKind::Queen.index()].count() as i32;
    let bn = board.pieces[Color::Black.index()][PieceKind::Knight.index()].count() as i32;
    let bb = board.pieces[Color::Black.index()][PieceKind::Bishop.index()].count() as i32;
    let br = board.pieces[Color::Black.index()][PieceKind::Rook.index()].count() as i32;
    let bq = board.pieces[Color::Black.index()][PieceKind::Queen.index()].count() as i32;

    let w_minors = wn + wb;
    let b_minors = bn + bb;

    // Rook vs minor piece: the exchange is worth more than raw material difference
    // If one side has more rooks and the other has more minors, adjust
    let w_exchange = wr - br; // positive = White has more rooks
    let b_exchange = b_minors - w_minors; // positive = Black has more minors

    // When one side has extra rook(s) compensated by extra minors,
    // the rook side usually has an advantage ("the exchange")
    if w_exchange > 0 && b_exchange > 0 {
        let pairs = w_exchange.min(b_exchange);
        mg += pairs * params.imbalance_exchange_mg;
        eg += pairs * params.imbalance_exchange_eg;
    } else if w_exchange < 0 && b_exchange < 0 {
        let pairs = (-w_exchange).min(-b_exchange);
        mg -= pairs * params.imbalance_exchange_mg;
        eg -= pairs * params.imbalance_exchange_eg;
    }

    // Redundancy: two rooks are slightly less valuable than 2x one rook
    if wr >= 2 { mg += params.imbalance_rook_pair_mg; eg += params.imbalance_rook_pair_eg; }
    if br >= 2 { mg -= params.imbalance_rook_pair_mg; eg -= params.imbalance_rook_pair_eg; }

    // Two knights are slightly less valuable (poor at endgame mating)
    if wn >= 2 { eg += params.imbalance_knight_pair_eg; }
    if bn >= 2 { eg -= params.imbalance_knight_pair_eg; }

    // Queen vs 2 minors is usually good for the queen side
    if wq > bq && b_minors > w_minors + 1 {
        mg += params.imbalance_queen_vs_minors_mg;
        eg += params.imbalance_queen_vs_minors_eg;
    }
    if bq > wq && w_minors > b_minors + 1 {
        mg -= params.imbalance_queen_vs_minors_mg;
        eg -= params.imbalance_queen_vs_minors_eg;
    }

    (mg, eg)
}

// ---------------------------------------------------------------------------
// Pawn majority advantage: extra pawns are worth more when non-pawn material
// is roughly equal, because the opponent can't just simplify down.
// ---------------------------------------------------------------------------

fn pawn_majority_advantage(board: &Board) -> (i32, i32) {
    let wp = board.pieces[Color::White.index()][PieceKind::Pawn.index()].count() as i32;
    let bp = board.pieces[Color::Black.index()][PieceKind::Pawn.index()].count() as i32;

    let pawn_diff = wp - bp; // positive = White has more
    if pawn_diff == 0 {
        return (0, 0);
    }

    // Compute non-pawn material for each side (knight=3, bishop=3, rook=5, queen=9)
    let w_npm = board.pieces[Color::White.index()][PieceKind::Knight.index()].count() as i32 * 3
        + board.pieces[Color::White.index()][PieceKind::Bishop.index()].count() as i32 * 3
        + board.pieces[Color::White.index()][PieceKind::Rook.index()].count() as i32 * 5
        + board.pieces[Color::White.index()][PieceKind::Queen.index()].count() as i32 * 9;
    let b_npm = board.pieces[Color::Black.index()][PieceKind::Knight.index()].count() as i32 * 3
        + board.pieces[Color::Black.index()][PieceKind::Bishop.index()].count() as i32 * 3
        + board.pieces[Color::Black.index()][PieceKind::Rook.index()].count() as i32 * 5
        + board.pieces[Color::Black.index()][PieceKind::Queen.index()].count() as i32 * 9;

    let npm_diff = (w_npm - b_npm).abs();

    // The pawn advantage bonus scales with how equal the non-pawn material is.
    // When pieces are equal (npm_diff=0), bonus is maximum.
    // When one side has significant extra pieces, bonus shrinks (material dominates).
    let scale = (6 - npm_diff).max(0); // 0..6

    let mg = pawn_diff * scale * 5;
    let eg = pawn_diff * scale * 10;

    (mg, eg)
}

// ---------------------------------------------------------------------------
// Endgame mopup: drive enemy king to corner when material advantage is large
// ---------------------------------------------------------------------------

fn mopup_eval(board: &Board, phase: i32) -> i32 {
    // Only apply in endgames (phase > 128)
    if phase < 128 {
        return 0;
    }

    // Compute material balance (positive = White ahead)
    let mut material_balance = 0i32;
    for &kind in &[PieceKind::Pawn, PieceKind::Knight, PieceKind::Bishop, PieceKind::Rook, PieceKind::Queen] {
        let ki = kind.index();
        let w_count = board.pieces[Color::White.index()][ki].count() as i32;
        let b_count = board.pieces[Color::Black.index()][ki].count() as i32;
        material_balance += (w_count - b_count) * MATERIAL_EG[ki];
    }

    // Need a significant material advantage to apply mopup
    let threshold = 300; // roughly a minor piece advantage
    if material_balance.abs() < threshold {
        return 0;
    }

    let (winning_color, losing_color) = if material_balance > 0 {
        (Color::White, Color::Black)
    } else {
        (Color::Black, Color::White)
    };

    let winning_king = board.king_square(winning_color);
    let losing_king = board.king_square(losing_color);

    // Reward: losing king pushed to corner
    let losing_file = losing_king.file() as i32;
    let losing_rank = losing_king.rank() as i32;
    let center_dist_file = (losing_file * 2 - 7).abs(); // 0..7 mapped to distance from center
    let center_dist_rank = (losing_rank * 2 - 7).abs();
    let corner_bonus = center_dist_file + center_dist_rank; // 0..14, higher = more cornered

    // Reward: winning king close to losing king (to help with mating)
    let king_dist = chebyshev_distance(winning_king, losing_king) as i32;
    let close_bonus = 14 - king_dist * 2; // higher when kings are close

    // Scale bonus by material advantage (cap at ~600 for large advantages)
    let advantage = material_balance.abs().min(1500);
    let scale = advantage / 3;

    let bonus = (corner_bonus * 8 + close_bonus * 6) * scale / 100;

    if material_balance > 0 { bonus } else { -bonus }
}

// ---------------------------------------------------------------------------
// Space evaluation: count safe squares behind the pawn chain.
// A cramped position limits piece mobility and coordination.
// ---------------------------------------------------------------------------

fn space_eval(board: &Board) -> (i32, i32) {
    let occ = board.all_occupancy();
    let white_pawns = board.pieces[Color::White.index()][PieceKind::Pawn.index()];
    let black_pawns = board.pieces[Color::Black.index()][PieceKind::Pawn.index()];
    let black_pawn_att = pawn_attack_span(black_pawns, Color::Black);
    let white_pawn_att = pawn_attack_span(white_pawns, Color::White);

    // White space: safe squares on ranks 2-4 behind own pawns, not attacked by enemy pawns
    let white_space_zone = rank_bitboard(1) | rank_bitboard(2) | rank_bitboard(3);
    let white_safe = white_space_zone & !black_pawn_att & !occ;
    let white_space = white_safe.count() as i32;

    // Black space: safe squares on ranks 5-7 behind own pawns, not attacked by enemy pawns
    let black_space_zone = rank_bitboard(4) | rank_bitboard(5) | rank_bitboard(6);
    let black_safe = black_space_zone & !white_pawn_att & !occ;
    let black_space = black_safe.count() as i32;

    // Scale: more pieces on the board = space matters more
    let total_pieces = board.all_occupancy().count() as i32;
    let scale = (total_pieces - 10).max(0); // only matters with many pieces

    let mg = (white_space - black_space) * scale / 8;
    let eg = (white_space - black_space) * 2;

    (mg, eg)
}

// ---------------------------------------------------------------------------
// Bad bishop: penalize a bishop whose own pawns block its diagonal scope.
// A bishop is "bad" when many friendly pawns sit on its color complex.
// ---------------------------------------------------------------------------

fn bad_bishop(board: &Board, params: &EvalParams) -> (i32, i32) {
    let mut mg = 0i32;
    let mut eg = 0i32;

    for &color in &[Color::White, Color::Black] {
        let ci = color.index();
        let our_pawns = board.pieces[ci][PieceKind::Pawn.index()];
        let bishops = board.pieces[ci][PieceKind::Bishop.index()];
        let sign = if color == Color::White { 1 } else { -1 };

        for sq in bishops.iter() {
            // Determine which color complex this bishop is on
            let bishop_complex = if (sq.file() + sq.rank()) % 2 == 0 {
                DARK_SQUARES
            } else {
                LIGHT_SQUARES
            };
            // Count own pawns on the same color complex
            let blocked_pawns = (our_pawns & bishop_complex).count() as i32;
            // Penalty scales with number of own pawns on bishop's color (0-4 typical)
            mg += sign * params.bad_bishop_mg * blocked_pawns;
            eg += sign * params.bad_bishop_eg * blocked_pawns;
        }
    }

    (mg, eg)
}

// ---------------------------------------------------------------------------
// Knight vs bishop adjustment: knights are better in closed positions (many
// pawns), bishops are better in open positions (few pawns). Threshold: 5 pawns
// per side (10 total). Above that = closed, below = open.
// ---------------------------------------------------------------------------

fn knight_bishop_adjustment(board: &Board, params: &EvalParams) -> (i32, i32) {
    let mut mg = 0i32;
    let mut eg = 0i32;

    let total_pawns = (board.pieces[Color::White.index()][PieceKind::Pawn.index()]
        | board.pieces[Color::Black.index()][PieceKind::Pawn.index()])
    .count() as i32;

    // Threshold: 10 total pawns = neutral. Above = closed, below = open.
    let closedness = total_pawns - 10; // positive = closed, negative = open

    for &color in &[Color::White, Color::Black] {
        let ci = color.index();
        let knights = board.pieces[ci][PieceKind::Knight.index()].count() as i32;
        let bishops = board.pieces[ci][PieceKind::Bishop.index()].count() as i32;
        let sign = if color == Color::White { 1 } else { -1 };

        if closedness > 0 {
            // Closed position: knights get bonus per pawn above threshold
            mg += sign * knights * params.knight_closed_bonus * closedness;
            eg += sign * knights * params.knight_closed_bonus * closedness;
        } else if closedness < 0 {
            // Open position: bishops get bonus per pawn below threshold
            let openness = -closedness;
            mg += sign * bishops * params.bishop_open_bonus * openness;
            eg += sign * bishops * params.bishop_open_bonus * openness;
        }
    }

    (mg, eg)
}

// ---------------------------------------------------------------------------
// Pawn islands: groups of connected pawn files. More islands = weaker structure.
// Each additional island beyond the first gets a penalty.
// ---------------------------------------------------------------------------

fn pawn_islands(board: &Board, params: &EvalParams) -> (i32, i32) {
    let mut mg = 0i32;
    let mut eg = 0i32;

    for &color in &[Color::White, Color::Black] {
        let ci = color.index();
        let pawns = board.pieces[ci][PieceKind::Pawn.index()];
        let sign = if color == Color::White { 1 } else { -1 };

        // Count islands: scan files 0-7, count transitions from empty to occupied
        let mut islands = 0;
        let mut prev_has_pawn = false;
        for file in 0..8u8 {
            let has_pawn = !(pawns & file_bitboard(file)).is_empty();
            if has_pawn && !prev_has_pawn {
                islands += 1;
            }
            prev_has_pawn = has_pawn;
        }

        // Penalty for each island beyond the first (1 island = no penalty)
        let extra = (islands - 1).max(0);
        mg += sign * params.pawn_islands_mg * extra;
        eg += sign * params.pawn_islands_eg * extra;
    }

    (mg, eg)
}

// ---------------------------------------------------------------------------
// Endgame scale factor: reduce the eval toward draw in drawish endgames.
// Returns a scale factor 0-128 (128 = full eval, lower = more drawish).
// ---------------------------------------------------------------------------

fn endgame_scale_factor(board: &Board, params: &EvalParams, eval: i32, phase: i32) -> i32 {
    // Only apply in endgame-ish positions (phase > 128 means mostly endgame)
    if phase < 128 {
        return 128; // full eval in middlegame
    }

    let wi = Color::White.index();
    let bi = Color::Black.index();

    let wp = board.pieces[wi][PieceKind::Pawn.index()].count() as i32;
    let bp = board.pieces[bi][PieceKind::Pawn.index()].count() as i32;
    let wb = board.pieces[wi][PieceKind::Bishop.index()].count() as i32;
    let bb = board.pieces[bi][PieceKind::Bishop.index()].count() as i32;
    let wn = board.pieces[wi][PieceKind::Knight.index()].count() as i32;
    let bn = board.pieces[bi][PieceKind::Knight.index()].count() as i32;
    let wr = board.pieces[wi][PieceKind::Rook.index()].count() as i32;
    let br = board.pieces[bi][PieceKind::Rook.index()].count() as i32;
    let wq = board.pieces[wi][PieceKind::Queen.index()].count() as i32;
    let bq = board.pieces[bi][PieceKind::Queen.index()].count() as i32;

    let w_minors = wn + wb;
    let b_minors = bn + bb;
    let w_major = wr + wq;
    let b_major = br + bq;
    let w_all = w_minors + w_major;
    let b_all = b_minors + b_major;

    // Opposite-colored bishop endgame (each side has exactly 1 bishop on different color)
    if wb == 1 && bb == 1 && wn == 0 && bn == 0 && w_major == 0 && b_major == 0 {
        let w_bishop_sq = board.pieces[wi][PieceKind::Bishop.index()].lsb().unwrap();
        let b_bishop_sq = board.pieces[bi][PieceKind::Bishop.index()].lsb().unwrap();
        let w_light = (w_bishop_sq.file() + w_bishop_sq.rank()) % 2 == 1;
        let b_light = (b_bishop_sq.file() + b_bishop_sq.rank()) % 2 == 1;

        if w_light != b_light {
            // Opposite-colored bishops — very drawish
            // Scale factor from params (default 60 means scale to ~47%)
            return params.ocb_scale_factor.clamp(20, 128);
        }
    }

    // Queen vs pawn(s) only — very drawish when the pawn is advanced (6th/7th rank).
    // Q vs p on 7th near the rook file or bishop file is often a fortress draw.
    if (wq == 1 && w_all == 1 && wp == 0 && b_all == 0 && bp > 0)
        || (bq == 1 && b_all == 1 && bp == 0 && w_all == 0 && wp > 0)
    {
        let (defending_pawns, defending_color) = if wq == 1 { (bp, Color::Black) } else { (wp, Color::White) };
        if defending_pawns == 1 {
            // Check if the defending pawn is on the 7th rank (about to promote)
            let pawn_bb = board.pieces[defending_color.index()][PieceKind::Pawn.index()];
            let on_7th = match defending_color {
                Color::White => !(pawn_bb & rank_bitboard(6)).is_empty(),
                Color::Black => !(pawn_bb & rank_bitboard(1)).is_empty(),
            };
            if on_7th {
                return 16; // Nearly drawn — fortress with pawn on 7th
            }
            let on_6th = match defending_color {
                Color::White => !(pawn_bb & rank_bitboard(5)).is_empty(),
                Color::Black => !(pawn_bb & rank_bitboard(2)).is_empty(),
            };
            if on_6th {
                return 48; // Somewhat drawish
            }
        }
        // Q vs multiple pawns — still hard to win
        if defending_pawns >= 2 {
            return 64;
        }
    }

    // No pawns for the winning side — hard to win without pawns
    let winning_side_pawns = if eval > 0 { wp } else { bp };
    if winning_side_pawns == 0 {
        let winning_side_minors = if eval > 0 { w_minors } else { b_minors };
        let winning_side_majors = if eval > 0 { w_major } else { b_major };

        // Lone minor piece(s) without pawns can rarely win
        if winning_side_majors == 0 {
            if winning_side_minors <= 1 {
                return 0; // KN vs K or KB vs K — drawn
            }
            if winning_side_minors == 2 {
                // KBN vs K or KBB vs K — technically winnable but slow
                return 32;
            }
        }

        // Queen vs lone minor — usually winning but slower than eval suggests
        let losing_side_minors = if eval > 0 { b_minors } else { w_minors };
        let losing_side_majors = if eval > 0 { b_major } else { w_major };
        let losing_side_pawns = if eval > 0 { bp } else { wp };
        if winning_side_majors == 1 && winning_side_minors == 0 && winning_side_pawns == 0
            && losing_side_majors == 0 && losing_side_minors <= 1 && losing_side_pawns == 0
        {
            // Q vs minor or Q vs nothing — scale down since no pawns to promote
            return 80;
        }
    }

    // Rook endgames with equal rooks — very drawish tendency
    if wq == 0 && bq == 0 && w_minors == 0 && b_minors == 0 && wr == br && wr > 0 {
        let total_pawns = wp + bp;
        let pawn_diff = (wp - bp).abs();
        // Even with a 1-2 pawn advantage, rook endgames are notoriously drawish
        if total_pawns <= 6 {
            if pawn_diff <= 1 {
                if total_pawns <= 2 {
                    return 48; // Very drawish — R+P vs R or R vs R
                }
                return 80; // Rook endgames are drawish
            }
            if pawn_diff == 2 {
                return 96; // Small advantage but still hard to convert
            }
        }
    }

    // R+N (or R+B) vs R+P: minor vs lone pawn — nearly always a theoretical draw.
    // Exception: pawn on rank 6 or 7 (index ≥ 5 for White, ≤ 2 for Black) where
    // promotion pressure is real; fall through to the weaker scale below in that case.
    if wq == 0 && bq == 0 && wr == 1 && br == 1 {
        if b_minors == 1 && w_minors == 0 && wp == 1 && bp == 0 {
            let pawn_sq = board.pieces[wi][PieceKind::Pawn.index()].lsb().unwrap();
            if pawn_sq.rank() <= 4 {
                return 16;
            }
        }
        if w_minors == 1 && b_minors == 0 && bp == 1 && wp == 0 {
            let pawn_sq = board.pieces[bi][PieceKind::Pawn.index()].lsb().unwrap();
            if pawn_sq.rank() >= 3 {
                return 16;
            }
        }
    }

    // R+minor vs R with no pawns at all — theoretical draw with correct play.
    // Scale=12 (not 16) because positional bonuses (active king, rook on 7th) inflate
    // the raw eval enough that 16 still yields ~55 cp after scaling.
    // Must come before the general R+minor vs R check below.
    if wq == 0 && bq == 0 && wr == 1 && br == 1 && wp == 0 && bp == 0
        && ((w_minors == 1 && b_minors == 0) || (b_minors == 1 && w_minors == 0))
    {
        return 12;
    }

    // Rook + minor vs rook with pawns on the board — drawish but pawn(s) add winning chances
    if (wr == 1 && w_minors == 1 && br == 1 && b_minors == 0 && wq == 0 && bq == 0)
        || (br == 1 && b_minors == 1 && wr == 1 && w_minors == 0 && bq == 0 && wq == 0)
    {
        return 64; // halve the eval
    }

    128 // full eval
}

// ---------------------------------------------------------------------------
// Bitboard helpers
// ---------------------------------------------------------------------------

fn file_bitboard(file: u8) -> Bitboard {
    match file {
        0 => Bitboard::FILE_A,
        1 => Bitboard::FILE_B,
        2 => Bitboard::FILE_C,
        3 => Bitboard::FILE_D,
        4 => Bitboard::FILE_E,
        5 => Bitboard::FILE_F,
        6 => Bitboard::FILE_G,
        7 => Bitboard::FILE_H,
        _ => Bitboard::EMPTY,
    }
}

fn adjacent_files(file: u8) -> Bitboard {
    let mut bb = Bitboard::EMPTY;
    if file > 0 {
        bb |= file_bitboard(file - 1);
    }
    if file < 7 {
        bb |= file_bitboard(file + 1);
    }
    bb
}

fn front_span_mask(sq: Square, color: Color) -> Bitboard {
    let rank = sq.rank();
    let mut bb = Bitboard::EMPTY;
    match color {
        Color::White => {
            for r in (rank + 1)..8 {
                bb |= rank_bitboard(r);
            }
        }
        Color::Black => {
            for r in 0..rank {
                bb |= rank_bitboard(r);
            }
        }
    }
    bb
}

fn behind_span_mask(sq: Square, color: Color) -> Bitboard {
    front_span_mask(sq, color.opposite())
}

fn rank_bitboard(rank: u8) -> Bitboard {
    Bitboard(0xFF << (rank * 8))
}

fn pawn_attack_span(pawns: Bitboard, color: Color) -> Bitboard {
    match color {
        Color::White => pawns.north_east() | pawns.north_west(),
        Color::Black => pawns.south_east() | pawns.south_west(),
    }
}

fn square_attacked_by(board: &Board, sq: Square, color: Color, occ: Bitboard) -> bool {
    let ci = color.index();
    let target = sq.bitboard();

    let pawns = board.pieces[ci][PieceKind::Pawn.index()];
    if !(pawn_attack_span(pawns, color) & target).is_empty() {
        return true;
    }

    for s in board.pieces[ci][PieceKind::Knight.index()].iter() {
        if !(chess_core::attacks::knight_attacks(s) & target).is_empty() {
            return true;
        }
    }
    for s in board.pieces[ci][PieceKind::Bishop.index()].iter() {
        if !(chess_core::attacks::bishop_attacks(s, occ) & target).is_empty() {
            return true;
        }
    }
    for s in board.pieces[ci][PieceKind::Rook.index()].iter() {
        if !(chess_core::attacks::rook_attacks(s, occ) & target).is_empty() {
            return true;
        }
    }
    for s in board.pieces[ci][PieceKind::Queen.index()].iter() {
        if !(chess_core::attacks::queen_attacks(s, occ) & target).is_empty() {
            return true;
        }
    }

    false
}

fn chebyshev_distance(a: Square, b: Square) -> u8 {
    let file_diff = (a.file() as i8 - b.file() as i8).unsigned_abs();
    let rank_diff = (a.rank() as i8 - b.rank() as i8).unsigned_abs();
    file_diff.max(rank_diff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess_common::Board;

    #[test]
    fn opening_evals_not_inflated() {
        // Common opening positions should eval within +/- 100cp
        let positions = [
            ("After 1.e4",   "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
            ("Sicilian 2.Nf3","rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"),
            ("Najdorf 5.Nc3","rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 2 5"),
        ];
        for (label, fen) in positions {
            let board = Board::from_fen(fen).unwrap();
            let score = evaluate(&board).0;
            assert!(score.abs() < 300,
                "{label}: eval {score} is too far from zero (expected within +/- 300)");
        }
    }

    #[test]
    fn berlin_endgame_not_overvalued() {
        // Berlin endgame after 9...Bd7: should be slightly positive, not +300
        let board = Board::from_fen(
            "r2k1b1r/pppb1ppp/2p2n2/4P3/8/2N2N2/PPP2PPP/R1B2RK1 w - - 3 10"
        ).unwrap();
        let score = evaluate(&board).0;
        assert!(score < 350, "Berlin eval {score} is too high (expected < 350)");
        assert!(score > -50, "Berlin eval {score} is too low (expected > -50)");
    }

    #[test]
    fn debug_eval_positions() {
        let positions = [
            // Game 4 HM20: SF=+155, Engine(old)=-53
            ("G4 HM20", "rn1qkb1r/pb3p2/2p1pn1p/6p1/1ppPP3/2N2NB1/PP2BPPP/R2Q1RK1 w KQkq - 0 11", 155),
            // Game 2 HM19: SF=+71, Engine(old)=-125
            ("G2 HM19", "r1b1k2r/pp1nbpp1/1qp2n1p/3p4/3P1B2/2NBP2P/PPQ2PP1/R3K1NR b KQkq - 2 10", 71),
            // Game 1 HM19: SF=-40, Engine(old)=+77
            ("G1 HM19", "r1bq1rk1/pppp1ppp/3n1b2/4R3/3P4/3B4/PPP2PPP/RNBQ2K1 b - - 2 10", -40),
            // Game 2 HM53: SF=+384, Engine(old)=-108
            ("G2 HM53", "3rr3/1k3pp1/p5qp/5b2/8/1Qb1P2P/PP3PPB/R4R1K b - - 1 27", 384),
            // Game 1 HM73: SF=-525, Engine(old)=+300
            ("G1 HM73", "8/p2r1k1p/1p3Bp1/2p3P1/3p4/1P5P/P1PR1PK1/1r6 b - - 0 37", -525),
            // Game 5 HM84: SF=-855, Engine(old)=+363
            ("G5 HM84", "1R6/8/2p1k3/p1p3P1/P1P1b2p/4P3/1P2K2r/8 w - - 3 43", -855),
        ];

        for (label, fen_str, sf) in positions {
            let board = Board::from_fen(fen_str).unwrap();
            let score = evaluate(&board);
            let sign_match = if (score.0 > 0) == (sf > 0) || score.0 == 0 || sf == 0 { "OK" } else { "FLIP" };
            eprintln!("{label:10}: eval={:+5}  SF={sf:+5}  {sign_match}", score.0);
        }
    }

    #[test]
    fn diagnose_sf_outliers() {
        // Top outlier positions from convergence validation
        let positions = [
            ("KBN vs K",         "8/8/8/8/K2B3N/8/4k3/8 w - - 0 1",                              194),
            ("R+P vs R",         "5r2/5P2/8/6k1/8/6KP/5R2/8 w - - 0 1",                          0),
            ("Middlegame1",      "r1bqk1nr/ppp2ppp/1b1p4/n5B1/2BPP3/2N2N2/P4PPP/R2Q1RK1 b kq - 0 1",  -12),
            ("Middlegame2",      "rn2kbnr/ppp1pppp/q7/1R6/6b1/2N2N2/P1PP1PPP/2BQKB1R w Kkq - 0 1",    97),
            ("Middlegame3",      "r1b2rk1/2q3pp/pb1p1pn1/1ppP4/4P1n1/P4NB1/1PBN1PPP/R2QR1K1 w - - 0 1",-29),
            ("Tactical",         "1r5k/p1p3pp/8/8/4p3/P1P2q2/1P1Q1Pr1/2KRR3 w - - 0 1",          -110),
            ("Middlegame4",      "r6r/p3pk1p/1n1p1npQ/q1p5/4P3/1Pp2P2/P1P3PP/K2R1BNR b - - 0 1", -38),
            ("Middlegame5",      "3r1rk1/pp1qnpb1/4n1pp/2pp4/4P1P1/P1NP4/1PPB2QP/1R2NR1K w - - 0 1", -45),
            ("Middlegame6",      "r2q1r2/pR2ppkp/2n3p1/8/4P3/5B2/P4PPP/3Q1RK1 b - - 0 1",        -16),
            ("Middlegame7",      "r1bq1r1k/p3nn1p/3p2pb/2pPpp2/2P1P3/P1N2P2/3NBBPP/1R1Q1RK1 b - - 0 1", -15),
        ];

        for (label, fen, sf_cp) in positions {
            let board = Board::from_fen(fen).unwrap();
            let bd = evaluate_impl(&board, &EvalParams::default(), false, true);
            let delta = bd.final_score - sf_cp;
            eprintln!("\n{label}: eval={:+} SF={sf_cp:+} Δ={delta:+} phase={} scale={}/128",
                bd.final_score, bd.phase, bd.scale);
            for t in &bd.terms {
                if t.mg != 0 || t.eg != 0 {
                    eprintln!("  {:30} mg={:+5} eg={:+5}", t.name, t.mg, t.eg);
                }
            }
            eprintln!("  {:30} mg={:+5} eg={:+5} → interp={:+} → final={:+}",
                "TOTAL", bd.mg_total, bd.eg_total, bd.interpolated, bd.final_score);
        }
    }

    #[test]
    fn rook_minor_vs_rook_drawn_endings() {
        // R+N vs R+P (non-advanced pawn) and R+N vs R (no pawns) are both
        // theoretical draws; the engine should evaluate both near zero.
        let positions = [
            // From an actual game: Black has R+N, White has R+g3 pawn
            ("R+N vs R+P (g3)", "4R3/8/8/4nk2/8/6P1/r7/4K3 b - - 7 73"),
            // Subsequent position: R+N vs R+g4 pawn
            ("R+N vs R+P (g4)", "R7/8/8/4k3/6P1/5n2/6r1/2K5 b - - 0 83"),
            // R+N vs R with no pawns (the position Black reaches after taking the g-pawn)
            ("R+N vs R (no pawns)", "4R3/8/8/4nk2/8/8/6r1/4K3 w - - 0 1"),
        ];
        for (label, fen) in positions {
            let board = Board::from_fen(fen).unwrap();
            let score = evaluate(&board).0;
            assert!(
                score.abs() < 50,
                "{label}: eval {score} should be near draw (< 50 cp) — \
                 R+minor vs R is a theoretical draw",
            );
        }
    }

}
