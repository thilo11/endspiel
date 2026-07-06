//! Syzygy endgame tablebase probing via pyrrhic-rs.
//!
//! Provides WDL (Win/Draw/Loss) probing integrated into the alpha-beta search.
//! Positions with no castling rights and at most `TB_LARGEST` pieces are probed
//! at every node, giving exact game-theoretic values for endgame positions.

use chess_common::{Bitboard, Board, Color, Move, PieceKind, Score, Square};
use pyrrhic_rs::{DtzProbeValue, EngineAdapter, Piece, TableBases, WdlProbeResult};

type ProbeInputs = (u64, u64, u64, u64, u64, u64, u64, u64, u32, bool);

// ---------------------------------------------------------------------------
// Engine adapter
// ---------------------------------------------------------------------------

/// Adapter that plugs this engine's magic-bitboard attack tables into pyrrhic-rs.
#[derive(Clone)]
pub struct ChessAdapter;

impl EngineAdapter for ChessAdapter {
    fn pawn_attacks(color: pyrrhic_rs::Color, square: u64) -> u64 {
        if square >= 64 {
            return 0;
        }
        let sq = Square(square as u8);
        let c = if color == pyrrhic_rs::Color::White {
            Color::White
        } else {
            Color::Black
        };
        chess_core::attacks::pawn_attacks(sq, c).0
    }

    fn knight_attacks(square: u64) -> u64 {
        if square >= 64 {
            return 0;
        }
        chess_core::attacks::knight_attacks(Square(square as u8)).0
    }

    fn bishop_attacks(square: u64, occupied: u64) -> u64 {
        if square >= 64 {
            return 0;
        }
        chess_core::attacks::bishop_attacks(Square(square as u8), Bitboard(occupied)).0
    }

    fn rook_attacks(square: u64, occupied: u64) -> u64 {
        if square >= 64 {
            return 0;
        }
        chess_core::attacks::rook_attacks(Square(square as u8), Bitboard(occupied)).0
    }

    fn queen_attacks(square: u64, occupied: u64) -> u64 {
        if square >= 64 {
            return 0;
        }
        chess_core::attacks::queen_attacks(Square(square as u8), Bitboard(occupied)).0
    }

    fn king_attacks(square: u64) -> u64 {
        if square >= 64 {
            return 0;
        }
        chess_core::attacks::king_attacks(Square(square as u8)).0
    }
}

// ---------------------------------------------------------------------------
// Public type alias
// ---------------------------------------------------------------------------

/// Tablebase handle parameterized with this engine's attack functions.
pub type SyzygyTB = TableBases<ChessAdapter>;

// ---------------------------------------------------------------------------
// Score constants
// ---------------------------------------------------------------------------

/// Score for a tablebase-proven win (below `MATE_THRESHOLD` so it is never
/// confused with a checkmate score, but well above normal material values).
pub const TB_WIN_SCORE: i32 = 28_000;

/// Score for a tablebase-proven loss.
pub const TB_LOSS_SCORE: i32 = -28_000;

// ---------------------------------------------------------------------------
// WDL probing
// ---------------------------------------------------------------------------

/// Convert a WDL probe result to a centipawn score from the side-to-move's
/// perspective.
///
/// | Result       | Score             | Meaning                                   |
/// |--------------|-------------------|-------------------------------------------|
/// | Win          |  28 000           | Forced win (50-move rule cannot save it)  |
/// | CursedWin    |      2            | Win but convertible only with 50-move help|
/// | Draw         |      0            | Drawn position                             |
/// | BlessedLoss  |     -2            | Loss but drawn by 50-move rule            |
/// | Loss         | -28 000           | Forced loss                               |
#[inline]
pub fn wdl_to_score(wdl: WdlProbeResult) -> i32 {
    match wdl {
        WdlProbeResult::Win => TB_WIN_SCORE,
        WdlProbeResult::CursedWin => 2,
        WdlProbeResult::Draw => 0,
        WdlProbeResult::BlessedLoss => -2,
        WdlProbeResult::Loss => TB_LOSS_SCORE,
    }
}

/// Return whether a score is one of the discrete WDL scores emitted by Syzygy.
#[inline]
pub fn is_wdl_score(score: i32) -> bool {
    matches!(score, TB_LOSS_SCORE | -2 | 0 | 2 | TB_WIN_SCORE)
}

/// Map Syzygy WDL sentinel scores to human-facing UCI display scores.
///
/// Search should continue using the raw internal sentinels; this mapping is
/// only for presentation so GUIs do not see values like `cp 28000`.
#[inline]
pub fn wdl_display_score(score: i32) -> Option<i32> {
    match score {
        TB_WIN_SCORE => Some(3_000),
        TB_LOSS_SCORE => Some(-3_000),
        2 => Some(20),
        -2 => Some(-20),
        0 => Some(0),
        _ => None,
    }
}

/// Probe the WDL tablebases for the given board position.
///
/// Returns `None` when the position cannot be probed:
/// - Any castling rights remain (Syzygy tablebases assume no castling).
/// - The total piece count exceeds the largest loaded tablebase.
/// - The probe itself fails (corrupt data, etc.).
pub fn probe_wdl(tb: &SyzygyTB, board: &Board) -> Option<WdlProbeResult> {
    // Syzygy does not cover positions where castling is still possible.
    if board.castling.0 != 0 {
        return None;
    }

    // Guard against invalid board states that would cause pyrrhic-rs to panic
    // (panic = "abort" → process dies, no unwinding possible).

    // Missing king: pyrrhic indexes piece bitboards with trailing_zeros(); an
    // empty bitboard yields 64, which is out of bounds for the 64-element
    // attack / encoding tables.
    let wk = board.pieces[Color::White.index()][PieceKind::King.index()];
    let bk = board.pieces[Color::Black.index()][PieceKind::King.index()];
    if wk.is_empty() || bk.is_empty() {
        return None;
    }

    let white = board.occupancy[Color::White.index()].0;
    let black = board.occupancy[Color::Black.index()].0;

    // Too many pieces: pyrrhic's position-encoding tables (BINOMIAL etc.) are
    // only valid for positions the loaded tablebases actually cover.  Passing a
    // position with more pieces than TB_LARGEST causes the hash lookup to find
    // an unrelated entry whose encoding metadata doesn't match, producing
    // out-of-bounds square indices (e.g. 116) in the BINOMIAL lookup.
    if (white | black).count_ones() > tb.max_pieces() {
        return None;
    }

    // Piece-type overlap guard: a TT-corrupted quiet move can land on an already-occupied
    // same-color square without clearing the existing piece from its type bitboard.
    // Occupancy is still correct (1 bit), but two piece-type boards share that square, so
    // pyrrhic's key computation overcounts pieces, finds the wrong TB entry (hash collision),
    // runs fill_squares on empty bitboards (poplsb→64), and after XOR transforms produces
    // sq=127 at BINOMIAL index k=4 with skips=4 → index 123 → OOB panic.
    //
    // Fix: sum of per-type piece counts must equal occupancy count for each side.
    let w_sum: u32 = board.pieces[Color::White.index()]
        .iter()
        .map(|bb| bb.0.count_ones())
        .sum();
    let b_sum: u32 = board.pieces[Color::Black.index()]
        .iter()
        .map(|bb| bb.0.count_ones())
        .sum();
    if w_sum != white.count_ones() || b_sum != black.count_ones() {
        return None;
    }

    // Cross-color occupancy overlap guard: a TT-corrupted "quiet" move can target a square
    // occupied by an enemy piece (flag says quiet → no capture removal). Without cleanup in
    // make_move the enemy's occupancy bit stays set, so `white & black != 0`. Pyrrhic
    // uses white and black occupancy independently; with overlap it overcounts material for
    // BOTH sides, finds the wrong TB entry, and eventually dereferences an invalid pointer
    // in decompress_pairs or accesses OFF_DIAG / BINOMIAL out of bounds → SEGV.
    if (white & black) != 0 {
        return None;
    }

    let piece = |ci: usize, ki: usize| board.pieces[ci][ki].0;

    let kings = piece(0, PieceKind::King.index()) | piece(1, PieceKind::King.index());
    let queens = piece(0, PieceKind::Queen.index()) | piece(1, PieceKind::Queen.index());
    let rooks = piece(0, PieceKind::Rook.index()) | piece(1, PieceKind::Rook.index());
    let bishops = piece(0, PieceKind::Bishop.index()) | piece(1, PieceKind::Bishop.index());
    let knights = piece(0, PieceKind::Knight.index()) | piece(1, PieceKind::Knight.index());
    let pawns = piece(0, PieceKind::Pawn.index()) | piece(1, PieceKind::Pawn.index());

    // En-passant target square (0 = none).
    //
    // Guard: a TT hash collision can produce a DoublePawnPush flag on a
    // non-pawn piece, setting board.en_passant to a square where no actual
    // pawn double-pushed.  When we pass such an EP square to pyrrhic it
    // generates an internal "EP capture" that removes the wrong piece from
    // the side-occupancy bitboard.  The derived position then has
    // `piece_type_bb & side_bb == 0` for a type the TB entry expects, causing
    // poplsb(0) == 64 and an OOB panic inside pyrrhic's encoding tables.
    //
    // Fix: only forward the EP square if the pawn that double-pushed is
    // actually present at the expected square (ep±8).
    let ep: u32 = if let Some(ep_sq) = board.en_passant {
        let idx = ep_sq.0;
        let wp = board.pieces[Color::White.index()][PieceKind::Pawn.index()];
        let bp = board.pieces[Color::Black.index()][PieceKind::Pawn.index()];
        let valid = match ep_sq.rank() {
            2 => wp.is_set(Square(idx + 8)),
            5 => bp.is_set(Square(idx - 8)),
            _ => false,
        };
        if valid { idx as u32 } else { 0 }
    } else {
        0
    };
    let turn = board.side_to_move == Color::White;

    tb.probe_wdl(
        white, black, kings, queens, rooks, bishops, knights, pawns, ep, turn,
    )
    .ok()
}

fn validated_probe_inputs(tb: &SyzygyTB, board: &Board) -> Option<ProbeInputs> {
    if board.castling.0 != 0 {
        return None;
    }

    let wk = board.pieces[Color::White.index()][PieceKind::King.index()];
    let bk = board.pieces[Color::Black.index()][PieceKind::King.index()];
    if wk.is_empty() || bk.is_empty() {
        return None;
    }

    let white = board.occupancy[Color::White.index()].0;
    let black = board.occupancy[Color::Black.index()].0;
    if (white | black).count_ones() > tb.max_pieces() {
        return None;
    }

    let w_sum: u32 = board.pieces[Color::White.index()]
        .iter()
        .map(|bb| bb.0.count_ones())
        .sum();
    let b_sum: u32 = board.pieces[Color::Black.index()]
        .iter()
        .map(|bb| bb.0.count_ones())
        .sum();
    if w_sum != white.count_ones() || b_sum != black.count_ones() {
        return None;
    }
    if (white & black) != 0 {
        return None;
    }

    let piece = |ci: usize, ki: usize| board.pieces[ci][ki].0;
    let kings = piece(0, PieceKind::King.index()) | piece(1, PieceKind::King.index());
    let queens = piece(0, PieceKind::Queen.index()) | piece(1, PieceKind::Queen.index());
    let rooks = piece(0, PieceKind::Rook.index()) | piece(1, PieceKind::Rook.index());
    let bishops = piece(0, PieceKind::Bishop.index()) | piece(1, PieceKind::Bishop.index());
    let knights = piece(0, PieceKind::Knight.index()) | piece(1, PieceKind::Knight.index());
    let pawns = piece(0, PieceKind::Pawn.index()) | piece(1, PieceKind::Pawn.index());

    let ep: u32 = if let Some(ep_sq) = board.en_passant {
        let idx = ep_sq.0;
        let wp = board.pieces[Color::White.index()][PieceKind::Pawn.index()];
        let bp = board.pieces[Color::Black.index()][PieceKind::Pawn.index()];
        let valid = match ep_sq.rank() {
            2 => wp.is_set(Square(idx + 8)),
            5 => bp.is_set(Square(idx - 8)),
            _ => false,
        };
        if valid { idx as u32 } else { 0 }
    } else {
        0
    };

    let turn = board.side_to_move == Color::White;
    Some((
        white, black, kings, queens, rooks, bishops, knights, pawns, ep, turn,
    ))
}

#[cfg(test)]
pub(crate) fn syzygy_test_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

/// Probe DTZ information for the root position.
///
/// This requires exclusive access to the underlying Syzygy handle.
pub fn probe_root(tb: &SyzygyTB, board: &Board) -> Option<pyrrhic_rs::DtzProbeResult> {
    let (white, black, kings, queens, rooks, bishops, knights, pawns, ep, turn) =
        validated_probe_inputs(tb, board)?;

    tb.probe_root(
        white,
        black,
        kings,
        queens,
        rooks,
        bishops,
        knights,
        pawns,
        board.halfmove_clock as u32,
        ep,
        turn,
    )
    .ok()
}

fn decode_root_move(
    board: &Board,
    from_square: u8,
    to_square: u8,
    promotion: Piece,
) -> Option<Move> {
    let promo_kind = match promotion {
        Piece::Queen => Some(PieceKind::Queen),
        Piece::Rook => Some(PieceKind::Rook),
        Piece::Bishop => Some(PieceKind::Bishop),
        Piece::Knight => Some(PieceKind::Knight),
        Piece::Pawn | Piece::King => None,
    };

    chess_core::generate_legal_moves(board)
        .iter()
        .copied()
        .find(|m| {
            m.from_sq() == Square(from_square)
                && m.to_sq() == Square(to_square)
                && m.flag().promotion_piece() == promo_kind
        })
}

/// Ranked root-move guidance derived from a single DTZ root probe.
///
/// DTZ measures distance-to-zeroing (the next pawn move or capture), not distance
/// to mate, and it assumes optimal defence throughout. Converting a DTZ line into
/// a "mate in N" score is therefore unsound: one suboptimal-for-the-defender reply
/// invalidates the line, and an engine that believes it is mating can chase checks
/// straight into a threefold repetition (the lichess R2KlN7mg perpetual trap).
///
/// So instead of minting a mate score and PV, we expose the tablebase as a *root
/// ranking*: a root-relative score in the TB band (never a mate score) plus, when
/// the side to move is winning, the win-preserving legal moves ordered best-first
/// by DTZ. The search restricts its root to these moves and picks among them,
/// producing the PV and the reported score itself. This keeps DTZ's one real
/// strength — guaranteeing zeroing *progress*, which a search whose leaf probes
/// return a flat `TB_WIN_SCORE` cannot see — while letting the search handle the
/// things DTZ is blind to: tactics, game-history repetitions, and the actual line.
#[derive(Clone, Debug)]
pub struct RootTbRanking {
    /// Root-relative value in the TB band (`TB_WIN_SCORE` / draw / `TB_LOSS_SCORE`),
    /// graded by DTZ so faster conversions rank higher. Never a mate score.
    pub score: Score,
    /// WDL of the root position from the side-to-move's perspective.
    pub best_wdl: WdlProbeResult,
    /// When the root is a win, the win-preserving legal moves ordered best-first
    /// (the tablebase-recommended move, then ascending DTZ). Empty otherwise.
    pub winning_moves: Vec<Move>,
}

/// Rank the legal root moves using a single DTZ root probe.
///
/// Returns `None` when the root is outside tablebase range, is itself terminal
/// (mate/stalemate), or the probe otherwise fails — the search handles those
/// positions directly. See [`RootTbRanking`] for the contract.
pub fn rank_root_moves(tb: &SyzygyTB, board: &Board) -> Option<RootTbRanking> {
    let probe = probe_root(tb, board)?;

    // The recommended root move and the root WDL come from `probe.root`. A
    // terminal root (mate/stalemate) or a failed probe carries no ranking.
    let root = match probe.root {
        DtzProbeValue::DtzResult(r) => r,
        _ => return None,
    };
    let best_wdl = root.wdl;
    let recommended = decode_root_move(board, root.from_square, root.to_square, root.promotion);

    // Score in the TB band, graded by the best-play DTZ. `probe_root` already
    // folds the halfmove clock into the WDL, so a value of `Win` here is a win
    // that survives the 50-move rule; a cursed/blessed result is the 50-move
    // draw and scores accordingly.
    let score = match best_wdl {
        WdlProbeResult::Win => Score(TB_WIN_SCORE - root.dtz as i32),
        WdlProbeResult::CursedWin => Score(2),
        WdlProbeResult::Draw => Score::DRAW,
        WdlProbeResult::BlessedLoss => Score(-2),
        WdlProbeResult::Loss => Score(TB_LOSS_SCORE + root.dtz as i32),
    };

    // Win-preserving move list — only meaningful when the root itself is a win.
    let mut winning_moves: Vec<Move> = Vec::new();
    if matches!(best_wdl, WdlProbeResult::Win) {
        let mut scored: Vec<(Move, u16)> = Vec::new();
        for value in probe.moves.iter().copied().take(probe.num_moves) {
            if let DtzProbeValue::DtzResult(r) = value
                && matches!(r.wdl, WdlProbeResult::Win)
                && let Some(mv) = decode_root_move(board, r.from_square, r.to_square, r.promotion)
            {
                scored.push((mv, r.dtz));
            }
        }
        // Fastest conversion first (ascending DTZ), but keep the tablebase's own
        // recommended move at the very front so ties match the probe's choice.
        scored.sort_by_key(|&(mv, dtz)| (Some(mv) != recommended, dtz));
        winning_moves = scored.into_iter().map(|(mv, _)| mv).collect();
    }

    Some(RootTbRanking {
        score,
        best_wdl,
        winning_moves,
    })
}

#[cfg(test)]
mod tests {
    use super::{SyzygyTB, TB_WIN_SCORE, rank_root_moves, syzygy_test_lock};
    use chess_common::Board;
    use pyrrhic_rs::WdlProbeResult;
    use std::path::PathBuf;

    fn syzygy_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/syzygy")
    }

    #[test]
    fn rank_root_moves_reports_loss_without_minting_a_mate() {
        // A losing root: DTZ used to mint a (negative) mate score here. The
        // ranker must classify it as a loss in the TB band — never a mate — and
        // offer no winning moves; defence is left to the search.
        let _guard = syzygy_test_lock().lock().expect("lock syzygy test mutex");
        let path = syzygy_path();
        if !path.exists() {
            return;
        }

        let tb = SyzygyTB::new(path.to_string_lossy().as_ref()).expect("load syzygy tables");
        let board = Board::from_fen("8/8/P1b5/6p1/3K1k2/8/8/8 w - - 0 54").expect("valid FEN");
        let ranking = rank_root_moves(&tb, &board).expect("root TB ranking");

        assert!(
            matches!(ranking.best_wdl, WdlProbeResult::Loss),
            "expected Loss, got {:?}",
            ranking.best_wdl
        );
        assert!(
            ranking.score.0 < 0,
            "expected negative score, got {}",
            ranking.score.0
        );
        assert!(
            !ranking.score.is_mate(),
            "DTZ must not mint a mate score, got {}",
            ranking.score
        );
        assert!(
            ranking.winning_moves.is_empty(),
            "a losing root has no winning moves"
        );
    }

    #[test]
    fn rank_root_moves_ranks_precapture_tb_win() {
        // A winning root whose best move is the pawn-capturing a8c6. The ranker
        // must report a TB-win-band score (not a mate) and list a8c6 first.
        let _guard = syzygy_test_lock().lock().expect("lock syzygy test mutex");
        let path = syzygy_path();
        if !path.exists() {
            return;
        }

        let tb = SyzygyTB::new(path.to_string_lossy().as_ref()).expect("load syzygy tables");
        let board = Board::from_fen("b7/8/P1P5/6p1/3K1k2/8/8/8 b - - 0 53").expect("valid FEN");
        let ranking = rank_root_moves(&tb, &board).expect("root TB ranking");

        assert!(
            matches!(ranking.best_wdl, WdlProbeResult::Win),
            "expected Win, got {:?}",
            ranking.best_wdl
        );
        assert!(
            !ranking.score.is_mate(),
            "DTZ must not mint a mate score, got {}",
            ranking.score
        );
        assert!(
            ranking.score.0 > TB_WIN_SCORE - 1_000 && ranking.score.0 <= TB_WIN_SCORE,
            "expected TB-win-band score, got {}",
            ranking.score.0
        );
        assert_eq!(
            ranking.winning_moves.first().map(|m| m.to_uci()),
            Some("a8c6".to_string())
        );
    }

    #[test]
    fn rank_root_moves_returns_none_when_root_is_out_of_tb_range() {
        // The 32-piece start position is outside any loaded Syzygy table; the
        // ranker must report no guidance rather than fabricate one.
        let _guard = syzygy_test_lock().lock().expect("lock syzygy test mutex");
        let path = syzygy_path();
        if !path.exists() {
            return;
        }

        let tb = SyzygyTB::new(path.to_string_lossy().as_ref()).expect("load syzygy tables");
        let board = Board::default();
        let result = rank_root_moves(&tb, &board);
        assert!(
            result.is_none(),
            "32-piece root must not yield a TB-backed ranking, got {:?}",
            result.as_ref().map(|r| (
                r.score.0,
                r.winning_moves
                    .iter()
                    .map(|m| m.to_uci())
                    .collect::<Vec<_>>()
            ))
        );
    }
}
