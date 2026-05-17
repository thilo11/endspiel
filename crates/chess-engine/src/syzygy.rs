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
        if square >= 64 { return 0; }
        let sq = Square(square as u8);
        let c = if color == pyrrhic_rs::Color::White { Color::White } else { Color::Black };
        chess_core::attacks::pawn_attacks(sq, c).0
    }

    fn knight_attacks(square: u64) -> u64 {
        if square >= 64 { return 0; }
        chess_core::attacks::knight_attacks(Square(square as u8)).0
    }

    fn bishop_attacks(square: u64, occupied: u64) -> u64 {
        if square >= 64 { return 0; }
        chess_core::attacks::bishop_attacks(Square(square as u8), Bitboard(occupied)).0
    }

    fn rook_attacks(square: u64, occupied: u64) -> u64 {
        if square >= 64 { return 0; }
        chess_core::attacks::rook_attacks(Square(square as u8), Bitboard(occupied)).0
    }

    fn queen_attacks(square: u64, occupied: u64) -> u64 {
        if square >= 64 { return 0; }
        chess_core::attacks::queen_attacks(Square(square as u8), Bitboard(occupied)).0
    }

    fn king_attacks(square: u64) -> u64 {
        if square >= 64 { return 0; }
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
    let w_sum: u32 = board.pieces[Color::White.index()].iter().map(|bb| bb.0.count_ones()).sum();
    let b_sum: u32 = board.pieces[Color::Black.index()].iter().map(|bb| bb.0.count_ones()).sum();
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

    tb.probe_wdl(white, black, kings, queens, rooks, bishops, knights, pawns, ep, turn).ok()
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

    let w_sum: u32 = board.pieces[Color::White.index()].iter().map(|bb| bb.0.count_ones()).sum();
    let b_sum: u32 = board.pieces[Color::Black.index()].iter().map(|bb| bb.0.count_ones()).sum();
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
    Some((white, black, kings, queens, rooks, bishops, knights, pawns, ep, turn))
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
    ).ok()
}

fn decode_root_move(board: &Board, from_square: u8, to_square: u8, promotion: Piece) -> Option<Move> {
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

/// Follow the DTZ-recommended tablebase line until terminal mate or draw.
///
/// Returns a root-relative mate/draw score and the corresponding PV when the
/// tablebase line reaches a terminal result within `max_plies`.
fn solve_root_position_direct(tb: &SyzygyTB, board: &Board, max_plies: usize) -> Option<(Score, Vec<Move>)> {
    let mut board = board.clone();
    let root_side = board.side_to_move;
    let mut pv = Vec::new();

    for ply in 0..=max_plies {
        // DTZ does not account for the current game's position history. A move
        // the tablebase calls "optimal" may complete a threefold repetition or
        // trip the 50-move rule given what has already been played, in which
        // case the game is drawn regardless of what DTZ claims. Detect that
        // here and truncate the PV, so TB-backed lines don't over-claim mates
        // that the losing side could simply draw.
        if ply > 0 && (board.halfmove_clock >= 100 || board.is_repetition()) {
            return Some((Score::DRAW, pv));
        }

        let result = probe_root(tb, &board)?;
        match result.root {
            DtzProbeValue::Checkmate => {
                let score = if board.side_to_move == root_side {
                    Score(-Score::MATE.0 + ply as i32)
                } else {
                    Score(Score::MATE.0 - ply as i32)
                };
                return Some((score, pv));
            }
            DtzProbeValue::Stalemate => return Some((Score::DRAW, pv)),
            DtzProbeValue::Failed => return None,
            DtzProbeValue::DtzResult(root) => {
                let next_move = decode_root_move(&board, root.from_square, root.to_square, root.promotion)?;
                pv.push(next_move);
                board.make_move(next_move);
            }
        }
    }

    None
}

/// Solve a root tablebase position to mate/draw when possible.
///
/// Some winning roots are not converted to a terminal line directly by DTZ, but
/// a winning root move can lead to a child position that is. In that case we
/// still want to expose the concrete mate score and PV at the root.
pub fn solve_root_position(tb: &SyzygyTB, board: &Board, max_plies: usize) -> Option<(Score, Vec<Move>)> {
    // Require the root itself to be TB-probable. Without this gate, the
    // legal-move exploration below manufactures (Score::DRAW, [rep_move])
    // results from any move that happens to complete a threefold via game
    // history — even when the position is outside TB coverage entirely. That
    // bogus "TB solution" then overrides the real search eval at the root.
    probe_root(tb, board)?;

    let direct = solve_root_position_direct(tb, board, max_plies);

    if max_plies == 0 {
        return direct;
    }

    // Always explore legal root moves, not just as a fallback. DTZ picks the
    // distance-minimizing move ignoring game history, so:
    //   - if the root side is losing per TB, any legal move that completes a
    //     threefold or trips the 50-move rule is a draw claim they will take;
    //   - if the root side is winning per TB but the direct line hits a
    //     history repetition, a non-DTZ alternative may still win.
    let legal_moves = chess_core::generate_legal_moves(board);
    let mut best_solution: Option<(Score, Vec<Move>)> = direct;

    for mv in legal_moves.iter().copied() {
        let mut child = board.clone();
        child.make_move(mv);

        let (child_score, child_pv) = if child.halfmove_clock >= 100 || child.is_repetition() {
            (Score::DRAW, Vec::new())
        } else {
            match solve_root_position_direct(tb, &child, max_plies - 1) {
                Some((s, p)) => (s, p),
                None => continue,
            }
        };

        let mut pv = Vec::with_capacity(child_pv.len() + 1);
        pv.push(mv);
        pv.extend(child_pv);
        let score = Score(-child_score.0);

        if best_solution.as_ref().is_none_or(|(best_score, _)| score > *best_score) {
            best_solution = Some((score, pv));
        }
    }

    best_solution
}

#[cfg(test)]
mod tests {
    use super::{solve_root_position, syzygy_test_lock, SyzygyTB};
    use chess_common::Board;
    use std::path::PathBuf;

    fn syzygy_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/syzygy")
    }

    #[test]
    fn solve_root_position_reports_mate_for_reported_fen() {
        let _guard = syzygy_test_lock().lock().expect("lock syzygy test mutex");
        let path = syzygy_path();
        if !path.exists() {
            return;
        }

        let tb = SyzygyTB::new(path.to_string_lossy().as_ref()).expect("load syzygy tables");
        let board = Board::from_fen("8/8/P1b5/6p1/3K1k2/8/8/8 w - - 0 54").expect("valid FEN");
        let (score, pv) = solve_root_position(&tb, &board, 128).expect("root TB solution");

        assert!(score.is_mate(), "expected mate score, got {}", score.0);
        assert!(score.0 < 0, "expected losing mate score, got {}", score.0);
        assert_eq!(pv.first().map(|m| m.to_uci()), Some("d4c5".to_string()));
    }

    #[test]
    fn solve_root_position_promotes_precapture_tb_root_to_mate() {
        let _guard = syzygy_test_lock().lock().expect("lock syzygy test mutex");
        let path = syzygy_path();
        if !path.exists() {
            return;
        }

        let tb = SyzygyTB::new(path.to_string_lossy().as_ref()).expect("load syzygy tables");
        let board = Board::from_fen("b7/8/P1P5/6p1/3K1k2/8/8/8 b - - 0 53").expect("valid FEN");
        let (score, pv) = solve_root_position(&tb, &board, 128).expect("root TB solution");

        assert!(score.is_mate(), "expected mate score, got {}", score.0);
        assert!(score.0 > 0, "expected winning mate score, got {}", score.0);
        assert_eq!(pv.first().map(|m| m.to_uci()), Some("a8c6".to_string()));
    }

    #[test]
    fn solve_root_position_returns_none_when_root_is_out_of_tb_range() {
        // A 32-piece root is outside any loaded Syzygy table. Before the
        // probe_root gate in solve_root_position, a legal move whose resulting
        // position was threefold-repeated via game history would be reported
        // as a TB-authoritative DRAW, silently overriding the real search eval
        // at the root.
        let _guard = syzygy_test_lock().lock().expect("lock syzygy test mutex");
        let path = syzygy_path();
        if !path.exists() {
            return;
        }

        let tb = SyzygyTB::new(path.to_string_lossy().as_ref()).expect("load syzygy tables");

        let mut board = Board::default();
        // Knight-shuffle so a subsequent legal move returns to a position
        // that already appears twice in position_history (→ is_repetition).
        for uci in ["b1c3", "b8c6", "c3b1", "c6b8", "b1c3", "b8c6", "c3b1"] {
            let mv = chess_core::generate_legal_moves(&board)
                .iter()
                .find(|m| m.to_uci() == uci)
                .copied()
                .expect("legal knight-shuffle move");
            board.make_move(mv);
        }
        // Sanity: Black's c6b8 would complete the threefold against history.
        let trigger = chess_core::generate_legal_moves(&board)
            .iter()
            .find(|m| m.to_uci() == "c6b8")
            .copied()
            .expect("c6b8 is legal");
        let mut after = board.clone();
        after.make_move(trigger);
        assert!(after.is_repetition(), "c6b8 must create threefold for this test");

        let result = solve_root_position(&tb, &board, 128);
        assert!(
            result.is_none(),
            "32-piece root must not yield a TB-backed result, got {:?}",
            result.as_ref().map(|(s, pv)| (s.0, pv.iter().map(|m| m.to_uci()).collect::<Vec<_>>()))
        );
    }
}
