use chess_common::moves::{Move, MoveFlag};
use chess_common::{Board, Color, PieceKind};
use thiserror::Error;

use crate::attacks::is_square_attacked;
use crate::movegen::generate_legal_moves;

/// A position that is structurally parseable but cannot occur at the boundary
/// between two legal chess moves.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PositionError {
    #[error("expected exactly one {color:?} king, got {count}")]
    KingCount { color: Color, count: u32 },
    #[error("side not to move ({0:?}) is in check")]
    SideNotToMoveInCheck(Color),
}

/// Validate the legality assumptions required before move generation/search.
///
/// FEN parsing handles static structure. This additional check rejects a board
/// where the side that supposedly just moved left its own king in check. Such
/// a board lets move generation capture that king and corrupt all deeper state.
pub fn validate_position(board: &Board) -> Result<(), PositionError> {
    for color in [Color::White, Color::Black] {
        let count = board.pieces[color.index()][PieceKind::King.index()].count();
        if count != 1 {
            return Err(PositionError::KingCount { color, count });
        }
    }

    let previous = board.side_to_move.opposite();
    if is_square_attacked(board, board.king_square(previous), board.side_to_move) {
        return Err(PositionError::SideNotToMoveInCheck(previous));
    }
    Ok(())
}

/// Check if a specific move is legal in the given position.
///
/// This generates all legal moves and checks if the given move is among them.
/// For UCI move parsing, the move may not have the exact flag set (e.g., a UCI
/// move "e2e4" won't know if it's a double pawn push), so we match on from/to
/// squares and promotion piece.
pub fn is_legal_move(board: &Board, m: Move) -> bool {
    let legal_moves = generate_legal_moves(board);

    // First try an exact match
    if legal_moves.contains(&m) {
        return true;
    }

    // If no exact match, try matching by from/to and promotion piece.
    // This handles the case where a UCI move has MoveFlag::Normal but the
    // actual legal move has a specific flag like Capture or DoublePawnPush.
    let from = m.from_sq();
    let to = m.to_sq();
    let promo = m.flag().promotion_piece();

    legal_moves.iter().any(|legal| {
        move_matches_uci(board, *legal, from, to, promo)
    })
}

/// Find the legal move matching the given from/to/promotion, returning the
/// fully-flagged version. Returns `None` if no such legal move exists.
///
/// Accepts both standard king-destination castling (`e1g1`) and Chess960
/// king-takes-rook (`e1h1`). When both a king walk and a castle share the
/// same destination (king already next to c/g), the castle is preferred so
/// standard UCI `e1g1` still means O-O.
pub fn find_legal_move(board: &Board, m: Move) -> Option<Move> {
    find_legal_move_uci(board, m, false)
}

/// Like [`find_legal_move`], but when `chess960` is true a same-destination
/// king walk beats castling (castle is sent as king-takes-rook).
pub fn find_legal_move_uci(board: &Board, m: Move, chess960: bool) -> Option<Move> {
    let legal_moves = generate_legal_moves(board);
    let from = m.from_sq();
    let to = m.to_sq();
    let promo = m.flag().promotion_piece();

    let mut castle_via_rook = None;
    let mut castle_via_dest = None;
    let mut other = None;
    for &legal in legal_moves.iter() {
        if legal.from_sq() != from || legal.flag().promotion_piece() != promo {
            continue;
        }
        let is_castle = matches!(
            legal.flag(),
            MoveFlag::KingsideCastle | MoveFlag::QueensideCastle
        );
        if is_castle {
            let kingside = legal.flag() == MoveFlag::KingsideCastle;
            let color = if legal.from_sq().rank() == 0 {
                Color::White
            } else {
                Color::Black
            };
            if board.castle_rook(color, kingside) == to {
                castle_via_rook = Some(legal);
            }
            if legal.to_sq() == to {
                castle_via_dest = Some(legal);
            }
        } else if legal.to_sq() == to {
            other = Some(legal);
        }
    }

    if chess960 {
        castle_via_rook.or(other).or(castle_via_dest)
    } else {
        castle_via_dest.or(other).or(castle_via_rook)
    }
}

fn move_matches_uci(
    board: &Board,
    legal: Move,
    from: chess_common::Square,
    to: chess_common::Square,
    promo: Option<PieceKind>,
) -> bool {
    if legal.from_sq() != from || legal.flag().promotion_piece() != promo {
        return false;
    }
    if legal.to_sq() == to {
        return true;
    }
    if matches!(
        legal.flag(),
        MoveFlag::KingsideCastle | MoveFlag::QueensideCastle
    ) {
        let kingside = legal.flag() == MoveFlag::KingsideCastle;
        let color = if legal.from_sq().rank() == 0 {
            Color::White
        } else {
            Color::Black
        };
        return board.castle_rook(color, kingside) == to;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess_common::Square;
    use chess_common::moves::MoveFlag;

    #[test]
    fn test_e2e4_is_legal() {
        let board = Board::starting_position();
        let m = Move::new(
            Square::from_algebraic("e2").unwrap(),
            Square::from_algebraic("e4").unwrap(),
            MoveFlag::DoublePawnPush,
        );
        assert!(is_legal_move(&board, m));
    }

    #[test]
    fn test_e2e4_from_uci() {
        let board = Board::starting_position();
        // UCI doesn't know about DoublePawnPush flag, so it sends Normal
        let m = Move::from_uci("e2e4").unwrap();
        assert!(is_legal_move(&board, m));
    }

    #[test]
    fn test_invalid_move() {
        let board = Board::starting_position();
        let m = Move::new(
            Square::from_algebraic("e2").unwrap(),
            Square::from_algebraic("e5").unwrap(),
            MoveFlag::Normal,
        );
        assert!(!is_legal_move(&board, m));
    }

    #[test]
    fn test_find_legal_move_flags() {
        let board = Board::starting_position();
        let uci_move = Move::from_uci("e2e4").unwrap();
        let legal = find_legal_move(&board, uci_move).unwrap();
        assert_eq!(legal.flag(), MoveFlag::DoublePawnPush);
    }

    #[test]
    fn validates_normal_and_in_check_positions() {
        assert!(validate_position(&Board::starting_position()).is_ok());

        // White is in check and must respond; that is a valid position.
        let checked = Board::from_fen("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1").unwrap();
        assert!(validate_position(&checked).is_ok());
    }

    #[test]
    fn rejects_a_position_where_the_previous_side_is_in_check() {
        let board = Board::from_fen("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1").unwrap();
        assert_eq!(
            validate_position(&board),
            Err(PositionError::SideNotToMoveInCheck(Color::Black))
        );
    }

    #[test]
    fn chess960_king_takes_rook_matches_castle() {
        let board =
            Board::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1").unwrap();
        let rook_uci = Move::from_uci("e1h1").unwrap();
        let dest_uci = Move::from_uci("e1g1").unwrap();
        let via_rook = find_legal_move_uci(&board, rook_uci, true).unwrap();
        assert_eq!(via_rook.flag(), MoveFlag::KingsideCastle);
        let via_dest = find_legal_move_uci(&board, dest_uci, false).unwrap();
        assert_eq!(via_dest.flag(), MoveFlag::KingsideCastle);
        assert_eq!(via_rook, via_dest);
    }
}
