use chess_common::moves::Move;
use chess_common::Board;

use crate::movegen::generate_legal_moves;

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
        legal.from_sq() == from
            && legal.to_sq() == to
            && legal.flag().promotion_piece() == promo
    })
}

/// Find the legal move matching the given from/to/promotion, returning the
/// fully-flagged version. Returns `None` if no such legal move exists.
pub fn find_legal_move(board: &Board, m: Move) -> Option<Move> {
    let legal_moves = generate_legal_moves(board);
    let from = m.from_sq();
    let to = m.to_sq();
    let promo = m.flag().promotion_piece();

    legal_moves
        .iter()
        .find(|legal| {
            legal.from_sq() == from
                && legal.to_sq() == to
                && legal.flag().promotion_piece() == promo
        })
        .copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess_common::moves::MoveFlag;
    use chess_common::Square;

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
}
