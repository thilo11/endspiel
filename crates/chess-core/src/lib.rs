pub mod attacks;
pub mod magics;
pub mod movegen;
pub mod san;
pub mod validate;

use chess_common::{Board, Move};
use chess_common::moves::MoveList;

/// Generate all legal moves for the current position.
pub fn generate_legal_moves(board: &Board) -> MoveList {
    movegen::generate_legal_moves(board)
}

/// Generate pseudo-legal captures and promotions only (no legality check).
pub fn generate_pseudo_legal_captures(board: &Board) -> MoveList {
    movegen::generate_pseudo_legal_captures(board)
}

/// Generate pseudo-legal quiet moves only (no legality check).
pub fn generate_pseudo_legal_quiets(board: &Board) -> MoveList {
    movegen::generate_pseudo_legal_quiets(board)
}

/// Generate all pseudo-legal moves (no legality check).
pub fn generate_pseudo_legal_moves(board: &Board) -> MoveList {
    movegen::generate_pseudo_legal_moves(board)
}

/// Check if a move is legal in the given position.
pub fn is_legal_move(board: &Board, m: Move) -> bool {
    validate::is_legal_move(board, m)
}

/// Check if the side to move is in check.
pub fn is_in_check(board: &Board) -> bool {
    let king_sq = board.king_square(board.side_to_move);
    attacks::is_square_attacked(board, king_sq, board.side_to_move.opposite())
}
