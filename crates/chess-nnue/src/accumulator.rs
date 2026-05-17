use chess_common::{Board, Color, PieceKind, Square};

use crate::features::feature_index;
use crate::network::NnueNetwork;
use crate::HIDDEN_SIZE;

/// NNUE accumulator holding feature-transformed values for both perspectives.
///
/// `needs_refresh` is set to true when the accumulator is stale (e.g. after a
/// king move changes the king bucket).  The refresh is deferred until the
/// position is actually evaluated, so pruned nodes pay no refresh cost.
#[derive(Clone)]
pub struct Accumulator {
    pub white: [i16; HIDDEN_SIZE],
    pub black: [i16; HIDDEN_SIZE],
    pub needs_refresh: bool,
}

impl Default for Accumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl Accumulator {
    /// Create a zeroed accumulator (marked as needing a refresh).
    pub const fn new() -> Self {
        Self {
            white: [0i16; HIDDEN_SIZE],
            black: [0i16; HIDDEN_SIZE],
            needs_refresh: true,
        }
    }

    /// Full recompute from scratch using the board state.
    pub fn refresh(&mut self, board: &Board, net: &NnueNetwork) {
        self.needs_refresh = false;
        self.white = *net.ft_biases;
        self.black = *net.ft_biases;

        let white_king = board.king_square(Color::White);
        let black_king = board.king_square(Color::Black);

        for &color in &[Color::White, Color::Black] {
            for &kind in &PieceKind::ALL {
                let bb = board.pieces[color.index()][kind.index()];
                for sq in bb.iter() {
                    self.add_piece_inner(net, white_king, black_king, color, kind, sq);
                }
            }
        }
    }

    /// Add a piece's feature weights to both perspectives.
    #[inline]
    pub fn add_piece(
        &mut self,
        net: &NnueNetwork,
        white_king: Square,
        black_king: Square,
        color: Color,
        kind: PieceKind,
        sq: Square,
    ) {
        self.add_piece_inner(net, white_king, black_king, color, kind, sq);
    }

    /// Subtract a piece's feature weights from both perspectives.
    #[inline]
    pub fn sub_piece(
        &mut self,
        net: &NnueNetwork,
        white_king: Square,
        black_king: Square,
        color: Color,
        kind: PieceKind,
        sq: Square,
    ) {
        let w_idx = feature_index(Color::White, white_king, black_king, color, kind, sq);
        let b_idx = feature_index(Color::Black, white_king, black_king, color, kind, sq);

        let w_row = &net.ft_weights[w_idx];
        let b_row = &net.ft_weights[b_idx];

        for i in 0..HIDDEN_SIZE {
            self.white[i] -= w_row[i];
            self.black[i] -= b_row[i];
        }
    }

    #[inline]
    fn add_piece_inner(
        &mut self,
        net: &NnueNetwork,
        white_king: Square,
        black_king: Square,
        color: Color,
        kind: PieceKind,
        sq: Square,
    ) {
        let w_idx = feature_index(Color::White, white_king, black_king, color, kind, sq);
        let b_idx = feature_index(Color::Black, white_king, black_king, color, kind, sq);

        let w_row = &net.ft_weights[w_idx];
        let b_row = &net.ft_weights[b_idx];

        for i in 0..HIDDEN_SIZE {
            self.white[i] += w_row[i];
            self.black[i] += b_row[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incremental_matches_refresh() {
        let net = NnueNetwork::embedded();
        let board = Board::starting_position();
        let white_king = board.king_square(Color::White);
        let black_king = board.king_square(Color::Black);

        // Full refresh
        let mut acc_full = Accumulator::new();
        acc_full.refresh(&board, &net);

        // Incremental: start from biases, add pieces one by one
        let mut acc_inc = Accumulator::new();
        acc_inc.white = *net.ft_biases;
        acc_inc.black = *net.ft_biases;
        for &color in &[Color::White, Color::Black] {
            for &kind in &PieceKind::ALL {
                let bb = board.pieces[color.index()][kind.index()];
                for sq in bb.iter() {
                    acc_inc.add_piece(&net, white_king, black_king, color, kind, sq);
                }
            }
        }

        assert_eq!(acc_full.white, acc_inc.white);
        assert_eq!(acc_full.black, acc_inc.black);
    }

    #[test]
    fn add_sub_roundtrip() {
        let net = NnueNetwork::embedded();
        let board = Board::starting_position();
        let white_king = board.king_square(Color::White);
        let black_king = board.king_square(Color::Black);

        let mut acc = Accumulator::new();
        acc.refresh(&board, &net);
        let original_white = acc.white;
        let original_black = acc.black;

        // Add then subtract a piece — should return to original
        acc.add_piece(&net, white_king, black_king, Color::White, PieceKind::Queen, Square::new(3, 3));
        acc.sub_piece(&net, white_king, black_king, Color::White, PieceKind::Queen, Square::new(3, 3));

        assert_eq!(acc.white, original_white);
        assert_eq!(acc.black, original_black);
    }
}
