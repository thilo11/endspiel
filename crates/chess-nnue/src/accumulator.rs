use chess_common::{Board, Color, PieceKind, Square};

use crate::HIDDEN_SIZE;
use crate::features::feature_index;
use crate::network::NnueNetwork;

/// NNUE accumulator holding feature-transformed values for both perspectives.
///
/// `needs_refresh` is set to true when the accumulator is stale (e.g. after a
/// king move changes the king bucket).  The refresh is deferred until the
/// position is actually evaluated, so pruned nodes pay no refresh cost.
#[derive(Clone)]
pub struct Accumulator {
    pub white: [i16; HIDDEN_SIZE],
    pub black: [i16; HIDDEN_SIZE],
    needs_refresh: [bool; 2],
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
            needs_refresh: [true; 2],
        }
    }

    /// Full recompute from scratch using the board state.
    pub fn refresh(&mut self, board: &Board, net: &NnueNetwork) {
        self.refresh_perspective(board, net, Color::White);
        self.refresh_perspective(board, net, Color::Black);
    }

    /// Recompute only one perspective. A king move changes the moving side's
    /// feature bucket, but the opponent's bucket remains stable and can stay
    /// incrementally updated.
    pub fn refresh_perspective(&mut self, board: &Board, net: &NnueNetwork, perspective: Color) {
        let mut values = *net.ft_biases;

        let white_king = board.king_square(Color::White);
        let black_king = board.king_square(Color::Black);

        for &color in &[Color::White, Color::Black] {
            for &kind in &PieceKind::ALL {
                let bb = board.pieces[color.index()][kind.index()];
                for sq in bb.iter() {
                    let idx = feature_index(perspective, white_king, black_king, color, kind, sq);
                    let row = &net.ft_weights[idx];
                    for i in 0..HIDDEN_SIZE {
                        values[i] += row[i];
                    }
                }
            }
        }

        match perspective {
            Color::White => self.white = values,
            Color::Black => self.black = values,
        }
        self.needs_refresh[perspective.index()] = false;
    }

    #[inline]
    pub fn needs_refresh(&self, perspective: Color) -> bool {
        self.needs_refresh[perspective.index()]
    }

    #[inline]
    pub fn mark_refresh(&mut self, perspective: Color) {
        self.needs_refresh[perspective.index()] = true;
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
        if !self.needs_refresh(Color::White) {
            let idx = feature_index(Color::White, white_king, black_king, color, kind, sq);
            let row = &net.ft_weights[idx];
            for (value, &delta) in self.white.iter_mut().zip(row.iter()) {
                *value -= delta;
            }
        }
        if !self.needs_refresh(Color::Black) {
            let idx = feature_index(Color::Black, white_king, black_king, color, kind, sq);
            let row = &net.ft_weights[idx];
            for (value, &delta) in self.black.iter_mut().zip(row.iter()) {
                *value -= delta;
            }
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
        if !self.needs_refresh(Color::White) {
            let idx = feature_index(Color::White, white_king, black_king, color, kind, sq);
            let row = &net.ft_weights[idx];
            for (value, &delta) in self.white.iter_mut().zip(row.iter()) {
                *value += delta;
            }
        }
        if !self.needs_refresh(Color::Black) {
            let idx = feature_index(Color::Black, white_king, black_king, color, kind, sq);
            let row = &net.ft_weights[idx];
            for (value, &delta) in self.black.iter_mut().zip(row.iter()) {
                *value += delta;
            }
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
        acc_inc.needs_refresh = [false; 2];
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
        acc.add_piece(
            &net,
            white_king,
            black_king,
            Color::White,
            PieceKind::Queen,
            Square::new(3, 3),
        );
        acc.sub_piece(
            &net,
            white_king,
            black_king,
            Color::White,
            PieceKind::Queen,
            Square::new(3, 3),
        );

        assert_eq!(acc.white, original_white);
        assert_eq!(acc.black, original_black);
    }

    #[test]
    fn perspective_refresh_leaves_other_half_untouched() {
        let net = NnueNetwork::embedded();
        let board = Board::starting_position();

        let mut expected = Accumulator::new();
        expected.refresh(&board, &net);

        let mut acc = expected.clone();
        acc.white = [0; HIDDEN_SIZE];
        acc.black = [123; HIDDEN_SIZE];
        acc.mark_refresh(Color::White);
        acc.refresh_perspective(&board, &net, Color::White);

        assert_eq!(acc.white, expected.white);
        assert_eq!(acc.black, [123; HIDDEN_SIZE]);
        assert!(!acc.needs_refresh(Color::White));
    }
}
