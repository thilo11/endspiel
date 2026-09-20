use chess_common::{Board, CastlingRights, Color, PieceKind, Square};

use crate::HIDDEN_SIZE;
use crate::features::{board_state_feature_indices, feature_index, state_feature_indices};
#[cfg(target_arch = "x86_64")]
use crate::inference::{SimdBackend, simd_backend};
use crate::network::NnueNetwork;

#[inline]
fn add_row(values: &mut [i16; HIDDEN_SIZE], row: &[i16; HIDDEN_SIZE]) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        match simd_backend() {
            SimdBackend::Avx512Icl | SimdBackend::Avx512 => return add_row_avx512(values, row),
            SimdBackend::Avx2 => return add_row_avx2(values, row),
            SimdBackend::Scalar | SimdBackend::Neon => {}
        }
    }
    add_row_scalar(values, row);
}

#[inline]
fn sub_row(values: &mut [i16; HIDDEN_SIZE], row: &[i16; HIDDEN_SIZE]) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        match simd_backend() {
            SimdBackend::Avx512Icl | SimdBackend::Avx512 => return sub_row_avx512(values, row),
            SimdBackend::Avx2 => return sub_row_avx2(values, row),
            SimdBackend::Scalar | SimdBackend::Neon => {}
        }
    }
    sub_row_scalar(values, row);
}

#[inline]
fn replace_row(
    values: &mut [i16; HIDDEN_SIZE],
    old_row: &[i16; HIDDEN_SIZE],
    new_row: &[i16; HIDDEN_SIZE],
) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        match simd_backend() {
            SimdBackend::Avx512Icl | SimdBackend::Avx512 => {
                return replace_row_avx512(values, old_row, new_row);
            }
            SimdBackend::Avx2 => return replace_row_avx2(values, old_row, new_row),
            SimdBackend::Scalar | SimdBackend::Neon => {}
        }
    }
    for ((value, &old), &new) in values.iter_mut().zip(old_row).zip(new_row) {
        *value += new - old;
    }
}

#[inline]
fn add_row_scalar(values: &mut [i16; HIDDEN_SIZE], row: &[i16; HIDDEN_SIZE]) {
    for (value, &delta) in values.iter_mut().zip(row) {
        *value += delta;
    }
}

#[inline]
fn sub_row_scalar(values: &mut [i16; HIDDEN_SIZE], row: &[i16; HIDDEN_SIZE]) {
    for (value, &delta) in values.iter_mut().zip(row) {
        *value -= delta;
    }
}

#[cfg(target_arch = "x86_64")]
macro_rules! simd_row_op {
    ($name:ident, $feature:literal, $vector:ty, $load:ident, $store:ident, $op:ident, $lanes:expr) => {
        #[target_feature(enable = $feature)]
        unsafe fn $name(values: &mut [i16; HIDDEN_SIZE], row: &[i16; HIDDEN_SIZE]) {
            use std::arch::x86_64::*;
            unsafe {
                let mut i = 0;
                while i < HIDDEN_SIZE {
                    let lhs = $load(values.as_ptr().add(i) as *const $vector);
                    let rhs = $load(row.as_ptr().add(i) as *const $vector);
                    $store(values.as_mut_ptr().add(i) as *mut $vector, $op(lhs, rhs));
                    i += $lanes;
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
simd_row_op!(
    add_row_avx512,
    "avx512f,avx512bw",
    __m512i,
    _mm512_loadu_si512,
    _mm512_storeu_si512,
    _mm512_add_epi16,
    32
);
#[cfg(target_arch = "x86_64")]
simd_row_op!(
    sub_row_avx512,
    "avx512f,avx512bw",
    __m512i,
    _mm512_loadu_si512,
    _mm512_storeu_si512,
    _mm512_sub_epi16,
    32
);
#[cfg(target_arch = "x86_64")]
simd_row_op!(
    add_row_avx2,
    "avx2",
    __m256i,
    _mm256_loadu_si256,
    _mm256_storeu_si256,
    _mm256_add_epi16,
    16
);
#[cfg(target_arch = "x86_64")]
simd_row_op!(
    sub_row_avx2,
    "avx2",
    __m256i,
    _mm256_loadu_si256,
    _mm256_storeu_si256,
    _mm256_sub_epi16,
    16
);

#[cfg(target_arch = "x86_64")]
macro_rules! simd_replace_row {
    ($name:ident, $feature:literal, $vector:ty, $load:ident, $store:ident, $add:ident, $sub:ident, $lanes:expr) => {
        #[target_feature(enable = $feature)]
        unsafe fn $name(
            values: &mut [i16; HIDDEN_SIZE],
            old_row: &[i16; HIDDEN_SIZE],
            new_row: &[i16; HIDDEN_SIZE],
        ) {
            use std::arch::x86_64::*;
            unsafe {
                let mut i = 0;
                while i < HIDDEN_SIZE {
                    let value = $load(values.as_ptr().add(i) as *const $vector);
                    let old = $load(old_row.as_ptr().add(i) as *const $vector);
                    let new = $load(new_row.as_ptr().add(i) as *const $vector);
                    $store(
                        values.as_mut_ptr().add(i) as *mut $vector,
                        $add(value, $sub(new, old)),
                    );
                    i += $lanes;
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
simd_replace_row!(
    replace_row_avx512,
    "avx512f,avx512bw",
    __m512i,
    _mm512_loadu_si512,
    _mm512_storeu_si512,
    _mm512_add_epi16,
    _mm512_sub_epi16,
    32
);
#[cfg(target_arch = "x86_64")]
simd_replace_row!(
    replace_row_avx2,
    "avx2",
    __m256i,
    _mm256_loadu_si256,
    _mm256_storeu_si256,
    _mm256_add_epi16,
    _mm256_sub_epi16,
    16
);

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

    /// Inherit only usable parent values. Dirty perspectives will be rebuilt
    /// before evaluation, so their destination storage can remain untouched.
    pub fn copy_clean_from(&mut self, parent: &Self, invalidate: Option<Color>) {
        self.needs_refresh = parent.needs_refresh;
        if let Some(color) = invalidate {
            self.mark_refresh(color);
        }
        if !self.needs_refresh(Color::White) {
            self.white.copy_from_slice(&parent.white);
        }
        if !self.needs_refresh(Color::Black) {
            self.black.copy_from_slice(&parent.black);
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
                    add_row(&mut values, row);
                }
            }
        }

        for index in board_state_feature_indices(board, perspective) {
            let row = &net.ft_weights[index];
            add_row(&mut values, row);
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

    /// Apply castling/en-passant state changes to both clean perspectives.
    #[allow(clippy::too_many_arguments)]
    pub fn update_state(
        &mut self,
        net: &NnueNetwork,
        white_king: Square,
        black_king: Square,
        old_castling: CastlingRights,
        old_en_passant: Option<Square>,
        new_castling: CastlingRights,
        new_en_passant: Option<Square>,
    ) {
        for perspective in [Color::White, Color::Black] {
            if self.needs_refresh(perspective) {
                continue;
            }
            let old = state_feature_indices(
                perspective,
                white_king,
                black_king,
                old_castling,
                old_en_passant,
            );
            let new = state_feature_indices(
                perspective,
                white_king,
                black_king,
                new_castling,
                new_en_passant,
            );
            let values = match perspective {
                Color::White => &mut self.white,
                Color::Black => &mut self.black,
            };
            for (old_index, new_index) in old.into_iter().zip(new) {
                if old_index == new_index {
                    continue;
                }
                let old_row = &net.ft_weights[old_index];
                let new_row = &net.ft_weights[new_index];
                replace_row(values, old_row, new_row);
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
        if !self.needs_refresh(Color::White) {
            let idx = feature_index(Color::White, white_king, black_king, color, kind, sq);
            let row = &net.ft_weights[idx];
            sub_row(&mut self.white, row);
        }
        if !self.needs_refresh(Color::Black) {
            let idx = feature_index(Color::Black, white_king, black_king, color, kind, sq);
            let row = &net.ft_weights[idx];
            sub_row(&mut self.black, row);
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
            add_row(&mut self.white, row);
        }
        if !self.needs_refresh(Color::Black) {
            let idx = feature_index(Color::Black, white_king, black_king, color, kind, sq);
            let row = &net.ft_weights[idx];
            add_row(&mut self.black, row);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FEATURES_PER_BUCKET, NUM_BUCKETS, PIECE_FEATURES};

    #[test]
    fn inheriting_dirty_perspectives_remains_correct_after_refresh() {
        let board = Board::starting_position();
        let net = NnueNetwork::embedded();
        let mut parent = Accumulator::new();
        parent.refresh(&board, &net);
        let expected = parent.clone();
        parent.mark_refresh(Color::White);
        // Stale contents must never leak into a clean perspective.
        parent.white.fill(-123);
        let mut child = Accumulator::new();
        child.white.fill(456);
        child.black.fill(789);
        child.copy_clean_from(&parent, None);
        assert!(child.needs_refresh(Color::White));
        assert!(!child.needs_refresh(Color::Black));
        assert_eq!(child.black, expected.black);
        child.refresh_perspective(&board, &net, Color::White);
        assert_eq!(child.white, expected.white);

        // A king move can invalidate the other half while one is already dirty.
        child.copy_clean_from(&parent, Some(Color::Black));
        assert!(child.needs_refresh(Color::White));
        assert!(child.needs_refresh(Color::Black));
        child.refresh(&board, &net);
        assert_eq!(child.white, expected.white);
        assert_eq!(child.black, expected.black);

        child.copy_clean_from(&expected, None);
        assert!(!child.needs_refresh(Color::White));
        assert!(!child.needs_refresh(Color::Black));
    }

    #[test]
    fn dispatched_row_operations_match_scalar() {
        let original = std::array::from_fn(|i| (i as i16 % 97) - 48);
        let row = std::array::from_fn(|i| (i as i16 % 31) - 15);
        let replacement = std::array::from_fn(|i| (i as i16 % 43) - 21);

        let mut expected = original;
        add_row_scalar(&mut expected, &row);
        let mut actual = original;
        add_row(&mut actual, &row);
        assert_eq!(actual, expected);

        sub_row_scalar(&mut expected, &row);
        sub_row(&mut actual, &row);
        assert_eq!(actual, original);
        assert_eq!(actual, expected);

        for ((value, &old), &new) in expected.iter_mut().zip(&row).zip(&replacement) {
            *value += new - old;
        }
        replace_row(&mut actual, &row, &replacement);
        assert_eq!(actual, expected);
    }

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
        for perspective in [Color::White, Color::Black] {
            let values = match perspective {
                Color::White => &mut acc_inc.white,
                Color::Black => &mut acc_inc.black,
            };
            for index in board_state_feature_indices(&board, perspective) {
                add_row(values, &net.ft_weights[index]);
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
    fn incremental_state_change_matches_full_refresh_with_nonzero_state_weights() {
        let mut net = NnueNetwork::zeroed_for_test();
        for bucket in 0..NUM_BUCKETS {
            for slot in PIECE_FEATURES..FEATURES_PER_BUCKET {
                net.ft_weights[bucket * FEATURES_PER_BUCKET + slot]
                    .fill((slot - PIECE_FEATURES + 1) as i16);
            }
        }

        let before = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        let mut after = before.clone();
        after.castling = CastlingRights::NONE;
        after.en_passant = Some(Square::new(4, 5));

        let mut incremental = Accumulator::new();
        incremental.refresh(&before, &net);
        incremental.update_state(
            &net,
            before.king_square(Color::White),
            before.king_square(Color::Black),
            before.castling,
            before.en_passant,
            after.castling,
            after.en_passant,
        );

        let mut refreshed = Accumulator::new();
        refreshed.refresh(&after, &net);
        assert_eq!(incremental.white, refreshed.white);
        assert_eq!(incremental.black, refreshed.black);
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
