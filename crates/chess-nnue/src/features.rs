use chess_common::{Board, CastlingRights, Color, PieceKind, Square};

use crate::{FEATURES_PER_BUCKET, PIECE_FEATURES};

// FILE_FOLD maps file 0-7 → half_file 0-3, mirroring kingside onto queenside.
const FILE_FOLD: [usize; 8] = [0, 1, 2, 3, 3, 2, 1, 0];

/// Compute the feature index for a piece using king-relative (HalfKP) encoding.
///
/// Uses 32 fine-grained king buckets — one per half-board king square — with a
/// 768-feature base that keeps the two king roles distinct:
///
///   [0,   320) — friendly non-king pieces  (5 types × 64 sq)
///   [320, 384) — friendly king              (64 sq)
///   [384, 448) — enemy king                 (64 sq)
///   [448, 768) — enemy non-king pieces     (5 types × 64 sq)
///
/// This is the mirrored Bullet `ChessBuckets` mapping with an identity bucket
/// layout (bucket index = half-board king square).
///
/// - `perspective`: which side's accumulator we're updating
/// - `white_king`, `black_king`: absolute king squares (a1=0, h8=63)
/// - `piece_color`, `piece_kind`, `sq`: the piece being indexed
///
/// Returns a value in `[0, INPUT_SIZE)`.
#[inline]
pub fn feature_index(
    perspective: Color,
    white_king: Square,
    black_king: Square,
    piece_color: Color,
    piece_kind: PieceKind,
    sq: Square,
) -> usize {
    // King square in perspective frame: white = absolute, black = rank-flipped (^56)
    let king_idx = match perspective {
        Color::White => white_king.index(),
        Color::Black => black_king.index() ^ 56,
    };

    // Horizontal flip: if the king is on the kingside (file > 3), mirror the whole board
    let h_flip: usize = if king_idx % 8 > 3 { 7 } else { 0 };

    // Bucket: one per half-board king square (0-31).
    // FILE_FOLD collapses file 4-7 onto 3-0, giving 32 unique (rank, half-file) positions.
    let bucket = (king_idx / 8) * 4 + FILE_FOLD[king_idx % 8];

    // Square in perspective frame (rank-flipped for Black)
    let sq_idx = if perspective == Color::White {
        sq.index()
    } else {
        sq.index() ^ 56
    };

    // 768-feature base encoding (unmerged kings):
    //   Friendly non-king pieces use offset 0:   [0, 320).
    //   Friendly king uses offset 0:              [320, 384).
    //   Enemy king uses offset 64:                [384, 448).
    //   Enemy non-king pieces use offset 128:     [448, 768).
    let is_friendly = perspective == piece_color;
    let color_offset = match (is_friendly, piece_kind == PieceKind::King) {
        (true, _) => 0,
        (false, true) => 64,
        (false, false) => 448,
    };
    let base = color_offset + piece_kind.index() * 64 + (sq_idx ^ h_flip);

    bucket * FEATURES_PER_BUCKET + base
}

const FRIENDLY_CASTLING_OFFSET: usize = PIECE_FEATURES;
const ENEMY_CASTLING_OFFSET: usize = FRIENDLY_CASTLING_OFFSET + 4;
const EN_PASSANT_OFFSET: usize = ENEMY_CASTLING_OFFSET + 4;

/// Three categorical state features for one accumulator perspective: friendly
/// castling rights, enemy castling rights, and the current en-passant file.
pub fn state_feature_indices(
    perspective: Color,
    white_king: Square,
    black_king: Square,
    castling: CastlingRights,
    en_passant: Option<Square>,
) -> [usize; 3] {
    let king_idx = match perspective {
        Color::White => white_king.index(),
        Color::Black => black_king.index() ^ 56,
    };
    let horizontal_flip = if king_idx % 8 > 3 { 7 } else { 0 };
    let bucket = (king_idx / 8) * 4 + FILE_FOLD[king_idx % 8];
    let bucket_base = bucket * FEATURES_PER_BUCKET;

    let (friendly, enemy) = match perspective {
        Color::White => (castling.0 & 0b0011, (castling.0 >> 2) & 0b0011),
        Color::Black => ((castling.0 >> 2) & 0b0011, castling.0 & 0b0011),
    };
    let mirror_rights = |rights: u8| -> usize {
        let rights = usize::from(rights);
        if horizontal_flip == 0 {
            rights
        } else {
            ((rights & 1) << 1) | ((rights & 2) >> 1)
        }
    };
    let ep_category = en_passant.map_or(0, |square| {
        usize::from(square.file() ^ horizontal_flip as u8) + 1
    });

    [
        bucket_base + FRIENDLY_CASTLING_OFFSET + mirror_rights(friendly),
        bucket_base + ENEMY_CASTLING_OFFSET + mirror_rights(enemy),
        bucket_base + EN_PASSANT_OFFSET + ep_category,
    ]
}

pub fn board_state_feature_indices(board: &Board, perspective: Color) -> [usize; 3] {
    state_feature_indices(
        perspective,
        board.king_square(Color::White),
        board.king_square(Color::Black),
        board.castling,
        board.en_passant,
    )
}

#[cfg(test)]
mod tests {
    use crate::INPUT_SIZE;

    use super::*;

    #[test]
    fn feature_index_bounds() {
        let wk = Square::new(4, 0); // e1
        let bk = Square::new(4, 7); // e8
        for &perspective in &[Color::White, Color::Black] {
            for &piece_color in &[Color::White, Color::Black] {
                for &kind in &PieceKind::ALL {
                    for sq_idx in 0..64 {
                        let idx =
                            feature_index(perspective, wk, bk, piece_color, kind, Square(sq_idx));
                        assert!(idx < INPUT_SIZE, "feature index {idx} out of bounds");
                    }
                }
            }
        }
    }

    #[test]
    fn state_features_distinguish_lost_castling_rights() {
        let with_rights = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        let without_rights = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1").unwrap();
        assert_ne!(
            board_state_feature_indices(&with_rights, Color::White),
            board_state_feature_indices(&without_rights, Color::White)
        );
    }

    #[test]
    fn state_feature_indices_stay_in_bounds() {
        let board = Board::starting_position();
        for perspective in [Color::White, Color::Black] {
            assert!(
                board_state_feature_indices(&board, perspective)
                    .into_iter()
                    .all(|index| index < INPUT_SIZE)
            );
        }
    }

    #[test]
    fn mirrored_king_same_bucket() {
        // King on a1 (queenside) and h1 (kingside) should give the same bucket
        let wk_queenside = Square::new(0, 0); // a1
        let wk_kingside = Square::new(7, 0); // h1
        let bk = Square::new(4, 7); // e8
        let pawn_sq = Square::new(3, 3); // d4

        let idx_q = feature_index(
            Color::White,
            wk_queenside,
            bk,
            Color::White,
            PieceKind::Pawn,
            pawn_sq,
        );
        let idx_k = feature_index(
            Color::White,
            wk_kingside,
            bk,
            Color::White,
            PieceKind::Pawn,
            pawn_sq,
        );
        // Same bucket, but file mirrored: d4 (file 3) → e4 (file 4) when king on h1
        assert_eq!(
            idx_q / 768,
            idx_k / 768,
            "bucket should match for symmetric king positions"
        );
        assert_ne!(idx_q, idx_k, "piece square should be mirrored");
    }

    #[test]
    fn kings_use_distinct_feature_planes() {
        // Friendly and enemy kings at the same square must be distinguishable.
        let wk = Square::new(4, 0); // e1 (white king, perspective=White)
        let bk = Square::new(4, 7); // e8
        let sq = Square::new(3, 3); // d4

        let friendly_king_idx =
            feature_index(Color::White, wk, bk, Color::White, PieceKind::King, sq);
        let enemy_king_idx = feature_index(Color::White, wk, bk, Color::Black, PieceKind::King, sq);

        assert_ne!(
            friendly_king_idx, enemy_king_idx,
            "friendly and enemy kings should map to distinct feature planes"
        );
    }
}
