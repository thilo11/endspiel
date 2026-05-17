use chess_common::{Color, PieceKind, Square};

// FILE_FOLD maps file 0-7 → half_file 0-3, mirroring kingside onto queenside.
const FILE_FOLD: [usize; 8] = [0, 1, 2, 3, 3, 2, 1, 0];

/// Compute the feature index for a piece using king-relative (HalfKP) encoding.
///
/// Uses 32 fine-grained king buckets — one per half-board king square — with a
/// 704-feature base that merges both kings into a shared slot:
///
///   [0,   320) — friendly non-king pieces  (5 types × 64 sq)
///   [320, 384) — kings, both colors merged (64 sq)
///   [384, 704) — enemy non-king pieces     (5 types × 64 sq)
///
/// This matches the Bullet `ChessBucketsMergedKingsMirrored` feature mapping
/// with an identity bucket layout (bucket index = half-board king square).
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

    // 704-feature base encoding (merged kings):
    //   Kings of both colors share [320, 384) — offset 0, piece_kind.index()*64 = 320.
    //   Friendly non-king pieces use offset 0:   [0, 320).
    //   Enemy non-king pieces use offset 384:    [384, 704).
    let is_friendly = perspective == piece_color;
    let color_offset = if piece_kind == PieceKind::King || is_friendly { 0 } else { 384 };
    let base = color_offset + piece_kind.index() * 64 + (sq_idx ^ h_flip);

    bucket * 704 + base
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
        assert_eq!(idx_q / 704, idx_k / 704, "bucket should match for symmetric king positions");
        assert_ne!(idx_q, idx_k, "piece square should be mirrored");
    }

    #[test]
    fn kings_merged() {
        // Friendly and enemy kings at the same square should produce the same feature index
        let wk = Square::new(4, 0); // e1 (white king, perspective=White)
        let bk = Square::new(4, 7); // e8
        let sq = Square::new(3, 3); // d4

        let friendly_king_idx =
            feature_index(Color::White, wk, bk, Color::White, PieceKind::King, sq);
        let enemy_king_idx =
            feature_index(Color::White, wk, bk, Color::Black, PieceKind::King, sq);

        assert_eq!(
            friendly_king_idx, enemy_king_idx,
            "friendly and enemy kings should map to the same feature (merged kings)"
        );
    }
}
