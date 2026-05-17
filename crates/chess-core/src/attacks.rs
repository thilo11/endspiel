use chess_common::{Bitboard, Board, Color, PieceKind, Square};

// ─── Precomputed attack tables ───────────────────────────────────────────────

/// Knight attacks for each square.
static KNIGHT_ATTACKS: [Bitboard; 64] = {
    let mut table = [Bitboard::EMPTY; 64];
    let mut sq = 0u8;
    while sq < 64 {
        let bb = Bitboard(1u64 << sq);
        let mut attacks = 0u64;

        // All 8 knight jumps with file-wrapping guards
        // NNE: +17, file must not wrap from H to A
        attacks |= (bb.0 << 17) & !Bitboard::FILE_A.0;
        // NNW: +15, file must not wrap from A to H
        attacks |= (bb.0 << 15) & !Bitboard::FILE_H.0;
        // NEE: +10, must not wrap from G/H to A/B
        attacks |= (bb.0 << 10) & !(Bitboard::FILE_A.0 | Bitboard::FILE_B.0);
        // NWW: +6, must not wrap from A/B to G/H
        attacks |= (bb.0 << 6) & !(Bitboard::FILE_G.0 | Bitboard::FILE_H.0);
        // SSE: -15, file must not wrap from H to A
        attacks |= (bb.0 >> 15) & !Bitboard::FILE_A.0;
        // SSW: -17, file must not wrap from A to H
        attacks |= (bb.0 >> 17) & !Bitboard::FILE_H.0;
        // SEE: -6, must not wrap from G/H to A/B
        attacks |= (bb.0 >> 6) & !(Bitboard::FILE_A.0 | Bitboard::FILE_B.0);
        // SWW: -10, must not wrap from A/B to G/H
        attacks |= (bb.0 >> 10) & !(Bitboard::FILE_G.0 | Bitboard::FILE_H.0);

        table[sq as usize] = Bitboard(attacks);
        sq += 1;
    }
    table
};

/// King attacks for each square.
static KING_ATTACKS: [Bitboard; 64] = {
    let mut table = [Bitboard::EMPTY; 64];
    let mut sq = 0u8;
    while sq < 64 {
        let bb = Bitboard(1u64 << sq);
        let mut attacks = 0u64;

        attacks |= bb.0 << 8; // N
        attacks |= bb.0 >> 8; // S
        attacks |= (bb.0 << 1) & !Bitboard::FILE_A.0; // E
        attacks |= (bb.0 >> 1) & !Bitboard::FILE_H.0; // W
        attacks |= (bb.0 << 9) & !Bitboard::FILE_A.0; // NE
        attacks |= (bb.0 << 7) & !Bitboard::FILE_H.0; // NW
        attacks |= (bb.0 >> 7) & !Bitboard::FILE_A.0; // SE
        attacks |= (bb.0 >> 9) & !Bitboard::FILE_H.0; // SW

        table[sq as usize] = Bitboard(attacks);
        sq += 1;
    }
    table
};

/// White pawn attacks for each square.
static WHITE_PAWN_ATTACKS: [Bitboard; 64] = {
    let mut table = [Bitboard::EMPTY; 64];
    let mut sq = 0u8;
    while sq < 64 {
        let bb = Bitboard(1u64 << sq);
        let mut attacks = 0u64;
        attacks |= (bb.0 << 9) & !Bitboard::FILE_A.0; // NE
        attacks |= (bb.0 << 7) & !Bitboard::FILE_H.0; // NW
        table[sq as usize] = Bitboard(attacks);
        sq += 1;
    }
    table
};

/// Black pawn attacks for each square.
static BLACK_PAWN_ATTACKS: [Bitboard; 64] = {
    let mut table = [Bitboard::EMPTY; 64];
    let mut sq = 0u8;
    while sq < 64 {
        let bb = Bitboard(1u64 << sq);
        let mut attacks = 0u64;
        attacks |= (bb.0 >> 7) & !Bitboard::FILE_A.0; // SE
        attacks |= (bb.0 >> 9) & !Bitboard::FILE_H.0; // SW
        table[sq as usize] = Bitboard(attacks);
        sq += 1;
    }
    table
};

// ─── Public accessors ────────────────────────────────────────────────────────

#[inline]
pub fn knight_attacks(sq: Square) -> Bitboard {
    KNIGHT_ATTACKS[sq.index()]
}

#[inline]
pub fn king_attacks(sq: Square) -> Bitboard {
    KING_ATTACKS[sq.index()]
}

#[inline]
pub fn pawn_attacks(sq: Square, color: Color) -> Bitboard {
    match color {
        Color::White => WHITE_PAWN_ATTACKS[sq.index()],
        Color::Black => BLACK_PAWN_ATTACKS[sq.index()],
    }
}

// ─── Sliding piece attacks (magic bitboard lookups) ─────────────────────────

/// Compute bishop attacks from `sq` given `occupancy` (all pieces).
#[inline]
pub fn bishop_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
    crate::magics::bishop_attacks(sq, occupancy)
}

/// Compute rook attacks from `sq` given `occupancy` (all pieces).
#[inline]
pub fn rook_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
    crate::magics::rook_attacks(sq, occupancy)
}

/// Compute queen attacks (bishop + rook).
#[inline]
pub fn queen_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
    Bitboard(bishop_attacks(sq, occupancy).0 | rook_attacks(sq, occupancy).0)
}

// ─── Square attack detection ─────────────────────────────────────────────────

/// Check if `sq` is attacked by any piece of color `by_color`.
pub fn is_square_attacked(board: &Board, sq: Square, by_color: Color) -> bool {
    let ci = by_color.index();
    let occ = board.all_occupancy();

    // Pawn attacks: check if any pawn of `by_color` attacks `sq`.
    // A pawn of `by_color` attacks `sq` iff `sq` is in the pawn-attack set of that pawn.
    // Equivalently, we check if any `by_color` pawn is in the "reverse pawn attack" from `sq`.
    // The reverse pawn attack from sq (for by_color attacking) uses the *opposite* color direction.
    let pawn_attackers = pawn_attacks(sq, by_color.opposite()) & board.pieces[ci][PieceKind::Pawn.index()];
    if !pawn_attackers.is_empty() {
        return true;
    }

    // Knight attacks
    let knight_attackers = knight_attacks(sq) & board.pieces[ci][PieceKind::Knight.index()];
    if !knight_attackers.is_empty() {
        return true;
    }

    // King attacks
    let king_attackers = king_attacks(sq) & board.pieces[ci][PieceKind::King.index()];
    if !king_attackers.is_empty() {
        return true;
    }

    // Bishop/Queen (diagonal attacks)
    let diag_attackers = bishop_attacks(sq, occ)
        & (board.pieces[ci][PieceKind::Bishop.index()] | board.pieces[ci][PieceKind::Queen.index()]);
    if !diag_attackers.is_empty() {
        return true;
    }

    // Rook/Queen (straight attacks)
    let straight_attackers = rook_attacks(sq, occ)
        & (board.pieces[ci][PieceKind::Rook.index()] | board.pieces[ci][PieceKind::Queen.index()]);
    if !straight_attackers.is_empty() {
        return true;
    }

    false
}

/// Return a bitboard of all pieces of `by_color` that attack `sq`.
pub fn attackers_of(board: &Board, sq: Square, by_color: Color) -> Bitboard {
    let ci = by_color.index();
    let occ = board.all_occupancy();

    let pawns = pawn_attacks(sq, by_color.opposite()) & board.pieces[ci][PieceKind::Pawn.index()];
    let knights = knight_attacks(sq) & board.pieces[ci][PieceKind::Knight.index()];
    let king = king_attacks(sq) & board.pieces[ci][PieceKind::King.index()];
    let bishops_queens = bishop_attacks(sq, occ)
        & (board.pieces[ci][PieceKind::Bishop.index()] | board.pieces[ci][PieceKind::Queen.index()]);
    let rooks_queens = rook_attacks(sq, occ)
        & (board.pieces[ci][PieceKind::Rook.index()] | board.pieces[ci][PieceKind::Queen.index()]);

    pawns | knights | king | bishops_queens | rooks_queens
}

/// Return a bitboard of ALL pieces (both colors) that attack `sq`, using a
/// custom `occupancy` bitboard for sliding piece rays. This is needed by the
/// SEE algorithm to discover X-ray attackers after captures.
pub fn all_attackers_of(board: &Board, sq: Square, occupancy: Bitboard) -> Bitboard {
    let white_pawns = board.pieces[Color::White.index()][PieceKind::Pawn.index()];
    let black_pawns = board.pieces[Color::Black.index()][PieceKind::Pawn.index()];
    let knights = board.pieces[0][PieceKind::Knight.index()]
        | board.pieces[1][PieceKind::Knight.index()];
    let bishops = board.pieces[0][PieceKind::Bishop.index()]
        | board.pieces[1][PieceKind::Bishop.index()];
    let rooks = board.pieces[0][PieceKind::Rook.index()]
        | board.pieces[1][PieceKind::Rook.index()];
    let queens = board.pieces[0][PieceKind::Queen.index()]
        | board.pieces[1][PieceKind::Queen.index()];
    let kings = board.pieces[0][PieceKind::King.index()]
        | board.pieces[1][PieceKind::King.index()];

    // Pawn attacks use reverse direction lookup
    (pawn_attacks(sq, Color::Black) & white_pawns)
        | (pawn_attacks(sq, Color::White) & black_pawns)
        | (knight_attacks(sq) & knights)
        | (king_attacks(sq) & kings)
        | (bishop_attacks(sq, occupancy) & (bishops | queens))
        | (rook_attacks(sq, occupancy) & (rooks | queens))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knight_attacks_center() {
        // Knight on e4 should attack 8 squares
        let sq = Square::from_algebraic("e4").unwrap();
        let attacks = knight_attacks(sq);
        assert_eq!(attacks.count(), 8);
        assert!(attacks.is_set(Square::from_algebraic("d6").unwrap()));
        assert!(attacks.is_set(Square::from_algebraic("f6").unwrap()));
        assert!(attacks.is_set(Square::from_algebraic("c5").unwrap()));
        assert!(attacks.is_set(Square::from_algebraic("g5").unwrap()));
        assert!(attacks.is_set(Square::from_algebraic("c3").unwrap()));
        assert!(attacks.is_set(Square::from_algebraic("g3").unwrap()));
        assert!(attacks.is_set(Square::from_algebraic("d2").unwrap()));
        assert!(attacks.is_set(Square::from_algebraic("f2").unwrap()));
    }

    #[test]
    fn test_knight_attacks_corner() {
        // Knight on a1 should attack 2 squares
        let attacks = knight_attacks(Square::A1);
        assert_eq!(attacks.count(), 2);
        assert!(attacks.is_set(Square::from_algebraic("b3").unwrap()));
        assert!(attacks.is_set(Square::from_algebraic("c2").unwrap()));
    }

    #[test]
    fn test_king_attacks_center() {
        let sq = Square::from_algebraic("e4").unwrap();
        let attacks = king_attacks(sq);
        assert_eq!(attacks.count(), 8);
    }

    #[test]
    fn test_king_attacks_corner() {
        let attacks = king_attacks(Square::A1);
        assert_eq!(attacks.count(), 3);
    }

    #[test]
    fn test_pawn_attacks_white() {
        let sq = Square::from_algebraic("e4").unwrap();
        let attacks = pawn_attacks(sq, Color::White);
        assert_eq!(attacks.count(), 2);
        assert!(attacks.is_set(Square::from_algebraic("d5").unwrap()));
        assert!(attacks.is_set(Square::from_algebraic("f5").unwrap()));
    }

    #[test]
    fn test_pawn_attacks_black() {
        let sq = Square::from_algebraic("e4").unwrap();
        let attacks = pawn_attacks(sq, Color::Black);
        assert_eq!(attacks.count(), 2);
        assert!(attacks.is_set(Square::from_algebraic("d3").unwrap()));
        assert!(attacks.is_set(Square::from_algebraic("f3").unwrap()));
    }

    #[test]
    fn test_rook_attacks_empty_board() {
        let sq = Square::from_algebraic("e4").unwrap();
        let attacks = rook_attacks(sq, Bitboard::EMPTY);
        assert_eq!(attacks.count(), 14); // 7 horizontal + 7 vertical
    }

    #[test]
    fn test_bishop_attacks_empty_board() {
        let sq = Square::from_algebraic("e4").unwrap();
        let attacks = bishop_attacks(sq, Bitboard::EMPTY);
        assert_eq!(attacks.count(), 13);
    }

    #[test]
    fn test_rook_attacks_blocked() {
        let sq = Square::from_algebraic("e4").unwrap();
        // Place a blocker on e6 (north of e4)
        let blocker = Square::from_algebraic("e6").unwrap().bitboard();
        let attacks = rook_attacks(sq, blocker);
        // Should include e5 and e6 (blocker) but not e7, e8
        assert!(attacks.is_set(Square::from_algebraic("e5").unwrap()));
        assert!(attacks.is_set(Square::from_algebraic("e6").unwrap()));
        assert!(!attacks.is_set(Square::from_algebraic("e7").unwrap()));
    }

    #[test]
    fn test_is_square_attacked_starting_position() {
        let board = Board::starting_position();
        // e3 is attacked by white pawns d2 and f2, and white king e1
        assert!(is_square_attacked(&board, Square::from_algebraic("e3").unwrap(), Color::White));
        // e6 is attacked by black pawns d7 and f7, and black king e8
        assert!(is_square_attacked(&board, Square::from_algebraic("e6").unwrap(), Color::Black));
        // e4 is not attacked by white in starting position
        assert!(!is_square_attacked(&board, Square::from_algebraic("e4").unwrap(), Color::White));
    }
}
