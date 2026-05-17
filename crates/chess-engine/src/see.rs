//! Static Exchange Evaluation (SEE) using the swap algorithm.
//!
//! Determines whether a capture sequence on a square results in a material
//! gain at or above a given threshold, accounting for X-ray attacks through
//! captured pieces.

use chess_common::{Board, Color, Move, PieceKind, Square, Bitboard};
use chess_common::moves::MoveFlag;

/// SEE piece values (must be consistent with eval, but king gets a large sentinel).
const SEE_VALUES: [i32; 6] = [
    100,   // Pawn
    320,   // Knight
    330,   // Bishop
    500,   // Rook
    900,   // Queen
    20000, // King (sentinel — can't really be captured)
];

#[inline]
fn see_value(kind: PieceKind) -> i32 {
    SEE_VALUES[kind.index()]
}

/// Returns true if the SEE value of move `m` is >= `threshold`.
///
/// This uses the negamax swap algorithm: iteratively find the least valuable
/// attacker of each side, updating occupancy for X-ray discovery.
pub fn see_ge(board: &Board, m: Move, threshold: i32) -> bool {
    let from = m.from_sq();
    let to = m.to_sq();
    let flag = m.flag();

    // Determine the initial material gain of the capture.
    let mut swap = if flag == MoveFlag::EnPassant {
        SEE_VALUES[PieceKind::Pawn.index()]
    } else if flag.is_promotion() {
        let promo_val = see_value(flag.promotion_piece().unwrap());
        let victim_val = board
            .piece_at(to)
            .map(|p| see_value(p.kind))
            .unwrap_or(0);
        // Gain = victim + (promoted piece - pawn)
        victim_val + promo_val - SEE_VALUES[PieceKind::Pawn.index()]
    } else {
        board
            .piece_at(to)
            .map(|p| see_value(p.kind))
            .unwrap_or(0)
    };

    // Quick test: can we meet the threshold even in the best case?
    swap -= threshold;
    if swap < 0 {
        return false;
    }

    // Value of the piece we're putting at risk (the "next victim").
    let next_victim = if flag.is_promotion() {
        see_value(flag.promotion_piece().unwrap())
    } else {
        board
            .piece_at(from)
            .map(|p| see_value(p.kind))
            .unwrap_or(0)
    };

    // Even if we lose our piece, are we still above threshold?
    swap -= next_victim;
    if swap >= 0 {
        return true;
    }

    // -----------------------------------------------------------------------
    // Full iterative swap
    // -----------------------------------------------------------------------
    let mut occ = board.all_occupancy();
    occ = Bitboard(occ.0 ^ from.bitboard().0); // remove the initial attacker

    // For en passant, also remove the captured pawn.
    if flag == MoveFlag::EnPassant {
        let ep_sq = Square::new(to.file(), from.rank());
        occ = Bitboard(occ.0 ^ ep_sq.bitboard().0);
    }

    // Precompute combined bitboards for X-ray updates.
    let diag_sliders = board.pieces[0][PieceKind::Bishop.index()]
        | board.pieces[1][PieceKind::Bishop.index()]
        | board.pieces[0][PieceKind::Queen.index()]
        | board.pieces[1][PieceKind::Queen.index()];
    let orth_sliders = board.pieces[0][PieceKind::Rook.index()]
        | board.pieces[1][PieceKind::Rook.index()]
        | board.pieces[0][PieceKind::Queen.index()]
        | board.pieces[1][PieceKind::Queen.index()];

    let mut attackers = chess_core::attacks::all_attackers_of(board, to, occ) & occ;
    let mut stm = board.side_to_move.opposite(); // opponent recaptures first

    loop {
        let stm_attackers = attackers & board.occupancy[stm.index()];
        if stm_attackers.is_empty() {
            break;
        }

        // Find the least valuable attacker.
        let (attacker_sq, attacker_val) = least_valuable_attacker(board, stm_attackers, stm);

        // Negamax: flip perspective and subtract the new victim.
        swap = -swap - 1 - attacker_val;

        // Remove the attacker from occupancy (enables X-ray discovery).
        occ = Bitboard(occ.0 ^ attacker_sq.bitboard().0);

        // Discover new sliding attackers that were behind the removed piece.
        attackers |= chess_core::attacks::bishop_attacks(to, occ) & diag_sliders;
        attackers |= chess_core::attacks::rook_attacks(to, occ) & orth_sliders;
        attackers &= occ;

        stm = stm.opposite();

        if swap >= 0 {
            // King cannot capture if opponent still has attackers on this square.
            if attacker_val == SEE_VALUES[PieceKind::King.index()]
                && !(attackers & board.occupancy[stm.opposite().index()]).is_empty()
            {
                stm = stm.opposite();
            }
            break;
        }
    }

    // The side that cannot continue (or chose to stop because swap >= 0)
    // determines the result. If it's our opponent's turn (they ran out), we win.
    stm != board.side_to_move
}

/// Find the least valuable attacker among `attackers` for `color`.
/// Returns (square, see_value).
fn least_valuable_attacker(board: &Board, attackers: Bitboard, color: Color) -> (Square, i32) {
    let ci = color.index();
    const PIECE_ORDER: [PieceKind; 6] = [
        PieceKind::Pawn,
        PieceKind::Knight,
        PieceKind::Bishop,
        PieceKind::Rook,
        PieceKind::Queen,
        PieceKind::King,
    ];
    for &kind in &PIECE_ORDER {
        let piece_bb = attackers & board.pieces[ci][kind.index()];
        if !piece_bb.is_empty() {
            let sq = piece_bb.lsb().unwrap();
            return (sq, SEE_VALUES[kind.index()]);
        }
    }
    unreachable!("least_valuable_attacker called with empty attackers")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn board(fen: &str) -> Board {
        Board::from_fen(fen).unwrap()
    }

    fn uci_move(b: &Board, uci: &str) -> Move {
        let parsed = Move::from_uci(uci).unwrap();
        chess_core::validate::find_legal_move(b, parsed).unwrap()
    }

    #[test]
    fn test_see_winning_capture() {
        // Queen captures undefended pawn: SEE = +100
        let b = board("4k3/8/8/4p3/8/8/8/Q3K3 w - - 0 1");
        let m = uci_move(&b, "a1e5");
        assert!(see_ge(&b, m, 0));
        assert!(see_ge(&b, m, 100));
        assert!(!see_ge(&b, m, 101));
    }

    #[test]
    fn test_see_equal_capture() {
        // Knight takes knight defended by knight: NxN NxN → SEE = 0
        let b = board("4k3/8/5n2/3n4/8/4N3/8/4K3 w - - 0 1");
        let m = uci_move(&b, "e3d5");
        assert!(see_ge(&b, m, 0));   // NxN NxN = 0, which is >= 0
        assert!(!see_ge(&b, m, 1));  // NxN NxN = 0, which is NOT >= 1
    }

    #[test]
    fn test_see_losing_capture() {
        // Queen captures pawn defended by pawn: SEE = 100 - 900 = -800
        let b = board("4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1");
        let m = uci_move(&b, "d1d5");
        // QxPd5, then PxQ from e6: SEE = 100 - 900 = -800
        assert!(!see_ge(&b, m, 0));
        assert!(!see_ge(&b, m, -799));
        assert!(see_ge(&b, m, -800));
    }

    #[test]
    fn test_see_xray_discovery() {
        // Rook behind rook: after first rook captures, second rook backs it up
        // White: Rooks on a1 and a2, Black: Rook on a8
        // RxR on a8, then nothing defends. SEE = 500
        let b = board("r3k3/8/8/8/8/8/R7/R3K3 w - - 0 1");
        let m = uci_move(&b, "a2a8");
        // RxR, black has nothing. But wait, after Ra2xa8, Ra1 discovers attack on a8.
        // Actually Ra2xa8 means white rook on a2 captures Ra8. Then if black had
        // another piece, white Ra1 would x-ray. But black has nothing, so SEE = 500.
        assert!(see_ge(&b, m, 0));
        assert!(see_ge(&b, m, 500));
    }

    #[test]
    fn test_see_non_capture_always_true() {
        let b = board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
        let m = uci_move(&b, "e2e4");
        // Non-captures: threshold 0 should be true (gain = 0 >= 0)
        assert!(see_ge(&b, m, 0));
        assert!(!see_ge(&b, m, 1)); // gain = 0 < 1
    }
}
