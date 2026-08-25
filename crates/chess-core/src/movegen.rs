use chess_common::moves::{Move, MoveFlag, MoveList};
use chess_common::{Bitboard, Board, CastlingRights, Color, Piece, PieceKind, Square};

use crate::attacks::{
    bishop_attacks, is_square_attacked, king_attacks, knight_attacks, queen_attacks, rook_attacks,
};

/// Generate all legal moves for the current position.
pub fn generate_legal_moves(board: &Board) -> MoveList {
    let mut legal = MoveList::new();
    let pseudo = generate_pseudo_legal_moves(board);
    let us = board.side_to_move;

    for &m in pseudo.iter() {
        if is_legal(board, m, us) {
            legal.push(m);
        }
    }

    log::trace!("Generated {} legal moves", legal.len());
    legal
}

/// Check if a pseudo-legal move is legal (king is not left in check).
fn is_legal(board: &Board, m: Move, us: Color) -> bool {
    let them = us.opposite();

    // For castling, we already check intermediate squares in generation,
    // but we still need to verify the king doesn't end in check.
    // The pseudo-legal generator already ensures:
    //   - no pieces between king and rook
    //   - king is not currently in check
    //   - king does not pass through attacked squares
    // So castling moves are already fully validated.

    // Make the move on a copy and check if our king is in check.
    let mut test_board = board.clone();
    let prev_castling = test_board.castling;
    let prev_ep = test_board.en_passant;
    let prev_halfmove = test_board.halfmove_clock;
    let captured = test_board.make_move(m);

    let king_sq = test_board.king_square(us);
    let in_check = is_square_attacked(&test_board, king_sq, them);

    // Unmake is not strictly needed since we cloned, but let's not bother.
    let _ = (captured, prev_castling, prev_ep, prev_halfmove);

    !in_check
}

/// Generate all pseudo-legal moves for the current position.
/// These moves may leave the king in check (caller must filter).
pub fn generate_pseudo_legal_moves(board: &Board) -> MoveList {
    let mut moves = MoveList::new();
    let us = board.side_to_move;
    let ci = us.index();
    let our_occ = board.occupancy[ci];
    let their_occ = board.occupancy[us.opposite().index()];
    let all_occ = board.all_occupancy();

    generate_pawn_moves(board, us, our_occ, their_occ, all_occ, &mut moves);
    generate_knight_moves(board, us, our_occ, &mut moves);
    generate_bishop_moves(board, us, our_occ, their_occ, all_occ, &mut moves);
    generate_rook_moves(board, us, our_occ, their_occ, all_occ, &mut moves);
    generate_queen_moves(board, us, our_occ, their_occ, all_occ, &mut moves);
    generate_king_moves(board, us, our_occ, their_occ, all_occ, &mut moves);
    generate_castling_moves(board, us, all_occ, &mut moves);

    log::trace!("Generated {} pseudo-legal moves", moves.len());
    moves
}

fn generate_pawn_moves(
    board: &Board,
    us: Color,
    _our_occ: Bitboard,
    their_occ: Bitboard,
    all_occ: Bitboard,
    moves: &mut MoveList,
) {
    let ci = us.index();
    let pawns = board.pieces[ci][PieceKind::Pawn.index()];

    let (push_dir, start_rank, promo_rank): (i8, Bitboard, Bitboard) = match us {
        Color::White => (8, Bitboard::RANK_2, Bitboard::RANK_7),
        Color::Black => (-8, Bitboard::RANK_7, Bitboard::RANK_2),
    };

    let empty = !all_occ;

    // ── Single pawn push ──
    let single_push = match us {
        Color::White => pawns.north(),
        Color::Black => pawns.south(),
    } & empty;

    for to in single_push.iter() {
        let from = Square((to.0 as i8 - push_dir) as u8);
        if promo_rank.is_set(from) {
            // Promotion
            moves.push(Move::new(from, to, MoveFlag::PromoteQueen));
            moves.push(Move::new(from, to, MoveFlag::PromoteRook));
            moves.push(Move::new(from, to, MoveFlag::PromoteBishop));
            moves.push(Move::new(from, to, MoveFlag::PromoteKnight));
        } else {
            moves.push(Move::new(from, to, MoveFlag::Normal));
        }
    }

    // ── Double pawn push ──
    let double_push_candidates = match us {
        Color::White => (pawns & start_rank).north() & empty,
        Color::Black => (pawns & start_rank).south() & empty,
    };
    let double_push = match us {
        Color::White => double_push_candidates.north() & empty,
        Color::Black => double_push_candidates.south() & empty,
    };

    for to in double_push.iter() {
        let from = Square((to.0 as i8 - 2 * push_dir) as u8);
        moves.push(Move::new(from, to, MoveFlag::DoublePawnPush));
    }

    // ── Pawn captures ──
    let (attack_left, attack_right) = match us {
        Color::White => (pawns.north_west(), pawns.north_east()),
        Color::Black => (pawns.south_east(), pawns.south_west()),
    };

    let (left_delta, right_delta): (i8, i8) = match us {
        Color::White => (7, 9),
        Color::Black => (-7, -9),
    };

    // Left captures (NW for white, SE for black)
    for to in (attack_left & their_occ).iter() {
        let from = Square((to.0 as i8 - left_delta) as u8);
        if promo_rank.is_set(from) {
            moves.push(Move::new(from, to, MoveFlag::PromoteCaptureQueen));
            moves.push(Move::new(from, to, MoveFlag::PromoteCaptureRook));
            moves.push(Move::new(from, to, MoveFlag::PromoteCaptureBishop));
            moves.push(Move::new(from, to, MoveFlag::PromoteCaptureKnight));
        } else {
            moves.push(Move::new(from, to, MoveFlag::Capture));
        }
    }

    // Right captures (NE for white, SW for black)
    for to in (attack_right & their_occ).iter() {
        let from = Square((to.0 as i8 - right_delta) as u8);
        if promo_rank.is_set(from) {
            moves.push(Move::new(from, to, MoveFlag::PromoteCaptureQueen));
            moves.push(Move::new(from, to, MoveFlag::PromoteCaptureRook));
            moves.push(Move::new(from, to, MoveFlag::PromoteCaptureBishop));
            moves.push(Move::new(from, to, MoveFlag::PromoteCaptureKnight));
        } else {
            moves.push(Move::new(from, to, MoveFlag::Capture));
        }
    }

    // ── En passant ──
    if let Some(ep_sq) = board.en_passant {
        let ep_bb = ep_sq.bitboard();
        if !(attack_left & ep_bb).is_empty() {
            let from = Square((ep_sq.0 as i8 - left_delta) as u8);
            moves.push(Move::new(from, ep_sq, MoveFlag::EnPassant));
        }
        if !(attack_right & ep_bb).is_empty() {
            let from = Square((ep_sq.0 as i8 - right_delta) as u8);
            moves.push(Move::new(from, ep_sq, MoveFlag::EnPassant));
        }
    }
}

fn generate_knight_moves(board: &Board, us: Color, our_occ: Bitboard, moves: &mut MoveList) {
    let ci = us.index();
    let knights = board.pieces[ci][PieceKind::Knight.index()];
    let their_occ = board.occupancy[us.opposite().index()];

    for from in knights.iter() {
        let attacks = knight_attacks(from) & !our_occ;
        for to in attacks.iter() {
            if their_occ.is_set(to) {
                moves.push(Move::new(from, to, MoveFlag::Capture));
            } else {
                moves.push(Move::new(from, to, MoveFlag::Normal));
            }
        }
    }
}

fn generate_bishop_moves(
    board: &Board,
    us: Color,
    our_occ: Bitboard,
    their_occ: Bitboard,
    all_occ: Bitboard,
    moves: &mut MoveList,
) {
    let ci = us.index();
    let bishops = board.pieces[ci][PieceKind::Bishop.index()];

    for from in bishops.iter() {
        let attacks = bishop_attacks(from, all_occ) & !our_occ;
        for to in attacks.iter() {
            if their_occ.is_set(to) {
                moves.push(Move::new(from, to, MoveFlag::Capture));
            } else {
                moves.push(Move::new(from, to, MoveFlag::Normal));
            }
        }
    }
}

fn generate_rook_moves(
    board: &Board,
    us: Color,
    our_occ: Bitboard,
    their_occ: Bitboard,
    all_occ: Bitboard,
    moves: &mut MoveList,
) {
    let ci = us.index();
    let rooks = board.pieces[ci][PieceKind::Rook.index()];

    for from in rooks.iter() {
        let attacks = rook_attacks(from, all_occ) & !our_occ;
        for to in attacks.iter() {
            if their_occ.is_set(to) {
                moves.push(Move::new(from, to, MoveFlag::Capture));
            } else {
                moves.push(Move::new(from, to, MoveFlag::Normal));
            }
        }
    }
}

fn generate_queen_moves(
    board: &Board,
    us: Color,
    our_occ: Bitboard,
    their_occ: Bitboard,
    all_occ: Bitboard,
    moves: &mut MoveList,
) {
    let ci = us.index();
    let queens = board.pieces[ci][PieceKind::Queen.index()];

    for from in queens.iter() {
        let attacks = queen_attacks(from, all_occ) & !our_occ;
        for to in attacks.iter() {
            if their_occ.is_set(to) {
                moves.push(Move::new(from, to, MoveFlag::Capture));
            } else {
                moves.push(Move::new(from, to, MoveFlag::Normal));
            }
        }
    }
}

fn generate_king_moves(
    board: &Board,
    us: Color,
    our_occ: Bitboard,
    their_occ: Bitboard,
    _all_occ: Bitboard,
    moves: &mut MoveList,
) {
    let king_sq = board.king_square(us);
    let attacks = king_attacks(king_sq) & !our_occ;

    for to in attacks.iter() {
        if their_occ.is_set(to) {
            moves.push(Move::new(king_sq, to, MoveFlag::Capture));
        } else {
            moves.push(Move::new(king_sq, to, MoveFlag::Normal));
        }
    }
}

fn generate_castling_moves(board: &Board, us: Color, all_occ: Bitboard, moves: &mut MoveList) {
    let them = us.opposite();
    let king_from = board.king_square(us);

    // Cannot castle if in check. Chess960 also requires the king on the back rank.
    if is_square_attacked(board, king_from, them) {
        return;
    }
    let back = match us {
        Color::White => 0,
        Color::Black => 7,
    };
    if king_from.rank() != back {
        return;
    }

    for (kingside, flag, move_flag) in [
        (
            true,
            match us {
                Color::White => CastlingRights::WHITE_KINGSIDE,
                Color::Black => CastlingRights::BLACK_KINGSIDE,
            },
            MoveFlag::KingsideCastle,
        ),
        (
            false,
            match us {
                Color::White => CastlingRights::WHITE_QUEENSIDE,
                Color::Black => CastlingRights::BLACK_QUEENSIDE,
            },
            MoveFlag::QueensideCastle,
        ),
    ] {
        if !board.castling.has(flag) {
            continue;
        }
        let rook_from = board.castle_rook(us, kingside);
        if board.piece_at(rook_from) != Some(Piece::new(PieceKind::Rook, us)) {
            continue;
        }
        let king_to = Board::king_castle_to(us, kingside);
        let rook_to = Board::rook_castle_to(us, kingside);
        if !castle_path_clear(king_from, king_to, rook_from, rook_to, all_occ) {
            continue;
        }
        if !king_castle_path_safe(board, king_from, king_to, them) {
            continue;
        }
        moves.push(Move::new(king_from, king_to, move_flag));
    }
}

/// Squares strictly between `a` and `b` on the same rank.
fn between_on_rank(a: Square, b: Square) -> Bitboard {
    if a.rank() != b.rank() {
        return Bitboard::EMPTY;
    }
    let rank = a.rank();
    let (lo, hi) = if a.file() < b.file() {
        (a.file(), b.file())
    } else {
        (b.file(), a.file())
    };
    let mut bb = Bitboard::EMPTY;
    let mut file = lo + 1;
    while file < hi {
        bb = bb.set(Square::new(file, rank));
        file += 1;
    }
    bb
}

/// All squares the king and rook travel through, plus destinations, excluding
/// the two origin squares (the king and rook may occupy each other's path).
fn castle_path_clear(
    king_from: Square,
    king_to: Square,
    rook_from: Square,
    rook_to: Square,
    occ: Bitboard,
) -> bool {
    let origins = king_from.bitboard() | rook_from.bitboard();
    let mut need_empty = between_on_rank(king_from, king_to)
        | between_on_rank(rook_from, rook_to)
        | between_on_rank(king_from, rook_from);
    need_empty = need_empty.set(king_to).set(rook_to);
    let need_empty = Bitboard(need_empty.0 & !origins.0);
    (need_empty.0 & occ.0) == 0
}

/// The king may not start, pass through, or finish on an attacked square.
fn king_castle_path_safe(board: &Board, king_from: Square, king_to: Square, them: Color) -> bool {
    let rank = king_from.rank();
    let (lo, hi) = if king_from.file() <= king_to.file() {
        (king_from.file(), king_to.file())
    } else {
        (king_to.file(), king_from.file())
    };
    for file in lo..=hi {
        if is_square_attacked(board, Square::new(file, rank), them) {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Staged move generation: captures/promotions and quiet moves separately
// ---------------------------------------------------------------------------

/// Generate only pseudo-legal captures, en passant, and all promotions.
/// Used by quiescence search and staged move generation to avoid generating
/// quiet moves when they are not needed (e.g. when a beta cutoff happens
/// on a capture).
pub fn generate_pseudo_legal_captures(board: &Board) -> MoveList {
    let mut moves = MoveList::new();
    let us = board.side_to_move;
    let ci = us.index();
    let their_occ = board.occupancy[us.opposite().index()];
    let all_occ = board.all_occupancy();
    let empty = !all_occ;

    // ── Pawn captures, en passant, and promotions ──
    let pawns = board.pieces[ci][PieceKind::Pawn.index()];
    let (push_dir, promo_rank): (i8, Bitboard) = match us {
        Color::White => (8, Bitboard::RANK_7),
        Color::Black => (-8, Bitboard::RANK_2),
    };
    let (attack_left, attack_right) = match us {
        Color::White => (pawns.north_west(), pawns.north_east()),
        Color::Black => (pawns.south_east(), pawns.south_west()),
    };
    let (left_delta, right_delta): (i8, i8) = match us {
        Color::White => (7, 9),
        Color::Black => (-7, -9),
    };

    for to in (attack_left & their_occ).iter() {
        let from = Square((to.0 as i8 - left_delta) as u8);
        if promo_rank.is_set(from) {
            moves.push(Move::new(from, to, MoveFlag::PromoteCaptureQueen));
            moves.push(Move::new(from, to, MoveFlag::PromoteCaptureRook));
            moves.push(Move::new(from, to, MoveFlag::PromoteCaptureBishop));
            moves.push(Move::new(from, to, MoveFlag::PromoteCaptureKnight));
        } else {
            moves.push(Move::new(from, to, MoveFlag::Capture));
        }
    }
    for to in (attack_right & their_occ).iter() {
        let from = Square((to.0 as i8 - right_delta) as u8);
        if promo_rank.is_set(from) {
            moves.push(Move::new(from, to, MoveFlag::PromoteCaptureQueen));
            moves.push(Move::new(from, to, MoveFlag::PromoteCaptureRook));
            moves.push(Move::new(from, to, MoveFlag::PromoteCaptureBishop));
            moves.push(Move::new(from, to, MoveFlag::PromoteCaptureKnight));
        } else {
            moves.push(Move::new(from, to, MoveFlag::Capture));
        }
    }

    if let Some(ep_sq) = board.en_passant {
        let ep_bb = ep_sq.bitboard();
        if !(attack_left & ep_bb).is_empty() {
            let from = Square((ep_sq.0 as i8 - left_delta) as u8);
            moves.push(Move::new(from, ep_sq, MoveFlag::EnPassant));
        }
        if !(attack_right & ep_bb).is_empty() {
            let from = Square((ep_sq.0 as i8 - right_delta) as u8);
            moves.push(Move::new(from, ep_sq, MoveFlag::EnPassant));
        }
    }

    // Non-capture promotions (push to last rank — tactical)
    let single_push = match us {
        Color::White => pawns.north(),
        Color::Black => pawns.south(),
    } & empty;
    for to in single_push.iter() {
        let from = Square((to.0 as i8 - push_dir) as u8);
        if promo_rank.is_set(from) {
            moves.push(Move::new(from, to, MoveFlag::PromoteQueen));
            moves.push(Move::new(from, to, MoveFlag::PromoteRook));
            moves.push(Move::new(from, to, MoveFlag::PromoteBishop));
            moves.push(Move::new(from, to, MoveFlag::PromoteKnight));
        }
    }

    // ── Piece captures ──
    let knights = board.pieces[ci][PieceKind::Knight.index()];
    for from in knights.iter() {
        for to in (knight_attacks(from) & their_occ).iter() {
            moves.push(Move::new(from, to, MoveFlag::Capture));
        }
    }
    let bishops = board.pieces[ci][PieceKind::Bishop.index()];
    for from in bishops.iter() {
        for to in (bishop_attacks(from, all_occ) & their_occ).iter() {
            moves.push(Move::new(from, to, MoveFlag::Capture));
        }
    }
    let rooks = board.pieces[ci][PieceKind::Rook.index()];
    for from in rooks.iter() {
        for to in (rook_attacks(from, all_occ) & their_occ).iter() {
            moves.push(Move::new(from, to, MoveFlag::Capture));
        }
    }
    let queens = board.pieces[ci][PieceKind::Queen.index()];
    for from in queens.iter() {
        for to in (queen_attacks(from, all_occ) & their_occ).iter() {
            moves.push(Move::new(from, to, MoveFlag::Capture));
        }
    }
    let king_sq = board.king_square(us);
    for to in (king_attacks(king_sq) & their_occ).iter() {
        moves.push(Move::new(king_sq, to, MoveFlag::Capture));
    }

    moves
}

/// Generate only pseudo-legal quiet moves (non-captures, non-promotions).
/// Includes pawn pushes, piece non-capture moves, and castling.
pub fn generate_pseudo_legal_quiets(board: &Board) -> MoveList {
    let mut moves = MoveList::new();
    let us = board.side_to_move;
    let ci = us.index();
    let all_occ = board.all_occupancy();
    let empty = !all_occ;

    // ── Pawn pushes (non-promotion only) ──
    let pawns = board.pieces[ci][PieceKind::Pawn.index()];
    let (push_dir, start_rank, promo_rank): (i8, Bitboard, Bitboard) = match us {
        Color::White => (8, Bitboard::RANK_2, Bitboard::RANK_7),
        Color::Black => (-8, Bitboard::RANK_7, Bitboard::RANK_2),
    };

    let single_push = match us {
        Color::White => pawns.north(),
        Color::Black => pawns.south(),
    } & empty;

    for to in single_push.iter() {
        let from = Square((to.0 as i8 - push_dir) as u8);
        if !promo_rank.is_set(from) {
            moves.push(Move::new(from, to, MoveFlag::Normal));
        }
    }

    let double_push_candidates = match us {
        Color::White => (pawns & start_rank).north() & empty,
        Color::Black => (pawns & start_rank).south() & empty,
    };
    let double_push = match us {
        Color::White => double_push_candidates.north() & empty,
        Color::Black => double_push_candidates.south() & empty,
    };
    for to in double_push.iter() {
        let from = Square((to.0 as i8 - 2 * push_dir) as u8);
        moves.push(Move::new(from, to, MoveFlag::DoublePawnPush));
    }

    // ── Piece quiet moves ──
    let knights = board.pieces[ci][PieceKind::Knight.index()];
    for from in knights.iter() {
        for to in (knight_attacks(from) & empty).iter() {
            moves.push(Move::new(from, to, MoveFlag::Normal));
        }
    }
    let bishops = board.pieces[ci][PieceKind::Bishop.index()];
    for from in bishops.iter() {
        for to in (bishop_attacks(from, all_occ) & empty).iter() {
            moves.push(Move::new(from, to, MoveFlag::Normal));
        }
    }
    let rooks = board.pieces[ci][PieceKind::Rook.index()];
    for from in rooks.iter() {
        for to in (rook_attacks(from, all_occ) & empty).iter() {
            moves.push(Move::new(from, to, MoveFlag::Normal));
        }
    }
    let queens = board.pieces[ci][PieceKind::Queen.index()];
    for from in queens.iter() {
        for to in (queen_attacks(from, all_occ) & empty).iter() {
            moves.push(Move::new(from, to, MoveFlag::Normal));
        }
    }
    let king_sq = board.king_square(us);
    for to in (king_attacks(king_sq) & empty).iter() {
        moves.push(Move::new(king_sq, to, MoveFlag::Normal));
    }

    // ── Castling ──
    generate_castling_moves(board, us, all_occ, &mut moves);

    moves
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_starting_position_moves() {
        let board = Board::starting_position();
        let moves = generate_legal_moves(&board);
        // Starting position has exactly 20 legal moves
        assert_eq!(
            moves.len(),
            20,
            "Starting position should have 20 legal moves, got {}",
            moves.len()
        );
    }

    #[test]
    fn test_position_after_e4() {
        let board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1").unwrap();
        let moves = generate_legal_moves(&board);
        // After 1.e4, Black has 20 legal moves
        assert_eq!(moves.len(), 20);
    }

    #[test]
    fn test_castling_available() {
        // Position where white can castle both sides
        let board = Board::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1").unwrap();
        let moves = generate_legal_moves(&board);
        let castling_moves: Vec<_> = moves
            .iter()
            .filter(|m| {
                m.flag() == MoveFlag::KingsideCastle || m.flag() == MoveFlag::QueensideCastle
            })
            .collect();
        assert_eq!(
            castling_moves.len(),
            2,
            "Expected 2 castling moves, got {:?}",
            castling_moves
        );
    }

    #[test]
    fn test_en_passant() {
        // White pawn on e5, black pawn just pushed d7-d5
        let board = Board::from_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
            .unwrap();
        let moves = generate_legal_moves(&board);
        let ep_moves: Vec<_> = moves
            .iter()
            .filter(|m| m.flag() == MoveFlag::EnPassant)
            .collect();
        assert_eq!(ep_moves.len(), 1, "Expected 1 en passant move");
        assert_eq!(ep_moves[0].to_sq(), Square::from_algebraic("d6").unwrap());
    }

    #[test]
    fn test_promotion() {
        // White pawn on e7, can promote (black king on a8, not blocking e8)
        let board = Board::from_fen("k7/4P3/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let moves = generate_legal_moves(&board);
        let promo_moves: Vec<_> = moves.iter().filter(|m| m.flag().is_promotion()).collect();
        // 4 promotions (queen, rook, bishop, knight)
        assert_eq!(promo_moves.len(), 4);
    }

    #[test]
    fn test_promotion_with_capture() {
        // White pawn on e7, can capture-promote on d8 and f8
        let board = Board::from_fen("3rkr2/4P3/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let moves = generate_legal_moves(&board);
        let promo_cap_moves: Vec<_> = moves
            .iter()
            .filter(|m| m.flag().is_promotion() && m.flag().is_capture())
            .collect();
        // 2 capture-promotions * 4 pieces = 8
        assert_eq!(promo_cap_moves.len(), 8);
    }

    #[test]
    fn test_no_castling_through_check() {
        // Rook on f8 attacks f1, preventing kingside castle
        let board = Board::from_fen("1k3r2/8/8/8/8/8/8/R3K2R w KQ - 0 1").unwrap();
        let moves = generate_legal_moves(&board);
        let ks_castle: Vec<_> = moves
            .iter()
            .filter(|m| m.flag() == MoveFlag::KingsideCastle)
            .collect();
        assert!(
            ks_castle.is_empty(),
            "Should not be able to castle kingside through attacked f1"
        );
    }

    #[test]
    fn test_no_castling_in_check() {
        // King in check, cannot castle
        let board = Board::from_fen("1k2r3/8/8/8/8/8/8/R3K2R w KQ - 0 1").unwrap();
        let moves = generate_legal_moves(&board);
        let castling_moves: Vec<_> = moves
            .iter()
            .filter(|m| {
                m.flag() == MoveFlag::KingsideCastle || m.flag() == MoveFlag::QueensideCastle
            })
            .collect();
        assert!(castling_moves.is_empty(), "Should not castle when in check");
    }

    #[test]
    fn test_checkmate_no_moves() {
        // Scholar's mate position - black is checkmated
        let board =
            Board::from_fen("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
                .unwrap();
        let moves = generate_legal_moves(&board);
        assert_eq!(
            moves.len(),
            0,
            "Checkmate position should have 0 legal moves"
        );
    }

    #[test]
    fn test_stalemate() {
        // Stalemate position: black king on a8, white queen on b6, white king on c8
        // Actually let's use a simpler stalemate
        let board = Board::from_fen("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1").unwrap();
        let moves = generate_legal_moves(&board);
        assert_eq!(
            moves.len(),
            0,
            "Stalemate position should have 0 legal moves"
        );
    }

    /// Perft test: count leaf nodes at a given depth.
    fn perft(board: &Board, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }

        let moves = generate_legal_moves(board);
        if depth == 1 {
            return moves.len() as u64;
        }

        let mut nodes = 0u64;
        for &m in moves.iter() {
            let mut b = board.clone();
            b.make_move(m);
            nodes += perft(&b, depth - 1);
        }
        nodes
    }

    #[test]
    fn test_perft_starting_depth_1() {
        let board = Board::starting_position();
        assert_eq!(perft(&board, 1), 20);
    }

    #[test]
    fn test_perft_starting_depth_2() {
        let board = Board::starting_position();
        assert_eq!(perft(&board, 2), 400);
    }

    #[test]
    fn test_perft_starting_depth_3() {
        let board = Board::starting_position();
        assert_eq!(perft(&board, 3), 8902);
    }

    #[test]
    fn test_perft_starting_depth_4() {
        let board = Board::starting_position();
        assert_eq!(perft(&board, 4), 197_281);
    }

    #[test]
    fn test_perft_kiwipete_depth_1() {
        // "Kiwipete" position - good test for tactical move generation
        let board =
            Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
                .unwrap();
        assert_eq!(perft(&board, 1), 48);
    }

    #[test]
    fn test_perft_kiwipete_depth_2() {
        let board =
            Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
                .unwrap();
        assert_eq!(perft(&board, 2), 2039);
    }

    #[test]
    fn test_perft_kiwipete_depth_3() {
        let board =
            Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
                .unwrap();
        assert_eq!(perft(&board, 3), 97862);
    }

    // Chess960 perft positions from chessprogramming.org (Andrew Grant / Ethereal).
    #[test]
    fn test_perft_chess960_grant_1() {
        let board = Board::from_fen(
            "bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9",
        )
        .unwrap();
        assert_eq!(perft(&board, 1), 21);
        assert_eq!(perft(&board, 2), 528);
        assert_eq!(perft(&board, 3), 12_189);
    }

    #[test]
    fn test_perft_chess960_king_on_rook_file() {
        // King already on f, rooks on e and g: G/E castling, rook already on dest.
        let board =
            Board::from_fen("b1q1rrkb/pppppppp/3nn3/8/P7/1PPP4/4PPPP/BQNNRKRB w GE - 1 9")
                .unwrap();
        assert_eq!(perft(&board, 1), 20);
        assert_eq!(perft(&board, 2), 479);
        assert_eq!(perft(&board, 3), 10_471);
    }

    #[test]
    fn test_perft_chess960_double_rook_same_side() {
        let board =
            Board::from_fen("qn1rkrbb/pp1p1ppp/2p1p3/3n4/4P2P/2NP4/PPP2PP1/Q1NRKRBB w FDfd - 1 9")
                .unwrap();
        assert_eq!(perft(&board, 1), 24);
        assert_eq!(perft(&board, 2), 585);
        assert_eq!(perft(&board, 3), 14_769);
    }

    #[test]
    fn test_chess960_castle_make_unmake_hash() {
        // King on f, rook on g: O-O swaps them (king to g, rook to f).
        let fen = "b1q1rrkb/pppppppp/3nn3/8/P7/1PPP4/4PPPP/BQNNRKRB w GE - 1 9";
        let mut board = Board::from_fen(fen).unwrap();
        let hash_before = board.hash;
        let castle = generate_legal_moves(&board)
            .iter()
            .copied()
            .find(|m| m.flag() == MoveFlag::KingsideCastle)
            .expect("kingside castle");
        let prev_castling = board.castling;
        let prev_ep = board.en_passant;
        let prev_halfmove = board.halfmove_clock;
        let captured = board.make_move(castle);
        assert_eq!(board.king_square(Color::White), Square::G1);
        assert_eq!(
            board.piece_at(Square::F1).map(|p| p.kind),
            Some(PieceKind::Rook)
        );
        assert_eq!(board.hash, board.compute_hash());
        board.unmake_move(castle, captured, prev_castling, prev_ep, prev_halfmove);
        assert_eq!(board.hash, hash_before);
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn test_shredder_fen_standard_perft() {
        let board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w AHah - 0 1").unwrap();
        assert_eq!(perft(&board, 3), 8902);
    }

    // -----------------------------------------------------------------------
    // Tests for staged (capture / quiet) generators
    // -----------------------------------------------------------------------

    /// Captures + quiets must cover exactly the same moves as the full
    /// pseudo-legal generator (union must match, no overlaps except promos
    /// which we deliberately put in captures).
    #[test]
    fn test_staged_generators_cover_all_moves() {
        let positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
            "k7/4P3/8/8/8/8/8/4K3 w - - 0 1",    // promotion
            "3rkr2/4P3/8/8/8/8/8/4K3 w - - 0 1", // promo-capture
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3", // en passant
            "k7/2Q5/1K6/8/8/8/8/8 b - - 0 1",    // stalemate
        ];

        for fen in &positions {
            let board = Board::from_fen(fen).unwrap();
            let all = generate_pseudo_legal_moves(&board);
            let caps = generate_pseudo_legal_captures(&board);
            let qts = generate_pseudo_legal_quiets(&board);

            // Every capture/promotion from caps must be in all
            for &m in caps.iter() {
                assert!(
                    all.iter().any(|a| *a == m),
                    "capture move {} from captures-gen not found in all pseudo-legal (fen: {})",
                    m,
                    fen,
                );
            }
            // Every quiet from qts must be in all
            for &m in qts.iter() {
                assert!(
                    all.iter().any(|a| *a == m),
                    "quiet move {} from quiets-gen not found in all pseudo-legal (fen: {})",
                    m,
                    fen,
                );
            }
            // Union must match total count (promotions appear only in caps)
            assert_eq!(
                caps.len() + qts.len(),
                all.len(),
                "capture ({}) + quiet ({}) ≠ all ({}) for fen: {}",
                caps.len(),
                qts.len(),
                all.len(),
                fen,
            );
        }
    }
}
