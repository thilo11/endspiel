//! SAN (Standard Algebraic Notation) move parser.

use chess_common::moves::MoveFlag;
use chess_common::{Board, Move, PieceKind, Square};

use crate::generate_legal_moves;

/// Parse a SAN move string and return the matching legal move, or `None`.
///
/// Accepts standard SAN including:
/// - Piece moves: `Nf3`, `Bxe5`, `Qd2d4` (disambiguation)
/// - Pawn moves: `e4`, `exd5`, `e8=Q`, `e8Q`
/// - Castling: `O-O`, `O-O-O` (also `0-0`, `0-0-0`)
/// - Trailing annotations stripped: `+`, `#`, `!`, `?`
pub fn san_to_move(board: &Board, san: &str) -> Option<Move> {
    let san = san.trim_end_matches(['+', '#', '!', '?']).trim();
    if san.is_empty() {
        return None;
    }

    // Castling
    if san == "O-O-O" || san == "0-0-0" {
        return find_castle(board, false);
    }
    if san == "O-O" || san == "0-0" {
        return find_castle(board, true);
    }

    let (body, promo) = strip_promo(san)?;
    let body_bytes = body.as_bytes();
    let len = body_bytes.len();
    if len < 2 {
        return None;
    }

    // Piece type from first character (uppercase = piece; otherwise pawn)
    let (piece, rest_start) = match body_bytes[0] {
        b'N' => (PieceKind::Knight, 1),
        b'B' => (PieceKind::Bishop, 1),
        b'R' => (PieceKind::Rook, 1),
        b'Q' => (PieceKind::Queen, 1),
        b'K' => (PieceKind::King, 1),
        _ => (PieceKind::Pawn, 0),
    };

    // Strip capture marker 'x'
    let rest: String = body[rest_start..].chars().filter(|&c| c != 'x').collect();
    let rlen = rest.len();
    if rlen < 2 {
        return None;
    }

    // Destination square: last two characters
    let dest = Square::from_algebraic(&rest[rlen - 2..])?;
    let (from_file, from_rank) = parse_disambig(&rest[..rlen - 2]);

    for &m in generate_legal_moves(board).as_slice() {
        if m.to_sq() != dest {
            continue;
        }
        let from = m.from_sq();
        let p = match board.piece_at(from) {
            Some(p) => p,
            None => continue,
        };
        if p.kind != piece || p.color != board.side_to_move {
            continue;
        }
        if m.flag().promotion_piece() != promo {
            continue;
        }
        if let Some(f) = from_file
            && from.file() != f
        {
            continue;
        }
        if let Some(r) = from_rank
            && from.rank() != r
        {
            continue;
        }
        return Some(m);
    }
    None
}

/// Strip promotion suffix from SAN body.
/// Returns (body_without_promo, promotion_piece).
/// Handles "e8=Q" and "e8Q" styles.
fn strip_promo(san: &str) -> Option<(&str, Option<PieceKind>)> {
    let b = san.as_bytes();
    let n = b.len();
    // "e8=Q" style
    if n >= 4 && b[n - 2] == b'=' {
        let promo = promo_from_char(b[n - 1] as char);
        return Some((&san[..n - 2], promo));
    }
    // "e8Q" style: last char is piece letter, second-to-last is rank '1' or '8'
    if n >= 3
        && matches!(b[n - 2] as char, '1' | '8')
        && matches!(b[n - 1] as char, 'Q' | 'R' | 'B' | 'N')
    {
        let promo = promo_from_char(b[n - 1] as char);
        return Some((&san[..n - 1], promo));
    }
    Some((san, None))
}

fn promo_from_char(c: char) -> Option<PieceKind> {
    match c {
        'Q' => Some(PieceKind::Queen),
        'R' => Some(PieceKind::Rook),
        'B' => Some(PieceKind::Bishop),
        'N' => Some(PieceKind::Knight),
        _ => None,
    }
}

/// Parse disambiguation string (file, rank, or both).
/// Returns (file 0–7, rank 0–7), each optional.
fn parse_disambig(s: &str) -> (Option<u8>, Option<u8>) {
    let mut file = None;
    let mut rank = None;
    for c in s.chars() {
        if ('a'..='h').contains(&c) {
            file = Some(c as u8 - b'a');
        }
        if ('1'..='8').contains(&c) {
            rank = Some(c as u8 - b'1');
        }
    }
    (file, rank)
}

fn find_castle(board: &Board, kingside: bool) -> Option<Move> {
    let target = if kingside {
        MoveFlag::KingsideCastle
    } else {
        MoveFlag::QueensideCastle
    };
    generate_legal_moves(board)
        .as_slice()
        .iter()
        .copied()
        .find(|m| m.flag() == target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess_common::Board;

    #[test]
    fn test_pawn_move() {
        let board = Board::starting_position();
        let m = san_to_move(&board, "e4").unwrap();
        assert_eq!(m.to_sq(), Square::from_algebraic("e4").unwrap());
    }

    #[test]
    fn test_knight_move() {
        let board = Board::starting_position();
        let m = san_to_move(&board, "Nf3").unwrap();
        assert_eq!(m.to_sq(), Square::from_algebraic("f3").unwrap());
    }

    #[test]
    fn test_castling_kingside() {
        let board = Board::from_fen("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4").unwrap();
        let m = san_to_move(&board, "O-O").unwrap();
        assert_eq!(m.flag(), MoveFlag::KingsideCastle);
    }

    #[test]
    fn test_promotion() {
        let board = Board::from_fen("8/P7/8/8/8/8/8/4K2k w - - 0 1").unwrap();
        let m = san_to_move(&board, "a8=Q").unwrap();
        assert_eq!(m.flag().promotion_piece(), Some(PieceKind::Queen));
    }

    #[test]
    fn test_annotations_stripped() {
        let board = Board::starting_position();
        let m = san_to_move(&board, "e4!").unwrap();
        assert_eq!(m.to_sq(), Square::from_algebraic("e4").unwrap());
    }
}
