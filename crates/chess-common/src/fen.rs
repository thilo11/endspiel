use crate::board::Board;
use crate::types::{Bitboard, CastlingRights, Color, Piece, PieceKind, Square};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum FenError {
    #[error("invalid FEN: {0}")]
    Invalid(String),
    #[error("invalid FEN: expected 4 to 6 parts, got {0}")]
    WrongPartCount(usize),
    #[error("invalid piece placement in FEN")]
    BadPiecePlacement,
}

impl Board {
    /// Parse a FEN string into a Board.
    pub fn from_fen(fen: &str) -> Result<Self, FenError> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if !(4..=6).contains(&parts.len()) {
            return Err(FenError::WrongPartCount(parts.len()));
        }

        let mut board = Board {
            pieces: [[Bitboard::EMPTY; PieceKind::COUNT]; 2],
            mailbox: [None; 64],
            occupancy: [Bitboard::EMPTY; 2],
            side_to_move: Color::White,
            castling: CastlingRights::NONE,
            en_passant: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            hash: 0,
            position_history: Vec::new(),
        };

        // Parse piece placement
        let ranks: Vec<&str> = parts[0].split('/').collect();
        if ranks.len() != 8 {
            return Err(FenError::BadPiecePlacement);
        }

        for (rank_idx, rank_str) in ranks.iter().enumerate() {
            let rank = 7 - rank_idx as u8; // FEN starts from rank 8
            let mut file: u8 = 0;
            for c in rank_str.chars() {
                if let Some(skip) = c.to_digit(10).filter(|&n| (1..=8).contains(&n)) {
                    file = file
                        .checked_add(skip as u8)
                        .filter(|&f| f <= 8)
                        .ok_or(FenError::BadPiecePlacement)?;
                } else {
                    let color = if c.is_uppercase() {
                        Color::White
                    } else {
                        Color::Black
                    };
                    let kind = PieceKind::from_char(c)
                        .ok_or_else(|| FenError::Invalid(format!("unknown piece char: {c}")))?;
                    if file >= 8 {
                        return Err(FenError::BadPiecePlacement);
                    }
                    let sq = Square::new(file, rank);
                    board.set_piece(sq, Piece::new(kind, color));
                    file += 1;
                }
            }
            if file != 8 {
                return Err(FenError::BadPiecePlacement);
            }
        }

        // Parse side to move
        board.side_to_move = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err(FenError::Invalid(format!("bad side to move: {}", parts[1]))),
        };

        // Parse castling rights. Reject unknown or duplicate characters instead
        // of silently accepting a damaged field.
        board.castling = if parts[2] == "-" {
            CastlingRights::NONE
        } else {
            let mut rights = CastlingRights::NONE;
            for c in parts[2].chars() {
                let flag = match c {
                    'K' => CastlingRights::WHITE_KINGSIDE,
                    'Q' => CastlingRights::WHITE_QUEENSIDE,
                    'k' => CastlingRights::BLACK_KINGSIDE,
                    'q' => CastlingRights::BLACK_QUEENSIDE,
                    _ => {
                        return Err(FenError::Invalid(format!(
                            "bad castling rights: {}",
                            parts[2]
                        )));
                    }
                };
                if rights.has(flag) {
                    return Err(FenError::Invalid(format!("duplicate castling right: {c}")));
                }
                rights = rights.add(flag);
            }
            rights
        };

        // Parse en passant
        board.en_passant = if parts[3] == "-" {
            None
        } else {
            Some(
                Square::from_algebraic(parts[3])
                    .ok_or_else(|| FenError::Invalid(format!("bad en passant: {}", parts[3])))?,
            )
        };

        // Parse halfmove clock (optional)
        if parts.len() > 4 {
            board.halfmove_clock = parts[4]
                .parse()
                .map_err(|_| FenError::Invalid(format!("bad halfmove clock: {}", parts[4])))?;
        }

        // Parse fullmove number (optional)
        if parts.len() > 5 {
            board.fullmove_number = parts[5]
                .parse()
                .map_err(|_| FenError::Invalid(format!("bad fullmove number: {}", parts[5])))?;
            if board.fullmove_number == 0 {
                return Err(FenError::Invalid(
                    "fullmove number must be at least 1".to_string(),
                ));
            }
        }

        validate_structure(&board)?;

        board.hash = board.compute_hash();

        Ok(board)
    }

    /// Convert this board to a FEN string.
    pub fn to_fen(&self) -> String {
        let mut fen = String::with_capacity(80);

        // Piece placement
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let sq = Square::new(file, rank);
                match self.piece_at(sq) {
                    Some(piece) => {
                        if empty > 0 {
                            fen.push(char::from_digit(empty, 10).unwrap());
                            empty = 0;
                        }
                        fen.push(piece.kind.to_char(piece.color));
                    }
                    None => {
                        empty += 1;
                    }
                }
            }
            if empty > 0 {
                fen.push(char::from_digit(empty, 10).unwrap());
            }
            if rank > 0 {
                fen.push('/');
            }
        }

        fen.push(' ');

        // Side to move
        fen.push(match self.side_to_move {
            Color::White => 'w',
            Color::Black => 'b',
        });

        fen.push(' ');

        // Castling
        fen.push_str(&self.castling.to_fen());

        fen.push(' ');

        // En passant
        match self.en_passant {
            Some(sq) => fen.push_str(&sq.to_algebraic()),
            None => fen.push('-'),
        }

        fen.push(' ');

        // Halfmove clock
        fen.push_str(&self.halfmove_clock.to_string());

        fen.push(' ');

        // Fullmove number
        fen.push_str(&self.fullmove_number.to_string());

        fen
    }
}

/// Validate invariants assumed by move generation, NNUE and Syzygy. Keeping
/// impossible FENs out here is essential in release builds: those hot paths
/// deliberately index by king and piece locations without rechecking the
/// complete board at every node.
fn validate_structure(board: &Board) -> Result<(), FenError> {
    for color in [Color::White, Color::Black] {
        let ci = color.index();
        let kings = board.pieces[ci][PieceKind::King.index()].count();
        if kings != 1 {
            return Err(FenError::Invalid(format!(
                "expected exactly one {color:?} king, got {kings}"
            )));
        }

        let pawns = board.pieces[ci][PieceKind::Pawn.index()];
        if pawns.count() > 8 {
            return Err(FenError::Invalid(format!("too many {color:?} pawns")));
        }
        if !(pawns & (Bitboard::RANK_1 | Bitboard::RANK_8)).is_empty() {
            return Err(FenError::Invalid(format!(
                "{color:?} pawn on first or eighth rank"
            )));
        }
        if board.occupancy[ci].count() > 16 {
            return Err(FenError::Invalid(format!("too many {color:?} pieces")));
        }
    }

    let white_king = board.pieces[Color::White.index()][PieceKind::King.index()]
        .lsb()
        .expect("king count checked above");
    let black_king = board.pieces[Color::Black.index()][PieceKind::King.index()]
        .lsb()
        .expect("king count checked above");
    if white_king.file().abs_diff(black_king.file()) <= 1
        && white_king.rank().abs_diff(black_king.rank()) <= 1
    {
        return Err(FenError::Invalid("kings may not be adjacent".to_string()));
    }

    validate_castling_pieces(board)?;
    validate_en_passant(board)?;
    Ok(())
}

fn validate_castling_pieces(board: &Board) -> Result<(), FenError> {
    let white_king = Piece::new(PieceKind::King, Color::White);
    let black_king = Piece::new(PieceKind::King, Color::Black);
    let white_rook = Piece::new(PieceKind::Rook, Color::White);
    let black_rook = Piece::new(PieceKind::Rook, Color::Black);

    for (flag, king_sq, king, rook_sq, rook, name) in [
        (
            CastlingRights::WHITE_KINGSIDE,
            Square::E1,
            white_king,
            Square::H1,
            white_rook,
            "K",
        ),
        (
            CastlingRights::WHITE_QUEENSIDE,
            Square::E1,
            white_king,
            Square::A1,
            white_rook,
            "Q",
        ),
        (
            CastlingRights::BLACK_KINGSIDE,
            Square::E8,
            black_king,
            Square::H8,
            black_rook,
            "k",
        ),
        (
            CastlingRights::BLACK_QUEENSIDE,
            Square::E8,
            black_king,
            Square::A8,
            black_rook,
            "q",
        ),
    ] {
        if board.castling.has(flag)
            && (board.piece_at(king_sq) != Some(king) || board.piece_at(rook_sq) != Some(rook))
        {
            return Err(FenError::Invalid(format!(
                "castling right {name} has no king and rook on their home squares"
            )));
        }
    }
    Ok(())
}

fn validate_en_passant(board: &Board) -> Result<(), FenError> {
    let Some(target) = board.en_passant else {
        return Ok(());
    };
    if board.piece_at(target).is_some() {
        return Err(FenError::Invalid(
            "en passant target square is occupied".to_string(),
        ));
    }

    let (expected_rank, pawn_rank, origin_rank, pawn) = match board.side_to_move {
        Color::White => (5, 4, 6, Piece::new(PieceKind::Pawn, Color::Black)),
        Color::Black => (2, 3, 1, Piece::new(PieceKind::Pawn, Color::White)),
    };
    if target.rank() != expected_rank {
        return Err(FenError::Invalid(format!(
            "bad en passant rank for {:?} to move",
            board.side_to_move
        )));
    }

    let pawn_sq = Square::new(target.file(), pawn_rank);
    let origin_sq = Square::new(target.file(), origin_rank);
    if board.piece_at(pawn_sq) != Some(pawn) || board.piece_at(origin_sq).is_some() {
        return Err(FenError::Invalid(
            "en passant target is inconsistent with a double pawn push".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_starting_position_fen_roundtrip() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let board = Board::from_fen(fen).unwrap();
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn test_mid_game_fen_roundtrip() {
        let fen = "r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3";
        let board = Board::from_fen(fen).unwrap();
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn test_en_passant_fen() {
        let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
        let board = Board::from_fen(fen).unwrap();
        assert_eq!(
            board.en_passant,
            Some(Square::from_algebraic("e3").unwrap())
        );
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn rejects_missing_or_duplicate_kings() {
        assert!(Board::from_fen("8/8/8/8/8/8/8/8 w - - 0 1").is_err());
        assert!(Board::from_fen("4k3/8/8/8/8/8/4K3/4K3 w - - 0 1").is_err());
    }

    #[test]
    fn rejects_impossible_piece_placement_without_panicking() {
        for fen in [
            "4k3/8/8/8/8/8/8/P3K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/9K w - - 0 1",
            "4k3/8/8/8/8/8/8/80K w - - 0 1",
        ] {
            assert!(Board::from_fen(fen).is_err(), "accepted {fen}");
        }
    }

    #[test]
    fn rejects_invalid_castling_and_en_passant_state() {
        for fen in [
            "4k3/8/8/8/8/8/8/4K3 w K - 0 1",
            "4k3/8/8/8/8/8/8/4K3 w X - 0 1",
            "4k3/8/8/8/4P3/8/8/4K3 b - e4 0 1",
            "4k3/8/8/8/8/8/8/4K3 b - e3 0 1",
        ] {
            assert!(Board::from_fen(fen).is_err(), "accepted {fen}");
        }
    }

    #[test]
    fn accepts_four_field_epd_but_rejects_extra_fields_and_move_zero() {
        assert!(Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - -").is_ok());
        assert!(Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1 extra").is_err());
        assert!(Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 0").is_err());
    }
}
