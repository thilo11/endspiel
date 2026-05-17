use crate::board::Board;
use crate::types::{Bitboard, CastlingRights, Color, Piece, PieceKind, Square};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum FenError {
    #[error("invalid FEN: {0}")]
    Invalid(String),
    #[error("invalid FEN: expected 6 parts, got {0}")]
    WrongPartCount(usize),
    #[error("invalid piece placement in FEN")]
    BadPiecePlacement,
}

impl Board {
    /// Parse a FEN string into a Board.
    pub fn from_fen(fen: &str) -> Result<Self, FenError> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
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
                if let Some(skip) = c.to_digit(10) {
                    file += skip as u8;
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

        // Parse castling rights
        board.castling = CastlingRights::from_fen(parts[2]);

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
        }

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
        assert_eq!(board.en_passant, Some(Square::from_algebraic("e3").unwrap()));
        assert_eq!(board.to_fen(), fen);
    }
}
