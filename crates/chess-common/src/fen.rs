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
            castle_rooks: Board::STANDARD_CASTLE_ROOKS,
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

        // Parse castling rights. Accepts standard FEN (KQkq), X-FEN, and
        // Shredder-FEN (AHah / file letters). Rook origins are stored on the
        // board so Chess960 can castle from non-a/h files.
        parse_castling_field(&mut board, parts[2])?;

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

        // Castling (X-FEN: K/Q when the rook is on h/a, otherwise the file letter)
        fen.push_str(&castling_fen(self));

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

fn parse_castling_field(board: &mut Board, field: &str) -> Result<(), FenError> {
    board.castle_rooks = Board::STANDARD_CASTLE_ROOKS;
    if field == "-" {
        board.castling = CastlingRights::NONE;
        return Ok(());
    }

    let mut rights = CastlingRights::NONE;
    for c in field.chars() {
        let (color, kingside, rook_file) = match c {
            'K' => (
                Color::White,
                true,
                outer_rook_file(board, Color::White, true)?,
            ),
            'Q' => (
                Color::White,
                false,
                outer_rook_file(board, Color::White, false)?,
            ),
            'k' => (
                Color::Black,
                true,
                outer_rook_file(board, Color::Black, true)?,
            ),
            'q' => (
                Color::Black,
                false,
                outer_rook_file(board, Color::Black, false)?,
            ),
            'A'..='H' => rook_file_castle(board, Color::White, c as u8 - b'A')?,
            'a'..='h' => rook_file_castle(board, Color::Black, c as u8 - b'a')?,
            _ => {
                return Err(FenError::Invalid(format!(
                    "bad castling rights: {field}"
                )));
            }
        };
        let flag = castle_flag(color, kingside);
        if rights.has(flag) {
            return Err(FenError::Invalid(format!("duplicate castling right: {c}")));
        }
        rights = rights.add(flag);
        let rank = back_rank(color);
        board.castle_rooks[Board::castle_rook_index(color, kingside)] =
            Square::new(rook_file, rank);
    }
    board.castling = rights;
    Ok(())
}

fn back_rank(color: Color) -> u8 {
    match color {
        Color::White => 0,
        Color::Black => 7,
    }
}

fn castle_flag(color: Color, kingside: bool) -> u8 {
    match (color, kingside) {
        (Color::White, true) => CastlingRights::WHITE_KINGSIDE,
        (Color::White, false) => CastlingRights::WHITE_QUEENSIDE,
        (Color::Black, true) => CastlingRights::BLACK_KINGSIDE,
        (Color::Black, false) => CastlingRights::BLACK_QUEENSIDE,
    }
}

/// X-FEN / standard FEN: K/Q names the outermost rook on that side of the king.
fn outer_rook_file(board: &Board, color: Color, kingside: bool) -> Result<u8, FenError> {
    let king = board.king_square(color);
    let rank = back_rank(color);
    if king.rank() != rank {
        return Err(FenError::Invalid(
            "castling king is not on the back rank".to_string(),
        ));
    }
    let rooks = board.pieces[color.index()][PieceKind::Rook.index()];
    let mut found: Option<u8> = None;
    for sq in rooks.iter() {
        if sq.rank() != rank {
            continue;
        }
        if kingside && sq.file() > king.file() {
            found = Some(found.map_or(sq.file(), |f| f.max(sq.file())));
        } else if !kingside && sq.file() < king.file() {
            found = Some(found.map_or(sq.file(), |f| f.min(sq.file())));
        }
    }
    found.ok_or_else(|| {
        FenError::Invalid(
            "castling right has no rook on that side of the king".to_string(),
        )
    })
}

/// Shredder-FEN / X-FEN file letter: the rook on that back-rank file.
fn rook_file_castle(
    board: &Board,
    color: Color,
    file: u8,
) -> Result<(Color, bool, u8), FenError> {
    let king = board.king_square(color);
    let rank = back_rank(color);
    if king.rank() != rank {
        return Err(FenError::Invalid(
            "castling king is not on the back rank".to_string(),
        ));
    }
    if file == king.file() {
        return Err(FenError::Invalid(
            "castling rook file must not be the king's file".to_string(),
        ));
    }
    let kingside = file > king.file();
    Ok((color, kingside, file))
}

/// X-FEN castling field: K/Q when the rook is on h/a, otherwise the file letter.
fn castling_fen(board: &Board) -> String {
    if board.castling.0 == 0 {
        return "-".to_string();
    }
    let mut s = String::with_capacity(4);
    let mut emit = |color: Color, kingside: bool, kq: char| {
        let rook = board.castle_rook(color, kingside);
        let standard_file = if kingside { 7 } else { 0 };
        if rook.file() == standard_file {
            s.push(kq);
        } else {
            let letter = (b'a' + rook.file()) as char;
            s.push(match color {
                Color::White => letter.to_ascii_uppercase(),
                Color::Black => letter,
            });
        }
    };
    if board.castling.has(CastlingRights::WHITE_KINGSIDE) {
        emit(Color::White, true, 'K');
    }
    if board.castling.has(CastlingRights::WHITE_QUEENSIDE) {
        emit(Color::White, false, 'Q');
    }
    if board.castling.has(CastlingRights::BLACK_KINGSIDE) {
        emit(Color::Black, true, 'k');
    }
    if board.castling.has(CastlingRights::BLACK_QUEENSIDE) {
        emit(Color::Black, false, 'q');
    }
    s
}

fn validate_castling_pieces(board: &Board) -> Result<(), FenError> {
    for (flag, color, kingside, name) in [
        (CastlingRights::WHITE_KINGSIDE, Color::White, true, "K"),
        (CastlingRights::WHITE_QUEENSIDE, Color::White, false, "Q"),
        (CastlingRights::BLACK_KINGSIDE, Color::Black, true, "k"),
        (CastlingRights::BLACK_QUEENSIDE, Color::Black, false, "q"),
    ] {
        if !board.castling.has(flag) {
            continue;
        }
        let king_sq = board.king_square(color);
        let rank = back_rank(color);
        if king_sq.rank() != rank {
            return Err(FenError::Invalid(format!(
                "castling right {name} has no king on the back rank"
            )));
        }
        let rook_sq = board.castle_rook(color, kingside);
        if rook_sq.rank() != rank
            || board.piece_at(rook_sq) != Some(Piece::new(PieceKind::Rook, color))
        {
            return Err(FenError::Invalid(format!(
                "castling right {name} has no rook on its home square"
            )));
        }
        if kingside && rook_sq.file() <= king_sq.file()
            || !kingside && rook_sq.file() >= king_sq.file()
        {
            return Err(FenError::Invalid(format!(
                "castling right {name} rook is on the wrong side of the king"
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

    #[test]
    fn shredder_fen_standard_start_roundtrips_as_kq() {
        // Shredder-FEN AHah is the standard start; we emit X-FEN KQkq.
        let board =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w AHah - 0 1").unwrap();
        assert_eq!(board.castle_rooks, Board::STANDARD_CASTLE_ROOKS);
        assert_eq!(
            board.to_fen(),
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        );
    }

    #[test]
    fn chess960_shredder_fen_stores_rook_files() {
        // BBQNNRKR: king g, rooks f and h. Shredder-FEN is HFhf; X-FEN emit is KFkf
        // because the h-file rook is written as K.
        let board =
            Board::from_fen("bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w HFhf - 0 1").unwrap();
        assert_eq!(board.castle_rook(Color::White, true), Square::H1);
        assert_eq!(board.castle_rook(Color::White, false), Square::F1);
        assert_eq!(board.castle_rook(Color::Black, true), Square::H8);
        assert_eq!(board.castle_rook(Color::Black, false), Square::F8);
        assert_eq!(
            board.to_fen(),
            "bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w KFkf - 0 1"
        );
    }

    #[test]
    fn chess960_kq_infers_outermost_rooks() {
        let board =
            Board::from_fen("bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w KQkq - 0 1").unwrap();
        assert_eq!(board.castle_rook(Color::White, true), Square::H1);
        assert_eq!(board.castle_rook(Color::White, false), Square::F1);
        // X-FEN: K (h-file rook) and F (not the a-file).
        assert_eq!(
            board.to_fen(),
            "bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w KFkf - 0 1"
        );
    }
}
