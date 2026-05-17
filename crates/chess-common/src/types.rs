use std::fmt;

/// A player color.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    #[inline]
    pub const fn opposite(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Color::White => write!(f, "White"),
            Color::Black => write!(f, "Black"),
        }
    }
}

/// A piece type (without color).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PieceKind {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl PieceKind {
    pub const COUNT: usize = 6;
    pub const ALL: [PieceKind; 6] = [
        PieceKind::Pawn,
        PieceKind::Knight,
        PieceKind::Bishop,
        PieceKind::Rook,
        PieceKind::Queen,
        PieceKind::King,
    ];

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Material value in centipawns.
    #[inline]
    pub const fn value(self) -> i32 {
        match self {
            PieceKind::Pawn => 100,
            PieceKind::Knight => 320,
            PieceKind::Bishop => 330,
            PieceKind::Rook => 500,
            PieceKind::Queen => 900,
            PieceKind::King => 0, // King has no material value
        }
    }

    pub fn from_char(c: char) -> Option<Self> {
        match c.to_ascii_lowercase() {
            'p' => Some(PieceKind::Pawn),
            'n' => Some(PieceKind::Knight),
            'b' => Some(PieceKind::Bishop),
            'r' => Some(PieceKind::Rook),
            'q' => Some(PieceKind::Queen),
            'k' => Some(PieceKind::King),
            _ => None,
        }
    }

    pub fn to_char(self, color: Color) -> char {
        let c = match self {
            PieceKind::Pawn => 'p',
            PieceKind::Knight => 'n',
            PieceKind::Bishop => 'b',
            PieceKind::Rook => 'r',
            PieceKind::Queen => 'q',
            PieceKind::King => 'k',
        };
        match color {
            Color::White => c.to_ascii_uppercase(),
            Color::Black => c,
        }
    }
}

/// A piece with its color.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Piece {
    pub kind: PieceKind,
    pub color: Color,
}

impl Piece {
    #[inline]
    pub const fn new(kind: PieceKind, color: Color) -> Self {
        Self { kind, color }
    }
}

/// A square on the board, represented as 0..63.
/// a1 = 0, b1 = 1, ..., h8 = 63
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Square(pub u8);

impl Square {
    pub const A1: Square = Square(0);
    pub const B1: Square = Square(1);
    pub const C1: Square = Square(2);
    pub const D1: Square = Square(3);
    pub const E1: Square = Square(4);
    pub const F1: Square = Square(5);
    pub const G1: Square = Square(6);
    pub const H1: Square = Square(7);
    pub const A8: Square = Square(56);
    pub const B8: Square = Square(57);
    pub const C8: Square = Square(58);
    pub const D8: Square = Square(59);
    pub const E8: Square = Square(60);
    pub const F8: Square = Square(61);
    pub const G8: Square = Square(62);
    pub const H8: Square = Square(63);

    #[inline]
    pub const fn new(file: u8, rank: u8) -> Self {
        debug_assert!(file < 8 && rank < 8);
        Square(rank * 8 + file)
    }

    #[inline]
    pub const fn file(self) -> u8 {
        self.0 % 8
    }

    #[inline]
    pub const fn rank(self) -> u8 {
        self.0 / 8
    }

    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Parse a square from algebraic notation (e.g., "e4").
    pub fn from_algebraic(s: &str) -> Option<Self> {
        let bytes = s.as_bytes();
        if bytes.len() != 2 {
            return None;
        }
        let file = bytes[0].wrapping_sub(b'a');
        let rank = bytes[1].wrapping_sub(b'1');
        if file < 8 && rank < 8 {
            Some(Square::new(file, rank))
        } else {
            None
        }
    }

    /// Convert to algebraic notation (e.g., "e4").
    pub fn to_algebraic(self) -> String {
        let file = (b'a' + self.file()) as char;
        let rank = (b'1' + self.rank()) as char;
        format!("{file}{rank}")
    }

    /// Return the bitboard with only this square set.
    #[inline]
    pub const fn bitboard(self) -> Bitboard {
        Bitboard(1u64 << self.0)
    }
}

impl fmt::Debug for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_algebraic())
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_algebraic())
    }
}

/// A 64-bit bitboard for efficient set operations on squares.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Bitboard(pub u64);

impl Bitboard {
    pub const EMPTY: Bitboard = Bitboard(0);
    pub const ALL: Bitboard = Bitboard(u64::MAX);

    pub const RANK_1: Bitboard = Bitboard(0xFF);
    pub const RANK_2: Bitboard = Bitboard(0xFF << 8);
    pub const RANK_3: Bitboard = Bitboard(0xFF << 16);
    pub const RANK_4: Bitboard = Bitboard(0xFF << 24);
    pub const RANK_5: Bitboard = Bitboard(0xFF << 32);
    pub const RANK_6: Bitboard = Bitboard(0xFF << 40);
    pub const RANK_7: Bitboard = Bitboard(0xFF << 48);
    pub const RANK_8: Bitboard = Bitboard(0xFF << 56);

    pub const FILE_A: Bitboard = Bitboard(0x0101_0101_0101_0101);
    pub const FILE_B: Bitboard = Bitboard(0x0202_0202_0202_0202);
    pub const FILE_C: Bitboard = Bitboard(0x0404_0404_0404_0404);
    pub const FILE_D: Bitboard = Bitboard(0x0808_0808_0808_0808);
    pub const FILE_E: Bitboard = Bitboard(0x1010_1010_1010_1010);
    pub const FILE_F: Bitboard = Bitboard(0x2020_2020_2020_2020);
    pub const FILE_G: Bitboard = Bitboard(0x4040_4040_4040_4040);
    pub const FILE_H: Bitboard = Bitboard(0x8080_8080_8080_8080);

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub const fn is_set(self, sq: Square) -> bool {
        self.0 & (1u64 << sq.0) != 0
    }

    #[inline]
    pub const fn set(self, sq: Square) -> Self {
        Bitboard(self.0 | (1u64 << sq.0))
    }

    #[inline]
    pub const fn clear(self, sq: Square) -> Self {
        Bitboard(self.0 & !(1u64 << sq.0))
    }

    #[inline]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    #[inline]
    pub const fn lsb(self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            Some(Square(self.0.trailing_zeros() as u8))
        }
    }

    #[inline]
    pub const fn pop_lsb(self) -> (Option<Square>, Bitboard) {
        if self.0 == 0 {
            (None, self)
        } else {
            let sq = Square(self.0.trailing_zeros() as u8);
            (Some(sq), Bitboard(self.0 & (self.0 - 1)))
        }
    }

    /// Iterate over all set squares.
    pub fn iter(self) -> BitboardIter {
        BitboardIter(self)
    }

    // Shift operations (with edge clipping)
    #[inline]
    pub const fn north(self) -> Self {
        Bitboard(self.0 << 8)
    }

    #[inline]
    pub const fn south(self) -> Self {
        Bitboard(self.0 >> 8)
    }

    #[inline]
    pub const fn east(self) -> Self {
        Bitboard((self.0 << 1) & !Bitboard::FILE_A.0)
    }

    #[inline]
    pub const fn west(self) -> Self {
        Bitboard((self.0 >> 1) & !Bitboard::FILE_H.0)
    }

    #[inline]
    pub const fn north_east(self) -> Self {
        Bitboard((self.0 << 9) & !Bitboard::FILE_A.0)
    }

    #[inline]
    pub const fn north_west(self) -> Self {
        Bitboard((self.0 << 7) & !Bitboard::FILE_H.0)
    }

    #[inline]
    pub const fn south_east(self) -> Self {
        Bitboard((self.0 >> 7) & !Bitboard::FILE_A.0)
    }

    #[inline]
    pub const fn south_west(self) -> Self {
        Bitboard((self.0 >> 9) & !Bitboard::FILE_H.0)
    }
}

impl std::ops::BitAnd for Bitboard {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Bitboard(self.0 & rhs.0)
    }
}

impl std::ops::BitOr for Bitboard {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Bitboard(self.0 | rhs.0)
    }
}

impl std::ops::BitXor for Bitboard {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Bitboard(self.0 ^ rhs.0)
    }
}

impl std::ops::Not for Bitboard {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Bitboard(!self.0)
    }
}

impl std::ops::BitAndAssign for Bitboard {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl std::ops::BitOrAssign for Bitboard {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        for rank in (0..8).rev() {
            write!(f, "  {} ", rank + 1)?;
            for file in 0..8 {
                let sq = Square::new(file, rank);
                if self.is_set(sq) {
                    write!(f, "X ")?;
                } else {
                    write!(f, ". ")?;
                }
            }
            writeln!(f)?;
        }
        writeln!(f, "    a b c d e f g h")
    }
}

/// Iterator over set bits in a bitboard.
pub struct BitboardIter(Bitboard);

impl Iterator for BitboardIter {
    type Item = Square;

    #[inline]
    fn next(&mut self) -> Option<Square> {
        let (sq, rest) = self.0.pop_lsb();
        self.0 = rest;
        sq
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.0.count() as usize;
        (n, Some(n))
    }
}

impl ExactSizeIterator for BitboardIter {}

/// Castling rights packed into a u8 bitmask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CastlingRights(pub u8);

impl CastlingRights {
    pub const NONE: Self = Self(0);
    pub const WHITE_KINGSIDE: u8 = 0b0001;
    pub const WHITE_QUEENSIDE: u8 = 0b0010;
    pub const BLACK_KINGSIDE: u8 = 0b0100;
    pub const BLACK_QUEENSIDE: u8 = 0b1000;
    pub const ALL: Self = Self(0b1111);

    #[inline]
    pub const fn has(self, flag: u8) -> bool {
        self.0 & flag != 0
    }

    #[inline]
    pub const fn remove(self, flag: u8) -> Self {
        Self(self.0 & !flag)
    }

    #[inline]
    pub const fn add(self, flag: u8) -> Self {
        Self(self.0 | flag)
    }

    pub fn from_fen(s: &str) -> Self {
        if s == "-" {
            return Self::NONE;
        }
        let mut rights = 0u8;
        for c in s.chars() {
            match c {
                'K' => rights |= Self::WHITE_KINGSIDE,
                'Q' => rights |= Self::WHITE_QUEENSIDE,
                'k' => rights |= Self::BLACK_KINGSIDE,
                'q' => rights |= Self::BLACK_QUEENSIDE,
                _ => {}
            }
        }
        Self(rights)
    }

    pub fn to_fen(self) -> String {
        if self.0 == 0 {
            return "-".to_string();
        }
        let mut s = String::with_capacity(4);
        if self.has(Self::WHITE_KINGSIDE) {
            s.push('K');
        }
        if self.has(Self::WHITE_QUEENSIDE) {
            s.push('Q');
        }
        if self.has(Self::BLACK_KINGSIDE) {
            s.push('k');
        }
        if self.has(Self::BLACK_QUEENSIDE) {
            s.push('q');
        }
        s
    }
}

/// Evaluation score in centipawns. Positive favors White.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Score(pub i32);

impl Score {
    pub const ZERO: Self = Self(0);
    pub const MATE: Self = Self(30_000);
    pub const NEG_MATE: Self = Self(-30_000);
    pub const DRAW: Self = Self(0);
    pub const INF: Self = Self(i32::MAX);
    pub const NEG_INF: Self = Self(i32::MIN + 1); // +1 to avoid overflow on negation

    #[inline]
    pub const fn is_mate(self) -> bool {
        self.0.abs() >= 29_000
    }

    #[inline]
    pub const fn centipawns(self) -> i32 {
        self.0
    }
}

impl fmt::Display for Score {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_mate() {
            let moves_to_mate = (Score::MATE.0 - self.0.abs() + 1) / 2;
            if self.0 > 0 {
                write!(f, "mate {moves_to_mate}")
            } else {
                write!(f, "mate -{moves_to_mate}")
            }
        } else {
            write!(f, "cp {}", self.0)
        }
    }
}

/// Game result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    Ongoing,
    WhiteWins,
    BlackWins,
    Draw(DrawReason),
}

/// Reason for a draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawReason {
    Stalemate,
    FiftyMoveRule,
    ThreefoldRepetition,
    InsufficientMaterial,
    Agreement,
}
