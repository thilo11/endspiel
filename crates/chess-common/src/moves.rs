use crate::types::{PieceKind, Square};
use std::fmt;

/// Flags for special move types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MoveFlag {
    Normal = 0,
    DoublePawnPush = 1,
    KingsideCastle = 2,
    QueensideCastle = 3,
    Capture = 4,
    EnPassant = 5,
    /// Promotion (possibly with capture, indicated by captured piece).
    PromoteKnight = 8,
    PromoteBishop = 9,
    PromoteRook = 10,
    PromoteQueen = 11,
    PromoteCaptureKnight = 12,
    PromoteCaptureBishop = 13,
    PromoteCaptureRook = 14,
    PromoteCaptureQueen = 15,
}

impl MoveFlag {
    #[inline]
    pub const fn is_capture(self) -> bool {
        matches!(
            self,
            MoveFlag::Capture
                | MoveFlag::EnPassant
                | MoveFlag::PromoteCaptureKnight
                | MoveFlag::PromoteCaptureBishop
                | MoveFlag::PromoteCaptureRook
                | MoveFlag::PromoteCaptureQueen
        )
    }

    #[inline]
    pub const fn is_promotion(self) -> bool {
        self as u8 >= 8
    }

    #[inline]
    pub fn promotion_piece(self) -> Option<PieceKind> {
        match self {
            MoveFlag::PromoteKnight | MoveFlag::PromoteCaptureKnight => Some(PieceKind::Knight),
            MoveFlag::PromoteBishop | MoveFlag::PromoteCaptureBishop => Some(PieceKind::Bishop),
            MoveFlag::PromoteRook | MoveFlag::PromoteCaptureRook => Some(PieceKind::Rook),
            MoveFlag::PromoteQueen | MoveFlag::PromoteCaptureQueen => Some(PieceKind::Queen),
            _ => None,
        }
    }
}

/// A chess move encoded compactly.
///
/// Encoding: from (6 bits) | to (6 bits) | flag (4 bits) = 16 bits.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Move(pub u16);

impl Move {
    pub const NULL: Move = Move(0);

    #[inline]
    pub const fn new(from: Square, to: Square, flag: MoveFlag) -> Self {
        Move(from.0 as u16 | ((to.0 as u16) << 6) | ((flag as u16) << 12))
    }

    #[inline]
    pub const fn from_sq(self) -> Square {
        Square((self.0 & 0x3F) as u8)
    }

    #[inline]
    pub const fn to_sq(self) -> Square {
        Square(((self.0 >> 6) & 0x3F) as u8)
    }

    #[inline]
    pub const fn flag_bits(self) -> u8 {
        ((self.0 >> 12) & 0xF) as u8
    }

    pub fn flag(self) -> MoveFlag {
        match self.flag_bits() {
            0 => MoveFlag::Normal,
            1 => MoveFlag::DoublePawnPush,
            2 => MoveFlag::KingsideCastle,
            3 => MoveFlag::QueensideCastle,
            4 => MoveFlag::Capture,
            5 => MoveFlag::EnPassant,
            8 => MoveFlag::PromoteKnight,
            9 => MoveFlag::PromoteBishop,
            10 => MoveFlag::PromoteRook,
            11 => MoveFlag::PromoteQueen,
            12 => MoveFlag::PromoteCaptureKnight,
            13 => MoveFlag::PromoteCaptureBishop,
            14 => MoveFlag::PromoteCaptureRook,
            15 => MoveFlag::PromoteCaptureQueen,
            _ => MoveFlag::Normal,
        }
    }

    #[inline]
    pub fn is_capture(self) -> bool {
        self.flag().is_capture()
    }

    #[inline]
    pub fn is_promotion(self) -> bool {
        self.flag().is_promotion()
    }

    #[inline]
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    /// Parse a UCI move string (e.g., "e2e4", "e7e8q").
    pub fn from_uci(s: &str) -> Option<Self> {
        if s.len() < 4 || s.len() > 5 {
            return None;
        }
        let from = Square::from_algebraic(&s[0..2])?;
        let to = Square::from_algebraic(&s[2..4])?;

        let flag = if s.len() == 5 {
            match s.as_bytes()[4] {
                b'n' | b'N' => MoveFlag::PromoteKnight,
                b'b' | b'B' => MoveFlag::PromoteBishop,
                b'r' | b'R' => MoveFlag::PromoteRook,
                b'q' | b'Q' => MoveFlag::PromoteQueen,
                _ => return None,
            }
        } else {
            MoveFlag::Normal
        };

        Some(Move::new(from, to, flag))
    }

    /// Convert to UCI string (e.g., "e2e4", "e7e8q", "0000" for null).
    pub fn to_uci(self) -> String {
        if self.is_null() {
            return "0000".to_string();
        }
        let mut s = format!("{}{}", self.from_sq(), self.to_sq());
        if let Some(promo) = self.flag().promotion_piece() {
            s.push(match promo {
                PieceKind::Knight => 'n',
                PieceKind::Bishop => 'b',
                PieceKind::Rook => 'r',
                PieceKind::Queen => 'q',
                _ => unreachable!(),
            });
        }
        s
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

/// A pre-sized list of moves (avoids heap allocation during move generation).
pub struct MoveList {
    moves: [Move; 256],
    len: usize,
}

impl MoveList {
    pub fn new() -> Self {
        Self {
            moves: [Move::NULL; 256],
            len: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, m: Move) {
        debug_assert!(self.len < 256);
        self.moves[self.len] = m;
        self.len += 1;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }

    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, Move> {
        self.as_slice().iter()
    }

    pub fn contains(&self, m: &Move) -> bool {
        self.as_slice().contains(m)
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for MoveList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}
