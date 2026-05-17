use crate::moves::{Move, MoveFlag};
use crate::types::{Bitboard, CastlingRights, Color, Piece, PieceKind, Square};
use std::fmt;
use std::sync::LazyLock;

/// The full state of a chess position.
#[derive(Clone, PartialEq, Eq)]
pub struct Board {
    /// Bitboards for each piece type per color: pieces[color][piece_kind]
    pub pieces: [[Bitboard; PieceKind::COUNT]; 2],
    /// Mailbox board for O(1) piece lookup by square.
    pub mailbox: [Option<Piece>; 64],
    /// Occupancy bitboard per color.
    pub occupancy: [Bitboard; 2],
    /// Side to move.
    pub side_to_move: Color,
    /// Castling rights.
    pub castling: CastlingRights,
    /// En passant target square (the square a pawn can capture to).
    pub en_passant: Option<Square>,
    /// Halfmove clock for the 50-move rule.
    pub halfmove_clock: u16,
    /// Fullmove number (starts at 1, incremented after Black moves).
    pub fullmove_number: u16,
    /// Zobrist hash for position identification.
    pub hash: u64,
    /// History of position hashes for repetition detection.
    pub position_history: Vec<u64>,
}

impl Board {
    /// Create a new board from the standard starting position.
    pub fn starting_position() -> Self {
        Self::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("starting FEN should always be valid")
    }

    /// Get the piece at a given square.
    #[inline]
    pub fn piece_at(&self, sq: Square) -> Option<Piece> {
        self.mailbox[sq.index()]
    }

    /// Combined occupancy of both sides.
    #[inline]
    pub fn all_occupancy(&self) -> Bitboard {
        self.occupancy[0] | self.occupancy[1]
    }

    /// Set a piece on the board.
    pub fn set_piece(&mut self, sq: Square, piece: Piece) {
        let ci = piece.color.index();
        let ki = piece.kind.index();
        self.pieces[ci][ki] = self.pieces[ci][ki].set(sq);
        self.occupancy[ci] = self.occupancy[ci].set(sq);
        self.mailbox[sq.index()] = Some(piece);
    }

    /// Remove a piece from the board.
    pub fn remove_piece(&mut self, sq: Square, piece: Piece) {
        let ci = piece.color.index();
        let ki = piece.kind.index();
        self.pieces[ci][ki] = self.pieces[ci][ki].clear(sq);
        self.occupancy[ci] = self.occupancy[ci].clear(sq);
        self.mailbox[sq.index()] = None;
    }

    /// Make a move on the board, returning the captured piece (if any).
    pub fn make_move(&mut self, m: Move) -> Option<Piece> {
        let from = m.from_sq();
        let to = m.to_sq();
        let flag = m.flag();
        let us = self.side_to_move;
        let them = us.opposite();

        // Save hash for repetition detection (pre-move hash)
        self.position_history.push(self.hash);

        // Snapshot state needed for incremental hash update
        let old_ep = self.en_passant;

        let moving_piece = match self.piece_at(from) {
            Some(p) => p,
            None => {
                log::error!("make_move: no piece at from square {} for move {}", from, m);
                self.position_history.pop();
                return None;
            }
        };
        let mut captured = None;

        // Handle capture (remove enemy piece at destination)
        if flag.is_capture()
            && flag != MoveFlag::EnPassant
            && let Some(cap) = self.piece_at(to)
        {
            self.remove_piece(to, cap);
            captured = Some(cap);
        }

        // Handle en passant capture
        if flag == MoveFlag::EnPassant {
            // Guard against TT hash collisions producing an EnPassant move where `to`
            // is on an impossible rank (rank 0 for white, rank 7 for black), which
            // would underflow the u8 rank arithmetic and produce Square(255+).
            let ep_capture_sq = match us {
                Color::White if to.rank() > 0 => Some(Square::new(to.file(), to.rank() - 1)),
                Color::Black if to.rank() < 7 => Some(Square::new(to.file(), to.rank() + 1)),
                _ => None, // Invalid rank from hash collision — skip ep capture
            };
            if let Some(sq) = ep_capture_sq {
                let cap = Piece::new(PieceKind::Pawn, them);
                self.remove_piece(sq, cap);
                captured = Some(cap);
            }
        }

        // Safety: clear any piece at `to` not already removed by the explicit capture
        // handling above.  In a legal game this is always a no-op.  Under a TT hash
        // collision a "quiet" move can target an enemy-occupied square; without this,
        // the enemy's piece-type and occupancy bits are never cleared, producing
        // cross-color occupancy overlap that corrupts Syzygy probing (both color
        // occupancies share the square → pyrrhic's material key overcounts pieces →
        // wrong TB entry → sq=64 from poplsb(0) → OOB in OFF_DIAG / BINOMIAL → SEGV).
        if let Some(ghost) = self.piece_at(to) {
            self.remove_piece(to, ghost);
        }

        // Move the piece
        self.remove_piece(from, moving_piece);

        // Handle promotion; track the piece kind that ends up on `to`
        let placed_kind = if flag.is_promotion() {
            let promo_kind = flag.promotion_piece().unwrap();
            self.set_piece(to, Piece::new(promo_kind, us));
            promo_kind
        } else {
            self.set_piece(to, moving_piece);
            moving_piece.kind
        };

        // Handle castling (move the rook)
        match flag {
            MoveFlag::KingsideCastle => {
                let (rook_from, rook_to) = match us {
                    Color::White => (Square::H1, Square::F1),
                    Color::Black => (Square::H8, Square::F8),
                };
                let rook = Piece::new(PieceKind::Rook, us);
                self.remove_piece(rook_from, rook);
                self.set_piece(rook_to, rook);
            }
            MoveFlag::QueensideCastle => {
                let (rook_from, rook_to) = match us {
                    Color::White => (Square::A1, Square::D1),
                    Color::Black => (Square::A8, Square::D8),
                };
                let rook = Piece::new(PieceKind::Rook, us);
                self.remove_piece(rook_from, rook);
                self.set_piece(rook_to, rook);
            }
            _ => {}
        }

        // Update en passant square
        // Guard: a TT hash collision can produce a DoublePawnPush from an impossible rank
        // (rank 7 for white, rank 0 for black), which would overflow the u8 rank arithmetic
        // and produce Square(64+) — an out-of-bounds index that causes a panic in attacks.rs.
        self.en_passant = if flag == MoveFlag::DoublePawnPush {
            match us {
                Color::White if from.rank() < 7 => Some(Square::new(from.file(), from.rank() + 1)),
                Color::Black if from.rank() > 0 => Some(Square::new(from.file(), from.rank() - 1)),
                _ => None, // Invalid rank from hash collision — suppress ep square
            }
        } else {
            None
        };

        // Save castling rights before they change, then update them
        let old_castling = self.castling;
        self.update_castling_rights(from, to);

        // Update halfmove clock
        if moving_piece.kind == PieceKind::Pawn || captured.is_some() {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }

        // Update fullmove number
        if us == Color::Black {
            self.fullmove_number += 1;
        }

        // Switch side to move
        self.side_to_move = them;

        // Incremental Zobrist hash update
        {
            let z = &*ZOBRIST;
            // Moving piece leaves source, arrives at destination
            self.hash ^= z.piece_sq[us.index()][moving_piece.kind.index()][from.index()];
            self.hash ^= z.piece_sq[us.index()][placed_kind.index()][to.index()];
            // Captured piece is removed from the board
            if let Some(cap) = captured {
                if flag == MoveFlag::EnPassant {
                    let ep_cap_sq = match us {
                        Color::White => Square::new(to.file(), to.rank() - 1),
                        Color::Black => Square::new(to.file(), to.rank() + 1),
                    };
                    self.hash ^= z.piece_sq[them.index()][PieceKind::Pawn.index()][ep_cap_sq.index()];
                } else {
                    self.hash ^= z.piece_sq[them.index()][cap.kind.index()][to.index()];
                }
            }
            // Castling rook teleports
            match flag {
                MoveFlag::KingsideCastle => {
                    let (rook_from, rook_to) = match us {
                        Color::White => (Square::H1, Square::F1),
                        Color::Black => (Square::H8, Square::F8),
                    };
                    self.hash ^= z.piece_sq[us.index()][PieceKind::Rook.index()][rook_from.index()];
                    self.hash ^= z.piece_sq[us.index()][PieceKind::Rook.index()][rook_to.index()];
                }
                MoveFlag::QueensideCastle => {
                    let (rook_from, rook_to) = match us {
                        Color::White => (Square::A1, Square::D1),
                        Color::Black => (Square::A8, Square::D8),
                    };
                    self.hash ^= z.piece_sq[us.index()][PieceKind::Rook.index()][rook_from.index()];
                    self.hash ^= z.piece_sq[us.index()][PieceKind::Rook.index()][rook_to.index()];
                }
                _ => {}
            }
            // En-passant file contribution changes
            if let Some(ep) = old_ep {
                self.hash ^= z.ep_file[ep.file() as usize];
            }
            if let Some(ep) = self.en_passant {
                self.hash ^= z.ep_file[ep.file() as usize];
            }
            // Castling rights change
            self.hash ^= z.castling[old_castling.0 as usize];
            self.hash ^= z.castling[self.castling.0 as usize];
            // Side to move flips
            self.hash ^= z.side;
        }

        captured
    }

    /// Unmake a move (requires the captured piece info).
    pub fn unmake_move(
        &mut self,
        m: Move,
        captured: Option<Piece>,
        prev_castling: CastlingRights,
        prev_en_passant: Option<Square>,
        prev_halfmove: u16,
    ) {
        let from = m.from_sq();
        let to = m.to_sq();
        let flag = m.flag();

        // Switch side back
        self.side_to_move = self.side_to_move.opposite();
        let us = self.side_to_move;
        let them = us.opposite();

        // Restore state
        self.castling = prev_castling;
        self.en_passant = prev_en_passant;
        self.halfmove_clock = prev_halfmove;
        if us == Color::Black {
            self.fullmove_number -= 1;
        }

        // Identify the piece at `to` BEFORE removing it
        let moved_piece_kind = if flag.is_promotion() {
            // The piece at `to` is the promoted piece, but the original was a pawn
            PieceKind::Pawn
        } else {
            // The piece at `to` is what was moved.
            // If the square is empty (board corruption from TT collision),
            // recover gracefully instead of panicking.
            match self.piece_at(to) {
                Some(p) => p.kind,
                None => {
                    log::error!(
                        "unmake_move: no piece at to square {} for move {}, recovering",
                        to, m
                    );
                    // Best-effort recovery: restore state and return.
                    self.side_to_move = us;
                    self.castling = prev_castling;
                    self.en_passant = prev_en_passant;
                    self.halfmove_clock = prev_halfmove;
                    if us == Color::Black {
                        self.fullmove_number += 1;
                    }
                    self.hash = self.position_history.pop().unwrap_or(0);
                    return;
                }
            }
        };

        // Remove piece from destination
        if flag.is_promotion() {
            let promo_kind = flag.promotion_piece().unwrap();
            self.remove_piece(to, Piece::new(promo_kind, us));
        } else {
            self.remove_piece(to, Piece::new(moved_piece_kind, us));
        }

        // Restore original piece to source square
        self.set_piece(from, Piece::new(moved_piece_kind, us));

        // Restore captured piece
        if flag == MoveFlag::EnPassant {
            // Same rank guards as make_move: hash collision could produce an EnPassant
            // unmake with `to` on an impossible rank, causing u8 underflow/overflow.
            let ep_sq = match us {
                Color::White if to.rank() > 0 => Some(Square::new(to.file(), to.rank() - 1)),
                Color::Black if to.rank() < 7 => Some(Square::new(to.file(), to.rank() + 1)),
                _ => None,
            };
            if let Some(sq) = ep_sq {
                self.set_piece(sq, Piece::new(PieceKind::Pawn, them));
            }
        } else if let Some(cap) = captured {
            self.set_piece(to, cap);
        }

        // Undo castling rook move
        match flag {
            MoveFlag::KingsideCastle => {
                let (rook_from, rook_to) = match us {
                    Color::White => (Square::H1, Square::F1),
                    Color::Black => (Square::H8, Square::F8),
                };
                let rook = Piece::new(PieceKind::Rook, us);
                self.remove_piece(rook_to, rook);
                self.set_piece(rook_from, rook);
            }
            MoveFlag::QueensideCastle => {
                let (rook_from, rook_to) = match us {
                    Color::White => (Square::A1, Square::D1),
                    Color::Black => (Square::A8, Square::D8),
                };
                let rook = Piece::new(PieceKind::Rook, us);
                self.remove_piece(rook_to, rook);
                self.set_piece(rook_from, rook);
            }
            _ => {}
        }

        // Restore hash from the snapshot saved by make_move
        self.hash = self.position_history.pop().unwrap_or(0);
    }

    fn update_castling_rights(&mut self, from: Square, to: Square) {
        // King moves remove both castling rights for that side
        if from == Square::E1 {
            self.castling = self
                .castling
                .remove(CastlingRights::WHITE_KINGSIDE | CastlingRights::WHITE_QUEENSIDE);
        }
        if from == Square::E8 {
            self.castling = self
                .castling
                .remove(CastlingRights::BLACK_KINGSIDE | CastlingRights::BLACK_QUEENSIDE);
        }
        // Rook moves or captures remove the relevant castling right
        if from == Square::A1 || to == Square::A1 {
            self.castling = self.castling.remove(CastlingRights::WHITE_QUEENSIDE);
        }
        if from == Square::H1 || to == Square::H1 {
            self.castling = self.castling.remove(CastlingRights::WHITE_KINGSIDE);
        }
        if from == Square::A8 || to == Square::A8 {
            self.castling = self.castling.remove(CastlingRights::BLACK_QUEENSIDE);
        }
        if from == Square::H8 || to == Square::H8 {
            self.castling = self.castling.remove(CastlingRights::BLACK_KINGSIDE);
        }
    }

    /// Compute the Zobrist hash from scratch for this position.
    /// Used for initialization (FEN parsing). During search the hash
    /// is maintained incrementally by make_move / unmake_move.
    pub fn compute_hash(&self) -> u64 {
        let z = &*ZOBRIST;
        let mut h = 0u64;
        for ci in 0..2 {
            for ki in 0..PieceKind::COUNT {
                for sq in self.pieces[ci][ki].iter() {
                    h ^= z.piece_sq[ci][ki][sq.index()];
                }
            }
        }
        h ^= z.castling[self.castling.0 as usize];
        if let Some(ep) = self.en_passant {
            h ^= z.ep_file[ep.file() as usize];
        }
        if self.side_to_move == Color::Black {
            h ^= z.side;
        }
        h
    }

    /// Apply the incremental Zobrist update for a null move.
    /// XORs out the current en-passant file contribution (if any) and flips
    /// the side-to-move bit.  Must be called BEFORE the caller clears
    /// `en_passant` and flips `side_to_move`.
    pub fn update_hash_for_null_move(&mut self) {
        let z = &*ZOBRIST;
        if let Some(ep) = self.en_passant {
            self.hash ^= z.ep_file[ep.file() as usize];
        }
        self.hash ^= z.side;
    }

    /// Check if the position has occurred before (for threefold repetition).
    pub fn is_repetition(&self) -> bool {
        let target = self.hash;
        let mut count = 0;
        for &h in self.position_history.iter().rev() {
            if h == target {
                count += 1;
                if count >= 2 {
                    return true;
                }
            }
        }
        false
    }

    /// Check if the current position has occurred at least once before in the
    /// current search path (indices >= `game_ply` in position_history).
    /// Using twofold detection in the search tree is standard practice: since
    /// the opponent can always force a third repetition, a second occurrence is
    /// sufficient to score the position as a draw.
    pub fn is_twofold_in_search(&self, game_ply: usize) -> bool {
        let target = self.hash;
        let len = self.position_history.len();
        if len <= game_ply {
            return false;
        }
        let search_len = len - game_ply;
        let limit = (self.halfmove_clock as usize).min(search_len);
        for i in 0..limit {
            if self.position_history[len - 1 - i] == target {
                return true;
            }
        }
        false
    }

    /// Check if the position has occurred at least once before (twofold).
    /// Only checks within the last `halfmove_clock` entries since pawn
    /// moves and captures make earlier positions unreachable.
    /// `game_ply` limits the check to the most recent `game_ply` entries.
    pub fn has_repeated(&self, game_ply: usize) -> bool {
        let target = self.hash;
        let len = self.position_history.len();
        let limit = (self.halfmove_clock as usize).min(len).min(game_ply);
        for i in 0..limit {
            if self.position_history[len - 1 - i] == target {
                return true;
            }
        }
        false
    }

    /// Returns the king square for the given color.
    pub fn king_square(&self, color: Color) -> Square {
        self.pieces[color.index()][PieceKind::King.index()]
            .lsb()
            .unwrap_or_else(|| {
                log::error!("king_square: no king for {:?}, board corrupted", color);
                // Return a sentinel square so the caller does not crash.
                // The search will detect an illegal position and back out.
                Square::A1
            })
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::starting_position()
    }
}

impl fmt::Debug for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        for rank in (0..8).rev() {
            write!(f, "  {} ", rank + 1)?;
            for file in 0..8 {
                let sq = Square::new(file, rank);
                match self.piece_at(sq) {
                    Some(piece) => write!(f, "{} ", piece.kind.to_char(piece.color))?,
                    None => write!(f, ". ")?,
                }
            }
            writeln!(f)?;
        }
        writeln!(f, "    a b c d e f g h")?;
        writeln!(f, "  Side to move: {}", self.side_to_move)?;
        writeln!(f, "  Castling: {}", self.castling.to_fen())?;
        writeln!(
            f,
            "  En passant: {}",
            self.en_passant
                .map(|s| s.to_algebraic())
                .unwrap_or_else(|| "-".to_string())
        )?;
        writeln!(f, "  Halfmove clock: {}", self.halfmove_clock)?;
        writeln!(f, "  Fullmove: {}", self.fullmove_number)
    }
}

// ---------------------------------------------------------------------------
// Zobrist hashing
// ---------------------------------------------------------------------------

struct ZobristKeys {
    /// piece_sq[color_idx][piece_kind_idx][square_idx]
    piece_sq: Box<[[[u64; 64]; 6]; 2]>,
    /// One key per castling-rights combination (4 bits → 16 values).
    castling: [u64; 16],
    /// One key per en-passant file (0 = a-file … 7 = h-file).
    ep_file: [u64; 8],
    /// XOR'd into the hash when it is Black's turn to move.
    side: u64,
}

static ZOBRIST: LazyLock<ZobristKeys> = LazyLock::new(|| {
    #[inline]
    fn splitmix(s: &mut u64) -> u64 {
        *s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut x = *s;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
        x ^ (x >> 31)
    }

    let mut s: u64 = 0x246C_CB2A_3B12_4D05; // fixed seed — must never change

    let mut piece_sq: Box<[[[u64; 64]; 6]; 2]> =
        vec![[[0u64; 64]; 6]; 2].into_boxed_slice().try_into().unwrap();
    for color in 0..2 {
        for kind in 0..6 {
            for sq in 0..64 {
                piece_sq[color][kind][sq] = splitmix(&mut s);
            }
        }
    }

    let mut castling = [0u64; 16];
    for c in castling.iter_mut() {
        *c = splitmix(&mut s);
    }

    let mut ep_file = [0u64; 8];
    for f in ep_file.iter_mut() {
        *f = splitmix(&mut s);
    }

    let side = splitmix(&mut s);

    ZobristKeys { piece_sq, castling, ep_file, side }
});
