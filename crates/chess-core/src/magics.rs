//! Magic bitboard attack generation for sliding pieces.
//!
//! Replaces O(n) ray iteration with O(1) table lookups for bishop and rook
//! attacks. Tables are lazily initialized on first use. Magic numbers are
//! found deterministically at init time using a fixed-seed PRNG.
//!
//! Total memory: ~840 KB (bishops ~42 KB, rooks ~800 KB).

use std::sync::LazyLock;

use chess_common::{Bitboard, Square};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Bishop attacks via magic bitboard lookup.
#[inline]
pub fn bishop_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
    let table = &*BISHOP_TABLE;
    let entry = &table.entries[sq.index()];
    let idx = entry.offset
        + ((occupancy.0 & entry.mask).wrapping_mul(entry.magic) >> entry.shift) as usize;
    table.attacks[idx]
}

/// Rook attacks via magic bitboard lookup.
#[inline]
pub fn rook_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
    let table = &*ROOK_TABLE;
    let entry = &table.entries[sq.index()];
    let idx = entry.offset
        + ((occupancy.0 & entry.mask).wrapping_mul(entry.magic) >> entry.shift) as usize;
    table.attacks[idx]
}

// ---------------------------------------------------------------------------
// Static tables (lazy init)
// ---------------------------------------------------------------------------

static BISHOP_TABLE: LazyLock<MagicTable> = LazyLock::new(|| build_table(true));
static ROOK_TABLE: LazyLock<MagicTable> = LazyLock::new(|| build_table(false));

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct MagicEntry {
    mask: u64,
    magic: u64,
    shift: u32,
    offset: usize,
}

struct MagicTable {
    entries: [MagicEntry; 64],
    attacks: Box<[Bitboard]>,
}

const EMPTY_ENTRY: MagicEntry = MagicEntry {
    mask: 0,
    magic: 0,
    shift: 64,
    offset: 0,
};

// ---------------------------------------------------------------------------
// Table construction
// ---------------------------------------------------------------------------

fn build_table(is_bishop: bool) -> MagicTable {
    let mut entries = [EMPTY_ENTRY; 64];
    let mut total_size = 0usize;

    // First pass: compute masks and allocate offsets.
    for (sq, entry) in entries.iter_mut().enumerate() {
        let mask = relevant_mask(sq, is_bishop);
        let bits = mask.count_ones();
        *entry = MagicEntry {
            mask,
            magic: 0,
            shift: 64 - bits,
            offset: total_size,
        };
        total_size += 1usize << bits;
    }

    let mut attacks = vec![Bitboard::EMPTY; total_size];
    let seed = if is_bishop {
        0xBEEF_CAFE_1234_5678
    } else {
        0xDEAD_FACE_8765_4321
    };
    let mut rng = Rng::new(seed);

    // Second pass: find magics and fill attack tables.
    for (sq, entry) in entries.iter_mut().enumerate() {
        let mask = entry.mask;
        let shift = entry.shift;
        let offset = entry.offset;
        let table_size = 1usize << (64 - shift);

        // Enumerate all occupancy subsets and their reference attacks.
        let subsets = enumerate_subsets(mask);
        let reference: Vec<u64> = subsets
            .iter()
            .map(|&occ| sliding_attacks_reference(sq, occ, is_bishop))
            .collect();

        let magic = find_magic(&subsets, &reference, mask, shift, table_size, &mut rng);
        entry.magic = magic;

        // Populate the attack table for this square.
        for (i, &subset) in subsets.iter().enumerate() {
            let idx = (subset.wrapping_mul(magic) >> shift) as usize;
            attacks[offset + idx] = Bitboard(reference[i]);
        }
    }

    MagicTable {
        entries,
        attacks: attacks.into_boxed_slice(),
    }
}

fn find_magic(
    subsets: &[u64],
    reference: &[u64],
    mask: u64,
    shift: u32,
    table_size: usize,
    rng: &mut Rng,
) -> u64 {
    // Sentinel value: no valid attack bitboard equals u64::MAX
    // (that would mean every square is attacked, impossible for one piece).
    let mut used = vec![u64::MAX; table_size];

    loop {
        let magic = rng.sparse();

        // Quick filter: mask * magic should spread enough bits into the top byte.
        if (mask.wrapping_mul(magic) & 0xFF00_0000_0000_0000).count_ones() < 6 {
            continue;
        }

        // Reset used table.
        for entry in used.iter_mut() {
            *entry = u64::MAX;
        }

        let mut ok = true;
        for (i, &subset) in subsets.iter().enumerate() {
            let idx = (subset.wrapping_mul(magic) >> shift) as usize;
            if used[idx] == u64::MAX {
                used[idx] = reference[i];
            } else if used[idx] != reference[i] {
                ok = false;
                break;
            }
        }

        if ok {
            return magic;
        }
    }
}

// ---------------------------------------------------------------------------
// Reference attack computation (used only during table init)
// ---------------------------------------------------------------------------

/// Compute sliding piece attacks using simple ray-stepping.
fn sliding_attacks_reference(sq: usize, occupancy: u64, is_bishop: bool) -> u64 {
    let dirs: &[(i8, i8)] = if is_bishop {
        &[(1, 1), (1, -1), (-1, 1), (-1, -1)]
    } else {
        &[(0, 1), (0, -1), (1, 0), (-1, 0)]
    };

    let file = (sq % 8) as i8;
    let rank = (sq / 8) as i8;
    let mut attacks = 0u64;

    for &(df, dr) in dirs {
        let mut f = file + df;
        let mut r = rank + dr;
        while (0..8).contains(&f) && (0..8).contains(&r) {
            let s = (r * 8 + f) as u32;
            attacks |= 1u64 << s;
            if occupancy & (1u64 << s) != 0 {
                break; // blocker found; include it but stop
            }
            f += df;
            r += dr;
        }
    }

    attacks
}

/// Compute the relevant occupancy mask for a square.
/// Excludes edge squares along each ray (they don't change the attack set
/// beyond themselves, since there are no further squares in that direction).
fn relevant_mask(sq: usize, is_bishop: bool) -> u64 {
    let dirs: &[(i8, i8)] = if is_bishop {
        &[(1, 1), (1, -1), (-1, 1), (-1, -1)]
    } else {
        &[(0, 1), (0, -1), (1, 0), (-1, 0)]
    };

    let file = (sq % 8) as i8;
    let rank = (sq / 8) as i8;
    let mut mask = 0u64;

    for &(df, dr) in dirs {
        let mut f = file + df;
        let mut r = rank + dr;
        while (0..8).contains(&f) && (0..8).contains(&r) {
            // If the next step would leave the board, this is an edge square — skip it.
            if f + df < 0 || f + df >= 8 || r + dr < 0 || r + dr >= 8 {
                break;
            }
            mask |= 1u64 << (r * 8 + f);
            f += df;
            r += dr;
        }
    }

    mask
}

/// Enumerate all subsets of a mask using the carry-rippler trick.
fn enumerate_subsets(mask: u64) -> Vec<u64> {
    let mut subsets = Vec::with_capacity(1 << mask.count_ones());
    let mut subset = 0u64;
    loop {
        subsets.push(subset);
        subset = subset.wrapping_sub(mask) & mask;
        if subset == 0 {
            break;
        }
    }
    subsets
}

// ---------------------------------------------------------------------------
// Deterministic PRNG for magic number generation
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Sparse random number (few bits set) — good magic candidates.
    fn sparse(&mut self) -> u64 {
        self.next() & self.next() & self.next()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bishop_attacks_empty_board() {
        let sq = Square::from_algebraic("e4").unwrap();
        let attacks = bishop_attacks(sq, Bitboard::EMPTY);
        assert_eq!(attacks.count(), 13);
    }

    #[test]
    fn test_rook_attacks_empty_board() {
        let sq = Square::from_algebraic("e4").unwrap();
        let attacks = rook_attacks(sq, Bitboard::EMPTY);
        assert_eq!(attacks.count(), 14);
    }

    #[test]
    fn test_rook_attacks_blocked() {
        let sq = Square::from_algebraic("e4").unwrap();
        let blocker = Square::from_algebraic("e6").unwrap().bitboard();
        let attacks = rook_attacks(sq, blocker);
        assert!(attacks.is_set(Square::from_algebraic("e5").unwrap()));
        assert!(attacks.is_set(Square::from_algebraic("e6").unwrap()));
        assert!(!attacks.is_set(Square::from_algebraic("e7").unwrap()));
    }

    #[test]
    fn test_bishop_attacks_corner() {
        let sq = Square::A1;
        let attacks = bishop_attacks(sq, Bitboard::EMPTY);
        assert_eq!(attacks.count(), 7);
        assert!(attacks.is_set(Square::from_algebraic("b2").unwrap()));
        assert!(attacks.is_set(Square::H8));
    }

    #[test]
    fn test_magic_matches_reference_all_squares() {
        for sq in 0..64u8 {
            let square = Square(sq);

            // Empty board
            let magic_b = bishop_attacks(square, Bitboard::EMPTY);
            let ref_b = sliding_attacks_reference(sq as usize, 0, true);
            assert_eq!(magic_b.0, ref_b, "bishop mismatch sq={} empty", sq);

            let magic_r = rook_attacks(square, Bitboard::EMPTY);
            let ref_r = sliding_attacks_reference(sq as usize, 0, false);
            assert_eq!(magic_r.0, ref_r, "rook mismatch sq={} empty", sq);

            // Full board
            let full = Bitboard::ALL;
            let magic_bf = bishop_attacks(square, full);
            let ref_bf = sliding_attacks_reference(sq as usize, full.0, true);
            assert_eq!(magic_bf.0, ref_bf, "bishop mismatch sq={} full", sq);

            let magic_rf = rook_attacks(square, full);
            let ref_rf = sliding_attacks_reference(sq as usize, full.0, false);
            assert_eq!(magic_rf.0, ref_rf, "rook mismatch sq={} full", sq);
        }
    }

    #[test]
    fn test_all_subsets_bishop() {
        for sq in 0..64usize {
            let mask = relevant_mask(sq, true);
            let subsets = enumerate_subsets(mask);
            for &subset in &subsets {
                let magic = bishop_attacks(Square(sq as u8), Bitboard(subset));
                let reference = sliding_attacks_reference(sq, subset, true);
                assert_eq!(
                    magic.0, reference,
                    "bishop subset mismatch sq={} occ={:#x}",
                    sq, subset
                );
            }
        }
    }

    #[test]
    fn test_all_subsets_rook() {
        for sq in 0..64usize {
            let mask = relevant_mask(sq, false);
            let subsets = enumerate_subsets(mask);
            for &subset in &subsets {
                let magic = rook_attacks(Square(sq as u8), Bitboard(subset));
                let reference = sliding_attacks_reference(sq, subset, false);
                assert_eq!(
                    magic.0, reference,
                    "rook subset mismatch sq={} occ={:#x}",
                    sq, subset
                );
            }
        }
    }
}
