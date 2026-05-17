//! Shared lock-free transposition table for Lazy SMP.
//!
//! Each entry is 16 bytes: two `AtomicU64` fields. XOR-based verification
//! detects torn reads without any locking. `Ordering::Relaxed` is used
//! throughout — the TT is a best-effort cache and occasional corruption
//! from races is acceptable.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

use chess_common::Move;

// ---------------------------------------------------------------------------
// TT entry flag (2 bits)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum TTFlag {
    Exact = 0,
    LowerBound = 1,
    UpperBound = 2,
}

impl TTFlag {
    fn from_u8(v: u8) -> Self {
        match v & 3 {
            0 => Self::Exact,
            1 => Self::LowerBound,
            _ => Self::UpperBound,
        }
    }
}

// ---------------------------------------------------------------------------
// Unpacked TT entry (for caller convenience)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct TTData {
    pub depth: u8,
    pub score: i32,
    pub flag: TTFlag,
    pub best_move: Move,
}

// ---------------------------------------------------------------------------
// Packed data layout (64 bits):
//   bits  0..15 : best_move (u16)
//   bits 16..31 : score (i16, clamped)
//   bits 32..39 : depth (u8)
//   bits 40..41 : flag (2 bits)
//   bits 42..49 : generation (8 bits)
//   bits 50..63 : unused
// ---------------------------------------------------------------------------

fn pack_data(depth: u8, score: i32, flag: TTFlag, best_move: Move, generation: u8) -> u64 {
    let m = best_move.0 as u64;
    let s = (score.clamp(-32000, 32000) as i16) as u16 as u64;
    let d = depth as u64;
    let f = flag as u64;
    let g = generation as u64;
    m | (s << 16) | (d << 32) | (f << 40) | (g << 42)
}

fn unpack_data(data: u64) -> (TTData, u8) {
    let best_move = Move((data & 0xFFFF) as u16);
    let score = ((data >> 16) & 0xFFFF) as u16 as i16 as i32;
    let depth = ((data >> 32) & 0xFF) as u8;
    let flag = TTFlag::from_u8(((data >> 40) & 3) as u8);
    let generation = ((data >> 42) & 0xFF) as u8;
    (
        TTData {
            depth,
            score,
            flag,
            best_move,
        },
        generation,
    )
}

// ---------------------------------------------------------------------------
// Atomic TT entry (16 bytes, cache-line friendly)
// ---------------------------------------------------------------------------

#[repr(C, align(16))]
struct AtomicTTEntry {
    /// Stores `hash_key XOR data` for torn-read detection.
    key: AtomicU64,
    /// Packed entry data.
    data: AtomicU64,
}

impl AtomicTTEntry {
    fn new() -> Self {
        Self {
            key: AtomicU64::new(0),
            data: AtomicU64::new(0),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared transposition table
// ---------------------------------------------------------------------------

pub struct SharedTT {
    entries: Box<[AtomicTTEntry]>,
    mask: usize,
    generation: AtomicU8,
}


impl SharedTT {
    /// Create a new transposition table with the given size in MB.
    pub fn new(size_mb: usize) -> Self {
        let entry_size = 16; // two u64
        let num_entries = (size_mb * 1024 * 1024) / entry_size;
        let num_entries = num_entries.next_power_of_two() / 2;
        let num_entries = num_entries.max(1024);

        let mut entries = Vec::with_capacity(num_entries);
        for _ in 0..num_entries {
            entries.push(AtomicTTEntry::new());
        }

        Self {
            entries: entries.into_boxed_slice(),
            mask: num_entries - 1,
            generation: AtomicU8::new(0),
        }
    }

    /// Probe the TT for the given hash. Returns `None` on miss or corrupted read.
    pub fn probe(&self, hash: u64) -> Option<TTData> {
        let idx = hash as usize & self.mask;
        let entry = &self.entries[idx];

        let data = entry.data.load(Ordering::Relaxed);
        let key = entry.key.load(Ordering::Relaxed);

        // XOR verification: the stored key should equal hash ^ data.
        // If a concurrent write tore either field, this will (very likely) fail.
        if key ^ data != hash {
            return None;
        }

        let (tt_data, _generation) = unpack_data(data);
        Some(tt_data)
    }

    /// Store an entry. Uses a generation + depth replacement scheme:
    /// - Always replace if same hash (refinement)
    /// - Replace if current generation is newer than stored
    /// - Replace if depth is >= stored depth within same generation
    pub fn store(&self, hash: u64, depth: u8, score: i32, flag: TTFlag, best_move: Move) {
        let idx = hash as usize & self.mask;
        let entry = &self.entries[idx];
        let generation = self.generation.load(Ordering::Relaxed);

        let data = pack_data(depth, score, flag, best_move, generation);

        // Check if we should replace
        let old_data = entry.data.load(Ordering::Relaxed);
        let old_key = entry.key.load(Ordering::Relaxed);

        if old_data != 0 {
            let old_hash = old_key ^ old_data;
            let (_old_entry, old_generation) = unpack_data(old_data);

            // Keep deeper entries from the current generation
            if old_hash != hash
                && old_generation == generation
                && _old_entry.depth >= depth + 2
            {
                return;
            }
        }

        entry.data.store(data, Ordering::Relaxed);
        entry.key.store(hash ^ data, Ordering::Relaxed);
    }

    /// Increment the generation counter. Call once per search.
    pub fn new_search(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    /// Approximate fill ratio in permille (0..1000).
    pub fn hashfull(&self) -> u16 {
        let sample = 1000.min(self.entries.len());
        let mut used = 0u32;
        let generation = self.generation.load(Ordering::Relaxed);

        for i in 0..sample {
            let data = self.entries[i].data.load(Ordering::Relaxed);
            if data != 0 {
                let (_entry, entry_generation) = unpack_data(data);
                if entry_generation == generation {
                    used += 1;
                }
            }
        }

        ((used as u64 * 1000) / sample as u64) as u16
    }

    /// Clear all entries.
    pub fn clear(&self) {
        for entry in self.entries.iter() {
            entry.key.store(0, Ordering::Relaxed);
            entry.data.store(0, Ordering::Relaxed);
        }
        self.generation.store(0, Ordering::Relaxed);
    }

    /// Resize the table. Returns a new SharedTT with the given size.
    /// The old table is dropped.
    pub fn resize(size_mb: usize) -> Self {
        Self::new(size_mb)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let m = Move(1234);
        let data = pack_data(12, -500, TTFlag::LowerBound, m, 7);
        let (entry, generation) = unpack_data(data);
        assert_eq!(entry.depth, 12);
        assert_eq!(entry.score, -500);
        assert_eq!(entry.flag, TTFlag::LowerBound);
        assert_eq!(entry.best_move, m);
        assert_eq!(generation, 7);
    }

    #[test]
    fn test_pack_unpack_positive_score() {
        let m = Move(4321);
        let data = pack_data(20, 31000, TTFlag::Exact, m, 255);
        let (entry, generation) = unpack_data(data);
        assert_eq!(entry.depth, 20);
        assert_eq!(entry.score, 31000);
        assert_eq!(entry.flag, TTFlag::Exact);
        assert_eq!(entry.best_move, m);
        assert_eq!(generation, 255);
    }

    #[test]
    fn test_store_and_probe() {
        let tt = SharedTT::new(1);
        let hash = 0xDEAD_BEEF_1234_5678u64;
        let m = Move(100);

        tt.store(hash, 8, 150, TTFlag::Exact, m);
        let result = tt.probe(hash);
        assert!(result.is_some());
        let entry = result.unwrap();
        assert_eq!(entry.depth, 8);
        assert_eq!(entry.score, 150);
        assert_eq!(entry.flag, TTFlag::Exact);
        assert_eq!(entry.best_move, m);
    }

    #[test]
    fn test_probe_miss() {
        let tt = SharedTT::new(1);
        assert!(tt.probe(12345).is_none());
    }

    #[test]
    fn test_hashfull() {
        let tt = SharedTT::new(1);
        assert_eq!(tt.hashfull(), 0);

        // Store some entries
        for i in 0..500u64 {
            tt.store(i * 7919, 5, 0, TTFlag::Exact, Move::NULL);
        }
        let hf = tt.hashfull();
        assert!(hf > 0, "hashfull should be > 0 after storing entries");
    }

    #[test]
    fn test_generation_replacement() {
        let tt = SharedTT::new(1);
        let hash = 0x1234_5678_ABCD_EF01u64;

        // Store at generation 0
        tt.store(hash, 10, 100, TTFlag::Exact, Move(200));

        // New search bumps generation
        tt.new_search();

        // Store at generation 1 — should replace even at lower depth
        tt.store(hash, 3, 50, TTFlag::UpperBound, Move(300));

        let entry = tt.probe(hash).unwrap();
        assert_eq!(entry.depth, 3);
        assert_eq!(entry.score, 50);
        assert_eq!(entry.best_move, Move(300));
    }

    #[test]
    fn test_clear() {
        let tt = SharedTT::new(1);
        tt.store(42, 5, 100, TTFlag::Exact, Move(10));
        assert!(tt.probe(42).is_some());

        tt.clear();
        assert!(tt.probe(42).is_none());
    }
}
