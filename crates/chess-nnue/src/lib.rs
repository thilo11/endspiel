pub mod accumulator;
pub mod features;
pub mod inference;
pub mod network;

pub const NUM_BUCKETS: usize = 32;
pub const INPUT_SIZE: usize = 768 * NUM_BUCKETS; // HalfKP: 768 features × 32 king-square buckets
pub const HIDDEN_SIZE: usize = 1024;
pub const PAIR_SIZE: usize = HIDDEN_SIZE / 2;
pub const PAIR_INPUT_SIZE: usize = PAIR_SIZE * 2;
pub const L1_SIZE: usize = 16;
pub const L2_SIZE: usize = 32;
/// Material-keyed output buckets: bucket = (piece_count - 2) / 4, matching
/// bullet's MaterialCount<8> (divisor = ceil(32/8) = 4).
pub const OUTPUT_BUCKETS: usize = 8;
pub const FT_QUANT: i32 = 127; // activation quantization (QA)
pub const NET_QUANT: i32 = 64; // dense-layer weight quantization (QB)

pub use accumulator::Accumulator;
pub use inference::nnue_evaluate;
pub use network::NnueNetwork;
