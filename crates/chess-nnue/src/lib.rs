pub mod accumulator;
pub mod features;
pub mod inference;
pub mod network;

pub const NUM_BUCKETS: usize = 32;
pub const INPUT_SIZE: usize = 768 * NUM_BUCKETS; // HalfKP: 768 features × 32 king-square buckets
pub const HIDDEN_SIZE: usize = 768;
/// Material-keyed output buckets: bucket = (piece_count - 2) / 4, matching
/// bullet's MaterialCount<8> (divisor = ceil(32/8) = 4).
pub const OUTPUT_BUCKETS: usize = 8;
pub const FT_QUANT: i32 = 127; // feature transformer quantization (QA)
pub const NET_QUANT: i32 = 64; // output layer quantization (QB)

pub use accumulator::Accumulator;
pub use inference::nnue_evaluate;
pub use network::NnueNetwork;
