pub mod accumulator;
pub mod features;
pub mod inference;
pub mod network;

pub const NUM_BUCKETS: usize = 32;
pub const INPUT_SIZE: usize = 704 * NUM_BUCKETS; // HalfKP: 704 features × 32 king-square buckets
pub const HIDDEN_SIZE: usize = 768;
pub const FT_QUANT: i32 = 127; // feature transformer quantization (QA)
pub const NET_QUANT: i32 = 64;  // output layer quantization (QB)

pub use accumulator::Accumulator;
pub use inference::nnue_evaluate;
pub use network::NnueNetwork;
