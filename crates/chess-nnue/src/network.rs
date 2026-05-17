use std::sync::{Arc, LazyLock};

use crate::{HIDDEN_SIZE, INPUT_SIZE};

/// NNUE network parameters.
///
/// Binary layout (little-endian, sequential, matches Bullet trainer output):
/// - ft_weights:     INPUT_SIZE × HIDDEN_SIZE i16 values  (QA=127)
/// - ft_biases:      HIDDEN_SIZE i16 values               (QA=127)
/// - output_weights: HIDDEN_SIZE × 2 i8 values            (QB=64)
/// - output_bias:    1 i16 value                          (QA×QB=8128)
///
/// Total: (INPUT_SIZE × HIDDEN_SIZE × 2) + (HIDDEN_SIZE × 4) + 2 bytes
pub struct NnueNetwork {
    pub ft_weights: Box<[[i16; HIDDEN_SIZE]; INPUT_SIZE]>,
    pub ft_biases: Box<[i16; HIDDEN_SIZE]>,
    pub output_weights: Box<[i8; HIDDEN_SIZE * 2]>,
    pub output_bias: i16,
}

/// Expected size of the network file in bytes.
pub const NET_FILE_SIZE: usize =
    INPUT_SIZE * HIDDEN_SIZE * 2   // ft_weights (i16)
    + HIDDEN_SIZE * 2              // ft_biases (i16)
    + HIDDEN_SIZE * 2              // output_weights (i8)
    + 2;                           // output_bias (i16)

impl NnueNetwork {
    /// Parse a network from raw bytes (little-endian sequential format).
    pub fn from_bytes(data: &[u8]) -> Result<Self, &'static str> {
        if data.len() < NET_FILE_SIZE {
            return Err("NNUE file too small");
        }

        let mut offset = 0;

        // Feature transformer weights: INPUT_SIZE × HIDDEN_SIZE i16
        //
        // Use vec! + into_boxed_slice to avoid materialising a 3 MB array on
        // the stack before boxing it (Box::new([…; N]) does so in debug builds,
        // overflowing the ~1 MB Windows default thread stack).
        let mut ft_weights: Box<[[i16; HIDDEN_SIZE]; INPUT_SIZE]> =
            vec![[0i16; HIDDEN_SIZE]; INPUT_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap();
        for row in ft_weights.iter_mut() {
            for val in row.iter_mut() {
                *val = i16::from_le_bytes([data[offset], data[offset + 1]]);
                offset += 2;
            }
        }

        // Feature transformer biases: HIDDEN_SIZE i16
        let mut ft_biases = Box::new([0i16; HIDDEN_SIZE]);
        for val in ft_biases.iter_mut() {
            *val = i16::from_le_bytes([data[offset], data[offset + 1]]);
            offset += 2;
        }

        // Output weights: HIDDEN_SIZE * 2 i8
        let mut output_weights = Box::new([0i8; HIDDEN_SIZE * 2]);
        for val in output_weights.iter_mut() {
            *val = data[offset] as i8;
            offset += 1;
        }

        // Output bias: i16
        let output_bias = i16::from_le_bytes([data[offset], data[offset + 1]]);

        Ok(Self {
            ft_weights,
            ft_biases,
            output_weights,
            output_bias,
        })
    }

    /// Return a shared reference to the embedded default network.
    pub fn embedded() -> Arc<NnueNetwork> {
        static NET: LazyLock<Arc<NnueNetwork>> = LazyLock::new(|| {
            let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/default.nnue"));
            Arc::new(NnueNetwork::from_bytes(bytes).expect("embedded NNUE net is invalid"))
        });
        Arc::clone(&NET)
    }

    /// Returns false if this is a zero-initialised placeholder (net not yet trained).
    pub fn is_trained(&self) -> bool {
        self.output_weights.iter().any(|&w| w != 0)
    }

    /// Load a network from a file path, falling back to embedded if path is empty.
    pub fn from_path(path: &str) -> Result<Arc<NnueNetwork>, String> {
        if path.is_empty() {
            return Ok(Self::embedded());
        }
        let data = std::fs::read(path).map_err(|e| format!("failed to read '{path}': {e}"))?;
        let net = Self::from_bytes(&data).map_err(|e| format!("invalid NNUE file '{path}': {e}"))?;
        Ok(Arc::new(net))
    }
}
