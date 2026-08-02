use std::sync::{Arc, LazyLock};

use crate::{HIDDEN_SIZE, INPUT_SIZE, L1_SIZE, L2_SIZE, OUTPUT_BUCKETS, PAIR_INPUT_SIZE};

pub const NET_MAGIC: &[u8; 8] = b"ESPNNUE2";
pub const NET_VERSION: u32 = 1;
pub const HEADER_SIZE: usize = 32;

/// Layer-stacked NNUE parameters.
///
/// Binary layout after the 32-byte versioned header:
/// - feature-transformer weights: INPUT_SIZE × HIDDEN_SIZE i16
/// - feature-transformer biases: HIDDEN_SIZE i16
/// - l1 weights: OUTPUT_BUCKETS × L1_SIZE × PAIR_INPUT_SIZE i8
/// - l1 biases: OUTPUT_BUCKETS × L1_SIZE i32
/// - l2 weights: OUTPUT_BUCKETS × L2_SIZE × L1_SIZE i8
/// - l2 biases: OUTPUT_BUCKETS × L2_SIZE i32
/// - l3 weights: OUTPUT_BUCKETS × L2_SIZE i8
/// - l3 biases: OUTPUT_BUCKETS i32
pub struct NnueNetwork {
    pub ft_weights: Box<[[i16; HIDDEN_SIZE]; INPUT_SIZE]>,
    pub ft_biases: Box<[i16; HIDDEN_SIZE]>,
    pub l1_weights: Box<[[i8; PAIR_INPUT_SIZE]]>,
    pub l1_biases: Box<[[i32; L1_SIZE]; OUTPUT_BUCKETS]>,
    pub l2_weights: Box<[[i8; L1_SIZE]]>,
    pub l2_biases: Box<[[i32; L2_SIZE]; OUTPUT_BUCKETS]>,
    pub l3_weights: Box<[[i8; L2_SIZE]; OUTPUT_BUCKETS]>,
    pub l3_biases: [i32; OUTPUT_BUCKETS],
}

pub const NET_FILE_SIZE: usize = HEADER_SIZE
    + INPUT_SIZE * HIDDEN_SIZE * 2
    + HIDDEN_SIZE * 2
    + OUTPUT_BUCKETS * L1_SIZE * PAIR_INPUT_SIZE
    + OUTPUT_BUCKETS * L1_SIZE * 4
    + OUTPUT_BUCKETS * L2_SIZE * L1_SIZE
    + OUTPUT_BUCKETS * L2_SIZE * 4
    + OUTPUT_BUCKETS * L2_SIZE
    + OUTPUT_BUCKETS * 4;

const PAD_ALIGN: usize = 64;

fn expected_header() -> [u8; HEADER_SIZE] {
    let mut header = [0u8; HEADER_SIZE];
    header[..8].copy_from_slice(NET_MAGIC);
    for (idx, value) in [
        NET_VERSION,
        INPUT_SIZE as u32,
        HIDDEN_SIZE as u32,
        OUTPUT_BUCKETS as u32,
        L1_SIZE as u32,
        L2_SIZE as u32,
    ]
    .into_iter()
    .enumerate()
    {
        let start = 8 + idx * 4;
        header[start..start + 4].copy_from_slice(&value.to_le_bytes());
    }
    header
}

impl NnueNetwork {
    pub fn from_bytes(data: &[u8]) -> Result<Self, &'static str> {
        if !(NET_FILE_SIZE..NET_FILE_SIZE + PAD_ALIGN).contains(&data.len()) {
            return Err("layer-stacked NNUE file has the wrong size");
        }
        if data[..HEADER_SIZE] != expected_header() {
            return Err("layer-stacked NNUE header or dimensions do not match this engine");
        }

        let mut offset = HEADER_SIZE;
        let read_i16 = |offset: &mut usize| {
            let value = i16::from_le_bytes([data[*offset], data[*offset + 1]]);
            *offset += 2;
            value
        };
        let read_i32 = |offset: &mut usize| {
            let value = i32::from_le_bytes([
                data[*offset],
                data[*offset + 1],
                data[*offset + 2],
                data[*offset + 3],
            ]);
            *offset += 4;
            value
        };

        let mut ft_weights: Box<[[i16; HIDDEN_SIZE]; INPUT_SIZE]> =
            vec![[0i16; HIDDEN_SIZE]; INPUT_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap();
        for row in ft_weights.iter_mut() {
            for value in row.iter_mut() {
                *value = read_i16(&mut offset);
            }
        }

        let mut ft_biases = Box::new([0i16; HIDDEN_SIZE]);
        for value in ft_biases.iter_mut() {
            *value = read_i16(&mut offset);
        }

        let mut l1_weights =
            vec![[0i8; PAIR_INPUT_SIZE]; OUTPUT_BUCKETS * L1_SIZE].into_boxed_slice();
        for row in l1_weights.iter_mut() {
            for value in row.iter_mut() {
                *value = data[offset] as i8;
                offset += 1;
            }
        }
        let mut l1_biases = Box::new([[0i32; L1_SIZE]; OUTPUT_BUCKETS]);
        for bucket in l1_biases.iter_mut() {
            for value in bucket.iter_mut() {
                *value = read_i32(&mut offset);
            }
        }

        let mut l2_weights = vec![[0i8; L1_SIZE]; OUTPUT_BUCKETS * L2_SIZE].into_boxed_slice();
        for row in l2_weights.iter_mut() {
            for value in row.iter_mut() {
                *value = data[offset] as i8;
                offset += 1;
            }
        }
        let mut l2_biases = Box::new([[0i32; L2_SIZE]; OUTPUT_BUCKETS]);
        for bucket in l2_biases.iter_mut() {
            for value in bucket.iter_mut() {
                *value = read_i32(&mut offset);
            }
        }

        let mut l3_weights = Box::new([[0i8; L2_SIZE]; OUTPUT_BUCKETS]);
        for row in l3_weights.iter_mut() {
            for value in row.iter_mut() {
                *value = data[offset] as i8;
                offset += 1;
            }
        }
        let mut l3_biases = [0i32; OUTPUT_BUCKETS];
        for value in &mut l3_biases {
            *value = read_i32(&mut offset);
        }
        debug_assert_eq!(offset, NET_FILE_SIZE);

        Ok(Self {
            ft_weights,
            ft_biases,
            l1_weights,
            l1_biases,
            l2_weights,
            l2_biases,
            l3_weights,
            l3_biases,
        })
    }

    pub fn embedded() -> Arc<NnueNetwork> {
        static NET: LazyLock<Arc<NnueNetwork>> = LazyLock::new(|| {
            let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/default.nnue"));
            Arc::new(NnueNetwork::from_bytes(bytes).expect("embedded NNUE net is invalid"))
        });
        Arc::clone(&NET)
    }

    pub fn is_trained(&self) -> bool {
        self.l3_weights
            .iter()
            .any(|row| row.iter().any(|&weight| weight != 0))
    }

    pub fn from_path(path: &str) -> Result<Arc<NnueNetwork>, String> {
        if path.is_empty() {
            return Ok(Self::embedded());
        }
        let data =
            std::fs::read(path).map_err(|error| format!("failed to read '{path}': {error}"))?;
        let net = Self::from_bytes(&data)
            .map_err(|error| format!("invalid NNUE file '{path}': {error}"))?;
        Ok(Arc::new(net))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blank_net(extra_padding: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; NET_FILE_SIZE + extra_padding];
        bytes[..HEADER_SIZE].copy_from_slice(&expected_header());
        bytes
    }

    #[test]
    fn accepts_exact_and_padded_sizes() {
        assert!(NnueNetwork::from_bytes(&blank_net(0)).is_ok());
        assert!(NnueNetwork::from_bytes(&blank_net(32)).is_ok());
    }

    #[test]
    fn rejects_bad_header_and_sizes() {
        assert!(NnueNetwork::from_bytes(&vec![0u8; NET_FILE_SIZE]).is_err());
        assert!(NnueNetwork::from_bytes(&blank_net(PAD_ALIGN)).is_err());
        assert!(NnueNetwork::from_bytes(&[]).is_err());
    }
}
