use std::sync::{Arc, LazyLock};

use crate::{
    FEATURES_PER_BUCKET, HIDDEN_SIZE, INPUT_SIZE, L1_SIZE, L2_SIZE, NUM_BUCKETS, OUTPUT_BUCKETS,
    PAIR_INPUT_SIZE, PIECE_FEATURES,
};

pub const NET_MAGIC: &[u8; 8] = b"ESPNNUE2";
pub const NET_VERSION: u32 = 2;
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

const LEGACY_INPUT_SIZE: usize = PIECE_FEATURES * NUM_BUCKETS;
const LEGACY_NET_FILE_SIZE: usize = HEADER_SIZE
    + LEGACY_INPUT_SIZE * HIDDEN_SIZE * 2
    + HIDDEN_SIZE * 2
    + OUTPUT_BUCKETS * L1_SIZE * PAIR_INPUT_SIZE
    + OUTPUT_BUCKETS * L1_SIZE * 4
    + OUTPUT_BUCKETS * L2_SIZE * L1_SIZE
    + OUTPUT_BUCKETS * L2_SIZE * 4
    + OUTPUT_BUCKETS * L2_SIZE
    + OUTPUT_BUCKETS * 4;

const PAD_ALIGN: usize = 64;

#[cfg(test)]
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
    #[cfg(test)]
    pub(crate) fn zeroed_for_test() -> Self {
        let mut bytes = vec![0u8; NET_FILE_SIZE];
        bytes[..HEADER_SIZE].copy_from_slice(&expected_header());
        Self::from_bytes(&bytes).unwrap()
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, &'static str> {
        if data.len() < HEADER_SIZE || data[..8] != *NET_MAGIC {
            return Err("layer-stacked NNUE header or dimensions do not match this engine");
        }
        let read_header_u32 = |index: usize| {
            let start = 8 + index * 4;
            u32::from_le_bytes(data[start..start + 4].try_into().unwrap())
        };
        let version = read_header_u32(0);
        let file_input_size = read_header_u32(1) as usize;
        let dimensions_match = read_header_u32(2) as usize == HIDDEN_SIZE
            && read_header_u32(3) as usize == OUTPUT_BUCKETS
            && read_header_u32(4) as usize == L1_SIZE
            && read_header_u32(5) as usize == L2_SIZE;
        let legacy = version == 1 && file_input_size == LEGACY_INPUT_SIZE;
        let current = version == NET_VERSION && file_input_size == INPUT_SIZE;
        let expected_size = if legacy {
            LEGACY_NET_FILE_SIZE
        } else {
            NET_FILE_SIZE
        };
        if !dimensions_match
            || (!legacy && !current)
            || !(expected_size..expected_size + PAD_ALIGN).contains(&data.len())
        {
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
        if legacy {
            // Old nets used 768 rows per king bucket. Scatter those rows into
            // the expanded state-aware layout; new state rows remain zero, so
            // the embedded production net retains bit-identical evaluation.
            for bucket in 0..NUM_BUCKETS {
                for piece_feature in 0..PIECE_FEATURES {
                    let row = bucket * FEATURES_PER_BUCKET + piece_feature;
                    for value in ft_weights[row].iter_mut() {
                        *value = read_i16(&mut offset);
                    }
                }
            }
        } else {
            for row in ft_weights.iter_mut() {
                for value in row.iter_mut() {
                    *value = read_i16(&mut offset);
                }
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
        debug_assert_eq!(offset, expected_size);

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

    fn legacy_blank_net() -> Vec<u8> {
        let mut bytes = vec![0u8; LEGACY_NET_FILE_SIZE];
        let mut header = expected_header();
        header[8..12].copy_from_slice(&1u32.to_le_bytes());
        header[12..16].copy_from_slice(&(LEGACY_INPUT_SIZE as u32).to_le_bytes());
        bytes[..HEADER_SIZE].copy_from_slice(&header);
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

    #[test]
    fn accepts_and_expands_legacy_piece_only_nets() {
        let net = NnueNetwork::from_bytes(&legacy_blank_net()).unwrap();
        for bucket in 0..NUM_BUCKETS {
            for state_feature in PIECE_FEATURES..FEATURES_PER_BUCKET {
                assert_eq!(
                    net.ft_weights[bucket * FEATURES_PER_BUCKET + state_feature],
                    [0; HIDDEN_SIZE]
                );
            }
        }
    }
}
