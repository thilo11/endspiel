use chess_common::Color;

use crate::accumulator::Accumulator;
use crate::network::NnueNetwork;
use crate::{
    FT_QUANT, HIDDEN_SIZE, MAX_L1_SIZE, MAX_L2_SIZE, NET_QUANT, OUTPUT_BUCKETS, PAIR_INPUT_SIZE,
    PAIR_SIZE,
};

#[inline]
pub fn output_bucket(piece_count: u32) -> usize {
    ((piece_count.saturating_sub(2) / 4) as usize).min(OUTPUT_BUCKETS - 1)
}

#[inline]
fn pairwise_crelu_sum(acc: &[i16; HIDDEN_SIZE], weights: &[i8]) -> i32 {
    debug_assert_eq!(weights.len(), PAIR_SIZE);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            return unsafe { pairwise_crelu_sum_avx512(acc, weights) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { pairwise_crelu_sum_avx2(acc, weights) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { pairwise_crelu_sum_neon(acc, weights) };
    }
    #[allow(unreachable_code)]
    pairwise_crelu_sum_scalar(acc, weights)
}

#[inline]
fn pairwise_crelu_sum_scalar(acc: &[i16; HIDDEN_SIZE], weights: &[i8]) -> i32 {
    let mut sum = 0i32;
    for idx in 0..PAIR_SIZE {
        let lhs = i32::from(acc[idx]).clamp(0, FT_QUANT);
        let rhs = i32::from(acc[idx + PAIR_SIZE]).clamp(0, FT_QUANT);
        sum += lhs * rhs * i32::from(weights[idx]);
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn pairwise_crelu_sum_avx512(acc: &[i16; HIDDEN_SIZE], weights: &[i8]) -> i32 {
    use std::arch::x86_64::*;

    let zero = _mm512_setzero_si512();
    let quant = _mm512_set1_epi16(FT_QUANT as i16);
    let mut sum = _mm512_setzero_si512();
    unsafe {
        let mut idx = 0;
        while idx < PAIR_SIZE {
            let lhs = _mm512_loadu_si512(acc.as_ptr().add(idx) as *const __m512i);
            let rhs = _mm512_loadu_si512(acc.as_ptr().add(idx + PAIR_SIZE) as *const __m512i);
            let lhs = _mm512_min_epi16(_mm512_max_epi16(lhs, zero), quant);
            let rhs = _mm512_min_epi16(_mm512_max_epi16(rhs, zero), quant);
            let products = _mm512_mullo_epi16(lhs, rhs);
            let packed_weights = _mm256_loadu_si256(weights.as_ptr().add(idx) as *const __m256i);
            let expanded_weights = _mm512_cvtepi8_epi16(packed_weights);
            sum = _mm512_add_epi32(sum, _mm512_madd_epi16(products, expanded_weights));
            idx += 32;
        }
        _mm512_reduce_add_epi32(sum)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pairwise_crelu_sum_avx2(acc: &[i16; HIDDEN_SIZE], weights: &[i8]) -> i32 {
    use std::arch::x86_64::*;

    let zero = _mm256_setzero_si256();
    let quant = _mm256_set1_epi16(FT_QUANT as i16);
    let mut sum = _mm256_setzero_si256();
    unsafe {
        let mut idx = 0;
        while idx < PAIR_SIZE {
            let lhs = _mm256_loadu_si256(acc.as_ptr().add(idx) as *const __m256i);
            let rhs = _mm256_loadu_si256(acc.as_ptr().add(idx + PAIR_SIZE) as *const __m256i);
            let lhs = _mm256_min_epi16(_mm256_max_epi16(lhs, zero), quant);
            let rhs = _mm256_min_epi16(_mm256_max_epi16(rhs, zero), quant);
            let products = _mm256_mullo_epi16(lhs, rhs);
            let packed_weights = _mm_loadu_si128(weights.as_ptr().add(idx) as *const __m128i);
            let expanded_weights = _mm256_cvtepi8_epi16(packed_weights);
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(products, expanded_weights));
            idx += 16;
        }

        let high = _mm256_extracti128_si256(sum, 1);
        let low = _mm256_castsi256_si128(sum);
        let sum128 = _mm_add_epi32(high, low);
        let sum64 = _mm_hadd_epi32(sum128, sum128);
        let sum32 = _mm_hadd_epi32(sum64, sum64);
        _mm_cvtsi128_si32(sum32)
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn pairwise_crelu_sum_neon(acc: &[i16; HIDDEN_SIZE], weights: &[i8]) -> i32 {
    use std::arch::aarch64::*;

    let zero = vdupq_n_s16(0);
    let quant = vdupq_n_s16(FT_QUANT as i16);
    let mut sum_low = vdupq_n_s32(0);
    let mut sum_high = vdupq_n_s32(0);
    unsafe {
        let mut idx = 0;
        while idx < PAIR_SIZE {
            let lhs = vld1q_s16(acc.as_ptr().add(idx));
            let rhs = vld1q_s16(acc.as_ptr().add(idx + PAIR_SIZE));
            let lhs = vminq_s16(vmaxq_s16(lhs, zero), quant);
            let rhs = vminq_s16(vmaxq_s16(rhs, zero), quant);
            let products = vmulq_s16(lhs, rhs);
            let expanded_weights = vmovl_s8(vld1_s8(weights.as_ptr().add(idx)));
            sum_low = vmlal_s16(
                sum_low,
                vget_low_s16(products),
                vget_low_s16(expanded_weights),
            );
            sum_high = vmlal_high_s16(sum_high, products, expanded_weights);
            idx += 8;
        }
        vaddvq_s32(vaddq_s32(sum_low, sum_high))
    }
}

#[inline]
fn dense_screlu_sum(values: &[i16], weights: &[i8]) -> i32 {
    debug_assert_eq!(values.len(), weights.len());
    values
        .iter()
        .zip(weights)
        .map(|(&value, &weight)| {
            let value = i32::from(value).clamp(0, FT_QUANT);
            value * value * i32::from(weight)
        })
        .sum()
}

#[inline]
fn quantised_activation(raw: i32) -> i16 {
    (raw / (FT_QUANT * NET_QUANT)).clamp(0, FT_QUANT) as i16
}

/// Evaluate the layer-stacked NNUE. Dense widths come from the loaded net.
#[inline]
pub fn nnue_evaluate(
    acc: &Accumulator,
    side_to_move: Color,
    net: &NnueNetwork,
    piece_count: u32,
) -> i32 {
    let (stm_acc, opp_acc) = match side_to_move {
        Color::White => (&acc.white, &acc.black),
        Color::Black => (&acc.black, &acc.white),
    };
    let bucket = output_bucket(piece_count);
    let l1_size = net.l1_size;
    let l2_size = net.l2_size;

    let mut l1 = [0i16; MAX_L1_SIZE];
    for (neuron, slot) in l1.iter_mut().enumerate().take(l1_size) {
        let weights = net.l1_row(bucket, neuron);
        let raw = pairwise_crelu_sum(stm_acc, &weights[..PAIR_SIZE])
            + pairwise_crelu_sum(opp_acc, &weights[PAIR_SIZE..PAIR_INPUT_SIZE])
            + net.l1_bias(bucket, neuron);
        *slot = quantised_activation(raw);
    }

    let mut l2 = [0i16; MAX_L2_SIZE];
    for (neuron, slot) in l2.iter_mut().enumerate().take(l2_size) {
        let raw = dense_screlu_sum(&l1[..l1_size], net.l2_row(bucket, neuron))
            + net.l2_bias(bucket, neuron);
        *slot = quantised_activation(raw);
    }

    let raw = dense_screlu_sum(&l2[..l2_size], net.l3_row(bucket)) + net.l3_biases[bucket];
    ((i64::from(raw) * 400) / i64::from(FT_QUANT * FT_QUANT * NET_QUANT)) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piece_count_buckets_cover_the_board() {
        assert_eq!(output_bucket(2), 0);
        assert_eq!(output_bucket(5), 0);
        assert_eq!(output_bucket(6), 1);
        assert_eq!(output_bucket(32), 7);
    }

    #[test]
    fn pairwise_simd_matches_scalar() {
        let mut acc = [0i16; HIDDEN_SIZE];
        let mut weights = [0i8; PAIR_SIZE];
        for (idx, value) in acc.iter_mut().enumerate() {
            *value = ((idx * 97 + 31) % 401) as i16 - 150;
        }
        for (idx, weight) in weights.iter_mut().enumerate() {
            *weight = ((idx * 29 + 7) % 127) as i8 - 63;
        }
        let scalar = pairwise_crelu_sum_scalar(&acc, &weights);
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                assert_eq!(scalar, unsafe { pairwise_crelu_sum_avx512(&acc, &weights) });
            }
            if is_x86_feature_detected!("avx2") {
                assert_eq!(scalar, unsafe { pairwise_crelu_sum_avx2(&acc, &weights) });
            }
        }
        #[cfg(target_arch = "aarch64")]
        assert_eq!(scalar, unsafe { pairwise_crelu_sum_neon(&acc, &weights) });
    }
}
