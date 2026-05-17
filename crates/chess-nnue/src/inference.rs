use chess_common::Color;

use crate::accumulator::Accumulator;
use crate::network::NnueNetwork;
use crate::{FT_QUANT, HIDDEN_SIZE, NET_QUANT};

/// SCReLU dot product for one perspective:
///   sum( clamp(acc[i], 0, FT_QUANT)² × weights[i] )
///
/// With FT_QUANT=127, squaring stays in i16 (127²=16129 ≤ 32767), enabling
/// AVX2 to process 16 values per iteration via mullo_epi16 + madd_epi16.
#[inline]
fn screlu_sum(acc: &[i16; HIDDEN_SIZE], weights: &[i8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // unsafe: #[target_feature] fns are unsafe because calling them without the
            // feature enabled is UB; runtime detection above guarantees AVX2 is present.
            return unsafe { screlu_sum_avx2(acc, weights) };
        }
    }
    // NEON is mandatory on all aarch64 targets — no runtime check needed.
    #[cfg(target_arch = "aarch64")]
    {
        // unsafe: same as AVX2 above; NEON is always available on aarch64.
        return unsafe { screlu_sum_neon(acc, weights) };
    }
    #[allow(unreachable_code)]
    screlu_sum_scalar(acc, weights)
}

#[inline]
fn screlu_sum_scalar(acc: &[i16; HIDDEN_SIZE], weights: &[i8]) -> i32 {
    let mut output: i32 = 0;
    for (&val, &wt) in acc.iter().zip(weights.iter()) {
        let clamped = (val as i32).clamp(0, FT_QUANT);
        output += clamped * clamped * i32::from(wt);
    }
    output
}

/// AVX2-accelerated SCReLU dot product.
///
/// Processes 16 × i16 per iteration (2× the old 8):
///   1. Load 16 × i16 accumulator values into a 256-bit register.
///   2. Clamp to [0, 127] in i16 space.
///   3. Square in i16 space (127² = 16129 ≤ 32767 — no overflow).
///   4. Sign-extend 16 × i8 weights to 16 × i16.
///   5. _mm256_madd_epi16: multiply adjacent pairs and sum to 8 × i32.
///   6. Accumulate and horizontally reduce.
// unsafe fn required by #[target_feature]: Rust mandates that functions compiled
// for a non-baseline feature are unsafe so callers must guarantee the feature exists.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn screlu_sum_avx2(acc: &[i16; HIDDEN_SIZE], weights: &[i8]) -> i32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(weights.len(), HIDDEN_SIZE);
    debug_assert_eq!(HIDDEN_SIZE % 16, 0);

    let zero = _mm256_setzero_si256();
    let quant = _mm256_set1_epi16(FT_QUANT as i16);
    let mut sum = _mm256_setzero_si256();

    unsafe {
        let mut i = 0;
        while i < HIDDEN_SIZE {
            // Load 16 × i16 accumulator (256 bits).
            let v = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);

            // Clamp to [0, 127] in i16 space.
            let clamped = _mm256_min_epi16(_mm256_max_epi16(v, zero), quant);

            // Square: 127² = 16129 ≤ i16::MAX — stays in i16.
            let sq = _mm256_mullo_epi16(clamped, clamped);

            // Load 16 × i8 weights (128 bits) and sign-extend to 16 × i16.
            let w = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                weights.as_ptr().add(i) as *const __m128i,
            ));

            // Multiply adjacent pairs and accumulate to i32:
            // madd(sq, w)[k] = sq[2k]*w[2k] + sq[2k+1]*w[2k+1]  (8 × i32)
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(sq, w));

            i += 16;
        }

        // Horizontal reduction: 8 × i32 → scalar.
        let hi = _mm256_extracti128_si256(sum, 1);
        let lo = _mm256_castsi256_si128(sum);
        let s = _mm_add_epi32(hi, lo);
        let s2 = _mm_hadd_epi32(s, s);
        let s3 = _mm_hadd_epi32(s2, s2);
        _mm_cvtsi128_si32(s3)
    }
}

/// NEON-accelerated SCReLU dot product (aarch64 — Apple Silicon, Windows ARM).
///
/// Processes 8 × i16 per iteration using 128-bit NEON registers.
/// With FT_QUANT=127, squaring stays in i16 (no widening needed for that step).
// unsafe fn required by #[target_feature]: same rationale as screlu_sum_avx2.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn screlu_sum_neon(acc: &[i16; HIDDEN_SIZE], weights: &[i8]) -> i32 {
    use std::arch::aarch64::*;

    debug_assert_eq!(weights.len(), HIDDEN_SIZE);

    let zero = vdupq_n_s16(0);
    let quant = vdupq_n_s16(FT_QUANT as i16);
    let mut sum_lo = vdupq_n_s32(0i32);
    let mut sum_hi = vdupq_n_s32(0i32);

    unsafe {
        let mut i = 0;
        while i < HIDDEN_SIZE {
            // Load 8 × i16 accumulator values.
            let v = vld1q_s16(acc.as_ptr().add(i));

            // Clamp to [0, 127] in i16 space.
            let clamped = vminq_s16(vmaxq_s16(v, zero), quant);

            // Square: 127² = 16129 ≤ i16::MAX — stays in i16.
            let sq = vmulq_s16(clamped, clamped);

            // Load 8 × i8 weights and sign-extend to 8 × i16.
            let w = vmovl_s8(vld1_s8(weights.as_ptr().add(i)));

            // Widen to i32 and multiply-accumulate.
            sum_lo = vmlal_s16(sum_lo, vget_low_s16(sq), vget_low_s16(w));
            sum_hi = vmlal_high_s16(sum_hi, sq, w);

            i += 8;
        }

        // Horizontal reduction.
        vaddvq_s32(vaddq_s32(sum_lo, sum_hi))
    }
}

/// Evaluate the position using the NNUE accumulator.
///
/// Returns score from the **side-to-move perspective** (positive = good for STM).
/// Uses SCReLU activation: clamp(x, 0, FT_QUANT)² then dot product with output weights.
#[inline]
pub fn nnue_evaluate(acc: &Accumulator, side_to_move: Color, net: &NnueNetwork) -> i32 {
    let (stm_acc, opp_acc) = match side_to_move {
        Color::White => (&acc.white, &acc.black),
        Color::Black => (&acc.black, &acc.white),
    };

    let output = screlu_sum(stm_acc, &net.output_weights[..HIDDEN_SIZE])
        + screlu_sum(opp_acc, &net.output_weights[HIDDEN_SIZE..]);

    (output / FT_QUANT + i32::from(net.output_bias)) * 400 / (FT_QUANT * NET_QUANT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accumulator::Accumulator;
    use crate::network::NnueNetwork;
    use chess_common::Board;

    #[test]
    fn starting_position_eval_is_reasonable() {
        let net = NnueNetwork::embedded();
        if !net.is_trained() {
            return; // zero-padded placeholder — skip until a real net is trained
        }
        let board = Board::starting_position();

        let mut acc = Accumulator::new();
        acc.refresh(&board, &net);

        let score = nnue_evaluate(&acc, Color::White, &net);
        // The 768-wide net has a larger output range than narrower nets; the
        // threshold here is a sanity check against completely broken evaluation,
        // not a calibration target.
        assert!(
            score.abs() < 1000,
            "starting position eval {score} is unreasonably large"
        );
    }

    /// SIMD and scalar paths must produce identical results.
    #[test]
    fn avx2_matches_scalar() {
        let net = NnueNetwork::embedded();
        let board = Board::starting_position();

        let mut acc = Accumulator::new();
        acc.refresh(&board, &net);

        let scalar_stm = screlu_sum_scalar(&acc.white, &net.output_weights[..HIDDEN_SIZE]);
        let scalar_opp = screlu_sum_scalar(&acc.black, &net.output_weights[HIDDEN_SIZE..]);

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            let simd_stm =
                unsafe { screlu_sum_avx2(&acc.white, &net.output_weights[..HIDDEN_SIZE]) };
            let simd_opp =
                unsafe { screlu_sum_avx2(&acc.black, &net.output_weights[HIDDEN_SIZE..]) };
            assert_eq!(scalar_stm, simd_stm, "AVX2 STM half mismatch");
            assert_eq!(scalar_opp, simd_opp, "AVX2 opponent half mismatch");
        }

        #[cfg(target_arch = "aarch64")]
        {
            let neon_stm =
                unsafe { screlu_sum_neon(&acc.white, &net.output_weights[..HIDDEN_SIZE]) };
            let neon_opp =
                unsafe { screlu_sum_neon(&acc.black, &net.output_weights[HIDDEN_SIZE..]) };
            assert_eq!(scalar_stm, neon_stm, "NEON STM half mismatch");
            assert_eq!(scalar_opp, neon_opp, "NEON opponent half mismatch");
        }

        // Suppress unused-variable warnings on targets without SIMD paths.
        let _ = (scalar_stm, scalar_opp);
    }
}
