use chess_common::Color;

use crate::accumulator::Accumulator;
use crate::network::NnueNetwork;
use crate::{FT_QUANT, HIDDEN_SIZE, NET_QUANT, OUTPUT_BUCKETS};

/// Material-keyed output bucket: (piece_count - 2) / 4 → 0..OUTPUT_BUCKETS.
/// Must match the trainer's bullet `MaterialCount<8>` (divisor = ceil(32/8) = 4).
#[inline]
pub fn output_bucket(piece_count: u32) -> usize {
    ((piece_count.saturating_sub(2) / 4) as usize).min(OUTPUT_BUCKETS - 1)
}

/// SCReLU dot product for one perspective:
///   sum( clamp(acc[i], 0, FT_QUANT)² × weights[i] )
///
/// With FT_QUANT=127, squaring stays in i16 (127²=16129 ≤ 32767), enabling
/// AVX2 to process 16 values per iteration via mullo_epi16 + madd_epi16.
#[inline]
fn screlu_sum(acc: &[i16; HIDDEN_SIZE], weights: &[i8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            // unsafe: runtime detection guarantees both features required by
            // the 512-bit i16/i8 kernel are available.
            return unsafe { screlu_sum_avx512(acc, weights) };
        }
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

/// AVX-512-accelerated SCReLU dot product.
///
/// Processes 32 × i16 per iteration, twice the AVX2 width. AVX-512BW supplies
/// the packed i8/i16 conversions and arithmetic; AVX-512F supplies the i32
/// accumulation and reduction.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn screlu_sum_avx512(acc: &[i16; HIDDEN_SIZE], weights: &[i8]) -> i32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(weights.len(), HIDDEN_SIZE);
    debug_assert_eq!(HIDDEN_SIZE % 32, 0);

    let zero = _mm512_setzero_si512();
    let quant = _mm512_set1_epi16(FT_QUANT as i16);
    let mut sum = _mm512_setzero_si512();

    unsafe {
        let mut i = 0;
        while i < HIDDEN_SIZE {
            let v = _mm512_loadu_si512(acc.as_ptr().add(i) as *const __m512i);
            let clamped = _mm512_min_epi16(_mm512_max_epi16(v, zero), quant);
            let sq = _mm512_mullo_epi16(clamped, clamped);

            let packed_weights = _mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i);
            let w = _mm512_cvtepi8_epi16(packed_weights);

            sum = _mm512_add_epi32(sum, _mm512_madd_epi16(sq, w));
            i += 32;
        }

        _mm512_reduce_add_epi32(sum)
    }
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
            let w =
                _mm256_cvtepi8_epi16(_mm_loadu_si128(weights.as_ptr().add(i) as *const __m128i));

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
/// Uses SCReLU activation: clamp(x, 0, FT_QUANT)² then dot product with the
/// output weights of the material-keyed bucket for `piece_count`.
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
    let weights = &net.output_weights[bucket];
    let output =
        screlu_sum(stm_acc, &weights[..HIDDEN_SIZE]) + screlu_sum(opp_acc, &weights[HIDDEN_SIZE..]);

    (output / FT_QUANT + i32::from(net.output_bias[bucket])) * 400 / (FT_QUANT * NET_QUANT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accumulator::Accumulator;
    use crate::network::NnueNetwork;
    use chess_common::Board;

    /// Net under test: point `NNUE_TEST_NET=<path>` at a candidate .nnue to run
    /// the i8-path guards (eval sanity + SIMD==scalar) against it before promotion;
    /// otherwise the embedded default net. Mirrors the deeper engine's DEEPER_TEST_NET.
    fn test_net() -> std::sync::Arc<NnueNetwork> {
        match std::env::var("NNUE_TEST_NET") {
            Ok(p) if !p.is_empty() => NnueNetwork::from_path(&p).expect("load NNUE_TEST_NET"),
            _ => NnueNetwork::embedded(),
        }
    }

    #[test]
    fn starting_position_eval_is_reasonable() {
        let net = test_net();
        if !net.is_trained() {
            return; // zero-padded placeholder — skip until a real net is trained
        }
        let board = Board::starting_position();

        let mut acc = Accumulator::new();
        acc.refresh(&board, &net);

        let piece_count = (board.occupancy[0].0 | board.occupancy[1].0).count_ones();
        let score = nnue_evaluate(&acc, Color::White, &net, piece_count);
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
    fn simd_matches_scalar() {
        let net = test_net();
        let board = Board::starting_position();

        let mut acc = Accumulator::new();
        acc.refresh(&board, &net);

        // Exercise every output bucket's weight row, not just the full-board one.
        for weights in net.output_weights.iter() {
            let scalar_stm = screlu_sum_scalar(&acc.white, &weights[..HIDDEN_SIZE]);
            let scalar_opp = screlu_sum_scalar(&acc.black, &weights[HIDDEN_SIZE..]);

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                    let simd_stm =
                        unsafe { screlu_sum_avx512(&acc.white, &weights[..HIDDEN_SIZE]) };
                    let simd_opp =
                        unsafe { screlu_sum_avx512(&acc.black, &weights[HIDDEN_SIZE..]) };
                    assert_eq!(scalar_stm, simd_stm, "AVX-512 STM half mismatch");
                    assert_eq!(scalar_opp, simd_opp, "AVX-512 opponent half mismatch");
                }
                if is_x86_feature_detected!("avx2") {
                    let simd_stm = unsafe { screlu_sum_avx2(&acc.white, &weights[..HIDDEN_SIZE]) };
                    let simd_opp = unsafe { screlu_sum_avx2(&acc.black, &weights[HIDDEN_SIZE..]) };
                    assert_eq!(scalar_stm, simd_stm, "AVX2 STM half mismatch");
                    assert_eq!(scalar_opp, simd_opp, "AVX2 opponent half mismatch");
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                let neon_stm = unsafe { screlu_sum_neon(&acc.white, &weights[..HIDDEN_SIZE]) };
                let neon_opp = unsafe { screlu_sum_neon(&acc.black, &weights[HIDDEN_SIZE..]) };
                assert_eq!(scalar_stm, neon_stm, "NEON STM half mismatch");
                assert_eq!(scalar_opp, neon_opp, "NEON opponent half mismatch");
            }

            // Suppress unused-variable warnings on targets without SIMD paths.
            let _ = (scalar_stm, scalar_opp);
        }
    }

    /// Bucket mapping must match bullet's MaterialCount<8>: (occ - 2) / 4.
    #[test]
    fn output_bucket_matches_material_count() {
        assert_eq!(output_bucket(2), 0);
        assert_eq!(output_bucket(5), 0);
        assert_eq!(output_bucket(6), 1);
        assert_eq!(output_bucket(17), 3);
        assert_eq!(output_bucket(29), 6);
        assert_eq!(output_bucket(30), 7);
        assert_eq!(output_bucket(32), 7);
    }
}
