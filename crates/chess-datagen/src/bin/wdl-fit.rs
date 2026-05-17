/// Fit the WDL normalization parameters (a, b) for the logistic model:
///
///   P(win | score) = sigmoid((score - a) / b)
///
/// where score is in centipawns from White's perspective and result is
/// 1.0 (white wins) / 0.5 (draw) / 0.0 (black wins).
///
/// After fitting, copy the printed values into chess-uci/src/handler.rs
/// (WDL_A and WDL_B constants).
///
/// Usage:
///   wdl-fit --input data/archive/archive_shuf.bin [--max-positions N]
use std::io::Read;

use bulletformat::{BulletFormat, ChessBoard};

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Binary cross-entropy loss for one bucket (fractional labels).
fn bce(p: f64, target: f64) -> f64 {
    let p = p.clamp(1e-9, 1.0 - 1e-9);
    -target * p.ln() - (1.0 - target) * (1.0 - p).ln()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut input: Option<String> = None;
    let mut max_positions: usize = usize::MAX;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" => {
                i += 1;
                input = Some(args[i].clone());
            }
            "--max-positions" | "-n" => {
                i += 1;
                max_positions = args[i].parse().expect("invalid --max-positions");
            }
            "--help" | "-h" => {
                eprintln!("Usage: wdl-fit --input <file.bin> [--max-positions N]");
                eprintln!();
                eprintln!("Reads bulletformat training data and fits the WDL logistic curve:");
                eprintln!("  P(win | score) = sigmoid((score - a) / b)");
                eprintln!();
                eprintln!("Copy the printed WDL_A and WDL_B values into chess-uci/src/handler.rs.");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let path = input.unwrap_or_else(|| {
        eprintln!("Error: --input <file.bin> is required");
        std::process::exit(1);
    });

    // Read positions in chunks; clip scores to [-4000, 4000] cp.
    const CLIP: i32 = 4000;
    const BIN_WIDTH: i32 = 25;
    const NUM_BINS: usize = (2 * CLIP / BIN_WIDTH) as usize + 1;

    let mut bin_sum = vec![0.0f64; NUM_BINS]; // sum of results
    let mut bin_count = vec![0u64; NUM_BINS]; // position count
    let mut total: usize = 0;

    let entry_size = std::mem::size_of::<ChessBoard>();
    let mut file = std::fs::File::open(&path)
        .unwrap_or_else(|e| { eprintln!("Cannot open '{}': {e}", path); std::process::exit(1) });

    let file_positions = file.metadata().map(|m| m.len() as usize / entry_size).unwrap_or(0);
    let total_expected = file_positions.min(max_positions);
    eprintln!("Reading {} positions from '{}'...", total_expected, path);

    const REPORT_EVERY: usize = 50_000_000;
    let mut next_report = REPORT_EVERY;

    let mut buf = vec![0u8; entry_size * 65536];
    'outer: loop {
        let bytes_read = file.read(&mut buf).expect("read error");
        if bytes_read == 0 { break; }

        let n = bytes_read / entry_size;
        for k in 0..n {
            // unsafe: bulletformat exposes no safe byte-deserialization API; ptr::read is
            // required. Safety: ChessBoard is repr(C), 32 bytes, no interior padding.
            let entry: ChessBoard = unsafe {
                std::ptr::read(buf[k * entry_size..].as_ptr() as *const ChessBoard)
            };

            let score = entry.score as i32;
            let result = entry.result() as f64; // 0.0 / 0.5 / 1.0

            let clipped = score.clamp(-CLIP, CLIP);
            let bin = ((clipped + CLIP) / BIN_WIDTH) as usize;
            let bin = bin.min(NUM_BINS - 1);

            bin_sum[bin] += result;
            bin_count[bin] += 1;
            total += 1;

            if total >= next_report {
                let pct = (100 * total).checked_div(total_expected).unwrap_or(0);
                eprint!("\r  {:.0}M positions read ({pct}%)   ", total as f64 / 1e6);
                next_report += REPORT_EVERY;
            }

            if total >= max_positions { break 'outer; }
        }
    }

    eprintln!("\r  {:.0}M positions read (100%)   ", total as f64 / 1e6);
    eprintln!("Done reading. Non-empty score buckets: {}",
        bin_count.iter().filter(|&&c| c > 0).count());

    // Build non-empty buckets: (mean_score, mean_result, weight).
    let buckets: Vec<(f64, f64, f64)> = (0..NUM_BINS)
        .filter(|&b| bin_count[b] > 0)
        .map(|b| {
            let mean_score = (-CLIP + b as i32 * BIN_WIDTH) as f64 + BIN_WIDTH as f64 / 2.0;
            let mean_result = bin_sum[b] / bin_count[b] as f64;
            let weight = bin_count[b] as f64;
            (mean_score, mean_result, weight)
        })
        .collect();

    // Fit with Adam on weighted cross-entropy.
    let mut a = 0.0f64;
    let mut b = 400.0f64;

    let (mut m_a, mut v_a) = (0.0f64, 0.0f64);
    let (mut m_b, mut v_b) = (0.0f64, 0.0f64);
    let (beta1, beta2, eps) = (0.9, 0.999, 1e-8);
    let lr = 1.0;

    for step in 1..=2_000usize {
        let mut grad_a = 0.0f64;
        let mut grad_b = 0.0f64;
        let mut total_weight = 0.0f64;

        for &(score, target, w) in &buckets {
            let z = (score - a) / b;
            let p = sigmoid(z);
            let err = p - target; // dBCE/dz = p - target
            grad_a += w * err * (-1.0 / b);
            grad_b += w * err * (-z / b);
            total_weight += w;
        }

        grad_a /= total_weight;
        grad_b /= total_weight;

        // Adam
        m_a = beta1 * m_a + (1.0 - beta1) * grad_a;
        v_a = beta2 * v_a + (1.0 - beta2) * grad_a * grad_a;
        m_b = beta1 * m_b + (1.0 - beta1) * grad_b;
        v_b = beta2 * v_b + (1.0 - beta2) * grad_b * grad_b;

        let t = step as f64;
        let m_hat_a = m_a / (1.0 - beta1.powf(t));
        let v_hat_a = v_a / (1.0 - beta2.powf(t));
        let m_hat_b = m_b / (1.0 - beta1.powf(t));
        let v_hat_b = v_b / (1.0 - beta2.powf(t));

        a -= lr * m_hat_a / (v_hat_a.sqrt() + eps);
        b -= lr * m_hat_b / (v_hat_b.sqrt() + eps);

        // b must stay positive (it's a scale factor)
        b = b.max(1.0);

        if step % 500 == 0 {
            let loss: f64 = buckets.iter()
                .map(|&(score, target, w)| w * bce(sigmoid((score - a) / b), target))
                .sum::<f64>() / total_weight;
            eprintln!("step {step:>6}: a={a:.2}  b={b:.2}  loss={loss:.6}");
        }
    }

    // Final loss
    let total_weight: f64 = buckets.iter().map(|&(_, _, w)| w).sum();
    let loss: f64 = buckets.iter()
        .map(|&(score, target, w)| w * bce(sigmoid((score - a) / b), target))
        .sum::<f64>() / total_weight;

    eprintln!("Final: a={a:.2}  b={b:.2}  loss={loss:.6}");
    eprintln!();
    eprintln!("Copy into chess-uci/src/handler.rs:");
    println!("const WDL_A: f64 = {a:.1};");
    println!("const WDL_B: f64 = {b:.1};");
}
