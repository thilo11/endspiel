use bulletformat::{BulletFormat, ChessBoard, DataLoader};

#[derive(Clone, Debug)]
struct Config {
    input: String,
    sample: u64,
    strict: bool,
}

#[derive(Default, Clone, Copy, Debug)]
struct Stats {
    total: u64,
    wins: u64,
    draws: u64,
    losses: u64,
    king_only: u64,
    high_abs_draws: u64,
    near_zero: u64,
    high_abs: u64,
    sum_abs_cp: u64,
    decisive_scored: u64,
    white_agree: u64,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();

    let mut cfg = Config {
        input: "../data/data_train_next.bin".to_string(),
        sample: 0,
        strict: false,
    };

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" => {
                i += 1;
                cfg.input = args.get(i).cloned().expect("missing value for --input");
            }
            "--sample" => {
                i += 1;
                cfg.sample = args
                    .get(i)
                    .expect("missing value for --sample")
                    .parse::<u64>()
                    .expect("invalid --sample");
            }
            "--strict" => {
                cfg.strict = true;
            }
            "--help" | "-h" => {
                eprintln!("Check quality of binary ChessBoard training data");
                eprintln!();
                eprintln!("Usage: quality_check [OPTIONS]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --input FILE   Input binary file (default: ../data/data_train_next.bin)");
                eprintln!("  --sample N     Inspect at most N positions (default: 0 = all)");
                eprintln!("  --strict       Exit non-zero if any quality warning triggers");
                std::process::exit(0);
            }
            other => panic!("unknown argument: {other}"),
        }
        i += 1;
    }

    cfg
}

#[inline]
fn is_draw(board: &ChessBoard) -> bool {
    board.result() == 0.5
}

#[inline]
fn is_win(board: &ChessBoard) -> bool {
    board.result() == 1.0
}

#[inline]
fn has_non_king_piece(board: &ChessBoard) -> bool {
    for (piece, _) in (*board).into_iter() {
        let kind = piece & 0b111;
        if kind != 5 {
            return true;
        }
    }
    false
}

fn ratio(n: u64, d: u64) -> f64 {
    if d == 0 { 0.0 } else { n as f64 / d as f64 }
}

#[inline]
fn sign_agrees(score_cp_white: i32, result: f32) -> bool {
    match result {
        1.0 => score_cp_white > 0,
        0.0 => score_cp_white < 0,
        _ => false,
    }
}

fn main() {
    let cfg = parse_args();

    let file_size = std::fs::metadata(&cfg.input)
        .unwrap_or_else(|e| panic!("cannot stat input '{}': {e}", cfg.input))
        .len();
    assert!(file_size.is_multiple_of(32), "input is not 32-byte aligned: {}", cfg.input);
    let available = file_size / 32;

    let loader = DataLoader::<ChessBoard>::new(&cfg.input, 1024).expect("failed to open input");
    let mut s = Stats::default();

    loader.map_positions(|b| {
        if cfg.sample > 0 && s.total >= cfg.sample {
            return;
        }

        s.total += 1;

        if is_win(b) {
            s.wins += 1;
        } else if is_draw(b) {
            s.draws += 1;
        } else {
            s.losses += 1;
        }

        if !has_non_king_piece(b) {
            s.king_only += 1;
        }

        let abs_cp = i32::from(b.score()).unsigned_abs() as u64;
        s.sum_abs_cp += abs_cp;

        if abs_cp <= 20 {
            s.near_zero += 1;
        }
        if abs_cp >= 1500 {
            s.high_abs += 1;
        }
        if is_draw(b) && abs_cp >= 250 {
            s.high_abs_draws += 1;
        }

        // Perspective consistency diagnostic on decisive positions with
        // meaningful score magnitude.
        let result = b.result();
        let cp = i32::from(b.score());
        if (result == 1.0 || result == 0.0) && abs_cp >= 40 {
            s.decisive_scored += 1;

            // bulletformat::ChessBoard stores score as White-relative cp.
            if sign_agrees(cp, result) {
                s.white_agree += 1;
            }
        }
    });

    assert!(s.total > 0, "no positions found in {}", cfg.input);

    let draw_ratio = ratio(s.draws, s.total);
    let win_ratio = ratio(s.wins, s.total);
    let loss_ratio = ratio(s.losses, s.total);
    let king_only_ratio = ratio(s.king_only, s.total);
    let near_zero_ratio = ratio(s.near_zero, s.total);
    let high_abs_ratio = ratio(s.high_abs, s.total);
    let high_abs_draw_ratio = ratio(s.high_abs_draws, s.draws);
    let mean_abs_cp = s.sum_abs_cp as f64 / s.total as f64;
    let white_agree_ratio = ratio(s.white_agree, s.decisive_scored);

    let mut warnings: Vec<String> = Vec::new();

    if draw_ratio > 0.70 {
        warnings.push(format!("draw ratio too high: {:.2}% (>70%)", 100.0 * draw_ratio));
    }
    if draw_ratio < 0.25 {
        warnings.push(format!("draw ratio unusually low: {:.2}% (<25%)", 100.0 * draw_ratio));
    }
    if king_only_ratio > 0.03 {
        warnings.push(format!("king-only positions high: {:.2}% (>3%)", 100.0 * king_only_ratio));
    }
    if near_zero_ratio > 0.55 {
        warnings.push(format!("too many near-zero evals: {:.2}% (>55%)", 100.0 * near_zero_ratio));
    }
    if mean_abs_cp < 45.0 {
        warnings.push(format!("mean |score| too low: {:.1} cp (<45)", mean_abs_cp));
    }
    if high_abs_ratio > 0.08 {
        warnings.push(format!("too many extreme scores: {:.2}% (>8%)", 100.0 * high_abs_ratio));
    }
    if s.decisive_scored >= 10_000 && white_agree_ratio < 0.60 {
        warnings.push(format!(
            "weak score/result consistency: white {:.2}% (<60%)",
            100.0 * white_agree_ratio,
        ));
    }

    eprintln!("Data quality report");
    eprintln!("  input: {}", cfg.input);
    eprintln!("  available positions: {}", available);
    eprintln!(
        "  scanned positions: {}{}",
        s.total,
        if cfg.sample > 0 {
            format!(" (sample cap: {})", cfg.sample)
        } else {
            "".to_string()
        }
    );
    eprintln!(
        "  W/D/L: {} / {} / {}  ({:.2}% / {:.2}% / {:.2}%)",
        s.wins,
        s.draws,
        s.losses,
        100.0 * win_ratio,
        100.0 * draw_ratio,
        100.0 * loss_ratio
    );
    eprintln!("  mean |score|: {:.1} cp", mean_abs_cp);
    eprintln!("  near-zero (|cp|<=20): {:.2}%", 100.0 * near_zero_ratio);
    eprintln!("  extreme (|cp|>=1500): {:.2}%", 100.0 * high_abs_ratio);
    eprintln!("  high-|cp| draws (|cp|>=250): {:.2}% of draws", 100.0 * high_abs_draw_ratio);
    eprintln!("  king-only positions: {:.2}%", 100.0 * king_only_ratio);
    eprintln!(
        "  score/result consistency (white-relative, decisive, |cp|>=40): {:.2}% (n={})",
        100.0 * white_agree_ratio,
        s.decisive_scored
    );

    if warnings.is_empty() {
        eprintln!("  quality: OK");
    } else {
        eprintln!("  quality: WARN ({})", warnings.len());
        for w in &warnings {
            eprintln!("    - {}", w);
        }
    }

    if cfg.strict && !warnings.is_empty() {
        std::process::exit(2);
    }
}
