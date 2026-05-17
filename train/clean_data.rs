use std::io::Write;

use bulletformat::{BulletFormat, ChessBoard, DataLoader};

const BUF_BATCH: usize = 65_536;

#[derive(Clone, Debug)]
struct Config {
    input: String,
    output: String,
    target_draw_ratio: Option<f64>,
    drop_king_only: bool,
    drop_high_abs_score_draws_cp: i32,
    dedup: bool,
}

#[derive(Default, Clone, Copy, Debug)]
struct Stats {
    total: u64,
    kept: u64,
    dropped: u64,
    dropped_king_only: u64,
    dropped_high_abs_draw: u64,
    dropped_dedup: u64,
    wins: u64,
    draws: u64,
    losses: u64,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();

    let mut cfg = Config {
        input: "../data.bin".to_string(),
        output: "../data_clean.bin".to_string(),
        target_draw_ratio: Some(0.60),
        drop_king_only: true,
        drop_high_abs_score_draws_cp: 250,
        dedup: false,
    };

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" => {
                i += 1;
                cfg.input = args.get(i).cloned().expect("missing value for --input");
            }
            "--output" | "-o" => {
                i += 1;
                cfg.output = args.get(i).cloned().expect("missing value for --output");
            }
            "--target-draw-ratio" => {
                i += 1;
                let s = args.get(i).expect("missing value for --target-draw-ratio");
                if s.eq_ignore_ascii_case("none") {
                    cfg.target_draw_ratio = None;
                } else {
                    let v: f64 = s.parse().expect("invalid --target-draw-ratio");
                    assert!(v > 0.0 && v < 1.0, "--target-draw-ratio must be in (0,1)");
                    cfg.target_draw_ratio = Some(v);
                }
            }
            "--drop-king-only" => {
                i += 1;
                cfg.drop_king_only = args
                    .get(i)
                    .expect("missing value for --drop-king-only")
                    .parse::<bool>()
                    .expect("invalid bool for --drop-king-only");
            }
            "--drop-high-abs-score-draws" => {
                i += 1;
                cfg.drop_high_abs_score_draws_cp = args
                    .get(i)
                    .expect("missing value for --drop-high-abs-score-draws")
                    .parse::<i32>()
                    .expect("invalid int for --drop-high-abs-score-draws");
                assert!(
                    cfg.drop_high_abs_score_draws_cp >= 0,
                    "--drop-high-abs-score-draws must be >= 0"
                );
            }
            "--dedup" => {
                cfg.dedup = true;
            }
            "--help" | "-h" => {
                eprintln!("Clean/rebalance binary ChessBoard training data");
                eprintln!();
                eprintln!("Usage: clean_data [OPTIONS]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --input FILE                    Input binary file (default: ../data.bin)");
                eprintln!("  --output FILE                   Output binary file (default: ../data_clean.bin)");
                eprintln!("  --target-draw-ratio F|none      Downsample draws toward target ratio (default: 0.60)");
                eprintln!("  --drop-king-only true|false     Drop king-only records (default: true)");
                eprintln!("  --drop-high-abs-score-draws N   Drop draw labels with |score| >= N cp (default: 250)");
                eprintln!("  --dedup                         Drop duplicate positions (keeps first occurrence)");
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

#[inline]
fn hash64_fnv1a(data: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut h = OFFSET;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(PRIME);
    }
    h
}

#[inline]
fn board_hash(board: &ChessBoard) -> u64 {
    let one = [*board];
    hash64_fnv1a(ChessBoard::as_bytes_slice(&one))
}

fn pass1(cfg: &Config) -> Stats {
    let mut s = Stats::default();
    let loader = DataLoader::<ChessBoard>::new(&cfg.input, 1024).expect("failed to open input");

    loader.map_positions(|b| {
        s.total += 1;

        if cfg.drop_king_only && !has_non_king_piece(b) {
            s.dropped += 1;
            s.dropped_king_only += 1;
            return;
        }

        if is_draw(b) && cfg.drop_high_abs_score_draws_cp > 0 && i32::from(b.score()).unsigned_abs() as i32 >= cfg.drop_high_abs_score_draws_cp {
            s.dropped += 1;
            s.dropped_high_abs_draw += 1;
            return;
        }

        s.kept += 1;
        if is_win(b) {
            s.wins += 1;
        } else if is_draw(b) {
            s.draws += 1;
        } else {
            s.losses += 1;
        }
    });

    s
}

fn pass2(cfg: &Config, keep_draw_prob: f64) -> Stats {
    use std::collections::HashSet;

    let loader = DataLoader::<ChessBoard>::new(&cfg.input, 1024).expect("failed to open input");
    let out_file = std::fs::File::create(&cfg.output).expect("failed to create output");
    let mut writer = std::io::BufWriter::new(out_file);

    let mut out_stats = Stats::default();
    let mut buf: Vec<ChessBoard> = Vec::with_capacity(BUF_BATCH);
    let mut seen: HashSet<u64> = HashSet::new();

    loader.map_positions(|b| {
        out_stats.total += 1;

        if out_stats.total % 50_000_000 == 0 {
            eprintln!("  ... {}M positions scanned", out_stats.total / 1_000_000);
        }

        if cfg.drop_king_only && !has_non_king_piece(b) {
            out_stats.dropped += 1;
            out_stats.dropped_king_only += 1;
            return;
        }

        if is_draw(b) && cfg.drop_high_abs_score_draws_cp > 0 && i32::from(b.score()).unsigned_abs() as i32 >= cfg.drop_high_abs_score_draws_cp {
            out_stats.dropped += 1;
            out_stats.dropped_high_abs_draw += 1;
            return;
        }

        if cfg.dedup {
            let h = board_hash(b);
            if !seen.insert(h) {
                out_stats.dropped += 1;
                out_stats.dropped_dedup += 1;
                return;
            }
        }

        if is_draw(b) && keep_draw_prob < 1.0 {
            let threshold = (keep_draw_prob * (u64::MAX as f64)) as u64;
            if board_hash(b) > threshold {
                out_stats.dropped += 1;
                return;
            }
        }

        out_stats.kept += 1;
        if is_win(b) {
            out_stats.wins += 1;
        } else if is_draw(b) {
            out_stats.draws += 1;
        } else {
            out_stats.losses += 1;
        }

        buf.push(*b);
        if buf.len() >= BUF_BATCH {
            writer
                .write_all(ChessBoard::as_bytes_slice(&buf))
                .expect("write failed");
            buf.clear();
        }
    });

    if !buf.is_empty() {
        writer
            .write_all(ChessBoard::as_bytes_slice(&buf))
            .expect("write failed");
    }
    writer.flush().expect("flush failed");

    out_stats
}

fn draw_ratio(draws: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        draws as f64 / total as f64
    }
}

fn main() {
    let cfg = parse_args();

    eprintln!("Input : {}", cfg.input);
    eprintln!("Output: {}", cfg.output);
    eprintln!("drop_king_only = {}", cfg.drop_king_only);
    eprintln!("drop_high_abs_score_draws_cp = {}", cfg.drop_high_abs_score_draws_cp);
    eprintln!("target_draw_ratio = {:?}", cfg.target_draw_ratio);
    eprintln!("dedup = {}", cfg.dedup);

    let before = pass1(&cfg);
    let clean_kept = before.wins + before.draws + before.losses;

    eprintln!();
    eprintln!("Pre-sampling stats (after hard filters):");
    eprintln!("  kept:   {}", clean_kept);
    eprintln!("  wins:   {}", before.wins);
    eprintln!("  draws:  {} ({:.2}%)", before.draws, 100.0 * draw_ratio(before.draws, clean_kept));
    eprintln!("  losses: {}", before.losses);
    eprintln!("  dropped_king_only: {}", before.dropped_king_only);
    eprintln!("  dropped_high_abs_draw: {}", before.dropped_high_abs_draw);
    eprintln!("  note: dedup is applied in pass2, not reflected here");

    let keep_draw_prob = if let Some(target) = cfg.target_draw_ratio {
        let non_draw = before.wins + before.losses;
        if before.draws == 0 || non_draw == 0 {
            1.0
        } else {
            let desired_draws = target / (1.0 - target) * non_draw as f64;
            (desired_draws / before.draws as f64).clamp(0.0, 1.0)
        }
    } else {
        1.0
    };

    eprintln!("  draw keep probability: {:.4}", keep_draw_prob);

    eprintln!();
    eprintln!("Writing output...");
    let after = pass2(&cfg, keep_draw_prob);
    let total_after = after.wins + after.draws + after.losses;

    eprintln!();
    eprintln!("Output stats:");
    eprintln!("  kept:   {}", total_after);
    eprintln!("  wins:   {}", after.wins);
    eprintln!("  draws:  {} ({:.2}%)", after.draws, 100.0 * draw_ratio(after.draws, total_after));
    eprintln!("  losses: {}", after.losses);
    eprintln!("  dropped_dedup: {}", after.dropped_dedup);

    let out_size = std::fs::metadata(&cfg.output)
        .expect("failed to stat output")
        .len();
    eprintln!("Output file size: {} bytes ({:.1}M positions)", out_size, out_size as f64 / 32.0 / 1_000_000.0);
}
