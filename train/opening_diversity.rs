use std::collections::HashSet;

use bulletformat::{BulletFormat, ChessBoard, DataLoader};

fn parse_args() -> (String, usize) {
    let args: Vec<String> = std::env::args().collect();
    let mut input = "../data/data_new.bin".to_string();
    let mut min_pieces: usize = 28;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" => {
                i += 1;
                input = args.get(i).cloned().expect("missing value for --input");
            }
            "--min-pieces" => {
                i += 1;
                min_pieces = args
                    .get(i)
                    .expect("missing value for --min-pieces")
                    .parse::<usize>()
                    .expect("invalid --min-pieces");
            }
            "--help" | "-h" => {
                eprintln!("Analyse opening diversity in binary ChessBoard training data");
                eprintln!();
                eprintln!("Usage: opening_diversity [OPTIONS]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --input FILE        Input file (default: ../data/data_new.bin)");
                eprintln!("  --min-pieces N      Min piece count to treat as 'opening' (default: 28)");
                std::process::exit(0);
            }
            other => panic!("unknown argument: {other}"),
        }
        i += 1;
    }

    (input, min_pieces)
}

fn count_pieces(board: &ChessBoard) -> usize {
    (*board).into_iter().count()
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

fn main() {
    let (input, min_pieces) = parse_args();

    eprintln!("Input: {input}");
    eprintln!("Min-pieces threshold for 'opening': {min_pieces}");
    eprintln!();

    let loader = DataLoader::<ChessBoard>::new(&input, 1024).expect("failed to open input");

    // Piece-count histogram (buckets 0..=32)
    let mut piece_hist = [0u64; 33];
    let mut total: u64 = 0;

    // Unique position hashes for opening positions
    let mut opening_hashes: HashSet<u64> = HashSet::new();

    loader.map_positions(|b| {
        total += 1;

        let n = count_pieces(b).min(32);
        piece_hist[n] += 1;

        if n >= min_pieces {
            let one = [*b];
            let h = hash64_fnv1a(ChessBoard::as_bytes_slice(&one));
            opening_hashes.insert(h);
        }
    });

    eprintln!("Total positions scanned: {total}");
    eprintln!();

    // --- piece-count distribution ---
    eprintln!("Piece-count distribution (opening = piece count >= {min_pieces}):");
    eprintln!("{:>6}  {:>12}  {:>7}", "pieces", "count", "%");
    for pc in (2..=32usize).rev() {
        if piece_hist[pc] == 0 {
            continue;
        }
        let pct = 100.0 * piece_hist[pc] as f64 / total as f64;
        let marker = if pc >= min_pieces { " <-- opening" } else { "" };
        eprintln!("{:>6}  {:>12}  {:>6.2}%{}", pc, piece_hist[pc], pct, marker);
    }

    let opening_total: u64 = piece_hist[min_pieces..=32].iter().sum();
    let opening_pct = 100.0 * opening_total as f64 / total as f64;
    let unique = opening_hashes.len();
    let dup_ratio = if opening_total == 0 {
        0.0
    } else {
        1.0 - unique as f64 / opening_total as f64
    };

    eprintln!();
    eprintln!("Opening positions (>= {min_pieces} pieces):");
    eprintln!("  total:          {opening_total} ({opening_pct:.2}% of all)");
    eprintln!("  unique (hashed):{unique}");
    eprintln!("  duplication:    {:.2}%", 100.0 * dup_ratio);
}
