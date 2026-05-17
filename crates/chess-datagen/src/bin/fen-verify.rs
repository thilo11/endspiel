//! Verify that all FENs in a file parse via the engine's `Board::from_fen`.
//! Reports count of passing/failing FENs and prints the first few failures.

use chess_common::Board;
use std::io::{BufRead, BufReader};

fn main() {
    let path = std::env::args().nth(1).expect("usage: fen-verify <file>");
    let file = std::fs::File::open(&path).expect("open");
    let reader = BufReader::new(file);

    let mut ok = 0usize;
    let mut bad = 0usize;
    let mut piece_hist = std::collections::BTreeMap::<u32, usize>::new();
    let mut failures: Vec<(usize, String, String)> = Vec::new();

    for (lineno, line) in reader.lines().enumerate() {
        let line = match line { Ok(l) => l, Err(_) => continue };
        let fen = line.trim();
        if fen.is_empty() { continue; }
        match Board::from_fen(fen) {
            Ok(b) => {
                ok += 1;
                let pc = (b.occupancy[0].0 | b.occupancy[1].0).count_ones();
                *piece_hist.entry(pc).or_default() += 1;
            }
            Err(e) => {
                bad += 1;
                if failures.len() < 5 {
                    failures.push((lineno + 1, fen.to_string(), format!("{e:?}")));
                }
            }
        }
    }

    println!("parsed OK : {ok}");
    println!("failed    : {bad}");
    println!("piece-count histogram:");
    for (pc, n) in &piece_hist {
        println!("  {pc:2}: {n}");
    }
    for (ln, fen, err) in &failures {
        println!("FAIL line {ln}: {fen}  -> {err}");
    }
    if bad > 0 { std::process::exit(1); }
}
