// Convert Endspiel training data from text to binary (ChessBoard) format.
//
// The text format is: "FEN | score_cp | result" (one position per line).
// The binary format is bulletformat::ChessBoard (32 bytes per position),
// which can be loaded efficiently at training time with DirectSequentialDataLoader.
//
// Usage:
//   cd train
//   cargo run --release --bin convert -- --input ../data.txt --output ../data.bin

use std::io::{BufRead, Write};

use bulletformat::{BulletFormat, ChessBoard};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut input = String::from("../data.txt");
    let mut output = String::from("../data.bin");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" => {
                i += 1;
                input = args[i].clone();
            }
            "--output" | "-o" => {
                i += 1;
                output = args[i].clone();
            }
            "--help" | "-h" => {
                eprintln!("Convert Endspiel training data from text to binary format");
                eprintln!();
                eprintln!("Usage: convert [OPTIONS]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --input FILE   Input text file (default: ../data.txt)");
                eprintln!("  --output FILE  Output binary file (default: ../data.bin)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    eprintln!("Converting '{input}' → '{output}'");

    let in_file = std::fs::File::open(&input)
        .unwrap_or_else(|e| { eprintln!("Failed to open input '{input}': {e}"); std::process::exit(1); });
    let out_file = std::fs::File::create(&output)
        .unwrap_or_else(|e| { eprintln!("Failed to create output '{output}': {e}"); std::process::exit(1); });

    let reader = std::io::BufReader::new(in_file);
    let mut writer = std::io::BufWriter::new(out_file);

    const BATCH: usize = 65_536;
    let mut buf: Vec<ChessBoard> = Vec::with_capacity(BATCH);
    let mut total: u64 = 0;
    let mut errors: u64 = 0;

    for (line_no, line) in reader.lines().enumerate() {
        let line = line.unwrap_or_else(|e| { eprintln!("IO error at line {line_no}: {e}"); std::process::exit(1); });
        if line.is_empty() {
            continue;
        }

        match line.parse::<ChessBoard>() {
            Ok(board) => {
                buf.push(board);
                total += 1;
                if buf.len() >= BATCH {
                    writer.write_all(ChessBoard::as_bytes_slice(&buf))
                        .unwrap_or_else(|e| { eprintln!("Write error: {e}"); std::process::exit(1); });
                    buf.clear();

                    if total.is_multiple_of(10_000_000) {
                        eprintln!("  {:.0}M positions converted...", total as f64 / 1_000_000.0);
                    }
                }
            }
            Err(e) => {
                errors += 1;
                if errors <= 5 {
                    eprintln!("Warning: line {line_no}: {e}");
                } else if errors == 6 {
                    eprintln!("(further parse errors suppressed)");
                }
            }
        }
    }

    if !buf.is_empty() {
        writer.write_all(ChessBoard::as_bytes_slice(&buf))
            .unwrap_or_else(|e| { eprintln!("Write error: {e}"); std::process::exit(1); });
    }
    writer.flush().unwrap_or_else(|e| { eprintln!("Flush error: {e}"); std::process::exit(1); });

    let size_mb = total * 32 / 1024 / 1024;
    eprintln!("Done. {total} positions → {output} ({size_mb} MB, {errors} parse errors)");
}
