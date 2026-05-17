/// Extract late-middlegame / early-endgame FENs from the Lichess eval database.
///
/// Reads a `.jsonl.zst` file (e.g. `lichess_db_eval.jsonl.zst`) line by line,
/// applies material and eval filters, and writes one FEN per line to an output
/// file.  The resulting file is suitable as input to `chess-datagen --start-fens`.
///
/// Filters applied (in order):
///   1. Castling field contains only standard chars (K/Q/k/q/-) — rejects Chess960
///   2. Exactly one white king and one black king — rejects broken FENs
///   3. Total material (both sides, kings excluded) in [--min-material, --max-material]
///   4. |first cp eval| ≤ --max-eval  (skipped if --max-eval 0; positions with only
///      a mate score are always rejected as already-decided)
///
/// Usage:
///   extract_fens --input FILE.jsonl.zst --output fens.txt [OPTIONS]
use std::io::{BufRead, BufReader, BufWriter, Write};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut input_path = String::new();
    let mut output_path = String::new();
    let mut min_material: i32 = 20;
    let mut max_material: i32 = 48;
    let mut max_eval: i32 = 200;
    let mut limit: u64 = u64::MAX;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => { i += 1; input_path = args[i].clone(); }
            "--output" => { i += 1; output_path = args[i].clone(); }
            "--min-material" => { i += 1; min_material = args[i].parse().expect("invalid --min-material"); }
            "--max-material" => { i += 1; max_material = args[i].parse().expect("invalid --max-material"); }
            "--max-eval" => { i += 1; max_eval = args[i].parse().expect("invalid --max-eval"); }
            "--limit" => { i += 1; limit = args[i].parse().expect("invalid --limit"); }
            "--help" | "-h" => {
                eprintln!("extract_fens — extract FENs from Lichess eval database");
                eprintln!();
                eprintln!("Usage: extract_fens --input FILE.jsonl.zst --output fens.txt [OPTIONS]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --input FILE        Lichess eval database (.jsonl.zst)");
                eprintln!("  --output FILE       Output FEN list (one per line)");
                eprintln!("  --min-material N    Min total material excl. kings (default: 20)");
                eprintln!("  --max-material N    Max total material excl. kings (default: 48)");
                eprintln!("  --max-eval N        Max |cp| to accept; 0 = no filter (default: 200)");
                eprintln!("  --limit N           Stop after writing N FENs (default: unlimited)");
                eprintln!();
                eprintln!("Material scale: P/p=1  N/n=B/b=3  R/r=5  Q/q=9  (kings excluded)");
                eprintln!("Starting material (both sides): 78.  Late middlegame: ~20–48.");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if input_path.is_empty() || output_path.is_empty() {
        eprintln!("Error: --input and --output are required");
        std::process::exit(1);
    }

    let file = std::fs::File::open(&input_path)
        .unwrap_or_else(|e| panic!("cannot open '{input_path}': {e}"));
    let decoder = zstd::Decoder::new(file)
        .unwrap_or_else(|e| panic!("cannot create zstd decoder: {e}"));
    let reader = BufReader::new(decoder);

    let out = std::fs::File::create(&output_path)
        .unwrap_or_else(|e| panic!("cannot create '{output_path}': {e}"));
    let mut writer = BufWriter::new(out);

    let mut lines_read: u64 = 0;
    let mut written: u64 = 0;

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        lines_read += 1;

        if lines_read.is_multiple_of(1_000_000) {
            eprintln!("  scanned {}M lines, kept {written} FENs", lines_read / 1_000_000);
        }

        let fen = match extract_fen(&line) {
            Some(f) => f,
            None => continue,
        };

        // 1. Reject Chess960: castling field must only contain K/Q/k/q/-
        if !castling_is_standard(fen) {
            continue;
        }

        // 2. Exactly one king per side
        if !has_valid_kings(fen) {
            continue;
        }

        // 3. Material range
        let mat = material_count(fen);
        if mat < min_material || mat > max_material {
            continue;
        }

        // 4. Eval filter (skip positions with only a mate score — already decided)
        if max_eval > 0 {
            match extract_cp(&line) {
                Some(cp) if cp.abs() <= max_eval => {}
                _ => continue,
            }
        }

        writeln!(writer, "{fen}").expect("write failed");
        written += 1;

        if written >= limit {
            break;
        }
    }

    writer.flush().expect("flush failed");
    eprintln!("Done. Scanned {lines_read} lines, wrote {written} FENs → {output_path}");
}

/// Extract the FEN string from a JSONL line (`"fen":"<value>"`).
fn extract_fen(line: &str) -> Option<&str> {
    const MARKER: &str = "\"fen\":\"";
    let start = line.find(MARKER)? + MARKER.len();
    let end = start + line[start..].find('"')?;
    Some(&line[start..end])
}

/// Extract the first centipawn eval from a JSONL line (`"cp":<value>`).
/// Returns None if no cp field exists (position has only a mate score).
fn extract_cp(line: &str) -> Option<i32> {
    const MARKER: &str = "\"cp\":";
    let start = line.find(MARKER)? + MARKER.len();
    let rest = &line[start..];
    let end = rest
        .find(|c: char| c != '-' && !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Count total material (both sides, kings excluded) from a FEN's piece-placement field.
/// P/p=1, N/n=B/b=3, R/r=5, Q/q=9.
fn material_count(fen: &str) -> i32 {
    fen.split_whitespace()
        .next()
        .unwrap_or("")
        .chars()
        .map(|c| match c {
            'P' | 'p' => 1,
            'N' | 'n' | 'B' | 'b' => 3,
            'R' | 'r' => 5,
            'Q' | 'q' => 9,
            _ => 0,
        })
        .sum()
}

/// Return true if the FEN's castling-rights field contains only standard chars
/// (K, Q, k, q, -).  Chess960 uses file letters (A–H, a–h) which would pass
/// through as syntactically valid but semantically wrong for standard chess.
fn castling_is_standard(fen: &str) -> bool {
    let castling = fen.split_whitespace().nth(2).unwrap_or("-");
    castling.chars().all(|c| matches!(c, 'K' | 'Q' | 'k' | 'q' | '-'))
}

/// Return true if the piece-placement field has exactly one white king and one
/// black king.  Rejects corrupted or placeholder FENs.
fn has_valid_kings(fen: &str) -> bool {
    let placement = fen.split_whitespace().next().unwrap_or("");
    let wk = placement.chars().filter(|&c| c == 'K').count();
    let bk = placement.chars().filter(|&c| c == 'k').count();
    wk == 1 && bk == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_count_starting() {
        // Starting position: Q+2R+2B+2N+8P per side = 39 × 2 = 78
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        assert_eq!(material_count(fen), 78);
    }

    #[test]
    fn material_count_endgame() {
        let fen = "8/4r3/2R2pk1/6pp/3P4/6P1/5K1P/8 b - -";
        // r=5, R=5, p×3=3 (f6/g5/h5), P×3=3 (d4/g3/h2) → 16
        assert_eq!(material_count(fen), 16);
    }

    #[test]
    fn castling_standard() {
        assert!(castling_is_standard("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"));
        assert!(castling_is_standard("8/8/8/8/8/8/8/8 w - -"));
    }

    #[test]
    fn castling_chess960_rejected() {
        // Chess960 uses file letters for castling rights
        assert!(!castling_is_standard("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w HAha - 4 4"));
        assert!(!castling_is_standard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w AHah - 0 1"));
    }

    #[test]
    fn valid_kings() {
        assert!(has_valid_kings("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"));
        assert!(!has_valid_kings("rnbqKbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - -")); // two white kings
        assert!(!has_valid_kings("rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - -")); // no black king
    }

    #[test]
    fn extract_fen_from_jsonl() {
        let line = r#"{"fen":"7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - -","evals":[{"pvs":[{"cp":69}]}]}"#;
        assert_eq!(extract_fen(line), Some("7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - -"));
    }

    #[test]
    fn extract_cp_from_jsonl() {
        let line = r#"{"fen":"...","evals":[{"pvs":[{"cp":69}]}]}"#;
        assert_eq!(extract_cp(line), Some(69));
        let line_neg = r#"{"fen":"...","evals":[{"pvs":[{"cp":-150}]}]}"#;
        assert_eq!(extract_cp(line_neg), Some(-150));
        let line_mate = r#"{"fen":"...","evals":[{"pvs":[{"mate":5}]}]}"#;
        assert_eq!(extract_cp(line_mate), None);
    }
}
