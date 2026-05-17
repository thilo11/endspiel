//! Empirical benchmark: compare static NNUE eval (depth 0) with searched evals
//! at depths 2, 4, 6, 8, 10 across datagen-like positions (after 8 random plies).
//!
//! Run with:
//!   cargo run --release -p chess-datagen --bin eval_bench [-- --threads N]

use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

use chess_common::{Board, Color};
use chess_core::generate_legal_moves;
use chess_engine::{Engine, SearchParams};
use chess_nnue::{Accumulator, NnueNetwork, nnue_evaluate};
use rand::{Rng, RngExt};

const NUM_POSITIONS: usize = 10_000;
const RANDOM_PLIES: usize = 8;
const DEPTHS: &[u8] = &[2, 4, 6, 8, 10];

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("Usage: eval_bench [OPTIONS]");
        eprintln!();
        eprintln!("Compares static NNUE eval (depth 0) against searched evals at depths");
        eprintln!("2, 4, 6, 8, 10 across {NUM_POSITIONS} datagen-like positions (8 random");
        eprintln!("opening plies). Prints mean absolute error vs depth 10 and the");
        eprintln!("incremental gain at each depth step.");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --threads N   Worker threads (default: logical CPU count)");
        eprintln!("  --help        Show this help message");
        std::process::exit(0);
    }

    let num_threads = args.windows(2)
        .find(|w| w[0] == "--threads")
        .and_then(|w| w[1].parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
        });

    eprintln!(
        "Generating {NUM_POSITIONS} positions, evaluating at depth 0 / {} ({num_threads} threads)...",
        DEPTHS.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" / ")
    );

    let net = NnueNetwork::embedded();

    // Generate all positions up front (fast — no search involved).
    let positions: Vec<Board> = {
        let mut rng = rand::rng();
        let mut out = Vec::with_capacity(NUM_POSITIONS);
        while out.len() < NUM_POSITIONS {
            if let Some(b) = random_position(&mut rng) {
                out.push(b);
            }
        }
        out
    };

    // Each sample: [d0, d2, d4, d6, d8, d10]
    let samples: Arc<Mutex<Vec<[i32; 6]>>> = Arc::new(Mutex::new(Vec::with_capacity(NUM_POSITIONS)));
    let done_counter = Arc::new(AtomicUsize::new(0));
    let pos_arc = Arc::new(positions);

    let chunk_size = NUM_POSITIONS.div_ceil(num_threads);

    let handles: Vec<_> = (0..num_threads).map(|t| {
        let net = Arc::clone(&net);
        let samples = Arc::clone(&samples);
        let done_counter = Arc::clone(&done_counter);
        let positions = Arc::clone(&pos_arc);

        std::thread::spawn(move || {
            let start = t * chunk_size;
            let end = (start + chunk_size).min(NUM_POSITIONS);
            if start >= NUM_POSITIONS {
                return;
            }

            let mut engine = Engine::with_hash(16);
            engine.set_threads(1);
            engine.set_use_nnue(true);

            let mut local: Vec<[i32; 6]> = Vec::with_capacity(end - start);

            for board in &positions[start..end] {
                // depth 0: pure NNUE static eval
                let mut acc = Accumulator::new();
                acc.refresh(board, &net);
                let raw_d0 = nnue_evaluate(&acc, board.side_to_move, &net);
                let d0 = stm_to_white(board, raw_d0);

                // searched depths — each with a fresh TT
                let mut evals = [0i32; 5];
                let mut skip = false;
                for (i, &depth) in DEPTHS.iter().enumerate() {
                    engine.clear_tt();
                    let result = engine.search(board, &search_params(depth), None);
                    if result.score.is_mate() {
                        skip = true;
                        break;
                    }
                    evals[i] = stm_to_white(board, result.score.centipawns());
                }
                if skip {
                    continue;
                }

                local.push([d0, evals[0], evals[1], evals[2], evals[3], evals[4]]);

                let n = done_counter.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_multiple_of(1000) {
                    eprintln!("  {n}/{NUM_POSITIONS}");
                }
            }

            samples.lock().unwrap().extend(local);
        })
    }).collect();

    for h in handles {
        h.join().expect("thread panicked");
    }

    let samples = Arc::try_unwrap(samples).unwrap().into_inner().unwrap();
    eprintln!("\nDone. {} positions.\n", samples.len());

    println!("=== Eval divergence vs depth 10 (ground truth): {} positions ===", samples.len());
    println!("(centipawns, white-relative, mean absolute difference vs d10)\n");

    let labels = ["d0", "d2", "d4", "d6", "d8"];
    println!("{:<8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "vs d10", "mean", "p50", "p75", "p90", "p95", "p99", "max");
    println!("{}", "-".repeat(76));
    for (col, label) in labels.iter().enumerate() {
        let mut diffs: Vec<i32> = samples.iter().map(|s| (s[col] - s[5]).abs()).collect();
        print_row(label, &mut diffs);
    }

    println!();
    println!("=== Incremental gain per depth step ===\n");

    let step_labels = ["d0→d2", "d2→d4", "d4→d6", "d6→d8", "d8→d10"];
    println!("{:<8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "step", "mean", "p50", "p75", "p90", "p95", "p99", "max");
    println!("{}", "-".repeat(76));
    for step in 0..5 {
        let mut diffs: Vec<i32> = samples.iter().map(|s| (s[step] - s[step + 1]).abs()).collect();
        print_row(step_labels[step], &mut diffs);
    }
}

fn search_params(depth: u8) -> SearchParams {
    SearchParams { max_depth: depth, use_nnue: true, contempt: 0, ..SearchParams::default() }
}

fn stm_to_white(board: &Board, score_stm: i32) -> i32 {
    if board.side_to_move == Color::White { score_stm } else { -score_stm }
}

fn random_position(rng: &mut impl Rng) -> Option<Board> {
    let mut board = Board::starting_position();
    for _ in 0..RANDOM_PLIES {
        let moves = generate_legal_moves(&board);
        if moves.is_empty() { return None; }
        let idx = rng.random_range(0..moves.len());
        board.make_move(moves.as_slice()[idx]);
    }
    if chess_core::is_in_check(&board) { return None; }
    Some(board)
}

fn print_row(label: &str, diffs: &mut [i32]) {
    diffs.sort_unstable();
    let n = diffs.len();
    let mean = diffs.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let pct = |p: f64| -> i32 {
        let idx = ((p / 100.0) * n as f64) as usize;
        diffs[idx.min(n - 1)]
    };
    println!("{:<8}  {:>8.1}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        label, mean, pct(50.0), pct(75.0), pct(90.0), pct(95.0), pct(99.0), diffs[n - 1]);
}
