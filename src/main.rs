use std::env;

/// Run the real main on a thread with an explicit 4 MB stack so that deep
/// search recursion doesn't overflow on platforms with small defaults
/// (e.g. Windows ARM64 where the default is ~1 MB).
fn main() {
    let builder = std::thread::Builder::new().stack_size(4 * 1024 * 1024);
    let handler = builder
        .spawn(real_main)
        .expect("failed to spawn main thread");
    handler.join().unwrap();
}

fn real_main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp_millis()
        .init();

    log::info!("Endspiel engine starting");

    let args: Vec<String> = env::args().collect();

    if args.len() > 1 && args[1] == "bench" {
        let depth = args
            .get(2)
            .and_then(|s| s.parse::<u8>().ok())
            .unwrap_or(BENCH_DEPTH);
        run_bench(depth);
        return;
    }

    let syzygy_path = args.windows(2).find_map(|w| {
        if w[0] == "--syzygy" { Some(w[1].as_str()) } else { None }
    });

    match syzygy_path {
        Some(path) => chess_uci::run_with_syzygy(path),
        None => chess_uci::run(),
    }
}

/// Default depth for `endspiel bench`. Kept fixed so the printed node
/// count is reproducible across runs (useful for verifying that a search
/// change didn't accidentally alter the node tree). The same command is
/// used as the PGO training workload in CI.
const BENCH_DEPTH: u8 = 14;

/// Fixed FEN set for `bench`. Mix of opening, complex middlegame, tactical,
/// and endgame positions so the profile covers diverse evaluation regimes.
const BENCH_FENS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "8/8/4k3/8/2p5/8/B2P1K2/8 w - - 0 1",
];

fn run_bench(depth: u8) {
    use chess_common::Board;
    use chess_engine::{Engine, SearchParams};

    println!("Running bench: depth {} across {} positions", depth, BENCH_FENS.len());

    let mut engine = Engine::with_hash(16);
    engine.set_threads(1);

    let mut total_nodes: u64 = 0;
    let start = std::time::Instant::now();

    for (i, fen) in BENCH_FENS.iter().enumerate() {
        let board = Board::from_fen(fen).expect("invalid bench FEN");
        engine.clear_tt();
        let params = SearchParams {
            max_depth: depth,
            ..SearchParams::default()
        };
        let pos_start = std::time::Instant::now();
        let result = engine.search(&board, &params, None);
        let pos_elapsed = pos_start.elapsed();
        println!(
            "  [{}/{}] depth {} nodes {:>10} time {:>6.2}s",
            i + 1,
            BENCH_FENS.len(),
            result.depth,
            result.nodes,
            pos_elapsed.as_secs_f64()
        );
        total_nodes = total_nodes.saturating_add(result.nodes);
    }

    let elapsed = start.elapsed();
    let nps = if elapsed.as_secs_f64() > 0.0 {
        (total_nodes as f64 / elapsed.as_secs_f64()) as u64
    } else {
        0
    };

    println!("===========================");
    println!("Nodes: {}", total_nodes);
    println!("Time:  {:.2}s", elapsed.as_secs_f64());
    println!("NPS:   {}", nps);
}
