mod game;

use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use bulletformat::{BulletFormat, ChessBoard};
use chess_engine::polyglot::OpeningBook;
use chess_engine::syzygy::SyzygyTB;
use rand::RngExt;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut num_games: u32 = 2_000_000;
    let mut depth: u8 = 16;
    let mut threads: u32 = 32;
    let mut output = String::from("data/data_new.bin");
    let mut hash_mb: usize = 16;
    let mut use_nnue: bool = true;
    let mut book_path: Option<String> = None;
    let mut syzygy_path: Option<String> = None;
    let mut start_fens_path: Option<String> = None;
    let mut sequential_fens: bool = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => {
                i += 1;
                num_games = args[i].parse().expect("invalid --games value");
            }
            "--depth" => {
                i += 1;
                depth = args[i].parse().expect("invalid --depth value");
            }
            "--threads" => {
                i += 1;
                threads = args[i].parse().expect("invalid --threads value");
            }
            "--output" => {
                i += 1;
                output = args[i].clone();
            }
            "--hash" => {
                i += 1;
                hash_mb = args[i].parse().expect("invalid --hash value");
            }
            "--nnue" => {
                i += 1;
                use_nnue = args[i].parse().expect("invalid --nnue value (true/false)");
            }
            "--book" => {
                i += 1;
                book_path = Some(args[i].clone());
            }
            "--syzygy" => {
                i += 1;
                syzygy_path = Some(args[i].clone());
            }
            "--start-fens" => {
                i += 1;
                start_fens_path = Some(args[i].clone());
            }
            "--sequential-fens" => {
                sequential_fens = true;
            }
            "--help" | "-h" => {
                eprintln!("Usage: chess-datagen [OPTIONS]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --games N          Number of games to generate (default: 2000000)");
                eprintln!("  --depth D          Search depth per move (default: 16)");
                eprintln!("  --threads T        Parallel game threads (default: 32)");
                eprintln!("  --output FILE      Output file path (default: data/data_new.bin)");
                eprintln!("  --hash MB          TT size in MB per thread (default: 16)");
                eprintln!("  --nnue BOOL        Use NNUE evaluation (default: true)");
                eprintln!("  --book FILE        Polyglot opening book (.bin) for diverse openings");
                eprintln!("  --syzygy DIR       Syzygy tablebase directory for exact endgame labels");
                eprintln!("  --start-fens FILE  FEN list (one per line) to use as starting positions");
                eprintln!("  --sequential-fens  Consume --start-fens in order (each FEN once, no random opening)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let start_fens: Arc<Vec<String>> = Arc::new(
        start_fens_path
            .as_deref()
            .map(|p| {
                let content = std::fs::read_to_string(p)
                    .unwrap_or_else(|e| panic!("failed to read --start-fens '{p}': {e}"));
                let fens: Vec<String> = content
                    .lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty())
                    .map(str::to_string)
                    .collect();
                eprintln!("Loaded {} starting FENs from {p}", fens.len());
                fens
            })
            .unwrap_or_default(),
    );

    let book = Arc::new(book_path.as_deref().and_then(|p| {
        let book = OpeningBook::open(Path::new(p));
        if book.is_none() {
            eprintln!("Warning: failed to load book from {p}, falling back to random openings");
        }
        book
    }));

    let eval_mode = if use_nnue { "NNUE" } else { "HCE" };
    if sequential_fens && start_fens.is_empty() {
        eprintln!("--sequential-fens requires --start-fens");
        std::process::exit(1);
    }
    let opening_mode = if !start_fens.is_empty() {
        let order = if sequential_fens { "sequential" } else { "random" };
        format!("start-fens ({} positions, {order})", start_fens.len())
    } else if book.is_some() {
        format!("book={}", book_path.as_deref().unwrap_or(""))
    } else {
        "random openings".to_string()
    };
    let syzygy_mode = syzygy_path.as_deref().unwrap_or("off");
    eprintln!(
        "Generating {num_games} games at depth {depth} with {threads} threads (hash {hash_mb} MB, eval {eval_mode}, {opening_mode}, syzygy={syzygy_mode})"
    );

    // Initialise Syzygy tablebases ONCE before spawning threads.
    // pyrrhic-rs uses a global C-library singleton (TB_INITIALIZED flag).
    // Creating a new SyzygyTB per game from multiple threads causes concurrent
    // tb_init / tb_free races that deadlock the C library.
    // Instead we init once here, then hand each thread a cheap Arc clone.
    let syzygy_tb: Arc<Option<SyzygyTB>> = Arc::new(
        syzygy_path.as_deref().and_then(|p| {
            match SyzygyTB::new(p) {
                Ok(tb) => Some(tb),
                Err(e) => {
                    eprintln!("Warning: failed to load Syzygy tablebases from {p}: {e:?}");
                    None
                }
            }
        })
    );

    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&output)
        .expect("failed to open output file");
    let writer = Arc::new(Mutex::new(BufWriter::new(file)));
    let games_done = Arc::new(AtomicU32::new(0));
    let positions_done = Arc::new(AtomicU32::new(0));

    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let writer = Arc::clone(&writer);
            let games_done = Arc::clone(&games_done);
            let positions_done = Arc::clone(&positions_done);
            let book = Arc::clone(&book);
            let start_fens = Arc::clone(&start_fens);
            // Clone the SyzygyTB handle (just bumps an Arc refcount inside).
            let tb = (*syzygy_tb).clone();
            std::thread::spawn(move || {
                let mut rng = rand::rng();

                // Create the engine once per thread with the requested hash
                // size up front.  Using Engine::with_hash avoids allocating
                // and immediately discarding the default 4 GB TT that
                // Engine::new() would create before set_hash_mb replaces it —
                // which caused OOM kills when spawning many threads.
                let mut engine = chess_engine::Engine::with_hash(hash_mb);
                engine.set_threads(1);
                engine.set_use_nnue(use_nnue);
                engine.set_syzygy_tb(tb);

                loop {
                    let game_idx = games_done.fetch_add(1, Ordering::Relaxed);
                    if game_idx >= num_games {
                        break;
                    }
                    // Sequential mode exits as soon as every FEN has been used
                    // exactly once, regardless of the --games ceiling.
                    if sequential_fens && (game_idx as usize) >= start_fens.len() {
                        break;
                    }

                    // Pick the starting FEN for this game (if any). In
                    // sequential mode the index is stable; random mode keeps
                    // historical behaviour (uniform sample + one random move
                    // so duplicate samples diverge).
                    let (start_fen, randomize_after_fen) = if start_fens.is_empty() {
                        (None, false)
                    } else if sequential_fens {
                        (Some(start_fens[game_idx as usize].as_str()), false)
                    } else {
                        let idx = rng.random_range(0..start_fens.len());
                        (Some(start_fens[idx].as_str()), true)
                    };

                    // Bump TT generation between games (avoids clearing 16 MB
                    // of memory per game; old entries are still correct since
                    // the XOR key check guarantees position identity).
                    engine.new_search_tt();
                    let game_boards = game::play_game(
                        &engine,
                        depth,
                        book.as_ref().as_ref(),
                        start_fen,
                        randomize_after_fen,
                        &mut rng,
                    );
                    let count = game_boards.len() as u32;

                    // Flush completed game to disk immediately
                    {
                        let mut w = writer.lock().unwrap();
                        w.write_all(ChessBoard::as_bytes_slice(&game_boards))
                            .expect("failed to write positions");
                    }

                    let total_positions = positions_done.fetch_add(count, Ordering::Relaxed) + count;
                    let done = game_idx + 1;

                    if done.is_multiple_of(100) || done == num_games {
                        eprintln!("Games: {done}/{num_games}  Positions: {total_positions}");
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("thread panicked");
    }

    // Final flush
    writer.lock().unwrap().flush().expect("failed to flush output");

    let total = positions_done.load(Ordering::Relaxed);
    eprintln!("Done. Wrote {total} positions to {output}");
}
