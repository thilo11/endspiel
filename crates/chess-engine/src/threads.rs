//! Lazy SMP thread pool for parallel search.
//!
//! Thread 0 is the "main" thread whose result is reported. Helper threads
//! search the same root position with depth diversity (Stockfish-style
//! offsets) and share results via the lock-free transposition table.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

use chess_common::{Board, Move, Score};
use chess_nnue::NnueNetwork;

use crate::polyglot::OpeningBook;
use crate::search;
use crate::syzygy::SyzygyTB;
use crate::tt::SharedTT;
use crate::{InfoCallback, SearchInfo, SearchParams, SearchResult};

// ---------------------------------------------------------------------------
// Per-thread node counter (cache-line padded to avoid false sharing)
// ---------------------------------------------------------------------------

#[repr(align(64))]
struct PaddedCounter {
    value: AtomicU64,
}

impl PaddedCounter {
    fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }
}

// ---------------------------------------------------------------------------
// Thread pool
// ---------------------------------------------------------------------------

pub struct ThreadPool {
    num_threads: usize,
}

fn active_thread_count(num_threads: usize, root_moves: usize) -> usize {
    if num_threads <= 1 || root_moves <= 1 {
        return 1;
    }

    // Lazy SMP still benefits from some overlap on narrow roots, but spawning
    // a large helper swarm for only a couple of legal moves mostly duplicates
    // work. Keep one extra thread of depth-diverse overlap for tiny roots.
    if root_moves <= 4 {
        return num_threads.min(root_moves + 1);
    }

    num_threads
}

impl ThreadPool {
    pub fn new(num_threads: usize) -> Self {
        Self {
            num_threads: num_threads.max(1),
        }
    }

    /// Run a Lazy SMP search. Thread 0 is the main thread that returns the
    /// result. Helper threads search with depth offsets and contribute to the
    /// shared TT.
    #[allow(clippy::too_many_arguments)]
    pub fn search(
        &self,
        board: &Board,
        params: &SearchParams,
        stop: &Arc<AtomicBool>,
        tt: &Arc<SharedTT>,
        info_callback: Option<InfoCallback>,
        net: &Arc<NnueNetwork>,
        syzygy_tb: Option<SyzygyTB>,
        root_tb_solution: Option<(Score, Vec<Move>)>,
        book: Option<Arc<OpeningBook>>,
    ) -> SearchResult {
        tt.new_search();

        if self.num_threads <= 1 {
            // Single-thread fast path: avoid SMP setup and root prechecks.
            return search::iterative_deepening(board, params, stop, tt, info_callback, 0, net, None, syzygy_tb, root_tb_solution, book);
        }

        let root_moves = chess_core::generate_legal_moves(board);
        let active_threads = active_thread_count(self.num_threads, root_moves.len());

        if active_threads <= 1 {
            // Single-thread fast path: no spawning overhead
            return search::iterative_deepening(board, params, stop, tt, info_callback, 0, net, None, syzygy_tb, root_tb_solution, book);
        }

        let counters: Arc<Vec<PaddedCounter>> = Arc::new(
            (0..active_threads).map(|_| PaddedCounter::new()).collect(),
        );

        // Spawn helper threads (1..N)
        let mut handles = Vec::with_capacity(active_threads - 1);

        for thread_id in 1..active_threads {
            let board = board.clone();
            let params = params.clone();
            let stop = Arc::clone(stop);
            let tt = Arc::clone(tt);
            let counters = Arc::clone(&counters);
            let net = Arc::clone(net);
            let tb = syzygy_tb.clone();
            let book = book.clone();

            let handle = thread::Builder::new()
                .stack_size(4 * 1024 * 1024) // 4 MB – prevent stack overflow on Windows ARM64
                .spawn(move || {
                // Helper threads run without info callback — they just
                // contribute to the TT.  Wrap in catch_unwind so a panic
                // in a helper doesn't poison the process.
                let search_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    search::iterative_deepening(
                        &board, &params, &stop, &tt, None, thread_id, &net, Some(&counters[thread_id].value), tb, None, book,
                    )
                }));
                if let Ok(result) = search_result {
                    counters[thread_id].value.store(result.nodes, Ordering::Relaxed);
                }
            }).expect("failed to spawn search thread");
            handles.push(handle);
        }

        // Main thread (thread 0) runs with info callback and wrapped node
        // reporting that includes all threads' nodes.
        let counters_for_main = Arc::clone(&counters);
        let total_threads = active_threads;

        let wrapped_cb: Option<InfoCallback> = info_callback.map(|cb| {
            let cb_counters = Arc::clone(&counters_for_main);
            let wrapped: InfoCallback = Box::new(move |info: &SearchInfo| {
                // Sum up all thread node counts for the info line
                let mut total_nodes = info.nodes;
                for i in 1..total_threads {
                    total_nodes += cb_counters[i].value.load(Ordering::Relaxed);
                }
                let total_nps_info = SearchInfo {
                    depth: info.depth,
                    seldepth: info.seldepth,
                    score: info.score,
                    nodes: total_nodes,
                    time_ms: info.time_ms,
                    pv: info.pv.clone(),
                    hashfull: info.hashfull,
                    multipv_line: info.multipv_line,
                };
                cb(&total_nps_info);
            });
            wrapped
        });

        // Wrap the main thread search in catch_unwind so that a panic
        // (e.g. from a TT torn-read producing a corrupt move) does not
        // kill the engine process.  On panic, fall back to the first
        // legal root move.
        let fallback_move = root_moves.iter().next().copied().unwrap_or(Move::NULL);

        let search_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            search::iterative_deepening(board, params, stop, tt, wrapped_cb, 0, net, Some(&counters[0].value), syzygy_tb, root_tb_solution, book)
        }));

        let result = match search_result {
            Ok(r) => r,
            Err(e) => {
                log::error!("Main search thread panicked: {:?}", e);
                SearchResult {
                    best_move: fallback_move,
                    score: Score(0),
                    depth: 0,
                    nodes: 0,
                    pv: if fallback_move.is_null() { vec![] } else { vec![fallback_move] },
                }
            }
        };
        counters[0].value.store(result.nodes, Ordering::Relaxed);

        // Stop all helper threads
        stop.store(true, Ordering::SeqCst);
        for handle in handles {
            let _ = handle.join();
        }

        // Total nodes from all threads
        let total_nodes: u64 = counters.iter().map(|c| c.value.load(Ordering::Relaxed)).sum();

        SearchResult {
            best_move: result.best_move,
            score: result.score,
            depth: result.depth,
            nodes: total_nodes,
            pv: result.pv,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::active_thread_count;

    #[test]
    fn single_move_forces_single_thread() {
        assert_eq!(active_thread_count(1, 1), 1);
        assert_eq!(active_thread_count(8, 0), 1);
        assert_eq!(active_thread_count(8, 1), 1);
    }

    #[test]
    fn tiny_root_caps_helper_swarm() {
        assert_eq!(active_thread_count(8, 2), 3);
        assert_eq!(active_thread_count(8, 3), 4);
        assert_eq!(active_thread_count(8, 4), 5);
    }

    #[test]
    fn wider_roots_keep_requested_threads() {
        assert_eq!(active_thread_count(8, 5), 8);
        assert_eq!(active_thread_count(4, 20), 4);
    }
}
