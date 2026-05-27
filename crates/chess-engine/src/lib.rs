pub mod eval;
pub mod polyglot;
pub mod search;
pub mod see;
pub mod syzygy;
pub mod threads;
pub mod tt;

use chess_common::{Board, Move, Score};
use chess_nnue::NnueNetwork;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use polyglot::OpeningBook;
use syzygy::SyzygyTB;
use threads::ThreadPool;
use tt::SharedTT;

/// Search result returned by the engine.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub best_move: Move,
    pub score: Score,
    pub depth: u8,
    pub nodes: u64,
    pub pv: Vec<Move>,
}

/// Search parameters.
#[derive(Debug, Clone)]
pub struct SearchParams {
    pub max_depth: u8,
    pub max_nodes: Option<u64>,
    pub move_time_ms: Option<u64>,
    pub white_time_ms: Option<u64>,
    pub black_time_ms: Option<u64>,
    pub white_inc_ms: Option<u64>,
    pub black_inc_ms: Option<u64>,
    pub moves_to_go: Option<u32>,
    pub infinite: bool,
    /// Whether to use NNUE evaluation instead of HCE.
    pub use_nnue: bool,
    /// Safety margin (ms) subtracted from allocated time to avoid flagging.
    pub move_overhead_ms: u64,
    /// Time management scaling factor (percentage). 100 = normal.
    pub slow_mover: u64,
    /// Contempt factor in centipawns. A positive value makes the engine
    /// slightly prefer avoiding draws (scored as -contempt from the current
    /// side-to-move's perspective). 0 = neutral.
    pub contempt: i32,
    /// Singular extension mode.
    /// 0 = off, 1 = conservative, 2 = aggressive.
    pub singular_ext_mode: u8,
    /// Number of best lines to search and report (MultiPV). 1 = normal PV.
    pub multi_pv: usize,
    /// Tunable search parameters (SPSA targets).
    pub tune: TuneParams,
}

/// Search parameters that can be tuned via SPSA.
/// Exposed as UCI spin options so external tuners (weather-factory etc.)
/// can drive them via `setoption`.
#[derive(Clone, Debug)]
pub struct TuneParams {
    /// LMR base ×100 (default 50 → 0.50)
    pub lmr_base: i32,
    /// LMR divisor ×100 (default 200 → 2.00)
    pub lmr_div: i32,
    /// Divisor for history score's influence on LMR reduction (default 4000)
    pub hist_lmr_div: i32,
    /// Reverse futility pruning margin per depth when improving (default 65)
    pub rfp_margin_imp: i32,
    /// Reverse futility pruning margin per depth when not improving (default 85)
    pub rfp_margin_noimp: i32,
    /// Futility pruning margin per depth when improving (default 95)
    pub fut_margin_imp: i32,
    /// Futility pruning margin per depth when not improving (default 65)
    pub fut_margin_noimp: i32,
    /// SEE quiet-move pruning margin per depth (default 50)
    pub see_quiet_margin: i32,
    /// Pawn-corrhist correction multiplier ×100 (default 100 → 1.00; 0 disables)
    pub corrhist_mult: i32,
}

impl Default for TuneParams {
    fn default() -> Self {
        Self {
            lmr_base: 24,
            lmr_div: 163,
            hist_lmr_div: 979,
            rfp_margin_imp: 41,
            rfp_margin_noimp: 84,
            fut_margin_imp: 41,
            fut_margin_noimp: 57,
            see_quiet_margin: 14,
            corrhist_mult: 100,
        }
    }
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            max_depth: 64,
            max_nodes: None,
            move_time_ms: None,
            white_time_ms: None,
            black_time_ms: None,
            white_inc_ms: None,
            black_inc_ms: None,
            moves_to_go: None,
            infinite: false,
            use_nnue: true,
            move_overhead_ms: 20,
            slow_mover: 100,
            contempt: 20,
            singular_ext_mode: 1,
            multi_pv: 1,
            tune: TuneParams::default(),
        }
    }
}

/// Callback for reporting search info (UCI info lines).
pub type InfoCallback = Box<dyn Fn(&SearchInfo) + Send>;

/// Search info reported during search.
#[derive(Debug, Clone)]
pub struct SearchInfo {
    pub depth: u8,
    pub seldepth: u8,
    pub score: Score,
    pub nodes: u64,
    pub time_ms: u64,
    pub pv: Vec<Move>,
    pub hashfull: u16,
    /// Which MultiPV line this info belongs to (1-based). Always 1 in single-PV mode.
    pub multipv_line: usize,
}

/// The main engine that performs search.
pub struct Engine {
    stop: Arc<AtomicBool>,
    tt: Arc<SharedTT>,
    thread_pool: ThreadPool,
    hash_mb: usize,
    num_threads: usize,
    /// Safety margin (ms) subtracted from allocated time to avoid flagging
    /// due to communication/USB lag. Tournament default: 20ms.
    move_overhead_ms: u64,
    /// Time management scaling factor (percentage). 100 = normal,
    /// >100 = think longer, <100 = think faster. Range: 10..300.
    slow_mover: u64,
    /// Contempt factor in centipawns (draw score = -contempt from side-to-move).
    contempt: i32,
    /// Singular extension mode: 0 = off, 1 = conservative, 2 = aggressive.
    singular_ext_mode: u8,
    /// Whether to use NNUE evaluation.
    use_nnue: bool,
    /// The active NNUE network (embedded or loaded from file).
    nnue_net: Arc<NnueNetwork>,
    /// Syzygy tablebase handle. `None` when no path has been set.
    syzygy_tb: Option<SyzygyTB>,
    /// Opening book loaded via `BookFile`. `None` means no book.
    book: Option<Arc<OpeningBook>>,
    /// Tunable search parameters.
    tune: TuneParams,
}

impl Engine {
    pub fn new() -> Self {
        // Start with a minimal TT; the UCI handler will resize to the
        // configured (or default) size at `isready` time, after all
        // `setoption` commands have been processed.  This avoids
        // allocating 4 GB × concurrency at startup before fastchess
        // has a chance to send `setoption name Hash value N`.
        Self::with_hash(1)
    }

    /// Create an engine with a specific hash size (MB).
    /// Prefer this over `new()` + `set_hash_mb()` when the desired hash size
    /// is known up front — avoids allocating and immediately discarding the
    /// default 4 GB transposition table.
    pub fn with_hash(hash_mb: usize) -> Self {
        let hash_mb = hash_mb.clamp(1, 131072);
        // Default to min(available_parallelism, 16).
        // Can be overridden via UCI "setoption name Threads value N".
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get().min(16))
            .unwrap_or(1);
        Self {
            stop: Arc::new(AtomicBool::new(false)),
            tt: Arc::new(SharedTT::new(hash_mb)),
            thread_pool: ThreadPool::new(num_threads),
            hash_mb,
            num_threads,
            move_overhead_ms: 20,
            slow_mover: 100,
            contempt: 20,
            singular_ext_mode: 1,
            nnue_net: NnueNetwork::embedded(),
            use_nnue: NnueNetwork::embedded().is_trained(),
            syzygy_tb: None,
            book: None,
            tune: TuneParams::default(),
        }
    }

    /// Get a handle to stop the search.
    pub fn stop_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stop)
    }

    /// Set the transposition table size in MB.
    pub fn set_hash_mb(&mut self, mb: usize) {
        let mb = mb.clamp(1, 131072);
        if mb != self.hash_mb {
            self.hash_mb = mb;
            self.tt = Arc::new(SharedTT::new(mb));
        }
    }

    /// Get the current hash size in MB.
    pub fn hash_mb(&self) -> usize {
        self.hash_mb
    }

    /// Set the number of search threads.
    pub fn set_threads(&mut self, threads: usize) {
        self.num_threads = threads.clamp(1, 256);
        self.thread_pool = ThreadPool::new(self.num_threads);
    }

    /// Get the current number of threads.
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Set the move overhead in milliseconds.
    pub fn set_move_overhead(&mut self, ms: u64) {
        self.move_overhead_ms = ms.clamp(0, 5000);
    }

    /// Get the current move overhead in ms.
    pub fn move_overhead_ms(&self) -> u64 {
        self.move_overhead_ms
    }

    /// Set the slow mover percentage (time management scaling).
    pub fn set_slow_mover(&mut self, pct: u64) {
        self.slow_mover = pct.clamp(10, 300);
    }

    /// Get the current slow mover percentage.
    pub fn slow_mover(&self) -> u64 {
        self.slow_mover
    }

    /// Set the contempt factor in centipawns. Draws are scored as -contempt
    /// from the current side-to-move's perspective. Range: 0..=100.
    pub fn set_contempt(&mut self, cp: i32) {
        self.contempt = cp.clamp(0, 100);
    }

    /// Get the current contempt factor.
    pub fn contempt(&self) -> i32 {
        self.contempt
    }

    /// Set singular extension mode: 0 = off, 1 = conservative, 2 = aggressive.
    pub fn set_singular_ext_mode(&mut self, mode: u8) {
        self.singular_ext_mode = mode.clamp(0, 2);
    }

    /// Get singular extension mode.
    pub fn singular_ext_mode(&self) -> u8 {
        self.singular_ext_mode
    }

    /// Load an opening book from `path` (Polyglot .bin, EPD, or PGN).
    /// An empty path clears any loaded book.
    pub fn set_book_file(&mut self, path: &str) -> Result<(), String> {
        let path = path.trim();
        if path.is_empty() {
            self.book = None;
            log::info!("BookFile cleared");
            return Ok(());
        }
        match OpeningBook::open(std::path::Path::new(path)) {
            Some(b) => {
                self.book = Some(Arc::new(b));
                log::info!("BookFile loaded from '{path}'");
                Ok(())
            }
            None => Err(format!("Failed to load opening book from '{path}'")),
        }
    }

    /// Return the loaded book handle, if any.
    pub fn book(&self) -> Option<Arc<OpeningBook>> {
        self.book.clone()
    }

    /// Set whether NNUE evaluation is used.
    pub fn set_use_nnue(&mut self, enabled: bool) {
        self.use_nnue = enabled;
    }

    /// Check if NNUE evaluation is enabled.
    pub fn use_nnue(&self) -> bool {
        self.use_nnue
    }

    /// Load a NNUE network from a file path. Pass an empty string to revert to the embedded net.
    pub fn set_nnue_file(&mut self, path: &str) -> Result<(), String> {
        let net = NnueNetwork::from_path(path)?;
        self.use_nnue = net.is_trained();
        self.nnue_net = net;
        Ok(())
    }

    /// Return the active NNUE network.
    pub fn nnue_net(&self) -> &Arc<NnueNetwork> {
        &self.nnue_net
    }

    /// Set the Syzygy tablebase handle directly from a pre-created instance.
    ///
    /// Use this when the tablebase was already initialized elsewhere (e.g. a
    /// shared handle cloned across threads) to avoid concurrent `tb_init`/
    /// `tb_free` races that can occur when each thread calls `set_syzygy_path`.
    pub fn set_syzygy_tb(&mut self, tb: Option<SyzygyTB>) {
        self.syzygy_tb = tb;
    }

    /// Temporarily take ownership of the loaded Syzygy handle.
    pub fn take_syzygy_tb(&mut self) -> Option<SyzygyTB> {
        self.syzygy_tb.take()
    }

    /// Load Syzygy tablebases from the given directory path.
    ///
    /// An empty path clears any previously loaded tablebases.
    /// Multiple directories can be separated by colons (`:`) on Unix or
    /// semicolons (`;`) on Windows.
    pub fn set_syzygy_path(&mut self, path: &str) -> Result<(), String> {
        // Drop existing handle first so the global singleton is freed before
        // re-initializing with the new path.
        self.syzygy_tb = None;

        let path = path.trim();
        if path.is_empty() {
            log::info!("Syzygy tablebases cleared");
            return Ok(());
        }

        match SyzygyTB::new(path) {
            Ok(tb) => {
                let max = tb.max_pieces();
                log::info!("Syzygy tablebases loaded from '{}' (max {} pieces)", path, max);
                self.syzygy_tb = Some(tb);
                Ok(())
            }
            Err(e) => Err(format!("Failed to load Syzygy tablebases from '{path}': {e:?}")),
        }
    }

    /// Return the loaded tablebase handle, if any.
    pub fn syzygy_tb(&self) -> Option<&SyzygyTB> {
        self.syzygy_tb.as_ref()
    }

    /// Get the current tune params.
    pub fn tune(&self) -> &TuneParams {
        &self.tune
    }

    /// Set a single tune parameter by name. Returns false if the name is unknown.
    pub fn set_tune_param(&mut self, name: &str, value: i32) -> bool {
        match name {
            "lmr_base"          => { self.tune.lmr_base          = value; true }
            "lmr_div"           => { self.tune.lmr_div            = value; true }
            "hist_lmr_div"      => { self.tune.hist_lmr_div       = value; true }
            "rfp_margin_imp"    => { self.tune.rfp_margin_imp     = value; true }
            "rfp_margin_noimp"  => { self.tune.rfp_margin_noimp   = value; true }
            "fut_margin_imp"    => { self.tune.fut_margin_imp     = value; true }
            "fut_margin_noimp"  => { self.tune.fut_margin_noimp   = value; true }
            "see_quiet_margin"  => { self.tune.see_quiet_margin   = value; true }
            "corrhist_mult"     => { self.tune.corrhist_mult      = value; true }
            _ => false,
        }
    }

    /// Run a search on the given position.
    pub fn search(
        &self,
        board: &Board,
        params: &SearchParams,
        info_callback: Option<InfoCallback>,
    ) -> SearchResult {
        // Inject current tune params into search params so threads get them.
        let mut p = params.clone();
        p.tune = self.tune.clone();
        self.stop.store(false, Ordering::SeqCst);
        self.thread_pool.search(board, &p, &self.stop, &self.tt, info_callback, &self.nnue_net, self.syzygy_tb.clone(), None, self.book.clone(), None)
    }

    /// Signal the engine to stop searching.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    /// Clear the transposition table (e.g., on ucinewgame).
    pub fn clear_tt(&self) {
        self.tt.clear();
    }

    /// Bump the TT generation without clearing entries (cheap alternative to
    /// `clear_tt` between datagen games — old entries remain valid since the
    /// XOR key check already guarantees correctness).
    pub fn new_search_tt(&self) {
        self.tt.new_search();
    }

    /// Get a shared reference to the transposition table.
    pub fn shared_tt(&self) -> Arc<SharedTT> {
        Arc::clone(&self.tt)
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}
