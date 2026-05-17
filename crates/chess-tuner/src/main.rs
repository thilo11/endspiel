// ===========================================================================
// Chess Texel Tuner
//
// Adam optimizer that tunes evaluation parameters by minimizing the mean
// squared error between the engine's static eval (mapped to a win probability
// via logistic function) and Stockfish's evaluation of the same positions.
//
// Key design decisions:
//   - Material values are FROZEN by default.  They define the centipawn scale
//     that all search pruning margins (futility, razoring, probcut, SEE) are
//     calibrated for.  Tuning them rescales everything and breaks search.
//     Use --tune-material to opt in (with tight bounds).
//   - Every parameter has a min/max constraint.  Known-sign terms (mobility,
//     bishop pair, threats, etc.) are clamped appropriately.
//   - Uses Adam optimizer with finite-difference gradients for fast convergence.
//   - Non-quiet positions (in check, tactical best move) are filtered by default.
//   - Extreme positions (|cp| > 1000) are filtered by default since they
//     contribute little to differentiating parameters.
// ===========================================================================

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use chess_common::{Board, Square};
use chess_core::is_in_check;
use chess_engine::eval::{evaluate_with_params, EvalParams};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Data structures for parsing the Lichess eval JSONL
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct EvalRecord {
    fen: String,
    evals: Vec<EvalEntry>,
}

#[derive(Deserialize)]
struct EvalEntry {
    depth: u32,
    pvs: Vec<PvEntry>,
}

#[derive(Deserialize)]
struct PvEntry {
    #[serde(default)]
    cp: Option<i64>,
    #[serde(default)]
    mate: Option<i64>,
    #[serde(default)]
    line: Option<String>,
}

// ---------------------------------------------------------------------------
// Tuning position: a board + target win probability
// ---------------------------------------------------------------------------

struct TuningPosition {
    board: Board,
    target: f64,
}

// ---------------------------------------------------------------------------
// Per-parameter constraint and step tracking
// ---------------------------------------------------------------------------

struct ParamConstraint {
    /// Minimum allowed value (inclusive).
    min: i32,
    /// Maximum allowed value (inclusive).
    max: i32,
    /// Step size (unused by Adam, retained for potential coordinate descent).
    #[allow(dead_code)]
    step: i32,
    /// Initial step size (for reporting).
    #[allow(dead_code)]
    initial_step: i32,
    /// Whether this parameter is frozen (not tuned).
    frozen: bool,
}

impl ParamConstraint {
    fn new(min: i32, max: i32, step: i32) -> Self {
        Self { min, max, step, initial_step: step, frozen: false }
    }

    fn frozen() -> Self {
        Self { min: i32::MIN, max: i32::MAX, step: 0, initial_step: 0, frozen: true }
    }

    fn clamp(&self, val: i32) -> i32 {
        val.clamp(self.min, self.max)
    }
}

/// Build constraints for all tunable parameters.
///
/// Parameter layout (from EvalParams):
///   0..5   = material_mg [P, N, B, R, Q]
///   5..10  = material_eg [P, N, B, R, Q]
///   10..394  = pst_mg (6 pieces × 64 squares)
///   394..778 = pst_eg (6 pieces × 64 squares)
///   778..    = scalar params (61 total)
fn build_constraints(tune_material: bool, freeze_pst: bool, freeze_mobility: bool) -> Vec<ParamConstraint> {
    let total = EvalParams::param_count();
    let mut constraints = Vec::with_capacity(total);

    // === Material (indices 0..10) ===
    // Standard PeSTO values: MG=[82,337,365,477,1025], EG=[94,281,297,512,936]
    // Even when tuning, clamp tightly to preserve search scale.
    for i in 0..10 {
        if !tune_material {
            constraints.push(ParamConstraint::frozen());
        } else {
            // Allow ±15% deviation from default, step size 5
            let defaults = EvalParams::default();
            let base = defaults.get_param(i);
            let margin = (base.abs() as f64 * 0.15).max(10.0) as i32;
            constraints.push(ParamConstraint::new(base - margin, base + margin, 5));
        }
    }

    // === PST values (indices 10..778) ===
    // PSTs are relative adjustments; clamp to ±300 to prevent absurd values.
    // Use step size 3.  When --freeze-pst is set, freeze all PST values at
    // their defaults so only scalar params are tuned.
    for _ in 10..778 {
        if freeze_pst {
            constraints.push(ParamConstraint::frozen());
        } else {
            constraints.push(ParamConstraint::new(-300, 300, 3));
        }
    }

    // === Scalar params (indices 778..) ===
    // Each scalar has semantics that dictate its sign and range.
    // We use the v0.2.2 hand-tuned defaults as anchors — the bounds are
    // centered on those values with enough room to tune, but NOT so wide
    // that the optimizer can zero out entire feature categories.
    // Scalar index = overall index - 778
    let num_scalars = total - 778; // NUM_SCALAR_PARAMS (currently 79)
    for scalar_idx in 0..num_scalars {
        let c = match scalar_idx {
            // Pawn structure penalties (should be ≤ 0)
            // doubled_pawn_mg/eg, isolated_pawn_mg/eg,
            // doubled_isolated_mg/eg, backward_pawn_mg/eg
            //   v0.2.2: -10/-20, -15/-25, -15/-20, -15/-20
            0..=7 => ParamConstraint::new(-50, -2, 2),

            // Passed pawn bonuses (should be > 0)
            // passed_pawn_base_mg: v0.2.2 = 10
            8 => ParamConstraint::new(3, 40, 2),
            // passed_pawn_base_eg: widened range
            9 => ParamConstraint::new(3, 80, 2),
            // passed_pawn_adv_mg: v0.2.2 = 3 (per advancement²)
            10 => ParamConstraint::new(1, 15, 1),
            // passed_pawn_adv_eg: widened range
            11 => ParamConstraint::new(1, 30, 1),
            // connected_passer_base_mg: v0.2.2 = 8
            12 => ParamConstraint::new(2, 30, 2),
            // connected_passer_base_eg: widened range
            13 => ParamConstraint::new(2, 50, 2),
            // connected_passer_adv_mg: v0.2.2 = 2
            14 => ParamConstraint::new(1, 12, 1),
            // connected_passer_adv_eg: widened range
            15 => ParamConstraint::new(1, 25, 1),
            // rook_behind_passer_mg: v0.2.2 = 10
            16 => ParamConstraint::new(3, 50, 2),
            // rook_behind_passer_eg: widened range
            17 => ParamConstraint::new(3, 100, 2),

            // Blocked passer (penalty, ≤ 0)
            // v0.2.2: -5/-15
            18..=19 => ParamConstraint::new(-40, -2, 2),

            // Bishop pair (bonus, > 0)
            // v0.2.2: 30/35
            20..=21 => ParamConstraint::new(10, 60, 3),

            // Rook open file (bonus, > 0)
            // v0.2.2: 25/15
            22..=23 => ParamConstraint::new(5, 50, 2),
            // Rook semi-open file: v0.2.2 = 12/8
            24..=25 => ParamConstraint::new(3, 30, 2),

            // Rook on 7th rank (bonus, > 0)
            // v0.2.2: 20/45
            26..=27 => ParamConstraint::new(5, 80, 3),

            // King safety: shield bonuses (> 0)
            // v0.2.2: 12/6
            28..=29 => ParamConstraint::new(2, 30, 2),

            // King safety: open/semi-open/center penalties (< 0)
            // v0.2.2: -25/-15/-30
            30..=32 => ParamConstraint::new(-60, -5, 3),

            // King safety: attacker weights (> 0)
            // v0.2.2: knight=30, bishop=30, rook=50, queen=100
            33..=36 => ParamConstraint::new(10, 200, 5),

            // Mobility per-square (> 0)
            // v0.2.2: N=4/4, B=5/5, R=2/4, Q=2/3
            // MG floors are higher to prevent the tuner from collapsing
            // mobility into PSTs (a known Texel pathology).
            37..=44 if freeze_mobility => ParamConstraint::frozen(),
            37 => ParamConstraint::new(3, 12, 1), // mobility_knight_mg (min 3)
            38 => ParamConstraint::new(1, 12, 1), // mobility_knight_eg
            39 => ParamConstraint::new(4, 12, 1), // mobility_bishop_mg (min 4)
            40 => ParamConstraint::new(1, 12, 1), // mobility_bishop_eg
            41 => ParamConstraint::new(2, 12, 1), // mobility_rook_mg (min 2)
            42 => ParamConstraint::new(1, 12, 1), // mobility_rook_eg
            43 => ParamConstraint::new(2, 12, 1), // mobility_queen_mg (min 2)
            44 => ParamConstraint::new(1, 12, 1), // mobility_queen_eg

            // Center control bonuses (> 0)
            // v0.2.2: pawn=20, knight=10, bishop=6
            45 => ParamConstraint::new(5, 40, 2),
            46 => ParamConstraint::new(3, 25, 2),
            47 => ParamConstraint::new(2, 18, 1),

            // Connectivity bonuses (> 0)
            // pawn_protected_knight=15, knight_outpost=15
            48..=49 => ParamConstraint::new(5, 35, 2),

            // Threat bonuses (> 0)
            // v0.2.2: pawn_minor=40/50, pawn_rook=60/60,
            //   minor_rook=30/35, piece_queen=50/50, hanging=20/25
            50..=59 => ParamConstraint::new(5, 120, 3),

            // Tempo (> 0, small)
            // v0.2.2: 8
            60 => ParamConstraint::new(3, 20, 1),

            // Bad bishop (negative penalty per own pawn on bishop's color)
            // v0.2.2: -3/-5
            61 => ParamConstraint::new(-15, -1, 1),
            62 => ParamConstraint::new(-20, -1, 1),

            // Knight closed bonus / bishop open bonus (per pawn, positive)
            // v0.2.2: 5/5
            63..=64 => ParamConstraint::new(2, 15, 1),

            // Pawn islands (negative penalty per extra island)
            // v0.2.2: -8/-12
            65 => ParamConstraint::new(-20, -3, 1),
            66 => ParamConstraint::new(-25, -3, 1),

            // OCB scale factor (0-128, default 60)
            67 => ParamConstraint::new(20, 100, 5),

            // King-passer distance (EG only, positive)
            // king_passer_own_eg: bonus per (7 - own_king_dist) * (1 + adv) / 4
            68 => ParamConstraint::new(1, 15, 1),
            // king_passer_enemy_eg: bonus per enemy_king_dist * (1 + adv) / 4, widened
            69 => ParamConstraint::new(1, 40, 1),

            // Connected passer quadratic term (per advancement²)
            // connected_passer_sq_mg
            70 => ParamConstraint::new(0, 8, 1),
            // connected_passer_sq_eg: widened range
            71 => ParamConstraint::new(1, 30, 1),

            // Material imbalance params
            // imbalance_exchange_mg: rook vs minor, MG
            72 => ParamConstraint::new(20, 100, 5),
            // imbalance_exchange_eg: rook vs minor, EG
            73 => ParamConstraint::new(40, 150, 5),
            // imbalance_rook_pair_mg: redundant rook pair penalty (negative)
            74 => ParamConstraint::new(-40, -2, 2),
            // imbalance_rook_pair_eg
            75 => ParamConstraint::new(-40, -2, 2),
            // imbalance_knight_pair_eg: two-knight penalty (negative)
            76 => ParamConstraint::new(-25, -1, 1),
            // imbalance_queen_vs_minors_mg
            77 => ParamConstraint::new(10, 60, 3),
            // imbalance_queen_vs_minors_eg
            78 => ParamConstraint::new(10, 60, 3),

            _ => ParamConstraint::new(-200, 200, 3),
        };
        constraints.push(c);
    }

    assert_eq!(constraints.len(), total);
    constraints
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct Config {
    data_path: PathBuf,
    max_positions: usize,
    max_epochs: usize,
    min_depth: u32,
    output_path: Option<PathBuf>,
    /// If set, apply params from this JSON file to eval.rs and exit.
    apply_path: Option<PathBuf>,
    /// Path to eval.rs (for --apply).
    eval_rs_path: PathBuf,
    /// Allow tuning material values (frozen by default).
    tune_material: bool,
    /// Freeze PST values at defaults so only scalar params are tuned.
    freeze_pst: bool,
    /// Freeze mobility params at their defaults so the tuner can't collapse them.
    freeze_mobility: bool,
    /// Filter positions with |cp| > this threshold.
    max_cp: i64,
    /// L2 regularization strength.  Adds lambda * sum((param - default)^2) / N
    /// to the MSE, penalizing drift from the starting values.  This prevents
    /// the optimizer from zeroing out features that the PSTs can partially
    /// absorb.  Default: 1e-7 (gentle regularization; increase to 1e-6 for
    /// stronger anchoring).
    l2_lambda: f64,
    /// Filter non-quiet positions (in check or best move is capture).
    quiet_only: bool,
    /// Adam optimizer learning rate.
    learning_rate: f64,
    /// Filter positions where |engine eval| exceeds this threshold.
    /// Removes positions where our eval wildly disagrees with Stockfish.
    max_eval: i32,
    /// If set, run validation on a held-out set instead of tuning.
    validate_path: Option<PathBuf>,
    /// Path to stockfish binary for cross-check (default: "stockfish").
    sf_path: String,
    /// Milliseconds per position for Stockfish cross-check (default: 100).
    /// Set to 0 to use depth-based search instead (sf_depth).
    sf_movetime: u64,
    /// Depth for deterministic Stockfish cross-check (default: 20).
    /// Used when sf_movetime is 0.
    sf_depth: u32,
    /// Enable iterative tune→validate→repeat convergence loop.
    converge: bool,
    /// Max rounds for convergence loop (default: 10).
    max_rounds: usize,
    /// Target: SF MAE must be ≤ this (cp) to pass (default: 120).
    target_mae: f64,
    /// Target: SF correlation must be ≥ this to pass (default: 0.60).
    target_correlation: f64,
    /// Target: SF outlier % must be ≤ this to pass (default: 25.0).
    target_outlier_pct: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data_path: PathBuf::from("games/lichess_db_eval.jsonl.zst"),
            max_positions: 2_000_000,
            max_epochs: 200,
            min_depth: 30,
            output_path: None,
            apply_path: None,
            eval_rs_path: PathBuf::from("crates/chess-engine/src/eval.rs"),
            tune_material: false,
            freeze_pst: false,
            freeze_mobility: false,
            max_cp: 1000,
            l2_lambda: 1e-7,
            quiet_only: true,
            learning_rate: 2.0,
            max_eval: 2000,
            validate_path: None,
            sf_path: String::from("stockfish"),
            sf_movetime: 0,
            sf_depth: 20,
            converge: false,
            max_rounds: 10,
            target_mae: 155.0,
            target_correlation: 0.38,
            target_outlier_pct: 35.0,
        }
    }
}

fn parse_args() -> Config {
    let mut config = Config::default();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => {
                i += 1;
                config.data_path = PathBuf::from(&args[i]);
            }
            "--positions" => {
                i += 1;
                config.max_positions = args[i].parse().expect("invalid --positions");
            }
            "--epochs" => {
                i += 1;
                config.max_epochs = args[i].parse().expect("invalid --epochs");
            }
            "--min-depth" => {
                i += 1;
                config.min_depth = args[i].parse().expect("invalid --min-depth");
            }
            "--output" => {
                i += 1;
                config.output_path = Some(PathBuf::from(&args[i]));
            }
            "--apply" => {
                i += 1;
                config.apply_path = Some(PathBuf::from(&args[i]));
            }
            "--eval-path" => {
                i += 1;
                config.eval_rs_path = PathBuf::from(&args[i]);
            }
            "--tune-material" => {
                config.tune_material = true;
            }
            "--freeze-pst" => {
                config.freeze_pst = true;
            }
            "--freeze-mobility" => {
                config.freeze_mobility = true;
            }
            "--max-cp" => {
                i += 1;
                config.max_cp = args[i].parse().expect("invalid --max-cp");
            }
            "--l2-lambda" => {
                i += 1;
                config.l2_lambda = args[i].parse().expect("invalid --l2-lambda");
            }
            "--no-quiet-filter" => {
                config.quiet_only = false;
            }
            "--learning-rate" => {
                i += 1;
                config.learning_rate = args[i].parse().expect("invalid --learning-rate");
            }
            "--max-eval" => {
                i += 1;
                config.max_eval = args[i].parse().expect("invalid --max-eval");
            }
            "--validate" => {
                i += 1;
                config.validate_path = Some(PathBuf::from(&args[i]));
            }
            "--sf-path" => {
                i += 1;
                config.sf_path = args[i].clone();
            }
            "--sf-movetime" => {
                i += 1;
                config.sf_movetime = args[i].parse().expect("invalid --sf-movetime");
            }
            "--sf-depth" => {
                i += 1;
                config.sf_depth = args[i].parse().expect("invalid --sf-depth");
                // Using depth mode — disable movetime
                config.sf_movetime = 0;
            }
            "--converge" => {
                config.converge = true;
            }
            "--max-rounds" => {
                i += 1;
                config.max_rounds = args[i].parse().expect("invalid --max-rounds");
            }
            "--target-mae" => {
                i += 1;
                config.target_mae = args[i].parse().expect("invalid --target-mae");
            }
            "--target-correlation" => {
                i += 1;
                config.target_correlation = args[i].parse().expect("invalid --target-correlation");
            }
            "--target-outlier-pct" => {
                i += 1;
                config.target_outlier_pct = args[i].parse().expect("invalid --target-outlier-pct");
            }
            "--help" | "-h" => {
                eprintln!("Usage: chess-tuner [OPTIONS]");
                eprintln!("  --data PATH         Path to lichess_db_eval.jsonl.zst");
                eprintln!("  --positions N        Max positions to load (default: 2000000)");
                eprintln!("  --epochs N           Max tuning epochs (default: 200)");
                eprintln!("  --min-depth N        Min Stockfish depth (default: 30)");
                eprintln!("  --output PATH        Save params JSON to file");
                eprintln!("  --apply PATH         Apply params from JSON to eval.rs and exit");
                eprintln!("  --eval-path PATH     Path to eval.rs (default: crates/chess-engine/src/eval.rs)");
                eprintln!("  --tune-material      Also tune material values (frozen by default)");
                eprintln!("  --freeze-pst         Freeze PST values, tune only scalar params");
                eprintln!("  --freeze-mobility    Freeze mobility params at defaults");
                eprintln!("  --max-cp N           Filter positions with |cp| > N (default: 1000)");
                eprintln!("  --max-eval N         Filter positions with |engine eval| > N (default: 2000)");
                eprintln!("  --l2-lambda F        L2 regularization strength (default: 1e-7, 0 = off)");
                eprintln!("  --no-quiet-filter    Disable filtering of non-quiet positions");
                eprintln!("  --learning-rate F    Adam optimizer learning rate (default: 2.0)");
                eprintln!("  --validate PATH      Validate params from JSON (loss + SF cross-check)");
                eprintln!("  --sf-path PATH       Path to stockfish binary (default: stockfish)");
                eprintln!("  --sf-movetime MS     Stockfish time per position in ms (0=use depth)");
                eprintln!("  --sf-depth D         Stockfish depth (deterministic, default: 20)");
                eprintln!("  --converge           Iterative tune+validate loop until thresholds met");
                eprintln!("  --max-rounds N       Max convergence rounds (default: 10)");
                eprintln!("  --target-mae CP      Target SF MAE in cp (default: 155)");
                eprintln!("  --target-correlation F  Target SF correlation (default: 0.38)");
                eprintln!("  --target-outlier-pct F  Target SF outlier %% (default: 35.0)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }
    config
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

/// Convert centipawn score to win probability using the logistic function.
/// P(win) = 1 / (1 + 10^(-cp/400))
fn cp_to_win_prob(cp: f64) -> f64 {
    1.0 / (1.0 + 10f64.powf(-cp / 400.0))
}

fn load_positions(config: &Config) -> Vec<TuningPosition> {
    let start = Instant::now();
    eprintln!(
        "Loading positions from {} (max {}, min depth {}, max |cp| {})...",
        config.data_path.display(),
        config.max_positions,
        config.min_depth,
        config.max_cp,
    );

    let file = File::open(&config.data_path).expect("Failed to open data file");
    let decoder = zstd::Decoder::new(file).expect("Failed to create zstd decoder");
    let reader = BufReader::with_capacity(1 << 20, decoder);

    let mut positions = Vec::new();
    let mut lines_read = 0u64;
    let mut skipped_mate = 0u64;
    let mut skipped_depth = 0u64;
    let mut skipped_parse = 0u64;
    let mut skipped_extreme = 0u64;
    let mut skipped_in_check = 0u64;
    let mut skipped_tactical = 0u64;

    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(_) => continue,
        };
        lines_read += 1;

        if lines_read.is_multiple_of(1_000_000) {
            eprintln!(
                "  Read {}M lines, {} positions collected...",
                lines_read / 1_000_000,
                positions.len()
            );
        }

        // Early exit if we have enough
        if positions.len() >= config.max_positions {
            break;
        }

        let record: EvalRecord = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(_) => {
                skipped_parse += 1;
                continue;
            }
        };

        // Find the highest-depth eval
        let best_eval = record
            .evals
            .iter()
            .filter(|e| e.depth >= config.min_depth)
            .max_by_key(|e| e.depth);

        let eval_entry = match best_eval {
            Some(e) => e,
            None => {
                skipped_depth += 1;
                continue;
            }
        };

        // Get the first PV's score (best move eval)
        let pv = match eval_entry.pvs.first() {
            Some(p) => p,
            None => continue,
        };

        // Skip mate scores
        if pv.mate.is_some() {
            skipped_mate += 1;
            continue;
        }

        let cp = match pv.cp {
            Some(c) => c,
            None => continue,
        };

        // Filter extreme evaluations — already-decided positions add noise
        if cp.abs() > config.max_cp {
            skipped_extreme += 1;
            continue;
        }

        // Parse the FEN
        let board = match Board::from_fen(&record.fen) {
            Ok(b) => b,
            Err(_) => {
                skipped_parse += 1;
                continue;
            }
        };

        // Filter non-quiet positions
        if config.quiet_only {
            // Skip positions where side to move is in check
            if is_in_check(&board) {
                skipped_in_check += 1;
                continue;
            }

            // Skip positions where the best move is a capture
            if let Some(ref line_str) = pv.line {
                let best_move = line_str.split_whitespace().next().unwrap_or("");
                if best_move.len() >= 4 {
                    let to_str = &best_move[2..4];
                    if let Some(to_sq) = Square::from_algebraic(to_str) {
                        let is_capture = board.piece_at(to_sq).is_some()
                            || board.en_passant == Some(to_sq);
                        if is_capture {
                            skipped_tactical += 1;
                            continue;
                        }
                    }
                }
            }
        }

        // Convert cp to win probability
        // Both our eval and Lichess cp are from White's perspective — no flip needed
        let target = cp_to_win_prob(cp as f64);

        positions.push(TuningPosition { board, target });
    }

    let elapsed = start.elapsed();
    let total_skipped = skipped_mate + skipped_depth + skipped_parse + skipped_extreme
        + skipped_in_check + skipped_tactical;
    eprintln!(
        "Loaded {} positions in {:.1}s ({} lines read, {} skipped: {} mate, {} depth<{}, {} |cp|>{}, {} in-check, {} tactical, {} parse errors)",
        positions.len(),
        elapsed.as_secs_f64(),
        lines_read,
        total_skipped,
        skipped_mate,
        skipped_depth,
        config.min_depth,
        skipped_extreme,
        config.max_cp,
        skipped_in_check,
        skipped_tactical,
        skipped_parse,
    );

    positions
}

// ---------------------------------------------------------------------------
// Error function
// ---------------------------------------------------------------------------

/// Compute mean squared error between predicted and target win probabilities.
/// `k` is the scaling factor for the logistic function.
/// Optionally adds L2 regularization: lambda * sum((param_i - default_i)^2) / N_params
/// to penalize deviation from starting values.
fn mean_squared_error(
    positions: &[TuningPosition],
    params: &EvalParams,
    k: f64,
    l2_lambda: f64,
    defaults: &EvalParams,
    constraints: &[ParamConstraint],
) -> f64 {
    let sum: f64 = positions
        .par_iter()
        .map(|pos| {
            let eval = evaluate_with_params(&pos.board, params).0 as f64;
            let predicted = 1.0 / (1.0 + 10f64.powf(-eval / k));
            let diff = predicted - pos.target;
            diff * diff
        })
        .sum();

    let mut mse = sum / positions.len() as f64;

    // L2 regularization: penalize drift from defaults on non-frozen params
    if l2_lambda > 0.0 {
        let mut l2_sum = 0.0;
        let mut l2_count = 0;
        for (i, c) in constraints.iter().enumerate() {
            if c.frozen {
                continue;
            }
            let diff = (params.get_param(i) - defaults.get_param(i)) as f64;
            l2_sum += diff * diff;
            l2_count += 1;
        }
        if l2_count > 0 {
            mse += l2_lambda * l2_sum / l2_count as f64;
        }
    }

    mse
}

/// Find the optimal scaling factor K by ternary search.
fn find_optimal_k(
    positions: &[TuningPosition],
    params: &EvalParams,
    l2_lambda: f64,
    defaults: &EvalParams,
    constraints: &[ParamConstraint],
) -> f64 {
    eprintln!("Finding optimal K...");
    let mut lo = 100.0;
    let mut hi = 8000.0;

    for _ in 0..50 {
        let mid1 = lo + (hi - lo) / 3.0;
        let mid2 = hi - (hi - lo) / 3.0;
        let e1 = mean_squared_error(positions, params, mid1, l2_lambda, defaults, constraints);
        let e2 = mean_squared_error(positions, params, mid2, l2_lambda, defaults, constraints);
        if e1 < e2 {
            hi = mid2;
        } else {
            lo = mid1;
        }
    }

    let k = (lo + hi) / 2.0;
    let mse = mean_squared_error(positions, params, k, l2_lambda, defaults, constraints);
    eprintln!("Optimal K = {k:.2}, MSE = {mse:.8}");
    k
}

// ---------------------------------------------------------------------------
// Adam optimizer
// ---------------------------------------------------------------------------

fn adam_optimize(
    positions: &[TuningPosition],
    params: &mut EvalParams,
    mut k: f64,
    config: &Config,
    constraints: &[ParamConstraint],
) {
    let total_params = EvalParams::param_count();
    let tunable_indices: Vec<usize> = (0..total_params)
        .filter(|&i| !constraints[i].frozen)
        .collect();
    let tunable_count = tunable_indices.len();
    let defaults = EvalParams::default();
    let l2 = config.l2_lambda;
    let lr = config.learning_rate;

    // Adam hyperparameters
    let beta1 = 0.9;
    let beta2 = 0.999;
    let eps = 1e-8;

    eprintln!(
        "Starting Adam optimizer: {} total params, {} tunable, {} frozen, {} epochs max, lr={}, L2 lambda={:.1e}",
        total_params,
        tunable_count,
        total_params - tunable_count,
        config.max_epochs,
        lr,
        l2,
    );

    // Shadow parameters in floating-point for sub-integer momentum tracking
    let mut shadow: Vec<f64> = (0..total_params)
        .map(|i| params.get_param(i) as f64)
        .collect();

    // Adam moment vectors (only for tunable params, indexed by position in tunable_indices)
    let mut m = vec![0.0f64; tunable_count];
    let mut v = vec![0.0f64; tunable_count];

    let mut best_error = mean_squared_error(positions, params, k, l2, &defaults, constraints);
    eprintln!("Initial MSE: {best_error:.8}");

    let mut prev_error = best_error;
    let mut stagnant_epochs = 0u32;

    for epoch in 0..config.max_epochs {
        let epoch_start = Instant::now();
        let t = (epoch + 1) as f64; // 1-indexed for bias correction

        // Re-optimize K every 10 epochs
        if epoch > 0 && epoch % 10 == 0 {
            k = find_optimal_k(positions, params, l2, &defaults, constraints);
        }

        // Compute gradients via central finite differences (ε=1)
        let mut grad = vec![0.0f64; tunable_count];
        let base_error = mean_squared_error(positions, params, k, l2, &defaults, constraints);

        for (j, &i) in tunable_indices.iter().enumerate() {
            let original = params.get_param(i);

            let plus = constraints[i].clamp(original + 1);
            let minus = constraints[i].clamp(original - 1);

            if plus == minus {
                // At a boundary where we can't perturb in either direction
                grad[j] = 0.0;
                continue;
            }

            if plus != original && minus != original {
                // Central difference
                params.set_param(i, plus);
                let err_plus = mean_squared_error(positions, params, k, l2, &defaults, constraints);
                params.set_param(i, minus);
                let err_minus = mean_squared_error(positions, params, k, l2, &defaults, constraints);
                params.set_param(i, original);
                grad[j] = (err_plus - err_minus) / 2.0;
            } else if plus != original {
                // Forward difference (at lower bound)
                params.set_param(i, plus);
                let err_plus = mean_squared_error(positions, params, k, l2, &defaults, constraints);
                params.set_param(i, original);
                grad[j] = err_plus - base_error;
            } else {
                // Backward difference (at upper bound)
                params.set_param(i, minus);
                let err_minus = mean_squared_error(positions, params, k, l2, &defaults, constraints);
                params.set_param(i, original);
                grad[j] = base_error - err_minus;
            }
        }

        // Adam update
        for (j, &i) in tunable_indices.iter().enumerate() {
            m[j] = beta1 * m[j] + (1.0 - beta1) * grad[j];
            v[j] = beta2 * v[j] + (1.0 - beta2) * grad[j] * grad[j];

            let m_hat = m[j] / (1.0 - beta1.powf(t));
            let v_hat = v[j] / (1.0 - beta2.powf(t));

            shadow[i] -= lr * m_hat / (v_hat.sqrt() + eps);

            // Clamp to constraint bounds
            shadow[i] = shadow[i].clamp(constraints[i].min as f64, constraints[i].max as f64);

            // Apply rounded value to params
            params.set_param(i, shadow[i].round() as i32);
        }

        let current_error = mean_squared_error(positions, params, k, l2, &defaults, constraints);
        let delta = prev_error - current_error;
        let elapsed = epoch_start.elapsed();

        // Count how many params changed from defaults
        let changed = tunable_indices
            .iter()
            .filter(|&&i| params.get_param(i) != defaults.get_param(i))
            .count();

        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        eprintln!(
            "Epoch {:3}: MSE={current_error:.8} (Δ={delta:+.8}), changed={changed}/{tunable_count}, |grad|={grad_norm:.2e}, time={:.1}s",
            epoch + 1,
            elapsed.as_secs_f64()
        );

        // Save intermediate results
        if let Some(ref path) = config.output_path {
            save_params(params, path);
        }

        // Convergence check: stop when relative improvement is negligible.
        // A threshold of 1e-5 (0.001% of current error) for 5 consecutive
        // epochs typically catches the point of diminishing returns around
        // epoch 40-80 without leaving significant gains on the table.
        let relative_improvement = if current_error > 0.0 {
            delta / current_error
        } else {
            0.0
        };
        if relative_improvement < 1e-5 {
            stagnant_epochs += 1;
        } else {
            stagnant_epochs = 0;
        }

        if stagnant_epochs >= 5 {
            eprintln!(
                "Converged: <0.001% relative improvement for 5 consecutive epochs after {} epochs.",
                epoch + 1
            );
            break;
        }

        best_error = current_error;
        prev_error = current_error;
    }

    // Print convergence summary
    let changed = tunable_indices
        .iter()
        .filter(|&&i| params.get_param(i) != defaults.get_param(i))
        .count();
    eprintln!(
        "\nConverged: {changed}/{tunable_count} params changed from defaults, final MSE={best_error:.8}"
    );
}

// ---------------------------------------------------------------------------
// Serialization for resumability
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct SavedParams {
    values: Vec<i32>,
    names: Vec<String>,
}

fn save_params(params: &EvalParams, path: &PathBuf) {
    let total = EvalParams::param_count();
    let mut values = Vec::with_capacity(total);
    let mut names = Vec::with_capacity(total);
    for i in 0..total {
        values.push(params.get_param(i));
        names.push(params.param_name(i));
    }

    let saved = SavedParams { values, names };
    let file = File::create(path).expect("Failed to create output file");
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &saved).expect("Failed to write params");
}

fn load_params(path: &PathBuf) -> Option<EvalParams> {
    let file = File::open(path).ok()?;
    let reader = BufReader::new(file);
    let saved: SavedParams = serde_json::from_reader(reader).ok()?;

    let mut params = EvalParams::default();
    for (i, val) in saved.values.iter().enumerate() {
        if i < EvalParams::param_count() {
            params.set_param(i, *val);
        }
    }
    Some(params)
}

// ---------------------------------------------------------------------------
// Apply tuned params to eval.rs source code
// ---------------------------------------------------------------------------

const PST_CONST_NAMES: [&str; 12] = [
    "PST_PAWN_MG", "PST_KNIGHT_MG", "PST_BISHOP_MG",
    "PST_ROOK_MG", "PST_QUEEN_MG", "PST_KING_MG",
    "PST_PAWN_EG", "PST_KNIGHT_EG", "PST_BISHOP_EG",
    "PST_ROOK_EG", "PST_QUEEN_EG", "PST_KING_EG",
];

fn format_pst_array(name: &str, values: &[i32; 64]) -> String {
    let mut s = String::new();
    s.push_str("#[rustfmt::skip]\n");
    s.push_str(&format!("const {name}: [i32; 64] = [\n"));
    for rank in 0..8 {
        s.push_str("    ");
        for file in 0..8 {
            let val = values[rank * 8 + file];
            s.push_str(&format!("{val:4},"));
            if file < 7 {
                s.push(' ');
            }
        }
        s.push('\n');
    }
    s.push_str("];\n");
    s
}

fn generate_pst_section(params: &EvalParams) -> String {
    let mut s = String::new();
    s.push_str("// @tuner:pst_start\n");
    // MG tables
    for (i, name) in PST_CONST_NAMES[..6].iter().enumerate() {
        s.push_str(&format_pst_array(name, &params.pst_mg[i]));
        s.push('\n');
    }
    // EG tables
    for (i, name) in PST_CONST_NAMES[6..].iter().enumerate() {
        s.push_str(&format_pst_array(name, &params.pst_eg[i]));
        s.push('\n');
    }
    s.push_str("const PST_MG: [[i32; 64]; 6] = [\n");
    s.push_str("    PST_PAWN_MG, PST_KNIGHT_MG, PST_BISHOP_MG, PST_ROOK_MG, PST_QUEEN_MG, PST_KING_MG,\n");
    s.push_str("];\n\n");
    s.push_str("const PST_EG: [[i32; 64]; 6] = [\n");
    s.push_str("    PST_PAWN_EG, PST_KNIGHT_EG, PST_BISHOP_EG, PST_ROOK_EG, PST_QUEEN_EG, PST_KING_EG,\n");
    s.push_str("];\n");
    s.push_str("// @tuner:pst_end");
    s
}

fn generate_material_section(params: &EvalParams) -> String {
    let mut s = String::new();
    s.push_str("// @tuner:material_start\n");
    s.push_str(&format!(
        "const _MATERIAL_MG: [i32; 6] = [{}, {}, {}, {}, {}, 0]; // P N B R Q K\n",
        params.material_mg[0], params.material_mg[1], params.material_mg[2],
        params.material_mg[3], params.material_mg[4],
    ));
    s.push_str(&format!(
        "const MATERIAL_EG: [i32; 6] = [{}, {}, {}, {}, {}, 0];\n",
        params.material_eg[0], params.material_eg[1], params.material_eg[2],
        params.material_eg[3], params.material_eg[4],
    ));
    s.push_str("// @tuner:material_end");
    s
}

fn generate_defaults_section(params: &EvalParams) -> String {
    let mut s = String::new();
    s.push_str("// @tuner:defaults_start\n");
    s.push_str("impl Default for EvalParams {\n");
    s.push_str("    fn default() -> Self {\n");
    s.push_str("        Self {\n");
    s.push_str(&format!("            material_mg: {:?},\n", params.material_mg));
    s.push_str(&format!("            material_eg: {:?},\n", params.material_eg));
    s.push_str("            pst_mg: PST_MG,\n");
    s.push_str("            pst_eg: PST_EG,\n");
    s.push_str(&format!("            doubled_pawn_mg: {},\n", params.doubled_pawn_mg));
    s.push_str(&format!("            doubled_pawn_eg: {},\n", params.doubled_pawn_eg));
    s.push_str(&format!("            isolated_pawn_mg: {},\n", params.isolated_pawn_mg));
    s.push_str(&format!("            isolated_pawn_eg: {},\n", params.isolated_pawn_eg));
    s.push_str(&format!("            doubled_isolated_mg: {},\n", params.doubled_isolated_mg));
    s.push_str(&format!("            doubled_isolated_eg: {},\n", params.doubled_isolated_eg));
    s.push_str(&format!("            backward_pawn_mg: {},\n", params.backward_pawn_mg));
    s.push_str(&format!("            backward_pawn_eg: {},\n", params.backward_pawn_eg));
    s.push_str(&format!("            passed_pawn_base_mg: {},\n", params.passed_pawn_base_mg));
    s.push_str(&format!("            passed_pawn_base_eg: {},\n", params.passed_pawn_base_eg));
    s.push_str(&format!("            passed_pawn_adv_mg: {},\n", params.passed_pawn_adv_mg));
    s.push_str(&format!("            passed_pawn_adv_eg: {},\n", params.passed_pawn_adv_eg));
    s.push_str(&format!("            connected_passer_base_mg: {},\n", params.connected_passer_base_mg));
    s.push_str(&format!("            connected_passer_base_eg: {},\n", params.connected_passer_base_eg));
    s.push_str(&format!("            connected_passer_adv_mg: {},\n", params.connected_passer_adv_mg));
    s.push_str(&format!("            connected_passer_adv_eg: {},\n", params.connected_passer_adv_eg));
    s.push_str(&format!("            rook_behind_passer_mg: {},\n", params.rook_behind_passer_mg));
    s.push_str(&format!("            rook_behind_passer_eg: {},\n", params.rook_behind_passer_eg));
    s.push_str(&format!("            blocked_passer_mg: {},\n", params.blocked_passer_mg));
    s.push_str(&format!("            blocked_passer_eg: {},\n", params.blocked_passer_eg));
    s.push_str(&format!("            king_passer_own_eg: {},\n", params.king_passer_own_eg));
    s.push_str(&format!("            king_passer_enemy_eg: {},\n", params.king_passer_enemy_eg));
    s.push_str(&format!("            connected_passer_sq_mg: {},\n", params.connected_passer_sq_mg));
    s.push_str(&format!("            connected_passer_sq_eg: {},\n", params.connected_passer_sq_eg));
    s.push_str(&format!("            bishop_pair_base_mg: {},\n", params.bishop_pair_base_mg));
    s.push_str(&format!("            bishop_pair_base_eg: {},\n", params.bishop_pair_base_eg));
    s.push_str(&format!("            rook_open_file_mg: {},\n", params.rook_open_file_mg));
    s.push_str(&format!("            rook_open_file_eg: {},\n", params.rook_open_file_eg));
    s.push_str(&format!("            rook_semi_open_mg: {},\n", params.rook_semi_open_mg));
    s.push_str(&format!("            rook_semi_open_eg: {},\n", params.rook_semi_open_eg));
    s.push_str(&format!("            rook_seventh_mg: {},\n", params.rook_seventh_mg));
    s.push_str(&format!("            rook_seventh_eg: {},\n", params.rook_seventh_eg));
    s.push_str(&format!("            doubled_rook_file_mg: {},\n", params.doubled_rook_file_mg));
    s.push_str(&format!("            doubled_rook_file_eg: {},\n", params.doubled_rook_file_eg));
    s.push_str(&format!("            doubled_rook_7th_mg: {},\n", params.doubled_rook_7th_mg));
    s.push_str(&format!("            doubled_rook_7th_eg: {},\n", params.doubled_rook_7th_eg));
    s.push_str(&format!("            trapped_bishop_mg: {},\n", params.trapped_bishop_mg));
    s.push_str(&format!("            trapped_bishop_eg: {},\n", params.trapped_bishop_eg));
    s.push_str(&format!("            ks_shield_1: {},\n", params.ks_shield_1));
    s.push_str(&format!("            ks_shield_2: {},\n", params.ks_shield_2));
    s.push_str(&format!("            ks_open_file: {},\n", params.ks_open_file));
    s.push_str(&format!("            ks_semi_open: {},\n", params.ks_semi_open));
    s.push_str(&format!("            ks_center_king: {},\n", params.ks_center_king));
    s.push_str(&format!("            ks_knight_weight: {},\n", params.ks_knight_weight));
    s.push_str(&format!("            ks_bishop_weight: {},\n", params.ks_bishop_weight));
    s.push_str(&format!("            ks_rook_weight: {},\n", params.ks_rook_weight));
    s.push_str(&format!("            ks_queen_weight: {},\n", params.ks_queen_weight));
    s.push_str(&format!("            mobility_knight_mg: {},\n", params.mobility_knight_mg));
    s.push_str(&format!("            mobility_knight_eg: {},\n", params.mobility_knight_eg));
    s.push_str(&format!("            mobility_bishop_mg: {},\n", params.mobility_bishop_mg));
    s.push_str(&format!("            mobility_bishop_eg: {},\n", params.mobility_bishop_eg));
    s.push_str(&format!("            mobility_rook_mg: {},\n", params.mobility_rook_mg));
    s.push_str(&format!("            mobility_rook_eg: {},\n", params.mobility_rook_eg));
    s.push_str(&format!("            mobility_queen_mg: {},\n", params.mobility_queen_mg));
    s.push_str(&format!("            mobility_queen_eg: {},\n", params.mobility_queen_eg));
    s.push_str(&format!("            center_pawn_bonus: {},\n", params.center_pawn_bonus));
    s.push_str(&format!("            center_knight_bonus: {},\n", params.center_knight_bonus));
    s.push_str(&format!("            center_bishop_bonus: {},\n", params.center_bishop_bonus));
    s.push_str(&format!("            pawn_protected_knight: {},\n", params.pawn_protected_knight));
    s.push_str(&format!("            knight_outpost: {},\n", params.knight_outpost));
    s.push_str(&format!("            threat_pawn_minor_mg: {},\n", params.threat_pawn_minor_mg));
    s.push_str(&format!("            threat_pawn_minor_eg: {},\n", params.threat_pawn_minor_eg));
    s.push_str(&format!("            threat_pawn_rook_mg: {},\n", params.threat_pawn_rook_mg));
    s.push_str(&format!("            threat_pawn_rook_eg: {},\n", params.threat_pawn_rook_eg));
    s.push_str(&format!("            threat_minor_rook_mg: {},\n", params.threat_minor_rook_mg));
    s.push_str(&format!("            threat_minor_rook_eg: {},\n", params.threat_minor_rook_eg));
    s.push_str(&format!("            threat_piece_queen_mg: {},\n", params.threat_piece_queen_mg));
    s.push_str(&format!("            threat_piece_queen_eg: {},\n", params.threat_piece_queen_eg));
    s.push_str(&format!("            threat_hanging_mg: {},\n", params.threat_hanging_mg));
    s.push_str(&format!("            threat_hanging_eg: {},\n", params.threat_hanging_eg));
    s.push_str(&format!("            tempo: {},\n", params.tempo));
    s.push_str(&format!("            bad_bishop_mg: {},\n", params.bad_bishop_mg));
    s.push_str(&format!("            bad_bishop_eg: {},\n", params.bad_bishop_eg));
    s.push_str(&format!("            knight_closed_bonus: {},\n", params.knight_closed_bonus));
    s.push_str(&format!("            bishop_open_bonus: {},\n", params.bishop_open_bonus));
    s.push_str(&format!("            pawn_islands_mg: {},\n", params.pawn_islands_mg));
    s.push_str(&format!("            pawn_islands_eg: {},\n", params.pawn_islands_eg));
    s.push_str(&format!("            ocb_scale_factor: {},\n", params.ocb_scale_factor));
    s.push_str(&format!("            imbalance_exchange_mg: {},\n", params.imbalance_exchange_mg));
    s.push_str(&format!("            imbalance_exchange_eg: {},\n", params.imbalance_exchange_eg));
    s.push_str(&format!("            imbalance_rook_pair_mg: {},\n", params.imbalance_rook_pair_mg));
    s.push_str(&format!("            imbalance_rook_pair_eg: {},\n", params.imbalance_rook_pair_eg));
    s.push_str(&format!("            imbalance_knight_pair_eg: {},\n", params.imbalance_knight_pair_eg));
    s.push_str(&format!("            imbalance_queen_vs_minors_mg: {},\n", params.imbalance_queen_vs_minors_mg));
    s.push_str(&format!("            imbalance_queen_vs_minors_eg: {},\n", params.imbalance_queen_vs_minors_eg));
    s.push_str("        }\n");
    s.push_str("    }\n");
    s.push_str("}\n");
    s.push_str("// @tuner:defaults_end");
    s
}

/// Replace content between `start_marker` and `end_marker` (inclusive) with `replacement`.
/// Verify that `generate_defaults_section` emits every field in `EvalParams`.
///
/// We generate the defaults, then check that every scalar field name from
/// the struct appears in the output.  This catches forgotten fields after
/// adding new eval params.
#[test]
fn test_generate_defaults_covers_all_fields() {
    let params = EvalParams::default();
    let section = generate_defaults_section(&params);

    // Collect every field name that appears in the param_name() list.
    // Skip PST entries (they use the PST_MG / PST_EG constants) and
    // material arrays (they use the inline array syntax).
    let total = EvalParams::param_count();
    let mut missing = Vec::new();
    for i in 0..total {
        let name = params.param_name(i);
        // PST and material params are emitted differently
        if name.starts_with("pst_") || name.starts_with("material_") {
            continue;
        }
        if !section.contains(&format!("{name}:")) {
            missing.push(name);
        }
    }
    assert!(
        missing.is_empty(),
        "generate_defaults_section is missing fields: {missing:?}\n\
         Add them to the function in main.rs"
    );
}

fn replace_between_markers(source: &str, start_marker: &str, end_marker: &str, replacement: &str) -> Result<String, String> {
    let start = source.find(start_marker)
        .ok_or_else(|| format!("Marker not found: {start_marker}"))?;
    let end = source.find(end_marker)
        .ok_or_else(|| format!("Marker not found: {end_marker}"))?;
    let end = end + end_marker.len();

    let mut result = String::with_capacity(source.len());
    result.push_str(&source[..start]);
    result.push_str(replacement);
    result.push_str(&source[end..]);
    Ok(result)
}

fn apply_params_to_eval_rs(params: &EvalParams, eval_rs_path: &PathBuf) {
    eprintln!("Applying tuned params to {}", eval_rs_path.display());

    let source = std::fs::read_to_string(eval_rs_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", eval_rs_path.display()));

    let source = replace_between_markers(
        &source, "// @tuner:material_start", "// @tuner:material_end",
        &generate_material_section(params),
    ).expect("Failed to replace material section");

    let source = replace_between_markers(
        &source, "// @tuner:pst_start", "// @tuner:pst_end",
        &generate_pst_section(params),
    ).expect("Failed to replace PST section");

    let source = replace_between_markers(
        &source, "// @tuner:defaults_start", "// @tuner:defaults_end",
        &generate_defaults_section(params),
    ).expect("Failed to replace defaults section");

    std::fs::write(eval_rs_path, source)
        .unwrap_or_else(|e| panic!("Failed to write {}: {e}", eval_rs_path.display()));

    // Count changes
    let defaults = EvalParams::default();
    let total = EvalParams::param_count();
    let changed = (0..total).filter(|&i| defaults.get_param(i) != params.get_param(i)).count();
    eprintln!("Applied {changed}/{total} changed parameters to {}", eval_rs_path.display());
}

// ---------------------------------------------------------------------------
// Validation: clamp loaded params to constraints
// ---------------------------------------------------------------------------

/// Clamp all parameters to their constraint bounds.  Returns the number of
/// values that were clamped.
fn clamp_params_to_constraints(params: &mut EvalParams, constraints: &[ParamConstraint]) -> usize {
    let mut clamped = 0;
    for (i, c) in constraints.iter().enumerate() {
        if c.frozen {
            continue;
        }
        let val = params.get_param(i);
        let clamped_val = c.clamp(val);
        if clamped_val != val {
            params.set_param(i, clamped_val);
            clamped += 1;
        }
    }
    clamped
}

// ---------------------------------------------------------------------------
// Validation: post-tuning checks
// ---------------------------------------------------------------------------

/// Run Stockfish on a single FEN and return its eval in centipawns (from side-to-move).
/// Returns None if Stockfish can't be run or returns a mate score.
/// A persistent Stockfish process that can evaluate many positions without
/// restarting.  Communication is done line-by-line so we never close stdin
/// prematurely or race against search output.
struct StockfishProcess {
    child: std::process::Child,
    reader: BufReader<std::process::ChildStdout>,
}

impl StockfishProcess {
    fn start(sf_path: &str) -> Option<Self> {
        let mut child = Command::new(sf_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;

        let stdout = child.stdout.take()?;
        let mut sf = Self {
            child,
            reader: BufReader::new(stdout),
        };

        // Initialize UCI and wait for uciok
        sf.send("uci")?;
        sf.read_until_starts_with("uciok")?;
        sf.send("setoption name Skill Level value 20")?;
        sf.send("isready")?;
        sf.read_until_starts_with("readyok")?;
        Some(sf)
    }

    fn send(&mut self, cmd: &str) -> Option<()> {
        let stdin = self.child.stdin.as_mut()?;
        writeln!(stdin, "{cmd}").ok()?;
        stdin.flush().ok()
    }

    fn read_until_starts_with(&mut self, prefix: &str) -> Option<String> {
        let mut line = String::new();
        loop {
            line.clear();
            if self.reader.read_line(&mut line).ok()? == 0 {
                return None; // EOF
            }
            if line.trim_end().starts_with(prefix) {
                return Some(line);
            }
        }
    }

    /// Evaluate a single position, returning the score in centipawns from the
    /// side to move.  Returns `None` for mate scores or on communication error.
    fn eval(&mut self, fen: &str, movetime: u64) -> Option<i32> {
        self.send(&format!("position fen {fen}"))?;
        self.send(&format!("go movetime {movetime}"))?;
        self.read_search_result()
    }

    /// Evaluate using a fixed depth (deterministic).
    fn eval_depth(&mut self, fen: &str, depth: u32) -> Option<i32> {
        self.send(&format!("position fen {fen}"))?;
        self.send(&format!("go depth {depth}"))?;
        self.read_search_result()
    }

    /// Read SF output until "bestmove", returning the last score seen.
    fn read_search_result(&mut self) -> Option<i32> {
        // We prefer "score cp"; if the final line uses "score mate", convert to
        // a large centipawn value so the position isn't silently dropped.
        let mut last_cp = None;
        let mut line = String::new();
        loop {
            line.clear();
            if self.reader.read_line(&mut line).ok()? == 0 {
                break; // EOF
            }
            let trimmed = line.trim_end();
            if trimmed.starts_with("info") && trimmed.contains(" score ") {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                for (i, &part) in parts.iter().enumerate() {
                    if i > 0 && parts[i - 1] == "score"
                        && let Some(&val_str) = parts.get(i + 1)
                    {
                        if part == "cp" {
                            if let Ok(val) = val_str.parse::<i32>() {
                                last_cp = Some(val);
                            }
                        } else if part == "mate"
                            && let Ok(moves) = val_str.parse::<i32>()
                        {
                            // Convert mate-in-N to a large cp value
                            // (positive = winning, negative = losing)
                            last_cp = Some(if moves > 0 { 10_000 } else { -10_000 });
                        }
                    }
                }
            }
            if trimmed.starts_with("bestmove") {
                break;
            }
        }

        // Sync: make sure SF is ready for the next position
        let _ = self.send("isready");
        self.read_until_starts_with("readyok");

        last_cp
    }

    fn quit(mut self) {
        let _ = self.send("quit");
        let _ = self.child.wait();
    }
}

/// Compute Pearson correlation coefficient between two slices.
fn pearson_correlation(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        cov / denom
    }
}

/// Results from the validation pipeline (for programmatic convergence checks).
#[allow(dead_code)]
struct ValidationResult {
    train_mse: f64,
    val_mse: f64,
    ratio: f64,
    /// Stockfish cross-check: mean absolute error in centipawns.
    sf_mae: f64,
    /// Pearson correlation with Stockfish evals.
    sf_correlation: f64,
    /// Fraction of positions with |Δ| > 200 cp.
    sf_outlier_pct: f64,
    /// Number of positions evaluated by Stockfish.
    sf_evaluated: usize,
}

/// Run the full validation pipeline: loss check + Stockfish cross-check.
fn run_validation(config: &Config, params: &EvalParams) -> ValidationResult {
    let constraints = build_constraints(config.tune_material, config.freeze_pst, config.freeze_mobility);
    let defaults = EvalParams::default();

    // Load all positions
    let positions = load_positions(config);
    if positions.len() < 1000 {
        eprintln!("ERROR: Need at least 1000 positions for validation, got {}", positions.len());
        std::process::exit(1);
    }

    // Split: last 20k (or 10% if fewer) for validation, rest for "training" loss
    let val_count = 20_000.min(positions.len() / 10).max(500);
    let train_count = positions.len() - val_count;
    let train_positions = &positions[..train_count];
    let val_positions = &positions[train_count..];

    // Find optimal K using all positions
    let k = find_optimal_k(&positions, params, 0.0, &defaults, &constraints);

    // === Validation Loss ===
    let train_mse = mean_squared_error(train_positions, params, k, 0.0, &defaults, &constraints);
    let val_mse = mean_squared_error(val_positions, params, k, 0.0, &defaults, &constraints);
    let ratio = if train_mse > 0.0 { val_mse / train_mse } else { 1.0 };

    println!("\n=== Validation Report ===");
    println!("Training MSE:    {train_mse:.8} ({train_count} positions)");
    println!("Validation MSE:  {val_mse:.8} ({val_count} positions)");
    let status = if ratio > 1.05 { "WARNING: > 1.05, potential overfitting" } else { "OK: < 1.05" };
    println!("Ratio:           {ratio:.3} ({status})");

    // === Stockfish Cross-Check ===
    // Start a persistent Stockfish process
    let sf_process = StockfishProcess::start(&config.sf_path);

    if sf_process.is_none() {
        println!("\n=== Stockfish Cross-Check ===");
        println!("SKIPPED: stockfish not found at '{}' (use --sf-path to specify)", config.sf_path);
        return ValidationResult {
            train_mse, val_mse, ratio,
            sf_mae: f64::NAN, sf_correlation: 0.0, sf_outlier_pct: 100.0, sf_evaluated: 0,
        };
    }

    let mut sf = sf_process.unwrap();

    // Sample ~500 positions evenly from the validation set
    let sample_count = 500.min(val_positions.len());
    let step = val_positions.len() / sample_count;
    let sample_indices: Vec<usize> = (0..sample_count).map(|i| i * step).collect();

    let sf_label = if config.sf_movetime > 0 {
        format!("{}ms/pos", config.sf_movetime)
    } else {
        format!("depth {}", config.sf_depth)
    };
    println!("\n=== Stockfish Cross-Check ({sf_label}) ===");
    eprint!("Evaluating {} positions with Stockfish...", sample_count);

    let mut engine_evals = Vec::new();
    let mut sf_evals = Vec::new();
    let mut abs_errors = Vec::new();
    struct Outlier {
        fen: String,
        engine_cp: i32,
        sf_cp: i32,
        delta: i32,
    }
    let mut outliers: Vec<Outlier> = Vec::new();
    let mut evaluated = 0usize;

    for &idx in &sample_indices {
        let pos = &val_positions[idx];
        let fen = pos.board.to_fen();
        let engine_cp = evaluate_with_params(&pos.board, params).0;

        let sf_result = if config.sf_movetime > 0 {
            sf.eval(&fen, config.sf_movetime)
        } else {
            sf.eval_depth(&fen, config.sf_depth)
        };
        if let Some(sf_cp) = sf_result {
            // SF returns score from side-to-move; our eval is from White's perspective.
            // Convert SF to White's perspective.
            let sf_cp_white = if pos.board.side_to_move == chess_common::Color::White { sf_cp } else { -sf_cp };
            let delta = engine_cp - sf_cp_white;
            let abs_err = delta.abs();

            engine_evals.push(engine_cp as f64);
            sf_evals.push(sf_cp_white as f64);
            abs_errors.push(abs_err as f64);

            if abs_err > 200 {
                outliers.push(Outlier {
                    fen: fen.clone(),
                    engine_cp,
                    sf_cp: sf_cp_white,
                    delta,
                });
            }

            evaluated += 1;
            if evaluated.is_multiple_of(50) {
                eprint!(" {evaluated}");
            }
        }
    }
    eprintln!(" done.");

    sf.quit();

    if evaluated == 0 {
        println!("No positions could be evaluated by Stockfish.");
        return ValidationResult {
            train_mse, val_mse, ratio,
            sf_mae: f64::NAN, sf_correlation: 0.0, sf_outlier_pct: 100.0, sf_evaluated: 0,
        };
    }

    let mae = abs_errors.iter().sum::<f64>() / evaluated as f64;
    let correlation = pearson_correlation(&engine_evals, &sf_evals);
    let outlier_pct = outliers.len() as f64 / evaluated as f64 * 100.0;

    println!("Positions evaluated: {evaluated}");
    println!("Mean Absolute Error: {mae:.0} cp");
    println!("Correlation:         {correlation:.2}");
    println!(
        "Outliers (|Δ|>200):  {}/{} ({:.1}%)",
        outliers.len(),
        evaluated,
        outlier_pct,
    );

    let outlier_pct = outliers.len() as f64 / evaluated as f64 * 100.0;

    // Print up to 10 worst outliers
    outliers.sort_by_key(|o| std::cmp::Reverse(o.delta.abs()));
    for o in outliers.iter().take(10) {
        let sign = if o.delta >= 0 { "+" } else { "" };
        println!(
            "  {} engine={:+} SF={:+} Δ={sign}{}",
            o.fen, o.engine_cp, o.sf_cp, o.delta,
        );
    }

    ValidationResult {
        train_mse,
        val_mse,
        ratio,
        sf_mae: mae,
        sf_correlation: correlation,
        sf_outlier_pct: outlier_pct,
        sf_evaluated: evaluated,
    }
}

// ---------------------------------------------------------------------------
// Convergence helpers
// ---------------------------------------------------------------------------

fn convergence_passed(config: &Config, result: &ValidationResult) -> bool {
    result.sf_evaluated > 0
        && result.sf_mae <= config.target_mae
        && result.sf_correlation >= config.target_correlation
        && result.sf_outlier_pct <= config.target_outlier_pct
}

fn print_convergence_status(config: &Config, result: &ValidationResult) {
    println!("\n=== Convergence Status ===");
    let mae_ok  = result.sf_mae <= config.target_mae;
    let corr_ok = result.sf_correlation >= config.target_correlation;
    let out_ok  = result.sf_outlier_pct <= config.target_outlier_pct;
    println!(
        "MAE:         {:.0} cp  (target: ≤{:.0})  {}",
        result.sf_mae, config.target_mae, if mae_ok { "✓" } else { "✗" }
    );
    println!(
        "Correlation: {:.2}    (target: ≥{:.2})  {}",
        result.sf_correlation, config.target_correlation, if corr_ok { "✓" } else { "✗" }
    );
    println!(
        "Outliers:    {:.1}%   (target: ≤{:.1}%)  {}",
        result.sf_outlier_pct, config.target_outlier_pct, if out_ok { "✓" } else { "✗" }
    );
    if mae_ok && corr_ok && out_ok {
        println!("PASSED: all thresholds met.");
    } else {
        println!("NOT YET: thresholds not met.");
    }
}

/// Iterative convergence loop: tune → validate → check → repeat.
fn run_convergence_loop(config: &Config) {
    let output_path = config.output_path.clone().unwrap_or_else(|| {
        PathBuf::from("tuned_params.json")
    });

    eprintln!("Chess Tuner — Convergence Mode");
    eprintln!("===============================");
    eprintln!("Max rounds:        {}", config.max_rounds);
    eprintln!("Target MAE:        ≤{:.0} cp", config.target_mae);
    eprintln!("Target correlation: ≥{:.2}", config.target_correlation);
    eprintln!("Target outliers:   ≤{:.1}%", config.target_outlier_pct);
    eprintln!("Output:            {}", output_path.display());
    eprintln!();

    // Build constraints once
    let constraints = build_constraints(config.tune_material, config.freeze_pst, config.freeze_mobility);

    // Try to resume from existing params
    let mut params = if output_path.exists() {
        eprintln!("Resuming from {}", output_path.display());
        let mut p = load_params(&output_path).unwrap_or_else(|| {
            eprintln!("Failed to load saved params, using defaults");
            EvalParams::default()
        });
        let num_clamped = clamp_params_to_constraints(&mut p, &constraints);
        if num_clamped > 0 {
            eprintln!("  Clamped {num_clamped} resumed params to constraint bounds");
        }
        p
    } else {
        EvalParams::default()
    };

    let loop_start = Instant::now();

    for round in 1..=config.max_rounds {
        let round_start = Instant::now();
        eprintln!("\n{}", "=".repeat(60));
        eprintln!("  ROUND {round}/{}", config.max_rounds);
        eprintln!("{}\n", "=".repeat(60));

        // === Tune ===
        eprintln!("--- Tuning ---");
        let mut positions = load_positions(config);
        if positions.is_empty() {
            eprintln!("No positions loaded! Check data path.");
            std::process::exit(1);
        }

        // Filter extreme engine evals
        if config.max_eval < i32::MAX {
            let before = positions.len();
            positions.retain(|pos| {
                let eval = evaluate_with_params(&pos.board, &params).0.abs();
                eval <= config.max_eval
            });
            let filtered = before - positions.len();
            if filtered > 0 {
                eprintln!(
                    "Filtered {filtered} positions with |engine eval| > {} ({} remaining)",
                    config.max_eval, positions.len()
                );
            }
        }

        let k = find_optimal_k(&positions, &params, config.l2_lambda, &EvalParams::default(), &constraints);
        adam_optimize(&positions, &mut params, k, config, &constraints);
        save_params(&params, &output_path);
        drop(positions); // free memory before validation

        // === Validate ===
        eprintln!("\n--- Validation ---");
        let result = run_validation(config, &params);
        print_convergence_status(config, &result);

        let round_elapsed = round_start.elapsed();
        eprintln!("Round {round} completed in {:.0}s", round_elapsed.as_secs_f64());

        if convergence_passed(config, &result) {
            let total_elapsed = loop_start.elapsed();
            eprintln!(
                "\nConverged after {round} round(s) in {:.0}s total.",
                total_elapsed.as_secs_f64()
            );
            eprintln!("Final params saved to {}", output_path.display());
            return;
        }
    }

    let total_elapsed = loop_start.elapsed();
    eprintln!(
        "\nDid not converge after {} rounds ({:.0}s total).",
        config.max_rounds,
        total_elapsed.as_secs_f64()
    );
    eprintln!("Best params saved to {}", output_path.display());
    eprintln!("Consider: more epochs (--epochs), more data (--positions), or relaxed targets.");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let config = parse_args();

    // Handle --apply: load params from JSON and patch eval.rs
    if let Some(ref apply_path) = config.apply_path {
        let params = load_params(apply_path)
            .unwrap_or_else(|| panic!("Failed to load params from {}", apply_path.display()));
        apply_params_to_eval_rs(&params, &config.eval_rs_path);
        return;
    }

    // Handle --validate: run validation checks and exit
    if let Some(ref validate_path) = config.validate_path {
        let params = load_params(validate_path)
            .unwrap_or_else(|| panic!("Failed to load params from {}", validate_path.display()));
        eprintln!("Chess Tuner — Validation Mode");
        eprintln!("==============================");
        eprintln!("Params:     {}", validate_path.display());
        eprintln!("Data:       {}", config.data_path.display());
        eprintln!("Positions:  {}", config.max_positions);
        eprintln!("SF binary:  {}", config.sf_path);
        if config.sf_movetime > 0 {
            eprintln!("SF search:  {}ms/pos", config.sf_movetime);
        } else {
            eprintln!("SF search:  depth {}", config.sf_depth);
        }
        let result = run_validation(&config, &params);
        if config.converge {
            print_convergence_status(&config, &result);
        }
        return;
    }

    // Handle --converge: iterative tune → validate → check thresholds loop
    if config.converge {
        run_convergence_loop(&config);
        return;
    }

    eprintln!("Chess Texel Tuner");
    eprintln!("=================");
    eprintln!("Parameters: {} total, material: {}, mobility: {}, L2 lambda: {:.1e}",
        EvalParams::param_count(),
        if config.tune_material { "TUNABLE (bounded)" } else { "FROZEN" },
        if config.freeze_mobility { "FROZEN" } else { "TUNABLE (bounded)" },
        config.l2_lambda,
    );

    // Build constraints
    let constraints = build_constraints(config.tune_material, config.freeze_pst, config.freeze_mobility);

    // Try to resume from saved params
    let mut params = if let Some(ref path) = config.output_path {
        if path.exists() {
            eprintln!("Resuming from {}", path.display());
            let mut p = load_params(path).unwrap_or_else(|| {
                eprintln!("Failed to load saved params, using defaults");
                EvalParams::default()
            });
            // Clamp resumed params to current constraints
            let num_clamped = clamp_params_to_constraints(&mut p, &constraints);
            if num_clamped > 0 {
                eprintln!("  Clamped {num_clamped} resumed params to constraint bounds");
            }
            p
        } else {
            EvalParams::default()
        }
    } else {
        EvalParams::default()
    };

    // Load training data
    let mut positions = load_positions(&config);
    if positions.is_empty() {
        eprintln!("No positions loaded! Check data path.");
        std::process::exit(1);
    }

    // Filter positions where our eval wildly disagrees with Stockfish
    if config.max_eval < i32::MAX {
        let before = positions.len();
        positions.retain(|pos| {
            let eval = evaluate_with_params(&pos.board, &params).0.abs();
            eval <= config.max_eval
        });
        let filtered = before - positions.len();
        eprintln!(
            "Filtered {filtered} positions with |engine eval| > {} ({} remaining)",
            config.max_eval,
            positions.len()
        );
    }

    // Find optimal K
    let k = find_optimal_k(&positions, &params, config.l2_lambda, &EvalParams::default(), &constraints);

    // Run Adam optimizer
    adam_optimize(&positions, &mut params, k, &config, &constraints);

    // Output results
    eprintln!("\n=== Tuned Parameters ===\n");
    params.print_rust_code();

    if let Some(ref path) = config.output_path {
        save_params(&params, path);
        eprintln!("\nSaved to {}", path.display());
    }

    // Print changed params summary
    let defaults = EvalParams::default();
    let total = EvalParams::param_count();
    let mut changed = 0;
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "\n// Changed parameters:").ok();
    for i in 0..total {
        let old = defaults.get_param(i);
        let new = params.get_param(i);
        if old != new {
            writeln!(stdout, "//   {}: {} -> {}", params.param_name(i), old, new).ok();
            changed += 1;
        }
    }
    writeln!(stdout, "// Total changed: {changed}/{total}").ok();
}
