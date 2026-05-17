use bulletformat::ChessBoard;
use chess_common::{Board, Color, Move, PieceKind};
use chess_core::generate_legal_moves;
use chess_engine::{Engine, SearchParams, polyglot::OpeningBook};
use rand::{Rng, RngExt};

const REPETITION_SCORE_WINDOW: usize = 4;
const REPETITION_DISCARD_CP: i32 = 250;

/// Draw adjudication: declare draw if |score| < this threshold for this many
/// consecutive plies. Terminates shuffling endgames early without affecting
/// positions that will actually be recorded (scores near zero are already
/// low-signal for training).
const DRAW_ADJ_CP: i32 = 15;
const DRAW_ADJ_PLIES: u32 = 8;

/// A recorded position with its search score and eventual game result.
struct DataEntry {
    /// Position encoded as white-relative bitboards [white, black, P, N, B, R, Q, K].
    bbs: [u64; 8],
    /// Side to move: 0 = white, 1 = black.
    stm: usize,
    /// Score in centipawns from white's perspective.
    score_cp: i16,
    /// Game result: 1.0 (white wins), 0.0 (black wins), 0.5 (draw).
    result: f32,
}

impl DataEntry {
    fn from_board(board: &Board, score_white: i32) -> Self {
        let bbs = [
            board.occupancy[Color::White.index()].0,
            board.occupancy[Color::Black.index()].0,
            board.pieces[Color::White.index()][PieceKind::Pawn.index()].0
                | board.pieces[Color::Black.index()][PieceKind::Pawn.index()].0,
            board.pieces[Color::White.index()][PieceKind::Knight.index()].0
                | board.pieces[Color::Black.index()][PieceKind::Knight.index()].0,
            board.pieces[Color::White.index()][PieceKind::Bishop.index()].0
                | board.pieces[Color::Black.index()][PieceKind::Bishop.index()].0,
            board.pieces[Color::White.index()][PieceKind::Rook.index()].0
                | board.pieces[Color::Black.index()][PieceKind::Rook.index()].0,
            board.pieces[Color::White.index()][PieceKind::Queen.index()].0
                | board.pieces[Color::Black.index()][PieceKind::Queen.index()].0,
            board.pieces[Color::White.index()][PieceKind::King.index()].0
                | board.pieces[Color::Black.index()][PieceKind::King.index()].0,
        ];
        let stm = if board.side_to_move == Color::White { 0 } else { 1 };
        // Clamp to i16; mate scores are already filtered out by the caller.
        let score_cp = score_white.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        Self { bbs, stm, score_cp, result: 0.0 }
    }

    fn to_chess_board(&self) -> ChessBoard {
        ChessBoard::from_raw(self.bbs, self.stm, self.score_cp, self.result)
            .expect("valid chess position")
    }
}

/// Pick a book move using ln-softened weighted random selection driven by `rng`.
///
/// ln(w+1) strongly compresses skewed Polyglot weights so minority moves get a
/// fair share (e.g. weights 4900:400:100 → ln → 8:6:5 → ~42%:32%:26%).
fn pick_book_move(book: &OpeningBook, board: &Board, rng: &mut impl Rng) -> Option<Move> {
    let entries = book.probe(board);
    if entries.is_empty() {
        return None;
    }
    let weights: Vec<u32> = entries.iter().map(|(_, w)| ((*w as f64 + 1.0).ln() as u32).max(1)).collect();
    let total: u32 = weights.iter().sum();
    let mut pick = rng.random_range(0..total);
    for (i, (m, _)) in entries.iter().enumerate() {
        if pick < weights[i] {
            return Some(*m);
        }
        pick -= weights[i];
    }
    Some(entries[0].0)
}

fn recent_abs_score_avg_cp(entries: &[DataEntry], window: usize) -> i32 {
    let take = entries.len().min(window);
    if take == 0 {
        return 0;
    }

    let sum: i32 = entries[entries.len() - take..]
        .iter()
        .map(|e| e.score_cp.unsigned_abs() as i32)
        .sum();

    sum / take as i32
}

/// Play a single self-play game and return training positions in binary format.
///
/// If `start_fen` is `Some`, that FEN is used as the starting position and the
/// book/random-opening phase is skipped entirely.  `randomize_after_fen`
/// controls whether one random legal move is played on top of the FEN to
/// diverge duplicates (useful when the same FEN is sampled repeatedly; leave
/// off for one-shot sequential consumption of a FEN list).
///
/// If `start_fen` is `None` and `book` is provided, book moves are played at
/// the start of the game. Once the book runs out, random moves fill up to
/// `RANDOM_PLIES` total opening plies before the engine takes over.
///
/// The caller is responsible for creating and configuring `engine` (hash size,
/// threads, NNUE, Syzygy).  The TT is *not* cleared here; callers should call
/// `engine.clear_tt()` between games if a fresh table is desired.
pub fn play_game(
    engine: &Engine,
    depth: u8,
    book: Option<&OpeningBook>,
    start_fen: Option<&str>,
    randomize_after_fen: bool,
    rng: &mut impl Rng,
) -> Vec<ChessBoard> {
    // Total opening plies. With a book, up to BOOK_PLIES_MAX come from the
    // book; the remainder (and all plies when there is no book) are random.
    // Capping book depth prevents games from clustering in narrow theoretical
    // lines; the random fill ensures every game diverges into fresh territory.
    const RANDOM_PLIES: usize = 8;
    const BOOK_PLIES_MAX: usize = 4;
    const START_FEN_RANDOM_PLIES: usize = 1;

    let mut board = match start_fen {
        Some(fen) => {
            let mut b = match Board::from_fen(fen) {
                Ok(b) if (b.occupancy[0].0 | b.occupancy[1].0).count_ones() <= 32 => b,
                _ => return Vec::new(),
            };
            if randomize_after_fen {
                for _ in 0..START_FEN_RANDOM_PLIES {
                    let moves = generate_legal_moves(&b);
                    if moves.is_empty() {
                        return Vec::new();
                    }
                    b.make_move(moves.as_slice()[rng.random_range(0..moves.len())]);
                }
            }
            b
        }
        None => {
            // Standard opening: book moves (capped) then random fill.
            let mut b = Board::starting_position();
            let mut book_plies = 0usize;
            if let Some(book) = book {
                // Guard against cyclic book lines by tracking visited hashes.
                let mut visited = std::collections::HashSet::new();
                visited.insert(b.hash);
                while book_plies < BOOK_PLIES_MAX {
                    match pick_book_move(book, &b, rng) {
                        Some(m) => {
                            b.make_move(m);
                            book_plies += 1;
                            if b.position_history.len() > 60 || !visited.insert(b.hash) {
                                break;
                            }
                        }
                        None => break,
                    }
                }
            }
            // Fill remaining opening plies with random moves.
            for _ in book_plies..RANDOM_PLIES {
                let moves = generate_legal_moves(&b);
                if moves.is_empty() {
                    return Vec::new();
                }
                let idx = rng.random_range(0..moves.len());
                b.make_move(moves.as_slice()[idx]);
            }
            b
        }
    };

    let params = SearchParams {
        max_depth: depth,
        use_nnue: engine.use_nnue(),
        contempt: 0,
        ..SearchParams::default()
    };

    let mut entries: Vec<DataEntry> = Vec::new();
    let mut ply: u32 = 0;
    let mut consecutive_high_score: u32 = 0;
    let mut last_high_side: Option<Color> = None;
    let mut consecutive_low_score: u32 = 0;

    loop {
        let moves = generate_legal_moves(&board);

        // Checkmate or stalemate
        if moves.is_empty() {
            let in_check = chess_core::is_in_check(&board);
            let result = if in_check {
                // Checkmate: side to move lost
                if board.side_to_move == Color::White { 0.0 } else { 1.0 }
            } else {
                0.5 // Stalemate
            };
            backfill_results(&mut entries, result);
            break;
        }

        // Draw conditions
        if board.halfmove_clock >= 100 {
            backfill_results(&mut entries, 0.5);
            break;
        }
        if board.is_repetition() {
            // Only discard if the recent recorded scores were clearly non-zero:
            // that means the engine drew from a winning/losing position, so
            // backfilling result=0.5 would mislabel those positions.
            // We use an average over the last few recorded entries to avoid
            // overreacting to one noisy tactical score spike.
            let recent_abs_avg = recent_abs_score_avg_cp(&entries, REPETITION_SCORE_WINDOW);
            if recent_abs_avg >= REPETITION_DISCARD_CP {
                return Vec::new();
            }
            backfill_results(&mut entries, 0.5);
            break;
        }
        if ply > 400 {
            backfill_results(&mut entries, 0.5);
            break;
        }

        // Search
        let result = engine.search(&board, &params, None);
        let score_from_stm = result.score;

        // Convert score to white's perspective
        let score_white = if board.side_to_move == Color::White {
            score_from_stm.centipawns()
        } else {
            -score_from_stm.centipawns()
        };

        // Win adjudication: |score| >= 600 cp for 4 consecutive moves
        let leading_side = if score_white >= 600 {
            Some(Color::White)
        } else if score_white <= -600 {
            Some(Color::Black)
        } else {
            None
        };

        if let Some(side) = leading_side {
            if last_high_side == Some(side) {
                consecutive_high_score += 1;
            } else {
                consecutive_high_score = 1;
                last_high_side = Some(side);
            }
        } else {
            consecutive_high_score = 0;
            last_high_side = None;
        }

        if consecutive_high_score >= 4 {
            let result = if last_high_side == Some(Color::White) { 1.0 } else { 0.0 };
            backfill_results(&mut entries, result);
            break;
        }

        // Draw adjudication: score near zero for many consecutive plies.
        if score_white.abs() < DRAW_ADJ_CP {
            consecutive_low_score += 1;
        } else {
            consecutive_low_score = 0;
        }
        if consecutive_low_score >= DRAW_ADJ_PLIES {
            backfill_results(&mut entries, 0.5);
            break;
        }

        // Record position if quiet (not in check) and past random opening.
        // Clamp score to ±3000 cp: TB Win/Loss scores (±28000) saturate
        // sigmoid(score/400) to exactly 0/1 (zero gradient), corrupting
        // training.  Clamping preserves the endgame positions and their
        // decisive signal while keeping gradients non-zero.
        const MAX_RECORD_SCORE: i32 = 3000;
        if !chess_core::is_in_check(&board)
            && !score_from_stm.is_mate()
        {
            let clamped = score_white.clamp(-MAX_RECORD_SCORE, MAX_RECORD_SCORE);
            entries.push(DataEntry::from_board(&board, clamped));
        }

        // Play best move
        board.make_move(result.best_move);
        ply += 1;
    }

    entries.iter().map(DataEntry::to_chess_board).collect()
}

fn backfill_results(entries: &mut [DataEntry], result: f32) {
    for entry in entries.iter_mut() {
        entry.result = result;
    }
}
