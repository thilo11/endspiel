//! Regression: a forced move must not cost clock on the ponder route.
//!
//! lichess hXLfiVIo (2026-07-12, 300+5). After 31...Rxd1+ White had exactly one
//! legal move, 32.Bxd1 — and spent 14 seconds on it, in a game it finished on
//! 1:08. The search has a single-legal-move fast path, but it is gated on
//! `!infinite`, and `go ponder` runs the search in `infinite` mode so the move
//! can be held back until `ponderhit`. Nothing re-checked on the way out, so the
//! engine sleeps out its whole remaining allocation on a move it cannot choose.
//!
//! This drives the real UCI route end-to-end, which the unit tests in
//! `chess-uci` cannot: they check the decision, this checks that a live engine
//! process actually plays the move at once. It is invisible to the h2h gate,
//! because fastchess does not ponder.
//!
//! RELEASE ONLY. A debug engine spends ~8 seconds in search spin-up (zeroing the
//! history/correction tables) before it searches its first node, so it cannot
//! observe the stop flag promptly and any wall-clock assertion here is meaningless
//! — measured: 0.00s to answer `ponderhit` in release, 9-14s in debug, and the
//! debug figure does not scale with the clock, proving it is spin-up and not the
//! budget. CI runs `cargo test --workspace` in debug, so this is ignored there;
//! the debug-safe guard is `ponderhit_spends_no_clock_on_a_forced_move` in
//! `chess-uci`. Run it with `cargo test --release`.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

/// Position after 31...Rxd1+. Bxd1 (c2d1) is the *only* legal move.
const FORCED_FEN: &str = "2b3k1/5ppp/2n2n2/2N1p3/2P1P3/4B3/2B2PPP/3r2K1 w - - 0 32";

/// The clock White actually had. At 137s + 5s increment the move budget is on
/// the order of 15-20s, which is what the bug spent.
const GO_PONDER: &str = "go ponder wtime 137000 btime 246000 winc 5000 binc 5000";

/// Spawn a reader that forwards the first `bestmove ...` line off the engine's
/// stdout into a channel, so tests can both assert its absence (held back
/// during ponder) and await it with a timeout.
fn bestmove_channel(stdout: ChildStdout) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if line.starts_with("bestmove") {
                let _ = tx.send(line);
                return;
            }
        }
    });
    rx
}

/// Read `bestmove ...` off the engine's stdout, with a hard timeout so a
/// regression fails the test instead of hanging it.
fn bestmove_within(stdout: ChildStdout, timeout: Duration) -> Option<String> {
    bestmove_channel(stdout).recv_timeout(timeout).ok()
}

fn spawn_engine() -> Child {
    Command::new(env!("CARGO_BIN_EXE_endspiel"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn the endspiel binary")
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "debug builds spend ~8s in search spin-up; run with --release"
)]
fn ponderhit_on_a_forced_move_plays_at_once() {
    let mut engine = spawn_engine();
    let mut stdin = engine.stdin.take().expect("stdin");
    let stdout = engine.stdout.take().expect("stdout");

    writeln!(stdin, "uci").unwrap();
    writeln!(stdin, "isready").unwrap();
    writeln!(stdin, "position fen {FORCED_FEN}").unwrap();
    writeln!(stdin, "{GO_PONDER}").unwrap();
    stdin.flush().unwrap();

    // Let the ponder search get going, as it would on the opponent's clock.
    thread::sleep(Duration::from_millis(300));

    // The opponent plays the move we predicted: the pondered position is now
    // the real one, and our clock starts running.
    let hit = Instant::now();
    writeln!(stdin, "ponderhit").unwrap();
    stdin.flush().unwrap();

    // The bug sleeps out `alloc - elapsed`, which at this clock is ~19s — a wall
    // time it would take on any machine, since it is a sleep and not work. So the
    // threshold only has to sit well below that while clearing a release engine's
    // spin-up on a slow 2-core CI runner (0.2s here). 5s leaves ~4x margin both
    // ways. Measured from `ponderhit`, so process startup is not counted.
    let budget = Duration::from_secs(5);
    let best = bestmove_within(stdout, budget + Duration::from_secs(3));
    let elapsed = hit.elapsed();

    let _ = engine.kill();
    let _ = engine.wait();

    let best = best.expect("engine never answered ponderhit with a bestmove");
    assert!(
        best.starts_with("bestmove c2d1"),
        "the only legal move is Bxd1 (c2d1), got: {best}"
    );
    assert!(
        elapsed < budget,
        "forced move took {elapsed:?} after ponderhit; a single legal move must \
         cost no clock (the search's fast path is disabled while pondering)"
    );
}

/// Forward every stdout line into a channel (the bestmove-only channel above
/// cannot observe `info string tm` lines).
fn lines_channel(stdout: ChildStdout) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                return;
            }
        }
    });
    rx
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "debug builds spend ~8s in search spin-up; run with --release"
)]
fn ponderhit_activates_difficulty_time_management() {
    // A pondered search must get the same difficulty-adaptive soft-limit
    // machinery as a plain `go` (lichess gmsfUArc: five flat ~4.75s ponderhit
    // moves while the eval slid −0.7→−1.6 with 160s banked). The observable
    // is the time manager's own debug line: with ENDSPIEL_TM_DEBUG=1 the
    // soft-limit block prints `info string tm ...` once per completed
    // iteration — it must stay silent while pondering (no budget is running)
    // and start printing after `ponderhit`.
    let mut engine = Command::new(env!("CARGO_BIN_EXE_endspiel"))
        .env("ENDSPIEL_TM_DEBUG", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn the endspiel binary");
    let mut stdin = engine.stdin.take().expect("stdin");
    let lines = lines_channel(engine.stdout.take().expect("stdout"));

    writeln!(stdin, "uci").unwrap();
    writeln!(stdin, "isready").unwrap();
    // Real move history: probing by bare FEN gives game_ply == 0, which trips
    // the early-opening cap and distorts the budgets this test observes.
    writeln!(
        stdin,
        "position startpos moves e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 d2d3 f8c5"
    )
    .unwrap();
    writeln!(stdin, "go ponder wtime 60000 btime 60000 winc 0 binc 0").unwrap();
    stdin.flush().unwrap();

    // Let the ponder search run several iterations on the opponent's clock.
    thread::sleep(Duration::from_millis(900));
    let mut pre_hit = Vec::new();
    while let Ok(line) = lines.try_recv() {
        pre_hit.push(line);
    }
    assert!(
        !pre_hit.iter().any(|l| l.starts_with("bestmove")),
        "bestmove leaked mid-ponder"
    );
    assert!(
        !pre_hit.iter().any(|l| l.contains("info string tm")),
        "time manager ran during ponder; every time-based stop must be \
         suspended until ponderhit (got: {:?})",
        pre_hit
            .iter()
            .find(|l| l.contains("info string tm"))
            .unwrap()
    );

    writeln!(stdin, "ponderhit").unwrap();
    stdin.flush().unwrap();

    // From here the search self-manages: it must produce at least one tm
    // decision line and then a bestmove well inside the hard limit.
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut saw_tm = false;
    let mut best = None;
    while Instant::now() < deadline {
        match lines.recv_timeout(Duration::from_millis(200)) {
            Ok(line) => {
                if line.contains("info string tm") {
                    saw_tm = true;
                }
                if line.starts_with("bestmove") {
                    best = Some(line);
                    break;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    let _ = engine.kill();
    let _ = engine.wait();

    assert!(
        best.is_some(),
        "engine never answered ponderhit with a bestmove"
    );
    assert!(
        saw_tm,
        "no `info string tm` line after ponderhit: the soft-limit machinery \
         did not take over the pondered search (flat-timer regression)"
    );
}

/// White mates in 1 (Qg7#): the iterative-deepening loop breaks as soon as a
/// mate within 3 moves is proven, so the ponder search finishes in
/// milliseconds and the thread parks in the hold-back loop.
const MATE_IN_ONE_FEN: &str = "7k/8/5K2/8/8/8/8/6Q1 w - - 0 1";

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "debug builds spend ~8s in search spin-up; run with --release"
)]
fn ponderhit_after_search_completed_plays_at_once() {
    // Regression for the hold-back loop watching the shared stop flag: when a
    // ponder search resolves the position early (mate break here;
    // TB-restricted roots behave the same), pool.search stores `stop = true`
    // itself to shut down its helper threads — and a hold keyed on `stop`
    // then leaks the bestmove MID-PONDER, a UCI protocol violation. The move
    // must stay held until `ponderhit`, then play at once (not sleep out the
    // remaining allocation on a search that is no longer running).
    let mut engine = spawn_engine();
    let mut stdin = engine.stdin.take().expect("stdin");
    let stdout = engine.stdout.take().expect("stdout");

    writeln!(stdin, "uci").unwrap();
    writeln!(stdin, "isready").unwrap();
    writeln!(stdin, "position fen {MATE_IN_ONE_FEN}").unwrap();
    writeln!(stdin, "{GO_PONDER}").unwrap();
    stdin.flush().unwrap();

    let bestmove = bestmove_channel(stdout);

    // Give the search ample time to prove the mate and break out of the ID
    // loop; the move must still be held back for the whole wait.
    thread::sleep(Duration::from_millis(1500));
    if let Ok(early) = bestmove.try_recv() {
        let _ = engine.kill();
        let _ = engine.wait();
        panic!("bestmove leaked mid-ponder ({early}); it must be held until ponderhit/stop");
    }

    let hit = Instant::now();
    writeln!(stdin, "ponderhit").unwrap();
    stdin.flush().unwrap();

    // The budget-timer bug waits out `alloc - elapsed` (~19s at these clocks)
    // on a search that is no longer running. Same threshold reasoning as above.
    let budget = Duration::from_secs(5);
    let best = bestmove.recv_timeout(budget + Duration::from_secs(3)).ok();
    let elapsed = hit.elapsed();

    let _ = engine.kill();
    let _ = engine.wait();

    let best = best.expect("engine never answered ponderhit with a bestmove");
    assert!(
        best.starts_with("bestmove g1g7"),
        "Qg7# is mate in one, got: {best}"
    );
    assert!(
        elapsed < budget,
        "finished ponder search took {elapsed:?} after ponderhit; a completed \
         search must release its move at once, not sleep out the budget"
    );
}
