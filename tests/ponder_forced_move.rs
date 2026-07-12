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

/// Read `bestmove ...` off the engine's stdout, with a hard timeout so a
/// regression fails the test instead of hanging it.
fn bestmove_within(stdout: ChildStdout, timeout: Duration) -> Option<String> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if line.starts_with("bestmove") {
                let _ = tx.send(line);
                return;
            }
        }
    });
    rx.recv_timeout(timeout).ok()
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
