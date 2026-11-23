//! Regression: `go infinite` must not answer `bestmove` before `stop`.
//!
//! UCI has an infinite search run until `stop`, and GUIs rely on that: an
//! analysis GUI (Castellan) that sends `stop` waits for the `bestmove` it owes,
//! and a `bestmove` the engine already printed by itself never comes again — so
//! the next analysis's lines were taken for the stopped one's and dropped. The
//! search ends early in several positions (a short mate, the depth cap, no legal
//! move); the UCI layer must hold the move back in all of them, as it does for
//! `go ponder`.
//!
//! It also checks that `quit` ends a search without one.
//!
//! RELEASE ONLY, like `ponder_forced_move.rs`: a debug engine spends seconds in
//! search spin-up, so it would not answer in time even when correct.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// Rd8# — the search finds the mate at depth 1.
const MATE_IN_ONE: &str = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1";
/// Black is mated: there is no move to search.
const MATED: &str = "6rk/5Npp/8/8/8/8/8/6K1 b - - 0 1";
/// Bare kings: the search reaches its depth cap at once.
const BARE_KINGS: &str = "8/8/4k3/8/8/4K3/8/8 w - - 0 1";
const ITALIAN: &str = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5";

struct Engine {
    child: Child,
    stdin: ChildStdin,
    bestmoves: mpsc::Receiver<String>,
}

impl Engine {
    fn spawn() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_endspiel"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to spawn the endspiel binary");
        let stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let (tx, bestmoves) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if line.starts_with("bestmove") && tx.send(line).is_err() {
                    return;
                }
            }
        });
        let mut engine = Self {
            child,
            stdin,
            bestmoves,
        };
        engine.send("uci");
        engine.send("isready");
        engine
    }

    fn send(&mut self, command: &str) {
        writeln!(self.stdin, "{command}").unwrap();
        self.stdin.flush().unwrap();
    }

    /// The next `bestmove` within `timeout`, if any.
    fn bestmove_within(&self, timeout: Duration) -> Option<String> {
        self.bestmoves.recv_timeout(timeout).ok()
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "debug builds spend seconds in search spin-up; run with --release"
)]
fn infinite_search_answers_only_after_stop() {
    for (name, fen) in [
        ("mate in one", MATE_IN_ONE),
        ("mated", MATED),
        ("bare kings", BARE_KINGS),
    ] {
        let mut engine = Engine::spawn();
        engine.send(&format!("position fen {fen}"));
        engine.send("go infinite");
        let early = engine.bestmove_within(Duration::from_millis(1500));
        assert_eq!(early, None, "{name}: answered before stop");
        engine.send("stop");
        let answer = engine.bestmove_within(Duration::from_secs(5));
        assert!(answer.is_some(), "{name}: no bestmove after stop");
        // One `bestmove` per search: a second `stop` while idle answers nothing.
        engine.send("stop");
        let extra = engine.bestmove_within(Duration::from_millis(300));
        assert_eq!(extra, None, "{name}: answered a stop while idle");
    }
}

/// `go ponder infinite`: after `ponderhit` the search is an infinite one, so it
/// still runs until `stop`.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "debug builds spend seconds in search spin-up; run with --release"
)]
fn ponderhit_on_an_infinite_ponder_runs_until_stop() {
    let mut engine = Engine::spawn();
    engine.send(&format!("position fen {ITALIAN}"));
    engine.send("go ponder infinite");
    thread::sleep(Duration::from_millis(300));
    engine.send("ponderhit");
    let early = engine.bestmove_within(Duration::from_millis(1000));
    assert_eq!(early, None, "answered before stop");
    engine.send("stop");
    assert!(
        engine.bestmove_within(Duration::from_secs(5)).is_some(),
        "no bestmove after stop"
    );
}

/// `quit` during a search ends the process without a `bestmove`: nobody is
/// waiting for one.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "debug builds spend seconds in search spin-up; run with --release"
)]
fn quit_during_a_search_prints_no_bestmove() {
    let mut engine = Engine::spawn();
    engine.send(&format!("position fen {ITALIAN}"));
    engine.send("go infinite");
    thread::sleep(Duration::from_millis(300));
    engine.send("quit");
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while engine.child.try_wait().unwrap().is_none() {
        assert!(std::time::Instant::now() < deadline, "did not exit on quit");
        thread::sleep(Duration::from_millis(20));
    }
    let answer = engine.bestmove_within(Duration::from_millis(500));
    assert_eq!(answer, None, "printed a bestmove on quit");
}
