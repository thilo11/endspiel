//! Analysis over UCI: game-play settings must not weaken an analysis, a long
//! search must show it is still running, and `Clear Hash` must work.
//!
//! GUIs often keep one set of options per engine binary, so an analysis can
//! inherit `UCI_LimitStrength` from game play: at the beginner class that capped
//! `go infinite` at 150 nodes (depth 1). `go infinite` and `UCI_AnalyseMode`
//! now search at full strength.
//!
//! RELEASE ONLY, like `ponder_forced_move.rs`: a debug engine spends seconds in
//! search spin-up, so the wall-clock thresholds below would not hold.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

const ITALIAN: &str = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5";

struct Engine {
    child: Child,
    stdin: ChildStdin,
    lines: mpsc::Receiver<String>,
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
        let (tx, lines) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    return;
                }
            }
        });
        Self {
            child,
            stdin,
            lines,
        }
    }

    fn send(&mut self, command: &str) {
        writeln!(self.stdin, "{command}").unwrap();
        self.stdin.flush().unwrap();
    }

    /// The lines printed until one satisfies `done` (included), or `timeout`.
    fn lines_until(&self, timeout: Duration, done: impl Fn(&str) -> bool) -> Vec<String> {
        let deadline = Instant::now() + timeout;
        let mut lines = Vec::new();
        while let Some(left) = deadline.checked_duration_since(Instant::now()) {
            match self.lines.recv_timeout(left) {
                Ok(line) => {
                    let finished = done(&line);
                    lines.push(line);
                    if finished {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        lines
    }

    /// Every line printed within `duration`.
    fn lines_for(&self, duration: Duration) -> Vec<String> {
        let deadline = Instant::now() + duration;
        let mut lines = Vec::new();
        while let Some(left) = deadline.checked_duration_since(Instant::now()) {
            match self.lines.recv_timeout(left) {
                Ok(line) => lines.push(line),
                Err(_) => break,
            }
        }
        lines
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The deepest `depth` of the finished lines (`info … pv …`) in `lines`.
fn max_depth(lines: &[String]) -> u32 {
    lines
        .iter()
        .filter(|l| l.starts_with("info depth") && l.contains(" pv "))
        .filter_map(|l| l.split_whitespace().nth(2)?.parse().ok())
        .max()
        .unwrap_or(0)
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "debug builds spend seconds in search spin-up; run with --release"
)]
fn analysis_ignores_the_strength_limit() {
    let mut engine = Engine::spawn();
    engine.send("uci");
    engine.send("setoption name UCI_LimitStrength value true");
    engine.send("setoption name UCI_Elo value 1000");
    engine.send(&format!("position fen {ITALIAN}"));
    engine.send("go infinite");
    let lines = engine.lines_for(Duration::from_millis(1500));
    engine.send("stop");
    let depth = max_depth(&lines);
    assert!(
        depth >= 8,
        "go infinite stopped at depth {depth} under the strength limit"
    );
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "debug builds spend seconds in search spin-up; run with --release"
)]
fn a_long_search_reports_the_move_it_searches() {
    let mut engine = Engine::spawn();
    engine.send("uci");
    engine.send(&format!("position fen {ITALIAN}"));
    engine.send("go infinite");
    let lines = engine.lines_for(Duration::from_millis(5000));
    engine.send("stop");
    assert!(
        lines
            .iter()
            .any(|l| l.contains(" currmove ") && l.contains(" currmovenumber ")),
        "no currmove line in a 5 s search"
    );
}

#[test]
fn clear_hash_is_a_button_refused_during_a_search() {
    // A small table: clearing the default one takes seconds in a debug build.
    let mut engine = Engine::spawn();
    engine.send("uci");
    engine.send("setoption name Hash value 16");
    engine.send("isready");
    let lines = engine.lines_until(Duration::from_secs(30), |l| l == "readyok");
    assert!(
        lines
            .iter()
            .any(|l| l == "option name Clear Hash type button")
    );
    assert!(
        lines
            .iter()
            .any(|l| l == "option name UCI_AnalyseMode type check default false")
    );

    engine.send("setoption name Clear Hash");
    engine.send("isready");
    let idle = engine.lines_until(Duration::from_secs(30), |l| l == "readyok");
    assert_eq!(idle.last().map(String::as_str), Some("readyok"), "{idle:?}");
    assert!(
        !idle.iter().any(|l| l.contains("Clear Hash ignored")),
        "{idle:?}"
    );

    engine.send(&format!("position fen {ITALIAN}"));
    engine.send("go infinite");
    thread::sleep(Duration::from_millis(300));
    engine.send("setoption name Clear Hash");
    let busy = engine.lines_until(Duration::from_secs(10), |l| {
        l.contains("Clear Hash ignored")
    });
    engine.send("stop");
    assert!(
        busy.iter().any(|l| l.contains("Clear Hash ignored")),
        "{busy:?}"
    );
}
