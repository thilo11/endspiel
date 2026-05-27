# Endspiel

> **Endspiel** /ˈɛnt.ʃpiːl/ *n.* (German) &nbsp; **1.** the final, decisive game. &nbsp; **2.** the final phase of a chess game.

A UCI chess engine written in Rust — bitboards and move generation, the
alpha-beta search, the NNUE evaluation and its inference, the self-play
data generator, and the UCI front end are all hand-written from scratch,
with no external chess libraries. The network is trained **entirely on the
engine's own self-play** games scored by its own search; there is **no
external evaluation data** — no Stockfish or Leela labels. The only outside
ingredient is a set of raw opening positions used as self-play starting
points, never as training targets.

See [ABOUT.md](ABOUT.md) for project rationale, playing strength, and training details.

## Features

- **Built from scratch** — move generation, search, NNUE inference, the
  datagen tool and the UCI layer are all hand-rolled in Rust with no
  external chess libraries; the net is trained only on the engine's own
  self-play, with no external evaluation data (net training runs through
  the [Bullet](https://github.com/jw1912/bullet) trainer and Syzygy
  probing uses `pyrrhic-rs` — the only third-party pieces in the pipeline)
- **Full UCI compliance** — works in any UCI GUI (Arena, CuteChess, Fritz, Banksia, Scid, …)
- **NNUE evaluation** (default) — HalfKP 704×32→768×2→1 SCReLU, trained
  from scratch on ~2.6 billion self-play positions; the net is embedded
  in the binary, no extra files to ship
- **HCE fallback** — tapered hand-crafted evaluation (`UseNNUE=false`) with
  pawn hash, mobility, king safety, pawn structure, threats, space, and
  endgame scaling
- **Modern search** — alpha-beta + PVS with iterative deepening and
  aspiration windows, null move, reverse futility, futility, razoring,
  ProbCut, SEE pruning, LMR, LMP, IIR, singular and passed-pawn extensions,
  1- and 2-ply continuation history, capture history, and multi-facet
  correction history (pawn, non-pawn, minor/major, and continuation keys).
  History and correction tables persist across moves within a game
  (reset on `ucinewgame`)
- **Pondering** — thinks on the opponent's time (`go ponder` / `ponderhit`);
  enable via the `Ponder` UCI option in your GUI
- **Multi-threading** — Lazy SMP with depth diversity (`Threads` UCI option)
- **MultiPV** — up to 256 principal variations for analysis (`MultiPV` UCI option)
- **Syzygy tablebases** — WDL probing for 3–5 man endgames via `pyrrhic-rs`
- **Opening books** — load Polyglot `.bin`, EPD (`.epd`, with `bm` opcodes), or PGN (`.pgn`) at runtime; format is auto-detected by extension
- **WDL output** — optional `wdl W D L` annotation on each `info` line
  (`UCI_ShowWDL`), with the win/draw/loss mapping fit per net
- **Contempt** and configurable time management (`Move Overhead`, `Slow Mover`)
- **Performance** — native-CPU release builds; on a modern desktop the
  engine searches in the millions of nodes per second per thread and plays
  competitive bullet/blitz time controls comfortably
- **Cross-platform** — Linux x86_64/ARM64, Windows x86_64/ARM64, macOS Apple Silicon
- **Self-contained binary** — no runtime dependencies, no external net file

## Download

Prebuilt binaries are on the [Releases](https://github.com/thilo11/endspiel/releases) page:

| Platform | Binary | ISA / notes |
|----------|--------|-------------|
| Linux x86_64 (recommended) | `endspiel-linux-x64` | AVX2 / `x86-64-v3`, PGO-optimised |
| Linux x86_64 (AVX-512) | `endspiel-linux-x64-avx512` | `x86-64-v4` — faster on Zen 4/5, Sapphire Rapids, etc. |
| Windows x86_64 (recommended) | `endspiel-win-x64.exe` | AVX2 / `x86-64-v3`, PGO-optimised |
| Windows x86_64 (AVX-512) | `endspiel-win-x64-avx512.exe` | `x86-64-v4` — faster on CPUs with AVX-512 |
| Windows ARM64 | `endspiel-win-arm64.exe` | generic ARM64 |
| macOS Apple Silicon | `endspiel-mac-arm64` | `apple-m1`, PGO-optimised |
| Raspberry Pi 5 | `endspiel-linux-arm64-pi5` | `cortex-a76`, PGO-optimised; requires Pi OS Trixie (glibc ≥ 2.39) |

**Picking a build.** The recommended (`-v3`) builds run on essentially any
CPU sold in the last decade and are profile-guided-optimised for ~5–15%
extra throughput. If your CPU has AVX-512 (AMD Zen 4/5, recent Intel
Xeon, etc.), the `-avx512` build is typically 30–60% faster on NNUE
evaluation but will refuse to run on older hardware (illegal-instruction
crash). When in doubt, use the recommended build.

**Raspberry Pi 5.** Any RAM tier runs the engine; hash size is the only
thing that scales with it. Set `Hash` in your GUI rather than relying on
the 256 MB default:

| Pi 5 RAM | Recommended `Hash` | Notes |
|----------|--------------------|-------|
| 4 GB | 512–1024 MB | usable for blitz/rapid; leave ≥2 GB for the desktop/browser |
| 8 GB | 2048–4096 MB | sweet spot — full Pi-5 strength at all time controls |
| 16 GB | 4096–8192 MB | only worth it for deep analysis or running other workloads alongside |

Expect NPS roughly 5–10× lower than a modern x86 desktop, so plan for a
noticeable Elo drop at fixed time controls and compensate by giving the
engine more thinking time. Active cooling is recommended: under
sustained engine load the SoC will thermally throttle without a fan.

## Usage

Endspiel is a command-line program used through a UCI-compatible chess GUI (Arena, CuteChess, Fritz, Banksia, etc.). Point the GUI at the binary — no further setup is required.

**Bench** — runs a fixed-depth search across a small set of positions and
prints the total node count, elapsed time, and NPS. Useful as a sanity
check that the binary runs end-to-end:

```bash
./endspiel bench          # default depth 14
./endspiel bench 18       # deeper, for performance tuning
```

> **macOS users — run this once from a terminal before pointing a chess
> GUI at the binary.** The release binaries are not code-signed, so
> macOS Gatekeeper will quarantine them. If that's the case, this
> command surfaces the error clearly instead of leaving the GUI to
> silently fail to launch the engine. To clear the quarantine flag:
>
> ```bash
> chmod +x endspiel-mac-arm64
> xattr -d com.apple.quarantine endspiel-mac-arm64
> ```
>
> Then re-run `./endspiel-mac-arm64 bench` — once you get a Nodes/NPS
> line, the GUI will also be able to launch it.

## UCI Options

| Option | Default | Description |
|--------|---------|-------------|
| `Hash` | 256 | Transposition table size in MB |
| `Threads` | min(available, 16) | Search threads |
| `Move Overhead` | 20 | Time safety margin in ms |
| `Slow Mover` | 100 | Time usage scaling (%) — >100 thinks longer, <100 plays faster |
| `Ponder` | false | Think on the opponent's time; the GUI toggles this and drives `go ponder` / `ponderhit` |
| `Contempt` | 20 | Draw avoidance in centipawns |
| `SingularExt` | 1 | Singular extension: 0 = off, 1 = conservative, 2 = aggressive |
| `UseNNUE` | true | Use NNUE evaluation; false falls back to HCE |
| `EvalFile` | *(embedded)* | Path to an external `.nnue` / `quantised.bin` net |
| `BookFile` | *(disabled)* | Path to an opening book: Polyglot `.bin`, EPD `.epd`, or PGN `.pgn` (auto-detected by extension) |
| `SyzygyPath` | *(disabled)* | Path to Syzygy tablebase directory |
| `MultiPV` | 1 | Number of principal variations to report (1–256) |
| `UCI_ShowWDL` | false | Append `wdl <win> <draw> <loss>` (0–1000) to each info line |

Set `BookFile` or `SyzygyPath` to a valid path to enable; clear to disable. No separate toggle is needed.

### Notes

- **`Hash`** — increase for long time controls or analysis; watch `hashfull` in engine output (permille, so 950 = 95%).
- **`Threads`** — Lazy SMP; scaling is sub-linear. Stick to physical core count.
- **`EvalFile`** — load an alternate net at runtime without rebuilding. Clear to revert to the embedded net.
- **`SyzygyPath`** — WDL probing for 3–5 man endgames. Multiple directories: `:` on Linux/macOS, `;` on Windows.

> **Fritz 20 (Windows):** Fritz manages Syzygy and opening books through its own systems.
> Set the tablebase path in Fritz's settings — it forwards it to Endspiel automatically.
> To let Endspiel use its own `BookFile`, disable Fritz's opening book in the match settings.
> Verify loading via View → Engine Output: a successful load prints `info string BookFile loaded from '...'`.

## Build from Source

Requires Rust 1.95.0+.

```bash
cargo build --release
# binary: target/release/endspiel
```

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).
