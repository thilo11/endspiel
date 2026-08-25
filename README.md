# Endspiel

[![CI](https://github.com/thilo11/endspiel/actions/workflows/ci.yml/badge.svg)](https://github.com/thilo11/endspiel/actions/workflows/ci.yml)
[![Release](https://github.com/thilo11/endspiel/actions/workflows/release.yml/badge.svg)](https://github.com/thilo11/endspiel/actions/workflows/release.yml)
[![GitHub release](https://img.shields.io/github/v/release/thilo11/endspiel?logo=github&label=release)](https://github.com/thilo11/endspiel/releases/latest)
[![License](https://img.shields.io/github/license/thilo11/endspiel)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-stable-orange?logo=rust)](https://www.rust-lang.org)
[![Lichess](https://img.shields.io/badge/lichess-Endspiel%20%28Pi5%29-green?logo=lichess)](https://lichess.org/@/endspiel-pi)
[![Lichess endspiel-engine](https://img.shields.io/badge/lichess-endspiel--engine-green?logo=lichess)](https://lichess.org/@/endspiel-engine)

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
- **Chess960** — Fischer Random / Freestyle: X-FEN and Shredder-FEN, `UCI_Chess960` king-takes-rook castling
- **NNUE evaluation** (default) — state-aware HalfKP 785×32→(1024 pairwise 512)×2→16→32→1 (8 material-keyed output buckets), with castling rights and en passant in the input; trained from scratch on billions of self-play positions; the net is embedded in the binary, no extra files to ship. Older piece-only nets still load.
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
  enable via the `Ponder` UCI option in your GUI. A legal ponder move is
  still advertised when the PV is only one ply (book, tablebase, forced lines)
- **Multi-threading** — Lazy SMP with depth diversity (`Threads` UCI option)
- **MultiPV** — up to 256 principal variations for analysis (`MultiPV` UCI option)
- **Syzygy tablebases** — WDL/DTZ probing (up to 7-man) via `pyrrhic-rs`; at the root the engine stays on win- or draw-preserving moves instead of drifting into a loss
- **Opening books** — load Polyglot `.bin`, EPD (`.epd`, with `bm` opcodes), or PGN (`.pgn`) at runtime; format is auto-detected by extension
- **WDL output** — optional `wdl W D L` annotation on each `info` line
  (`UCI_ShowWDL`), with the win/draw/loss mapping fit per net
- **Contempt** and configurable time management (`Move Overhead`, `Slow Mover`)
- **Performance** — runtime-dispatched AVX-512 and AVX2 inference on x86-64,
  NEON on AArch64, and a scalar fallback; selected release builds also use
  profile-guided optimisation and fat LTO
- **Cross-platform** — Linux x86_64/ARM64, Windows x86_64/ARM64, macOS Apple Silicon
- **Self-contained binary** — no runtime dependencies, no external net file

## Download

Prebuilt binaries are on the [Releases](https://github.com/thilo11/endspiel/releases/latest) page. Current release: **v1.6.0**.

**x86-64** ships in three micro-architecture tiers (each faster than the one
below, all but `-v4` profile-guided-optimised). Pick the **highest your CPU
supports**:

| Tier | Linux | Windows | CPU requirement |
|------|-------|---------|-----------------|
| `v4` | `endspiel-linux-x64-v4` | `endspiel-win-x64-v4.exe` | **AVX-512** — AMD Zen 4/5, Intel Skylake-X / Ice Lake+ (typically 30–60% faster NNUE eval) |
| `v3` *(default)* | `endspiel-linux-x64-v3` | `endspiel-win-x64-v3.exe` | **AVX2** — Intel Haswell (2013)+, AMD Zen / Excavator+ |
| `v2` | `endspiel-linux-x64-v2` | `endspiel-win-x64-v2.exe` | **SSE4.2 + POPCNT** — pre-2013 mainstream + recent low-end Intel (Gemini/Jasper Lake) without AVX2 |

**Other platforms:**

| Platform | Binary | Notes |
|----------|--------|-------|
| macOS Apple Silicon | `endspiel-mac-arm64` | `apple-m1`, PGO-optimised |
| Windows ARM64 | `endspiel-win-arm64.exe` | generic ARM64 |
| Raspberry Pi 5 | `endspiel-linux-arm64-pi5` | `cortex-a76`, fat LTO + PGO; needs Raspberry Pi OS (Trixie / Debian 13) or newer — glibc ≥ 2.39 |
| Android arm64 | `endspiel-android-arm64.apk` | `arm64-v8a`, minSdk 24 — installs like an app; pick "Endspiel" as a UCI engine in DroidFish / Chess for Android |

**Picking an x86-64 build.** `-v3` (AVX2) is the safe default — it runs on
essentially any CPU sold since ~2013. Go up to `-v4` if your CPU has AVX-512
(AMD Zen 4/5, recent Intel) for a sizeable NNUE-eval speedup; drop to `-v2`
only for older or low-end (no-AVX2) hardware. If a build aborts immediately with an **illegal-
instruction** crash, your CPU lacks that tier's instructions — step down one
tier. On Linux you can check support with
`lscpu | grep -oE 'avx512f|avx2|sse4_2'` (highest match wins).

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
| `Hash` | auto (RAM + threads) | Transposition table size in MB |
| `Threads` | auto (cores) | Search threads |
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
| `OpeningVariety` | 0 | Opening spice: 0 = off; otherwise pick at random among MultiPV moves within this many centipawns of the best, for the first 8 plies |
| `UCI_ShowWDL` | false | Append `wdl <win> <draw> <loss>` (0–1000) to each info line |

Set `BookFile` or `SyzygyPath` to a valid path to enable; clear to disable. No separate toggle is needed.

### Notes

- **`Hash`** — increase for long time controls or analysis; watch `hashfull` in engine output (permille, so 950 = 95%). The default adapts to the machine: ~128 MB per search thread (more threads fill the table faster), capped at ~1/16 of available RAM, floor 16 MB. So it grows with both core count and RAM, and tracks an explicit `Threads` setting.
- **`Threads`** — Lazy SMP; scaling is sub-linear. Stick to physical core count. On Linux/Android the default is the performance-core count (the top CPU-frequency tier), which avoids the slow LITTLE cores and the thermal throttling they invite; elsewhere it's `min(available, 16)`.
- **`EvalFile`** — load an alternate net at runtime without rebuilding. Clear to revert to the embedded net.
- **`SyzygyPath`** — WDL/DTZ probing for up to 7-man endgames. Multiple directories: `:` on Linux/macOS, `;` on Windows.
- **`OpeningVariety`** — only affects the first 8 plies; 0 (default) always plays the best move.

> **Fritz 20 (Windows):** Fritz manages Syzygy and opening books through its own systems.
> Set the tablebase path in Fritz's settings — it forwards it to Endspiel automatically.
> To let Endspiel use its own `BookFile`, disable Fritz's opening book in the match settings.
> Verify loading via View → Engine Output: a successful load prints `info string BookFile loaded from '...'`.

## Build from Source

Requires Rust 1.97.1+.

```bash
cargo build --release
# binary: target/release/endspiel
```

### Android (arm64)

Cross-compile a standalone CLI binary for 64-bit Android. It speaks UCI over
stdin/stdout like the desktop build and runs under Termux or via `adb shell`.

```bash
rustup target add aarch64-linux-android
cargo install cargo-ndk
export ANDROID_NDK_HOME=/path/to/android-ndk   # r23+

scripts/build-android.sh
# binary: target/aarch64-linux-android/release/endspiel
```

On-device the engine auto-tunes its defaults: `Threads` is the performance-core
count of the SoC, and `Hash` is ~128 MB per such core, capped at ~1/16 of
available RAM (floor 16 MB). Both remain overridable via UCI `setoption`.

The raw binary is only convenient under Termux or `adb` — a non-rooted device
can't `exec` a binary off shared storage. For normal use, install the **Open
Exchange (OEX) engine APK** instead (`endspiel-android-arm64.apk` on the release
page): it installs like an app and is auto-discovered as a UCI engine by GUIs
such as DroidFish and Chess for Android. The APK wraps the same binary; the
release workflow builds it, and `android/oex/` documents building it by hand.

> **DroidFish doesn't list Endspiel?** DroidFish only scans for installed OEX
> engines when it has storage permission. Grant it under **Settings → Apps →
> DroidFish → Permissions → Files and media (Storage) → Allow**, then reopen the
> engine selector (**⋮ → Set engine…**) — Endspiel will appear next to Stockfish
> and CuckooChess.

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).

## Credits

Third-party software, tablebase/training tooling, and technique inspirations are
documented in [CREDITS.md](CREDITS.md).
