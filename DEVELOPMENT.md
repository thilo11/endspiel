# Endspiel — Developer Guide

## Project Layout

```
├── Cargo.toml                    # Workspace root + endspiel binary
├── src/main.rs                   # Entry point
├── crates/
│   ├── chess-common/             # Shared types: Board, Move, Bitboard, FEN
│   ├── chess-core/               # Move generation, attack tables, validation
│   ├── chess-engine/             # Search, HCE evaluation, Syzygy WDL probing
│   ├── chess-nnue/               # NNUE inference + embedded net (build.rs)
│   ├── chess-uci/                # UCI protocol handler
│   └── chess-tuner/              # HCE parameter tuner
├── scripts/                      # build & setup helpers (Android build, Syzygy download)
└── assets/                       # Gitignored — local resources (book, tablebases)
```

## Architecture

### Search (`chess-engine`)

Alpha-beta with iterative deepening and PVS.

- **Pruning**: null move, reverse futility, futility, razoring, SEE (captures + quiets), history pruning, ProbCut
- **Extensions**: check, singular (conservative/aggressive), passed pawn push
- **Reductions**: LMR, IIR
- **Move ordering**: TT move → good captures (MVV-LVA + capture history) → killers → counter move → history-sorted quiets → bad captures
- **History**: 1-ply and 2-ply continuation history
- **Quiescence**: SEE-based pruning
- **Time management**: complexity and volatility bonuses
- **SMP**: Lazy SMP with depth diversity

### Evaluation

Two modes, switchable at runtime via `UseNNUE`:

- **NNUE** (default): HalfKP 704×32→768×2→1 SCReLU, 32 per-square king buckets, embedded at compile time via `include_bytes!`
- **HCE**: tapered MG/EG with pawn hash, mobility, king safety, pawn structure (passed/doubled/isolated/backward/islands), threats, center control, connectivity, space, material imbalance, endgame scaling

### NNUE net embedding

`crates/chess-nnue/build.rs` copies `nets/default.nnue` into `OUT_DIR` at build time. If the file is missing or the wrong size it writes a zero buffer (engine falls back to HCE). Always `cargo build --release` after replacing the net file.

### Syzygy (`chess-engine/src/syzygy.rs`)

WDL probing via `pyrrhic-rs`. Fires at alpha-beta nodes when castling rights are gone and piece count ≤ loaded range. Returns Win/CursedWin/Draw/BlessedLoss/Loss. Guards against empty king bitboards before calling native code (necessary because the search uses pseudo-legal move generation and `panic = "abort"` is set).

### SPSA-tunable UCI options

In addition to the user-facing options in the README, the engine exposes search parameters (`LmrBase`, `LmrDiv`, `HistLmrDiv`, `RfpMarginImp`, `RfpMarginNoImp`, `FutMarginImp`, `FutMarginNoImp`, `SeeQuietMargin`) as UCI spin options for SPSA tuning. Run `uci` to see the current set with min/max/default; they are intentionally omitted from the user docs because they're tuning-only knobs.

## Build

```bash
cargo build --release              # endspiel binary
cargo build --release --workspace  # all binaries including datagen and tuner
```

### Native CPU optimisation

`.cargo/config.toml` is gitignored (machine-specific). For local AVX2/AVX-512/SIMD:

```toml
# .cargo/config.toml  (do not commit)
[build]
rustflags = ["-C", "target-cpu=native"]
```

### Release build matrix

CI (`.github/workflows/release.yml`, manual `workflow_dispatch` on a tag) builds
the following variants. x86-64 ships three micro-architecture tiers (v2–v4):

| Artifact | `target-cpu` | PGO | Notes |
|----------|--------------|-----|-------|
| `endspiel-linux-x64-v4` | `x86-64-v4` | no | AVX-512 (Zen 4/5, recent Xeon/Core) |
| `endspiel-linux-x64-v3` | `x86-64-v3` | yes | default Linux build (AVX2, ~2013+) |
| `endspiel-linux-x64-v2` | `x86-64-v2` | yes | SSE4.2 + POPCNT (no-AVX2 CPUs) |
| `endspiel-win-x64-v4.exe` | `x86-64-v4` | no | AVX-512 Windows build |
| `endspiel-win-x64-v3.exe` | `x86-64-v3` | yes | default Windows build |
| `endspiel-win-x64-v2.exe` | `x86-64-v2` | yes | SSE4.2 + POPCNT Windows build |
| `endspiel-win-arm64.exe` | `generic` | no | cross-built, no PGO |
| `endspiel-mac-arm64` | `apple-m1` | yes | macOS Apple Silicon |
| `endspiel-linux-arm64-pi5` | `cortex-a76` | yes | Raspberry Pi 5 (Raspberry Pi OS Trixie / Debian 13 or newer, glibc ≥ 2.39) |
| `endspiel-android-arm64.apk` | `generic` | no | Android arm64-v8a, minSdk 24 — Open Exchange engine APK (see `android/oex/`) |

PGO is a two-stage build: an instrumented binary is built with
`-Cprofile-generate`, then `endspiel bench` is run against it to produce
profile data, and a final build is done with `-Cprofile-use`. PGO is
skipped for the AVX-512 variants (the runner CPU may not support AVX-512
and the instrumented binary would crash with SIGILL) and for the
cross-built `win-arm64` target (can't execute on the x64 runner).

### Releasing a new version

Follow the Rust/Cargo convention of a **post-release `-dev` bump**, so
main always advertises a version that is unambiguously *not* a release.

1. On main, set `workspace.package.version` in `Cargo.toml` to the
   release version (drop the `-dev` suffix), e.g. `1.0.1-dev` → `1.0.1`.
2. Commit (`chore: release 1.0.1`), tag (`git tag v1.0.1`), push tag.
   The tag push does **not** trigger CI — `release.yml` is manual-only.
   Dispatch it explicitly with the tag as input to build the artifact
   matrix above and publish the release:
   `gh workflow run release.yml -f tag=v1.0.1` (or Actions > Release >
   Run workflow in the GitHub UI).
3. **Immediately** bump to the next patch with a `-dev` suffix
   (`1.0.1` → `1.0.2-dev`) as a separate commit
   (`chore: bump version to 1.0.2-dev`).

Pick the next *minor* (`1.1.0-dev`) instead of the next patch only when
the work already queued for the next cycle is known to be minor-worthy.

## Testing

```bash
cargo test --release --workspace
```

## Clippy

```bash
cargo clippy --workspace --all-targets
```

All clippy warnings must be resolved before merging. `#[allow(...)]` attributes are permitted only where the lint produces a false positive — add a comment explaining why.

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org):

```
<type>(<scope>): <short summary>

[optional body]
[optional footer]
```

Types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`, `ci`.
Scope is the crate short name, e.g. `engine`, `nnue`, `datagen`, `uci`.

Examples:
```
feat(engine): add passed pawn push extension
fix(nnue): correct column-major weight indexing for hidden layer 2
perf(datagen): raise default --hash to 16 MB
docs: restructure README for end users
```

Breaking changes: append `!` after the type/scope and add a `BREAKING CHANGE:` footer.

## Pull Requests

- All tests pass (`cargo test --release --workspace`)
- No clippy warnings (`cargo clippy --workspace --all-targets`)
- Bench node count is included in the PR description if search logic changed (see *Bench as a search-change diff* below)
- One logical change per PR — separate refactors from feature additions
- PR title follows the same Conventional Commits format as commit messages

### Bench as a search-change diff

`endspiel bench` runs a depth-14 search across 7 fixed positions on 1
thread with a fixed hash size. Because every input is pinned, the
total node count it prints is **deterministic** — run it twice on the
same binary and you get the same number.

This makes bench the standard quick check for "did my change actually
alter the search tree?":

1. Build the engine on the parent commit, run `endspiel bench`, note `Nodes: X`
2. Build the engine on your branch, run `endspiel bench`, note `Nodes: Y`
3. Interpret:
   - **X == Y** — your change did not affect the search tree at all.
     Either it's a behaviour-preserving refactor (good for a cleanup
     PR), or your change is dead / guarded behind a condition that
     never fires (bad for a feature PR — investigate).
   - **X != Y** — your change altered the search. Whether that's an
     *improvement* still needs game testing, but at least you know
     it's doing something.

Include the new node count in the PR description as
`Bench: <number>` — the chess engine convention.

### Promoting a New NNUE Net

A change that replaces `crates/chess-nnue/nets/default.nnue` must demonstrate
that the candidate is stronger than the current embedded net. The promotion
decision is made **head-to-head against the current embedded net**, by LOS:

1. **Promotion gate — self-play vs the embedded net** — fastchess match,
   candidate vs the current `default.nnue`:
   - At least **500 games** at `tc=10+0.1`, `Hash=64`, `Threads=1`
   - Promote only if the candidate wins with **LOS ≥ 99%**
   - Paste the final `Games / Wins / Losses / Draws / Elo / LOS` line in the PR

2. **Architecture / size changes** — if the net file size or layout changed,
   note it in the PR (the size is also checked by `crates/chess-nnue/build.rs`).

3. **WDL refit** — if the new net shifts the win-rate ↔ centipawn mapping,
   re-fit and include the updated `WDL_A` / `WDL_B` values in the PR.

The PR description, not this guide, is where the judgement call to ship lives.

## Syzygy Tablebases

Download 3–5 man tables (~350 MB):

```bash
bash scripts/download_syzygy.sh             # WDL + DTZ
bash scripts/download_syzygy.sh --wdl-only  # WDL only (~150 MB)
```

Files land in `assets/syzygy/` (gitignored).

Manual probe test (KRK, should return 28000 cp from depth 1):

```bash
(printf "uci\nisready\nsetoption name SyzygyPath value assets/syzygy\nucinewgame\nposition fen 8/8/8/8/4K3/8/4R3/7k w - - 0 1\ngo movetime 500\n"; sleep 2) \
  | ./target/release/endspiel
```

## Remark — HCE Tuning (`chess-tuner`)

> The hand-crafted evaluation is largely superseded by NNUE. It still lives
> in the tree as a fallback (`UseNNUE=false`, and when the embedded net is
> zeroed by `build.rs`), and the tuner below is kept for completeness, but
> it is not part of the active improvement path. New PRs should target the
> NNUE net (see *Promoting a New NNUE Net* above) rather than HCE
> parameters.

```bash
cargo build --release -p chess-tuner
target/release/chess-tuner --data assets/lichess_db_eval.jsonl.zst --epochs 200 --output params.json
target/release/chess-tuner --apply params.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data PATH` | `games/lichess_db_eval.jsonl.zst` | Eval dataset (jsonl.zst) |
| `--positions N` | 2000000 | Max positions to load |
| `--epochs N` | 200 | Tuning epochs |
| `--min-depth N` | 30 | Minimum dataset depth filter |
| `--output PATH` | — | Save parameters JSON |
| `--apply PATH` | — | Write parameters into engine source |
| `--tune-material` | off | Also tune material values |
| `--learning-rate F` | 2.0 | Optimizer LR |

`chess-tuner` has several additional flags (PST/mobility freezing, filtering
thresholds, SF cross-check, convergence loop). Run `chess-tuner --help` for
the full list.
