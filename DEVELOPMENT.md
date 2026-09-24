# Endspiel — Developer Guide

## Layout

```
├── Cargo.toml                    # Workspace root + endspiel binary
├── src/main.rs                   # Entry point
├── crates/
│   ├── chess-common/             # Shared types: Board, Move, Bitboard, FEN
│   ├── chess-core/               # Move generation, attack tables, validation
│   ├── chess-engine/             # Search, HCE evaluation, Syzygy WDL probing
│   ├── chess-nnue/               # NNUE inference + embedded net (build.rs)
│   └── chess-uci/                # UCI protocol handler
├── scripts/                      # build & setup helpers (Android, Syzygy download)
└── assets/                       # Gitignored — local resources
```

## Architecture

### Search (`chess-engine`)

Alpha-beta with iterative deepening and PVS:

- **Pruning**: null move, reverse futility, futility, razoring, SEE (captures + quiets), history pruning, ProbCut
- **Extensions**: passed-pawn push (7th rank / promotion). No check extension: an in-check node is only floored at depth 1, and checking moves are merely exempt from LMR and quiet pruning. A 2026-09 probe of a real check extension was mixed on the 47...Rh1 family and grew bench nodes ~58%. Singular extension is disabled (`singular_ext_mode = 0`): a 2026-09 h2h (same net, 10+0.1, 500 rounds, `default.nnue`) measured it at −18.4 ± 11.6 elo with LOS 0.09% for conservative (mode 1) vs off
- **Reductions**: LMR, IIR
- **Move ordering**: TT → good captures (MVV-LVA) → killers → counter → history-sorted quiets → bad captures
- **Quiescence**: SEE-based pruning; **SMP**: Lazy SMP with depth diversity

### Evaluation

Two backends:

- **NNUE** (default): HalfKP 785×32→(1024 pairwise 512)×2→16→32→1, 32 king buckets × 8 material output stacks. Embedded at compile time via `include_bytes!`; dense L1/L2 read from the net header (`1..=64`) so architecture-trial nets load without a rebuild.
- **HCE**: tapered MG/EG with pawn, mobility, king safety, pawn structure, threat, center, connectivity, space, and material-imbalance terms. Fallback when the embedded net is zeroed by `build.rs`. Superseded by NNUE; HCE parameter work is out of scope for new PRs.

### NNUE net embedding

`crates/chess-nnue/build.rs` copies `nets/default.nnue` into `OUT_DIR`. Missing/wrong size → zero buffer (HCE fallback). Always `cargo build --release` after replacing the net.

### Chess960

Rook origins live on `Board::castle_rooks` unless a FEN overrides them; internal castling still moves king→c/g and rook→d/f. FEN parsing accepts KQkq, X-FEN, and Shredder (`AHah`); emission is X-FEN. With `UCI_Chess960=true`, the UCI layer prints and parses king-takes-own-rook moves.

### Syzygy (`chess-engine/src/syzygy.rs`)

WDL probing via `pyrrhic-rs` at alpha-beta nodes when castling rights are gone and piece count ≤ loaded range. Empty-king-bitboard guard exists because move generation is pseudo-legal and `panic = "abort"` is set.

## Build

```bash
cargo build --release              # endspiel binary
```

### Native CPU optimisation / release contract

`.cargo/config.toml` is gitignored. For machine-local AVX/AVX-512 builds:

```toml
[build]
rustflags = ["-C", "target-cpu=native"]
```

**Release x86-64 binaries must stay on `-C target-cpu=x86-64-v2`** (SSE4.2 + POPCNT baseline). Advancing the whole binary to v3/v4/native defeats its portability floor. Only individually dispatched kernels (`#[target_feature]`) may use AVX2/AVX-512, with a portable reference path kept SSE4.2/POPCNT-clean and a scalar-equivalence test for each dispatched kernel. NNUE follows this pattern (baseline + AVX2 + AVX-512 variants, runtime dispatch; `endspiel bench` prints the selected tier).

### Release matrix

CI (`release.yml`, manual `workflow_dispatch` on tag):

| Artifact | target | LTO | PGO |
|----------|--------|-----|-----|
| linux-x64 / win-x64 | x86-64-v2 | thin | yes |
| win-arm64 | generic | thin | no (cross-built) |
| mac-arm64 | apple-m1 | thin | yes |
| linux-arm64-pi5 (Raspberry Pi 5, glibc ≥ 2.39) | cortex-a76 | fat | yes |
| android-arm64.apk | generic | thin | no (see `android/oex/`) |

### Releasing a version

Post-release `-dev` bump: main is always *not* a release.

1. On main, `workspace.package.version` → drop `-dev` (`1.0.1-dev` → `1.0.1`).
2. Commit `chore: release 1.0.1`, tag `v1.0.1`, push tag. **Tag push does NOT trigger CI** — dispatch manually with the tag as input:
   `gh workflow run release.yml -f tag=v1.0.1` (or Actions > Release > Run workflow).
3. Immediately bump to next patch `-dev` (`1.0.1` → `1.0.2-dev`) as a separate commit.

Pick a minor bump only when the queued work for the next cycle is known to be minor-worthy.

## Testing / Lint

```bash
cargo test --release --workspace          # required before merge
cargo clippy --workspace --all-targets    # zero warnings; #[allow] only for proven false positives, with a comment
cargo fmt                                 # on change
```

## Commit / PR conventions

[Conventional Commits](https://www.conventionalcommits.org) **one-liners only** — a single subject line, no body or footers: `<type>(<scope>): <summary>` with types `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`, `ci`; scope is the crate short name (`engine`, `nnue`, `uci`, …).

PR checklist:
- Tests green, clippy clean
- `Bench: <number>` in description when search logic changed (see below)
- One logical change per PR; title matches commit format

### Bench as a search-change diff

`endspiel bench` = depth-14 over 7 pinned positions, 1 thread, fixed hash → **deterministic** node count, used to detect whether the search tree changed:

1. `endspiel bench` on parent commit → `Nodes: X`; build branch, same → `Nodes: Y`.
2. `X == Y`: behaviour-neutral or dead/guarded code (investigate for features). `X != Y`: tree changed — improvement still needs game testing.

## NNUE net promotion gate

Replacing `crates/chess-nnue/nets/default.nnue` requires a fastchess self-play,
**candidate vs current embedded net**, `tc=10+0.1`, `Hash=64`, `Threads=1`,
**≥ 500 games**: promote only at **LOS ≥ 99%**. Paste the final `Games/Wins/Losses/Draws/Elo/LOS` line in the PR; note any architecture size change (`build.rs` checks it) and include updated `WDL_A`/`WDL_B` if the win-rate ↔ centipawn mapping shifted.

## Syzygy Tablebases

```bash
bash scripts/download_syzygy.sh             # WDL + DTZ (~350 MB)
bash scripts/download_syzygy.sh --wdl-only  # ~150 MB
```
Lands in `assets/syzygy/` (gitignored). KRK probe sanity check (`go movetime 500` should return 28000 cp from depth 1):

```bash
(printf "uci\nisready\nsetoption name SyzygyPath value assets/syzygy\nucinewgame\nposition fen 8/8/8/8/4K3/8/4R3/7k w - - 0 1\ngo movetime 500\n"; sleep 2) \
  | ./target/release/endspiel
```
