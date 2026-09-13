# About Endspiel

Endspiel is a **ground-up chess engine and a live experiment in
AI-assisted development**, with two goals running in parallel:

1. **Build a competitive chess engine end-to-end from scratch in Rust** —
   bitboards and move generation, alpha-beta search with modern pruning
   and reductions, an NNUE evaluation trained from scratch, a UCI front
   end, and Syzygy probing.

2. **See how far AI-assisted coding can push a solo developer.** A chess engine
   of this scope — move generator, modern search, an NNUE trained from scratch on
   billions of self-play positions, SPSA tuner, Syzygy integration, UCI
   compliance, cross-platform CI — would conventionally be a multi-year
   effort for a small team. The initial implementation was built solo with
   [Claude Code](https://www.anthropic.com/claude-code) as a pair-programming
   partner and came together in weeks; subsequent development also used Codex,
   Grok, and OpenCode. Whether that is the right comparison or not, the
   experience itself was a major part of the point: figuring out which tasks AI
   accelerates, which it changes the shape of, and which still need a human at
   the wheel.

## Playing strength and where it sits

The engine plays a strong game. It is well above the level of any human
player — including masters — and is comfortably useful as a sparring
partner and for position analysis.

Endspiel's bet is a from-scratch chess stack (no board library), a
layer-stacked king-bucketed net rather than a tiny `(768→N)×2→1` perspective
net, and binaries meant to be installed — a universal runtime-dispatched
x86-64 executable, Raspberry Pi 5, and Android builds — rather than one
`cargo build` on a VPS.
The x86-64 release has a portable v2 baseline and automatically uses AVX2 or
AVX-512-class NNUE kernels when the host supports them. Search is the usual
modern toolkit, reimplemented; see [CREDITS.md](CREDITS.md). Strength is well
above human play and short of the top of the engine lists.

## Training data

The bulk of the archive is Endspiel self-play: the engine's own games,
labelled with its own search scores. The current set is on the order of
**billions of positions** from tens of millions of games (mostly depth
10–12; openings from random prefixes and Lichess starting FENs). Training
runs from scratch on the accumulated mix — there is no fine-tune step in
the active pipeline.

That is not “zero external data.” Opening FENs seed games; they are not
eval targets. The current embedded net's mix also includes a public eval
dump as one source. What we do *not* do is ship someone else's network
file or train only on another engine's labels. The trainer is
[Bullet](https://github.com/jw1912/bullet); Syzygy probing is
`pyrrhic-rs`. Everything else — bitboards, move generation, search, NNUE
inference, UCI — is hand-written.

## Where the strength comes from now

The foundations are in place — move generation, modern search, NNUE
inference, Syzygy probing, SPSA tuning. From this point, closing
the remaining gap to the top of the Rust-engine charts is **mostly a
question of data and compute**, not of new code: more self-play games at
higher depth, larger archives, more training superbatches, more SPSA
iterations. The engine is structured to absorb that — every additional
billion positions and every additional training round goes through the
same pipeline and the embedded net is swapped in via a single rebuild.

## Further reading

- [README.md](README.md) — install, run, UCI options
- [DEVELOPMENT.md](DEVELOPMENT.md) — architecture, build, contributing
