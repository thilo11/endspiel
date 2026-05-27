# About Endspiel

Endspiel is first and foremost a **learning and craft project**, with two
goals running in parallel:

1. **Build a competitive chess engine end-to-end from scratch in Rust** —
   bitboards and move generation, alpha-beta search with modern pruning
   and reductions, an NNUE evaluation with its own training pipeline, a
   UCI front end, Syzygy probing, and a self-play data generator.

2. **See how far AI-assisted coding (specifically [Claude Code](https://www.anthropic.com/claude-code))
   actually pushes a solo developer.** A chess engine of this scope —
   move generator, modern search, an NNUE trained from scratch on
   billions of self-play positions, datagen + cleaning + mixing pipeline,
   SPSA tuner, Syzygy integration, UCI compliance, cross-platform CI —
   would conventionally be a multi-year effort for a small team. Built
   solo with Claude Code as a pair-programming partner, it came together
   in weeks. Whether that is the right comparison or not, the experience
   itself was a major part of the point: figuring out which tasks AI
   accelerates, which it changes the shape of, and which still need a
   human at the wheel.

## Playing strength and where it sits

The engine plays a strong game. It is well above the level of any human
player — including masters — and is comfortably useful as a sparring
partner and for position analysis.

As a rough indication (not a controlled benchmark), a recent match
against Stockfish limited to `UCI_Elo=3000`, played at `tc=10+0.1`,
1 thread, 64 MB hash:

```
Elo: +76.79 ± 28.05    nElo: +96.53 ± 34.05
LOS: 100.00 %          DrawRatio: 36.50 %     PairsRatio: 2.26
Games: 400  W 194  L 107  D 99   Points 243.5 (60.88 %)
Ptnml(0-2): [7, 32, 73, 43, 45]  WL/DD: 5.08
```

If you want the top of the Rust-engine charts, look at
[Reckless](https://github.com/codedeliveryservice/Reckless) or a
Stockfish derivative; if you want a readable, fully-formed Rust engine
that you can also actually play and analyse with, you're in the right
place.

## Training data

The embedded NNUE was trained from scratch on the engine's own self-play
output. The current training archive is **~2.6 billion positions** drawn
from roughly **25–30 million self-play games** (most generated at search
depth 10–12, with a smaller depth-12 set; openings sampled from random
prefixes and from Lichess opening positions). Every round of training
goes back to scratch on the full accumulated archive — there is no
fine-tune step in the active pipeline.

Crucially, the network learns from **no external evaluation data**. Unlike
most NNUE engines, it uses no Stockfish or Leela Chess Zero labels: every
training target is the engine's own search score on its own games. The
only outside ingredient is a set of raw opening positions (random prefixes
and Lichess opening FENs) used purely as self-play *starting points* —
they seed the games, never the training targets. The trainer itself is the
open-source [Bullet](https://github.com/jw1912/bullet) framework and
Syzygy probing uses `pyrrhic-rs`; everything else — bitboards and move
generation, search, NNUE inference, the datagen and data-cleaning
pipeline, and the UCI layer — is hand-written from scratch.

## Where the strength comes from now

The foundations are in place — move generation, modern search, NNUE
inference, the full training pipeline (datagen → clean → mix → train →
quantise → embed), Syzygy probing, SPSA tuning. From this point, closing
the remaining gap to the top of the Rust-engine charts is **mostly a
question of data and compute**, not of new code: more self-play games at
higher depth, larger archives, more training superbatches, more SPSA
iterations. The engine is structured to absorb that — every additional
billion positions and every additional training round goes through the
same pipeline and the embedded net is swapped in via a single rebuild.

## Further reading

- [README.md](README.md) — install, run, UCI options
- [TRAINING.md](TRAINING.md) — data generation and NNUE training pipeline
- [DEVELOPMENT.md](DEVELOPMENT.md) — architecture, build, contributing
