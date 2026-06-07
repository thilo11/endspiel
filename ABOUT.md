# About Endspiel

Endspiel is a **ground-up chess engine and a live experiment in
AI-assisted development**, with two goals running in parallel:

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

As a rough indication, a 400-game match against **Stockfish 17.1**
(Ubuntu package `17-1build1`) at a fixed `Skill Level 18`, played at
`tc=10+0.1`, 1 thread, 64 MB hash, openings from `openings.epd`:

```
Elo: -54.29 ± 24.39    nElo: -77.17 ± 34.05
LOS: 0.00 %            DrawRatio: 39.50 %     PairsRatio: 0.46
Games: 400  W 97  L 159  D 144   Points 169.0 (42.25 %)
Ptnml(0-2): [24, 59, 79, 31, 7]  WL/DD: 1.93
```

The engine trails a near-full-strength Stockfish but is still far above
human play. Strength is anchored to a fixed Stockfish `Skill Level`
against a pinned Stockfish version, rather than `UCI_Elo` /
`UCI_LimitStrength`, whose scale is RNG-calibrated and drifts between
Stockfish releases (so its numbers aren't comparable over time).

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
- [DEVELOPMENT.md](DEVELOPMENT.md) — architecture, build, contributing
