# About Endspiel

Endspiel is a **ground-up chess engine and a live experiment in
AI-assisted development**, with two goals running in parallel:

1. **Build a competitive chess engine end-to-end from scratch in Rust** —
   bitboards and move generation, alpha-beta search with modern pruning
   and reductions, an NNUE evaluation trained from scratch, a UCI front
   end, and Syzygy probing.

2. **See how far AI-assisted coding (specifically [Claude Code](https://www.anthropic.com/claude-code))
   actually pushes a solo developer.** A chess engine of this scope —
   move generator, modern search, an NNUE trained from scratch on
   billions of self-play positions, SPSA tuner, Syzygy integration, UCI
   compliance, cross-platform CI — would conventionally be a multi-year
   effort for a small team. Built solo with Claude Code as a pair-programming
   partner, it came together in weeks. Whether that is the right comparison
   or not, the experience itself was a major part of the point: figuring
   out which tasks AI accelerates, which it changes the shape of, and
   which still need a human at the wheel.

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

A Rust NNUE engine on Lichess is not unusual. Endspiel's bet is a
from-scratch chess stack (no board library), a layer-stacked king-bucketed
net rather than a tiny `(768→N)×2→1` perspective net, and binaries meant
to be installed — AVX-512 / AVX2 / SSE, Raspberry Pi 5, Android — rather
than one `cargo build` on a VPS. Search is the usual modern toolkit,
reimplemented; see [CREDITS.md](CREDITS.md). Strength is well above human
play and short of the top of the engine lists.

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
