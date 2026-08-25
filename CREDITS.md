# Credits & Acknowledgements

endspiel is an original chess engine, but it stands on the work of others — both
software it depends on and ideas it borrows. This file documents those sources.

## NNUE training

- **[bullet](https://github.com/jw1912/bullet)** (MIT) by Jamie Whiting — the
  trainer used to produce endspiel's NNUE networks. Weight loading
  deliberately matches bullet's on-disk formats, via the **`bulletformat`**
  crate.

## Endgame tablebase probing

- **[pyrrhic-rs](https://github.com/Algorhythm-sxv/pyrrhic-rs)** (MIT) by
  Algorhythm-sxv — the Syzygy probing code linked into endspiel. It is a Rust
  transliteration of the C **Pyrrhic** / **Fathom** library (Fathom © 2015 basil;
  modifications © 2016–2019 Jon Dart, © 2020 Andrew Grant). The tablebase files
  themselves are a separate, user-provided install — not shipped with endspiel.

## Search & evaluation inspiration (techniques, not code)

endspiel's alpha-beta search uses ideas that are common knowledge in the engine
community, several of which were pioneered or popularised by
**[Stockfish](https://github.com/official-stockfish/Stockfish)** (GPL-3.0):

- Lazy SMP with per-helper-thread depth diversity ("Stockfish-style" offsets);
- the usual pruning/reduction toolkit — null-move pruning, late move reductions,
  futility / reverse-futility pruning, razoring, singular extensions, and
  correction & continuation history;
- the displayed-centipawn convention (~100 cp ≈ one "WDL pawn").

These are algorithmic ideas and conventions, re-implemented from scratch in Rust.
**No Stockfish source code is included or ported into endspiel.**

## External tools (separate processes, not linked)

- **Stockfish** is optionally invoked as an *external binary* by the offline
  `chess-tuner` to cross-check endspiel's evaluations during net validation. This
  is a subprocess call to a separate program; no Stockfish code is linked into or
  distributed with endspiel.

## Android packaging

- **[Chess Engine Support Library](https://github.com/gkalab/chessenginesupport-androidlib)**
  (Apache-2.0) by Gerhard Kalab — the `ChessEngineProvider` class used by the
  Open Exchange (OEX) engine APK (`android/oex/`) is vendored from this library
  so chess GUIs (DroidFish, Chess for Android) can discover the bundled engine.
  Only that one file is included; the Apache-2.0 header is retained verbatim.

## Other notable dependencies

All permissive (MIT / Apache-2.0 / BSD-family): `rayon`, `zstd`, `rand`,
`serde_json`, `thiserror`, `log`, `env_logger`.

## Development

A substantial share of endspiel's implementation, debugging, and tooling was
carried out with **[Claude Code](https://claude.com/claude-code)**.
