# Credits & Acknowledgements

endspiel is an original chess engine, but it stands on the work of others — both
software it depends on and ideas it borrows. This file documents those sources.

## NNUE training

- **[bullet](https://github.com/jw1912/bullet)** (MIT) by Jamie Whiting — the
  trainer used to produce endspiel's NNUE networks. The engine implements its
  own network loading and inference code; bullet is not linked into or
  distributed with endspiel.

## Training data

- **[Leela Chess Zero](https://lczero.org)** — since 2026-09, endspiel's net is
  trained partly on position scores from Leela's public **T80** training data
  (January–June 2024, via the Stockfish-binpack conversion published as
  [`linrock/test80-2024`](https://huggingface.co/datasets/linrock/test80-2024)).
  Only positions and scores are used, as score-only labels next to endspiel's own
  self-play data; no Leela network or code is included. Thanks to the Leela
  project and everyone who contributed games to it.

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

## Android packaging

- **[Chess Engine Support Library](https://github.com/gkalab/chessenginesupport-androidlib)**
  (Apache-2.0) by Gerhard Kalab — the `ChessEngineProvider` class used by the
  Open Exchange (OEX) engine APK (`android/oex/`) is vendored from this library
  so chess GUIs (DroidFish, Chess for Android) can discover the bundled engine.
  Only that one file is included; the Apache-2.0 header is retained verbatim.

## Other notable dependencies

All permissive (MIT / Apache-2.0 / BSD-family): `rayon`, `zstd`, `serde`,
`serde_json`, `sysinfo`, `thiserror`, `log`, `env_logger`.

## Development

A substantial share of endspiel's initial implementation, debugging, and
tooling was carried out with
**[Claude Code](https://code.claude.com/docs/en/cli-usage)**
and further improved using other AI coding assistants, including
**[Codex](https://learn.chatgpt.com/docs/codex/cli)**,
**[Grok](https://x.ai/cli)**, and
**[OpenCode](https://opencode.ai/docs/cli/)**.
