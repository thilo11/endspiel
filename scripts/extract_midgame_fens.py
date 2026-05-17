#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "zstandard>=0.25.0",
# ]
# ///
"""Extract near-equal late-midgame FENs from `assets/lichess_db_eval.jsonl.zst`.

Reads JSONL (one {fen, evals: [...]} per line), keeps positions whose piece
count falls in a given range (proxy for game phase, since the eval DB has no
move numbers) and whose deepest engine analysis has |cp| <= max-abs-cp.
Mate scores are always skipped. Writes one FEN per line; scores are dropped
since the output is only used as self-play starting positions.

Typical use:
    scripts/extract_midgame_fens.py \\
        --output assets/midgame_fens.txt
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time

import zstandard as zstd


def count_pieces_fast(fen: str) -> int:
    board_part = fen.split(" ", 1)[0]
    return sum(1 for c in board_part if c.isalpha())


def best_cp(obj: dict) -> int | None:
    """Return the deepest eval record's first PV cp (best move).

    Returns None if no usable cp is present (missing evals, mate-only lines).
    """
    evals = obj.get("evals")
    if not evals:
        return None
    best = max(evals, key=lambda e: e.get("depth", 0))
    pvs = best.get("pvs")
    if not pvs:
        return None
    pv0 = pvs[0]
    cp = pv0.get("cp")
    if cp is None:
        # "mate" entries are never "almost equal" — drop them.
        return None
    return int(cp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="assets/lichess_db_eval.jsonl.zst")
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-pieces", type=int, default=20)
    ap.add_argument("--max-pieces", type=int, default=28)
    ap.add_argument("--max-abs-cp", type=int, default=50,
                    help="Skip entries whose deepest-eval cp exceeds this magnitude")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after N kept FENs (default: unlimited)")
    ap.add_argument("--dedupe", action="store_true",
                    help="Skip duplicate FENs (board+stm only)")
    ap.add_argument("--progress-every", type=int, default=1_000_000)
    args = ap.parse_args()

    seen: set[str] = set()
    kept = 0
    lines = 0
    skipped_pieces = 0
    skipped_eval = 0
    skipped_dup = 0
    t0 = time.time()

    out = open(args.output, "w", encoding="utf-8")
    try:
        with open(args.input, "rb") as raw:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(raw) as zr:
                text = io.TextIOWrapper(zr, encoding="utf-8", newline="")
                for line in text:
                    lines += 1
                    # Fast FEN extract — json.loads only if piece-count passes.
                    i = line.find('"fen":"')
                    if i < 0:
                        continue
                    j = line.find('"', i + 7)
                    if j < 0:
                        continue
                    fen = line[i + 7:j]

                    pieces = count_pieces_fast(fen)
                    if pieces < args.min_pieces or pieces > args.max_pieces:
                        skipped_pieces += 1
                        continue

                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        skipped_eval += 1
                        continue
                    cp = best_cp(obj)
                    if cp is None or abs(cp) > args.max_abs_cp:
                        skipped_eval += 1
                        continue

                    # lichess FENs are 4-field; pad to 6 so chess-datagen's
                    # Board::from_fen accepts them directly.
                    parts = fen.split()
                    if len(parts) == 4:
                        fen_full = fen + " 0 1"
                    else:
                        fen_full = fen

                    if args.dedupe:
                        key = " ".join(fen_full.split()[:2])
                        if key in seen:
                            skipped_dup += 1
                            continue
                        seen.add(key)

                    out.write(fen_full)
                    out.write("\n")
                    kept += 1

                    if args.limit and kept >= args.limit:
                        break

                    if lines % args.progress_every == 0:
                        elapsed = time.time() - t0
                        rate = lines / elapsed if elapsed > 0 else 0
                        print(
                            f"  lines={lines:,} kept={kept:,} "
                            f"rate={rate/1e3:.1f}k/s elapsed={elapsed/60:.1f}min",
                            file=sys.stderr, flush=True,
                        )
    finally:
        out.close()

    elapsed = time.time() - t0
    print(file=sys.stderr)
    print("=== STATS ===", file=sys.stderr)
    print(f"  lines read       : {lines:,}", file=sys.stderr)
    print(f"  skipped (pieces) : {skipped_pieces:,}", file=sys.stderr)
    print(f"  skipped (eval)   : {skipped_eval:,}", file=sys.stderr)
    print(f"  skipped (dup)    : {skipped_dup:,}", file=sys.stderr)
    print(f"  kept             : {kept:,}", file=sys.stderr)
    print(f"  elapsed          : {elapsed/60:.1f} min", file=sys.stderr)
    print(f"  output           : {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
