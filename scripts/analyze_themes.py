#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = ["zstandard>=0.25.0"]
# ///
"""Compare per-theme solve rate against baseline, using a fails tsv."""

import argparse
import csv
import io
import math
import sys

import zstandard as zstd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", default="assets/lichess_db_puzzle.csv.zst")
    ap.add_argument("--fails", required=True)
    ap.add_argument("--min-popularity", type=int, default=99)
    ap.add_argument("--min-rating", type=int, default=2200)
    ap.add_argument("--min-count", type=int, default=30)
    args = ap.parse_args()

    fail_ids = set()
    with open(args.fails) as f:
        for line in f:
            pid = line.split("\t", 1)[0].strip()
            if pid:
                fail_ids.add(pid)

    per_theme_total: dict[str, int] = {}
    per_theme_fail: dict[str, int] = {}
    total = 0
    total_fail = 0

    with open(args.puzzles, "rb") as raw:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(raw) as zr:
            text = io.TextIOWrapper(zr, encoding="utf-8", newline="")
            reader = csv.reader(text)
            header = next(reader)
            idx = {n: i for i, n in enumerate(header)}
            for row in reader:
                pop = int(row[idx["Popularity"]])
                rating = int(row[idx["Rating"]])
                if pop < args.min_popularity or rating < args.min_rating:
                    continue
                pid = row[idx["PuzzleId"]]
                themes = row[idx["Themes"]].split()
                failed = pid in fail_ids
                total += 1
                if failed:
                    total_fail += 1
                for t in themes:
                    per_theme_total[t] = per_theme_total.get(t, 0) + 1
                    if failed:
                        per_theme_fail[t] = per_theme_fail.get(t, 0) + 1

    base = total_fail / total if total else 0.0
    print(f"Baseline fail rate: {total_fail}/{total} = {100*base:.2f}%")
    print()

    rows = []
    for t, n in per_theme_total.items():
        if n < args.min_count:
            continue
        f_ = per_theme_fail.get(t, 0)
        rate = f_ / n
        # Wilson-ish z-score vs baseline for ordering
        se = math.sqrt(base * (1 - base) / n)
        z = (rate - base) / se if se > 0 else 0
        rows.append((t, n, f_, rate, z))

    rows.sort(key=lambda r: r[3], reverse=True)
    print(f"{'theme':<22} {'N':>6} {'fails':>6} {'fail%':>7} {'z':>6}")
    print("-" * 50)
    for t, n, f_, rate, z in rows:
        print(f"{t:<22} {n:>6} {f_:>6} {100*rate:>6.1f}% {z:>6.1f}")


if __name__ == "__main__":
    sys.exit(main())
