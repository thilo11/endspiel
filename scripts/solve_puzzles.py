#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "chess>=1.11.2",
#   "zstandard>=0.25.0",
# ]
# ///
"""Run lichess puzzles through the engine and report solve rate.

Puzzle format (lichess): PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,...
FEN is the position BEFORE the opponent's setup move.
Moves[0] is the opponent's move; Moves[1..] are the player's solution moves,
alternating opponent responses in between.

All player moves are "only moves" except: if the last player move delivers mate,
any mating move is accepted.
"""

from __future__ import annotations

import argparse
import csv
import io
import multiprocessing as mp
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import chess
import chess.engine
import zstandard as zstd


@dataclass
class Puzzle:
    puzzle_id: str
    fen: str
    moves: list[str]
    rating: int
    popularity: int


def iter_puzzles(path: str, min_popularity: int, min_rating: int, limit: int | None):
    with open(path, "rb") as raw:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(raw) as zr:
            text = io.TextIOWrapper(zr, encoding="utf-8", newline="")
            reader = csv.reader(text)
            header = next(reader)
            idx = {name: i for i, name in enumerate(header)}
            n = 0
            for row in reader:
                pop = int(row[idx["Popularity"]])
                if pop < min_popularity:
                    continue
                rating = int(row[idx["Rating"]])
                if rating < min_rating:
                    continue
                moves = row[idx["Moves"]].split()
                if len(moves) < 2:
                    continue
                yield Puzzle(
                    puzzle_id=row[idx["PuzzleId"]],
                    fen=row[idx["FEN"]],
                    moves=moves,
                    rating=rating,
                    popularity=pop,
                )
                n += 1
                if limit is not None and n >= limit:
                    return


def solve_puzzle(engine: chess.engine.SimpleEngine, puzzle: Puzzle, movetime_ms: int) -> tuple[bool, str]:
    """Return (solved, first-wrong-description-if-any)."""
    board = chess.Board(puzzle.fen)
    # Apply opponent's setup move.
    try:
        setup = chess.Move.from_uci(puzzle.moves[0])
    except ValueError:
        return False, f"bad setup uci {puzzle.moves[0]}"
    if setup not in board.legal_moves:
        return False, f"illegal setup {puzzle.moves[0]} in {puzzle.fen}"
    board.push(setup)

    limit = chess.engine.Limit(time=movetime_ms / 1000.0)

    solution = puzzle.moves[1:]
    for i, expected_uci in enumerate(solution):
        expected = chess.Move.from_uci(expected_uci)
        # Player-to-move indices are even (0, 2, ...); opponent replies at odd.
        if i % 2 == 0:
            result = engine.play(board, limit)
            played = result.move
            if played != expected:
                # Mate-in-one exception: if expected was final move and mates,
                # accept any mating move.
                is_last = i == len(solution) - 1
                if is_last:
                    board_copy = board.copy(stack=False)
                    board_copy.push(expected)
                    if board_copy.is_checkmate():
                        # Check whether engine's move also mates.
                        board_try = board.copy(stack=False)
                        if played in board.legal_moves:
                            board_try.push(played)
                            if board_try.is_checkmate():
                                return True, ""
                return False, f"move {i+1}: expected {expected_uci}, got {played.uci() if played else 'None'}"
            board.push(expected)
        else:
            # Opponent's forced reply from puzzle solution.
            if expected not in board.legal_moves:
                return False, f"illegal opp move {expected_uci}"
            board.push(expected)
    return True, ""


def worker(args_tuple):
    puzzles, engine_path, movetime_ms, threads, hash_mb, extra_opts, worker_id = args_tuple
    # Spawn engine once per worker.
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    cfg = {"Threads": threads, "Hash": hash_mb}
    cfg.update(extra_opts)
    try:
        engine.configure(cfg)
    except chess.engine.EngineError:
        pass
    results = []
    for p in puzzles:
        try:
            ok, why = solve_puzzle(engine, p, movetime_ms)
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as e:
            # Engine died; restart and mark as fail.
            try:
                engine.quit()
            except Exception:
                pass
            engine = chess.engine.SimpleEngine.popen_uci(engine_path)
            try:
                engine.configure(cfg)
            except chess.engine.EngineError:
                pass
            ok, why = False, f"engine error: {e}"
        results.append((p.puzzle_id, p.rating, p.popularity, ok, why))
    engine.quit()
    return results


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", default="assets/lichess_db_puzzle.csv.zst")
    ap.add_argument("--engine", default="target/release/endspiel")
    ap.add_argument("--min-popularity", type=int, default=99)
    ap.add_argument("--min-rating", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="cap number of puzzles")
    ap.add_argument("--movetime-ms", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1, help="Threads per engine")
    ap.add_argument("--hash-mb", type=int, default=64)
    ap.add_argument("--uci-option", action="append", default=[],
                    help="Extra UCI option name=value (repeatable)")
    ap.add_argument("--log-fails", default=None, help="write failing puzzle ids to this file")
    ap.add_argument("--progress-every", type=int, default=500)
    args = ap.parse_args()

    print(f"Loading puzzles (popularity >= {args.min_popularity}, rating >= {args.min_rating}, limit={args.limit})...", flush=True)
    t0 = time.time()
    puzzles = list(iter_puzzles(args.puzzles, args.min_popularity, args.min_rating, args.limit))
    print(f"Loaded {len(puzzles)} puzzles in {time.time()-t0:.1f}s", flush=True)

    if not puzzles:
        return 0

    # Split puzzles into chunks (not too many — one chunk per worker initially).
    chunk_size = max(50, len(puzzles) // (args.workers * 8))
    chunks = list(chunked(puzzles, chunk_size))
    print(f"Dispatching {len(chunks)} chunks of ~{chunk_size} to {args.workers} workers", flush=True)

    extra_opts: dict[str, object] = {}
    for spec in args.uci_option:
        if "=" not in spec:
            raise SystemExit(f"--uci-option must be name=value, got {spec!r}")
        name, _, val = spec.partition("=")
        name, val = name.strip(), val.strip()
        if val.lower() in ("true", "false"):
            extra_opts[name] = (val.lower() == "true")
        else:
            try:
                extra_opts[name] = int(val)
            except ValueError:
                extra_opts[name] = val

    task_args = [
        (chunk, args.engine, args.movetime_ms, args.threads, args.hash_mb, extra_opts, i)
        for i, chunk in enumerate(chunks)
    ]

    solved = 0
    total = 0
    fails: list[tuple[str, int, int, str]] = []
    t_start = time.time()
    by_rating_bucket = {}  # bucket -> (solved, total)
    with mp.Pool(args.workers) as pool:
        for results in pool.imap_unordered(worker, task_args):
            for pid, rating, pop, ok, why in results:
                total += 1
                if ok:
                    solved += 1
                else:
                    fails.append((pid, rating, pop, why))
                bucket = (rating // 200) * 200
                s, t = by_rating_bucket.get(bucket, (0, 0))
                by_rating_bucket[bucket] = (s + (1 if ok else 0), t + 1)
                if total % args.progress_every == 0:
                    elapsed = time.time() - t_start
                    rate = total / elapsed if elapsed > 0 else 0
                    eta = (len(puzzles) - total) / rate if rate > 0 else 0
                    print(f"  {total}/{len(puzzles)} solved={solved} ({100*solved/total:.1f}%) "
                          f"rate={rate:.1f}/s eta={eta/60:.1f}min", flush=True)

    elapsed = time.time() - t_start
    print()
    print(f"=== RESULT ===")
    print(f"Solved: {solved}/{total} = {100*solved/total:.2f}%")
    print(f"Elapsed: {elapsed:.1f}s ({total/elapsed:.1f} puzzles/s)")
    print()
    print("By rating bucket:")
    for bucket in sorted(by_rating_bucket):
        s, t = by_rating_bucket[bucket]
        print(f"  {bucket:>5}-{bucket+199:<5}: {s}/{t} = {100*s/t:.1f}%")

    if args.log_fails and fails:
        with open(args.log_fails, "w") as f:
            for pid, rating, pop, why in fails:
                f.write(f"{pid}\t{rating}\t{pop}\t{why}\n")
        print(f"\nWrote {len(fails)} failures to {args.log_fails}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
