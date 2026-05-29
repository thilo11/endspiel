#!/usr/bin/env python3
"""
SPSA tuner for Endspiel search parameters.

Uses the Simultaneous Perturbation Stochastic Approximation algorithm:
each iteration perturbs all parameters simultaneously in random ±1 directions,
plays a match with fastchess, and updates parameters from the Elo gradient.

Usage:
    python3 scripts/spsa.py [--iterations N] [--games-per-iter N] [--tc TC] [--concurrency N]

Typical run:
    python3 scripts/spsa.py                                      # 300 iters, 400 games, 2+0.02, c=14, lr=2.0
    python3 scripts/spsa.py --tc 1+0.01                          # faster, ~6 hrs on 32-thread machine
    python3 scripts/spsa.py --resume spsa_checkpoint.json        # resume after interruption

Output:
    - Progress printed to stdout after each iteration
    - Final parameters written to spsa_result.json
    - Checkpoint saved every 10 iterations to spsa_checkpoint.json
"""

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Parameter definitions: (name, UCI_option_name, initial, min, max, step)
# step = initial perturbation size (roughly 5-10% of expected range)
#
# IMPORTANT: the `init` column must stay in sync with TuneParams::default()
# in crates/chess-engine/src/lib.rs — SPSA starts from these values, so a
# stale init silently re-tunes a param away from its current best. After
# applying spsa_result.json back into lib.rs, update these inits too.
# ---------------------------------------------------------------------------
PARAMS = [
    # name              UCI name           init   min    max   step
    ("lmr_base",        "LmrBase",          24,    10,   200,   8),
    ("lmr_div",         "LmrDiv",          156,   100,   400,  20),
    ("hist_lmr_div",    "HistLmrDiv",     1055,   500, 20000, 500),
    ("rfp_margin_imp",  "RfpMarginImp",     70,    10,   300,  10),
    ("rfp_margin_noimp","RfpMarginNoImp",   98,    10,   300,  10),
    ("fut_margin_imp",  "FutMarginImp",     41,    10,   300,  10),
    ("fut_margin_noimp","FutMarginNoImp",   67,    10,   300,  10),
    ("see_quiet_margin","SeeQuietMargin",   13,    10,   200,   8),
    ("corrhist_mult",   "CorrHistMult",     82,     0,   300,  20),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def build_engine_options(param_values: dict) -> str:
    """Build 'option.Name=value' string for fastchess -engine."""
    parts = []
    for name, uci_name, *_ in PARAMS:
        parts.append(f"option.{uci_name}={int(round(param_values[name]))}")
    return " ".join(parts)


def run_match(engine_cmd: str, plus_opts: str, minus_opts: str,
              openings: str, tc: str, games: int, concurrency: int) -> dict:
    """Run a fastchess match between plus and minus param sets.
    Returns {'wins': W, 'draws': D, 'losses': L} from plus's perspective."""

    fastchess = os.environ.get("FASTCHESS") or shutil.which("fastchess")
    if not fastchess:
        fallback = os.path.expanduser("~/bin/fastchess-linux-x86-64/fastchess")
        if os.access(fallback, os.X_OK):
            fastchess = fallback
    if not fastchess:
        print("Could not find fastchess binary. Set $FASTCHESS or put it on PATH.", file=sys.stderr)
        sys.exit(1)

    cmd = [
        fastchess,
        "-engine", f"cmd={engine_cmd}", *plus_opts.split(), "option.Hash=8", "option.Threads=1", "name=plus",
        "-engine", f"cmd={engine_cmd}", *minus_opts.split(), "option.Hash=8", "option.Threads=1", "name=minus",
        "-openings", f"file={openings}", "format=epd", "order=random",
        "-each", f"tc={tc}",
        "-rounds", str(games // 2),
        "-concurrency", str(concurrency),
        "-recover",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print("  [WARN] fastchess timed out, skipping iteration", file=sys.stderr)
        return None

    # Parse final results line — use findall and take the last match to avoid
    # matching intermediate per-game update lines printed by fastchess.
    matches = re.findall(r"Games:\s*(\d+),\s*Wins:\s*(\d+),\s*Losses:\s*(\d+),\s*Draws:\s*(\d+)", output)
    if matches:
        _, w, l, d = matches[-1]
        return {"wins": int(w), "draws": int(d), "losses": int(l)}

    # Fallback: bare "Wins: W, Losses: L, Draws: D" (take last match)
    matches = re.findall(r"Wins:\s*(\d+),\s*Losses:\s*(\d+),\s*Draws:\s*(\d+)", output)
    if matches:
        w, l, d = matches[-1]
        result = {"wins": int(w), "draws": int(d), "losses": int(l)}
        n = result["wins"] + result["draws"] + result["losses"]
        if n == 0:
            print(f"  [WARN] fastchess returned 0 games played:\n{output[-500:]}", file=sys.stderr)
            return None
        if n < games * 0.9:
            print(f"  [WARN] only {n}/{games} games completed. fastchess output tail:\n{output[-800:]}", file=sys.stderr)
        return result

    # Last resort: parse Elo directly
    m = re.search(r"Elo:\s*([-\d.]+)\s*\+/-\s*([-\d.]+)", output)
    if m:
        elo = float(m.group(1))
        p = 1.0 / (1.0 + 10 ** (-elo / 400.0))
        w = int(p * games * 0.4)
        l = int((1 - p) * games * 0.4)
        d = games - w - l
        return {"wins": w, "draws": d, "losses": l, "elo": elo}

    print(f"  [WARN] Could not parse fastchess output:\n{output[-500:]}", file=sys.stderr)
    return None


def elo_from_wdl(wins, draws, losses):
    """Compute Elo difference from W/D/L."""
    n = wins + draws + losses
    if n == 0:
        return 0.0
    score = (wins + 0.5 * draws) / n
    score = clamp(score, 0.001, 0.999)
    return -400.0 * math.log10(1.0 / score - 1.0)


# ---------------------------------------------------------------------------
# SPSA update
# ---------------------------------------------------------------------------

def spsa_update(params: dict, deltas: dict, elo_diff: float,
                lr: float, step_sizes: dict) -> dict:
    """Apply one SPSA gradient step."""
    updated = dict(params)
    for name, uci_name, init, lo, hi, step in PARAMS:
        if elo_diff == 0:
            continue
        delta = deltas[name]  # ±step
        # Standard SPSA update: θ_i += a_k * (elo_diff / 2) * sign(delta)
        # Do NOT multiply by step — doing so creates wildly different effective
        # learning rates per param and causes small-step params to never move.
        direction = math.copysign(1.0, delta)
        updated[name] = clamp(
            params[name] + lr * (elo_diff / 2.0) * direction,
            lo, hi
        )
    return updated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SPSA tuner for Endspiel")
    parser.add_argument("--iterations",     type=int,   default=300,      help="SPSA iterations (default: 300)")
    parser.add_argument("--games-per-iter", type=int,   default=400,      help="Games per iteration, must be even (default: 400)")
    parser.add_argument("--tc",             type=str,   default="2+0.02", help="Time control (default: 2+0.02)")
    parser.add_argument("--concurrency",    type=int,   default=14,       help="fastchess concurrency (default: 14)")
    parser.add_argument("--engine",         type=str,   default="target/release/endspiel")
    parser.add_argument("--openings",       type=str,   default="assets/openings.epd")
    parser.add_argument("--resume",         type=str,   default=None,     help="Resume from checkpoint JSON")
    parser.add_argument("--lr",             type=float, default=2.0,      help="Learning rate / SPSA 'a' parameter (default: 2.0)")
    args = parser.parse_args()

    games = args.games_per_iter
    if games % 2 != 0:
        games += 1

    # Initial parameter values
    params = {name: float(init) for name, _, init, *_ in PARAMS}
    step_sizes = {name: float(step) for name, _, init, lo, hi, step in PARAMS}
    start_iter = 0

    if args.resume:
        with open(args.resume) as f:
            checkpoint = json.load(f)
        params = checkpoint["params"]
        start_iter = checkpoint.get("iteration", 0)
        if "games_per_iter" in checkpoint and args.games_per_iter == 200:
            games = checkpoint["games_per_iter"]
            print(f"Restored games-per-iter={games} from checkpoint")
        print(f"Resumed from iteration {start_iter}")

    print(f"SPSA tuning: {args.iterations} iterations, {games} games/iter, tc={args.tc}")
    print(f"Initial params: {json.dumps({k: round(v) for k, v in params.items()}, indent=2)}")
    print()

    best_params = dict(params)
    history = []

    for iteration in range(start_iter, start_iter + args.iterations):
        # Learning rate decay: a / (A + k)^alpha
        # a=lr, A=iterations/10, alpha=0.602
        a = args.lr
        A = args.iterations / 10.0
        alpha = 0.602
        lr_k = a / (A + iteration + 1) ** alpha

        # Random ±1 perturbation for each parameter
        deltas = {name: random.choice([-1, 1]) * step for name, _, init, lo, hi, step in PARAMS}

        plus_params  = {name: clamp(params[name] + deltas[name], lo, hi)
                        for name, _, init, lo, hi, step in PARAMS}
        minus_params = {name: clamp(params[name] - deltas[name], lo, hi)
                        for name, _, init, lo, hi, step in PARAMS}

        plus_opts  = build_engine_options(plus_params)
        minus_opts = build_engine_options(minus_params)

        result = run_match(
            args.engine, plus_opts, minus_opts,
            args.openings, args.tc, games, args.concurrency
        )

        if result is None:
            print(f"  iter {iteration+1}: skipped (match error)")
            continue

        elo_diff = elo_from_wdl(result["wins"], result["draws"], result["losses"])
        params = spsa_update(params, deltas, elo_diff, lr_k, step_sizes)

        history.append({"iteration": iteration + 1, "elo_diff": round(elo_diff, 2),
                         "params": {k: round(v) for k, v in params.items()}})

        print(f"iter {iteration+1:4d}  elo_diff={elo_diff:+.1f}  lr={lr_k:.4f}  "
              f"W={result['wins']} D={result['draws']} L={result['losses']}")
        print(f"  params: { {k: round(v) for k, v in params.items()} }")

        # Checkpoint every 10 iterations
        if (iteration + 1) % 10 == 0:
            checkpoint = {"iteration": iteration + 1, "params": params, "games_per_iter": games, "history": history}
            with open("spsa_checkpoint.json", "w") as f:
                json.dump(checkpoint, f, indent=2)
            print(f"  [checkpoint saved]")

    # Final result
    final = {k: round(v) for k, v in params.items()}
    print("\n=== Final parameters ===")
    print(json.dumps(final, indent=2))

    with open("spsa_result.json", "w") as f:
        json.dump({"params": final, "history": history}, f, indent=2)
    print("\nSaved to spsa_result.json")


if __name__ == "__main__":
    main()
