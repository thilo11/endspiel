#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CHECKPOINTS_DIR="$REPO_ROOT/train/checkpoints"
NAME="d16_pick"
START_SB=1
END_SB=""

FASTCHESS_BIN=""
ENGINE_CMD="$REPO_ROOT/target/release/endspiel"
BASELINE_NET="$REPO_ROOT/crates/chess-nnue/nets/default.nnue"
OPENINGS_FILE="$REPO_ROOT/assets/openings.epd"

HASH_MB=64
THREADS=1
ROUNDS=12
CONCURRENCY=8
TC="10+0.1"

print_help() {
  cat <<EOF
Sweep checkpoints against baseline and pick the best superbatch.

Usage:
  bash scripts/sweep_checkpoints.sh [options]

Options:
  --name ID              Checkpoint name prefix (default: $NAME)
  --start N              First superbatch to test (default: $START_SB)
  --end N                Last superbatch to test (default: auto-detect max existing)
  --checkpoints DIR      Checkpoints directory (default: train/checkpoints)
  --fastchess PATH       fastchess binary path (default: auto-detect)
  --engine PATH          Engine binary to test (default: target/release/endspiel)
  --baseline-net PATH    Baseline net for reference engine (default: crates/chess-nnue/nets/default.nnue)
  --openings FILE        EPD openings file (default: assets/openings.epd)
  --rounds N             Rounds per checkpoint (default: $ROUNDS)
  --concurrency N        Concurrent games (default: $CONCURRENCY)
  --tc TC                Time control (default: $TC)
  --hash MB              Hash for both engines (default: $HASH_MB)
  --threads N            Threads for both engines (default: $THREADS)
  -h, --help             Show help

Example:
  bash scripts/sweep_checkpoints.sh --name d16_pick --start 1 --end 12 --rounds 12 --concurrency 8
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --start) START_SB="$2"; shift 2 ;;
    --end) END_SB="$2"; shift 2 ;;
    --checkpoints) CHECKPOINTS_DIR="$2"; shift 2 ;;
    --fastchess) FASTCHESS_BIN="$2"; shift 2 ;;
    --engine) ENGINE_CMD="$2"; shift 2 ;;
    --baseline-net) BASELINE_NET="$2"; shift 2 ;;
    --openings) OPENINGS_FILE="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --tc) TC="$2"; shift 2 ;;
    --hash) HASH_MB="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

[[ "$START_SB" =~ ^[0-9]+$ ]] || { echo "--start must be an integer" >&2; exit 1; }
[[ "$ROUNDS" =~ ^[0-9]+$ ]] || { echo "--rounds must be an integer" >&2; exit 1; }
[[ "$CONCURRENCY" =~ ^[0-9]+$ ]] || { echo "--concurrency must be an integer" >&2; exit 1; }

if [[ -z "$FASTCHESS_BIN" ]]; then
  if command -v fastchess >/dev/null 2>&1; then
    FASTCHESS_BIN="$(command -v fastchess)"
  elif [[ -x "$HOME/bin/fastchess-linux-x86-64/fastchess" ]]; then
    FASTCHESS_BIN="$HOME/bin/fastchess-linux-x86-64/fastchess"
  else
    echo "Could not find fastchess binary. Use --fastchess PATH." >&2
    exit 1
  fi
fi

[[ -x "$FASTCHESS_BIN" ]] || { echo "fastchess not executable: $FASTCHESS_BIN" >&2; exit 1; }
[[ -x "$ENGINE_CMD" ]] || { echo "engine not executable: $ENGINE_CMD" >&2; exit 1; }
[[ -f "$BASELINE_NET" ]] || { echo "baseline net not found: $BASELINE_NET" >&2; exit 1; }
[[ -f "$OPENINGS_FILE" ]] || { echo "openings file not found: $OPENINGS_FILE" >&2; exit 1; }
[[ -d "$CHECKPOINTS_DIR" ]] || { echo "checkpoints dir not found: $CHECKPOINTS_DIR" >&2; exit 1; }

if [[ -z "$END_SB" ]]; then
  max_sb=-1
  for d in "$CHECKPOINTS_DIR/$NAME"-*/; do
    [[ -d "$d" ]] || continue
    b="$(basename "$d")"
    if [[ "$b" =~ ^${NAME}-([0-9]+)$ ]]; then
      n="${BASH_REMATCH[1]}"
      (( n > max_sb )) && max_sb="$n"
    fi
  done
  (( max_sb >= 0 )) || { echo "No checkpoints found for '$NAME'" >&2; exit 1; }
  END_SB="$max_sb"
fi

[[ "$END_SB" =~ ^[0-9]+$ ]] || { echo "--end must be an integer" >&2; exit 1; }
(( END_SB >= START_SB )) || { echo "--end must be >= --start" >&2; exit 1; }

best_sb=""
best_ratio="-1"
best_elo="-1e18"

printf "Sweep: name=%s range=%s..%s rounds=%s tc=%s\n" "$NAME" "$START_SB" "$END_SB" "$ROUNDS" "$TC"
printf "%-8s %-10s %-8s %-8s\n" "SB" "Points" "Score%" "Elo"
printf "%-8s %-10s %-8s %-8s\n" "--------" "----------" "--------" "--------"

for sb in $(seq "$START_SB" "$END_SB"); do
  cand_net="$CHECKPOINTS_DIR/$NAME-$sb/quantised.bin"
  if [[ ! -f "$cand_net" ]]; then
    continue
  fi

  out_file="$(mktemp)"
  "$FASTCHESS_BIN" \
    -engine cmd="$ENGINE_CMD" name="candidate" option.EvalFile="$cand_net" option.Hash="$HASH_MB" option.Threads="$THREADS" \
    -engine cmd="$ENGINE_CMD" name="baseline" option.EvalFile="$BASELINE_NET" option.Hash="$HASH_MB" option.Threads="$THREADS" \
    -openings file="$OPENINGS_FILE" format=epd order=random \
    -each tc="$TC" -rounds "$ROUNDS" -concurrency "$CONCURRENCY" -recover > "$out_file"

  parsed="$(python3 - "$out_file" <<'PY'
import re
import sys

txt = open(sys.argv[1], "r", encoding="utf-8", errors="ignore").read()
games = points = None
elo = "nan"

mg = re.search(r"Games:\s*(\d+).*?Points:\s*([0-9]+(?:\.[0-9]+)?)", txt, re.IGNORECASE | re.DOTALL)
if mg:
    games = int(mg.group(1))
    points = float(mg.group(2))

me = re.search(r"Elo:\s*([^\s,]+)", txt)
if me:
    elo = me.group(1)

if games is None or games <= 0 or points is None:
    print("nan nan nan")
else:
    ratio = points / games
    print(f"{points} {ratio} {elo}")
PY
)"

  rm -f "$out_file"

  points="$(awk '{print $1}' <<< "$parsed")"
  ratio="$(awk '{print $2}' <<< "$parsed")"
  elo="$(awk '{print $3}' <<< "$parsed")"

  if [[ "$ratio" == "nan" ]]; then
    printf "%-8s %-10s %-8s %-8s\n" "$sb" "-" "-" "-"
    continue
  fi

  score_pct="$(python3 - <<PY
r = float("$ratio") * 100.0
print(f"{r:.2f}")
PY
)"

  printf "%-8s %-10s %-8s %-8s\n" "$sb" "$points" "$score_pct" "$elo"

  cmp="$(python3 - <<PY
import math
ratio = float("$ratio")
best_ratio = float("$best_ratio")

def parse_elo(s: str) -> float:
    try:
        v = float(s)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return -1e18

elo = parse_elo("$elo")
best_elo = parse_elo("$best_elo")

if ratio > best_ratio + 1e-12:
    print("better")
elif abs(ratio - best_ratio) <= 1e-12 and elo > best_elo:
    print("better")
else:
    print("no")
PY
)"

  if [[ "$cmp" == "better" ]]; then
    best_sb="$sb"
    best_ratio="$ratio"
    best_elo="$elo"
  fi
done

[[ -n "$best_sb" ]] || { echo "No valid checkpoint results parsed." >&2; exit 2; }

best_net="$CHECKPOINTS_DIR/$NAME-$best_sb/quantised.bin"
best_pct="$(python3 - <<PY
print(f"{float('$best_ratio') * 100.0:.2f}")
PY
)"

echo
echo "Best checkpoint: $best_sb"
echo "Best score: $best_pct%"
echo "Best net: $best_net"
