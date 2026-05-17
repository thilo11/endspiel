#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ENGINE_CMD="$REPO_ROOT/target/release/endspiel"
STOCKFISH_CMD="/usr/games/stockfish"
OPENINGS_FILE="$REPO_ROOT/assets/openings.epd"
CHECKPOINTS_DIR="$REPO_ROOT/train/checkpoints"
CHECKPOINT_NAME="safeft"
SELECT_SB=""
NET_OVERRIDE=""
USE_BEST_LOSS=false
FASTCHESS_BIN=""

HASH_MB=64
THREADS=1
ROUNDS=100
CONCURRENCY=8
TC="10+0.1"
SF_ELO=3190
EARLY_STOP_ROUNDS=0
EARLY_STOP_MIN_SCORE=0.22

print_help() {
  cat <<EOF
Run fastchess with the newest checkpoint automatically.

Usage:
  bash scripts/fastchess_latest.sh [options] [-- <extra fastchess args>]

Options:
  --name ID           Checkpoint name prefix (default: $CHECKPOINT_NAME)
  --best-loss         Auto-pick saved checkpoint with lowest avg loss from log.txt
  --sb N              Use a specific superbatch number (e.g. 247)
  --net PATH          Use this exact net file (overrides --name/--sb)
  --checkpoints DIR   Checkpoints directory (default: train/checkpoints)
  --fastchess PATH    fastchess binary path (default: auto-detect)
  --engine PATH       Candidate engine command (default: target/release/endspiel)
  --stockfish PATH    Stockfish binary (default: /usr/games/stockfish)
  --openings FILE     EPD openings file (default: assets/openings.epd)
  --rounds N          Number of rounds (default: $ROUNDS)
  --concurrency N     Parallel games (default: $CONCURRENCY)
  --tc TC             Time control (default: $TC)
  --hash MB           Hash size for both engines (default: $HASH_MB)
  --threads N         Threads for both engines (default: $THREADS)
  --sf-elo N          Stockfish limited Elo (default: $SF_ELO)
  --early-stop-rounds N  Run an initial gate phase for N rounds (0 disables)
  --early-stop-games N   Compatibility alias: converted to rounds as ceil(N/2)
  --early-stop-min-score F  Minimum score ratio (0..1) required to continue (default: $EARLY_STOP_MIN_SCORE)
  -h, --help          Show this help

Examples:
  bash scripts/fastchess_latest.sh --name safeft
  bash scripts/fastchess_latest.sh --name d16_scratch --best-loss
  bash scripts/fastchess_latest.sh --name d16_ft --best-loss --rounds 100 --early-stop-rounds 15
  bash scripts/fastchess_latest.sh --name scratch --rounds 200 --concurrency 12
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) CHECKPOINT_NAME="$2"; shift 2 ;;
    --best-loss) USE_BEST_LOSS=true; shift ;;
    --sb) SELECT_SB="$2"; shift 2 ;;
    --net) NET_OVERRIDE="$2"; shift 2 ;;
    --checkpoints) CHECKPOINTS_DIR="$2"; shift 2 ;;
    --fastchess) FASTCHESS_BIN="$2"; shift 2 ;;
    --engine) ENGINE_CMD="$2"; shift 2 ;;
    --stockfish) STOCKFISH_CMD="$2"; shift 2 ;;
    --openings) OPENINGS_FILE="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --tc) TC="$2"; shift 2 ;;
    --hash) HASH_MB="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --sf-elo) SF_ELO="$2"; shift 2 ;;
    --early-stop-rounds) EARLY_STOP_ROUNDS="$2"; shift 2 ;;
    --early-stop-games)
      g="$2"
      [[ "$g" =~ ^[0-9]+$ ]] || { echo "--early-stop-games must be an integer" >&2; exit 1; }
      EARLY_STOP_ROUNDS=$(( (g + 1) / 2 ))
      shift 2
      ;;
    --early-stop-min-score) EARLY_STOP_MIN_SCORE="$2"; shift 2 ;;
    -h|--help)
      print_help
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

EXTRA_ARGS=("$@")

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
[[ -x "$STOCKFISH_CMD" ]] || { echo "stockfish not executable: $STOCKFISH_CMD" >&2; exit 1; }
[[ -f "$OPENINGS_FILE" ]] || { echo "openings file not found: $OPENINGS_FILE" >&2; exit 1; }
[[ -d "$CHECKPOINTS_DIR" ]] || { echo "checkpoints dir not found: $CHECKPOINTS_DIR" >&2; exit 1; }

find_latest_run_log() {
  local latest_log=""
  local latest_mtime=-1
  for d in "$CHECKPOINTS_DIR/$CHECKPOINT_NAME"-*/; do
    [[ -d "$d" ]] || continue
    local log="$d/log.txt"
    [[ -f "$log" ]] || continue
    local mt
    mt=$(stat -c %Y "$log" 2>/dev/null || echo 0)
    if (( mt > latest_mtime )); then
      latest_mtime="$mt"
      latest_log="$log"
    fi
  done
  [[ -n "$latest_log" ]] || return 1
  echo "$latest_log"
}

select_latest_saved_sb_from_log() {
  local log_path="$1"
  python3 - "$log_path" "$CHECKPOINTS_DIR" "$CHECKPOINT_NAME" <<'PY'
import os
import sys

log_path, checkpoints_dir, prefix = sys.argv[1:]
sbs = set()
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) != 3:
            continue
        try:
            sb = int(parts[0])
        except ValueError:
            continue
        sbs.add(sb)

saved = [sb for sb in sbs if os.path.exists(os.path.join(checkpoints_dir, f"{prefix}-{sb}", "quantised.bin"))]
if not saved:
    print("")
else:
    print(max(saved))
PY
}

select_best_loss_sb() {
  local log_path="$1"
  python3 - "$log_path" "$CHECKPOINTS_DIR" "$CHECKPOINT_NAME" <<'PY'
import os
import sys
from collections import defaultdict

log_path, checkpoints_dir, prefix = sys.argv[1:]
losses = defaultdict(list)

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) != 3:
            continue
        try:
            sb = int(parts[0])
            _batch = int(parts[1])
            loss = float(parts[2])
        except ValueError:
            continue
        losses[sb].append(loss)

if not losses:
    print("")
    raise SystemExit(0)

saved = []
for sb in losses:
    p = os.path.join(checkpoints_dir, f"{prefix}-{sb}", "quantised.bin")
    if os.path.exists(p):
        saved.append(sb)

if not saved:
    print("")
    raise SystemExit(0)

best_sb = min(saved, key=lambda sb: sum(losses[sb]) / len(losses[sb]))
print(best_sb)
PY
}

if [[ -n "$NET_OVERRIDE" ]]; then
  CAND_NET="$NET_OVERRIDE"
elif [[ -n "$SELECT_SB" ]]; then
  CAND_NET="$CHECKPOINTS_DIR/$CHECKPOINT_NAME-$SELECT_SB/quantised.bin"
elif [[ "$USE_BEST_LOSS" == true ]]; then
  latest_log="$(find_latest_run_log)" || {
    echo "No checkpoint logs found for name '$CHECKPOINT_NAME' in $CHECKPOINTS_DIR" >&2
    exit 1
  }
  best_sb="$(select_best_loss_sb "$latest_log")"
  [[ -n "$best_sb" ]] || { echo "Could not determine best-loss superbatch from $latest_log" >&2; exit 1; }

  CAND_NET="$CHECKPOINTS_DIR/$CHECKPOINT_NAME-$best_sb/quantised.bin"
  echo "Selected best-loss superbatch: $best_sb"
else
  latest_log="$(find_latest_run_log)" || {
    echo "No checkpoint logs found for name '$CHECKPOINT_NAME' in $CHECKPOINTS_DIR" >&2
    exit 1
  }
  latest_sb="$(select_latest_saved_sb_from_log "$latest_log")"
  [[ -n "$latest_sb" ]] || { echo "Could not determine latest saved superbatch from $latest_log" >&2; exit 1; }
  CAND_NET="$CHECKPOINTS_DIR/$CHECKPOINT_NAME-$latest_sb/quantised.bin"
fi

[[ -f "$CAND_NET" ]] || { echo "Candidate net not found: $CAND_NET" >&2; exit 1; }

[[ "$ROUNDS" =~ ^[0-9]+$ ]] || { echo "--rounds must be an integer" >&2; exit 1; }
[[ "$EARLY_STOP_ROUNDS" =~ ^[0-9]+$ ]] || { echo "--early-stop-rounds must be an integer" >&2; exit 1; }
python3 - <<PY >/dev/null || { echo "--early-stop-min-score must be a float in [0,1]" >&2; exit 1; }
v = float("$EARLY_STOP_MIN_SCORE")
assert 0.0 <= v <= 1.0
PY

echo "Using checkpoint: $CAND_NET"
echo "Running: $FASTCHESS_BIN"

run_match() {
  local rounds="$1"
  local label="$2"
  local out_file="$3"

  echo "[$label] Starting fastchess: rounds=$rounds"
  "$FASTCHESS_BIN" \
    -engine cmd="$ENGINE_CMD" name="candidate" \
            option.EvalFile="$CAND_NET" option.Hash="$HASH_MB" option.Threads="$THREADS" \
    -engine cmd="$STOCKFISH_CMD" name="sf${SF_ELO}" \
            option.UCI_LimitStrength=true option.UCI_Elo="$SF_ELO" option.Hash="$HASH_MB" option.Threads="$THREADS" \
    -openings file="$OPENINGS_FILE" format=epd order=random \
    -each tc="$TC" -rounds "$rounds" -concurrency "$CONCURRENCY" -recover \
    "${EXTRA_ARGS[@]}" | tee "$out_file"
}

extract_score_ratio() {
  local out_file="$1"
  python3 - "$out_file" <<'PY'
import re
import sys

path = sys.argv[1]
games = None
points = None
pat = re.compile(r"Games:\s*(\d+).*?Points:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = pat.search(line)
        if m:
            games = int(m.group(1))
            points = float(m.group(2))

if games is None or games <= 0:
    print("nan")
else:
    print(points / games)
PY
}

if (( EARLY_STOP_ROUNDS > 0 )) && (( ROUNDS > EARLY_STOP_ROUNDS )); then
  gate_out="$(mktemp)"
  run_match "$EARLY_STOP_ROUNDS" "gate" "$gate_out"
  gate_ratio="$(extract_score_ratio "$gate_out")"
  rm -f "$gate_out"

  if ! python3 - <<PY >/dev/null
ratio = float("$gate_ratio")
thr = float("$EARLY_STOP_MIN_SCORE")
assert ratio == ratio
raise SystemExit(0 if ratio >= thr else 1)
PY
  then
    echo "[gate] Early stop triggered: score ratio $gate_ratio < threshold $EARLY_STOP_MIN_SCORE"
    exit 2
  fi

  remaining=$(( ROUNDS - EARLY_STOP_ROUNDS ))
  echo "[gate] Passed: score ratio $gate_ratio >= threshold $EARLY_STOP_MIN_SCORE"
  echo "[gate] Continuing with remaining rounds: $remaining"
  main_out="$(mktemp)"
  run_match "$remaining" "main" "$main_out"
  rm -f "$main_out"
else
  main_out="$(mktemp)"
  run_match "$ROUNDS" "main" "$main_out"
  rm -f "$main_out"
fi
