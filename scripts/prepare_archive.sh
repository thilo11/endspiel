#!/usr/bin/env bash
#
# Prepare an archive file for use in the training pipeline.
# Run this once whenever you add new data to the archive.
# Produces a deduped, cleaned, shuffled file that next_round_pipeline.sh
# can consume directly without repeating these expensive operations.
#
# Pipeline: clean + dedup → shuffle → output

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
INPUT="$REPO_ROOT/data/archive/archive.bin"
OUTPUT="$REPO_ROOT/data/archive/archive_ready.bin"
WORK_DIR="$REPO_ROOT/.archive_prep"
# Dedup is OFF by default: the in-memory HashSet is O(unique_positions) in RAM
# (~24 bytes per u64 entry), which OOMs on multi-billion-position archives.
# Pure-random-opening data has ~1–2% duplication — not worth the RAM cost.
# Use --dedup to enable (fine for archives <~1B unique positions).
DEDUP=false
DRAW_RATIO=0.60
DROP_KING_ONLY=true
DROP_HIGH_ABS_DRAWS=0
MEM_MB=50000
NO_CLEANUP=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input|-i)             INPUT="$2";               shift 2 ;;
    --output|-o)            OUTPUT="$2";              shift 2 ;;
    --work-dir)             WORK_DIR="$2";            shift 2 ;;
    --dedup)                DEDUP=true;               shift ;;
    --no-dedup)             DEDUP=false;              shift ;;
    --draw-ratio)           DRAW_RATIO="$2";          shift 2 ;;
    --drop-king-only)       DROP_KING_ONLY="$2";      shift 2 ;;
    --drop-high-abs-draws)  DROP_HIGH_ABS_DRAWS="$2"; shift 2 ;;
    --mem-mb)               MEM_MB="$2";              shift 2 ;;
    --no-cleanup)           NO_CLEANUP=true;          shift ;;
    -h|--help)
      cat <<EOF
Prepare an archive file for use in next_round_pipeline.sh.
Run once whenever you add new data to the archive; the training pipeline
then uses the output directly without repeating these expensive steps.

Usage:
  bash scripts/prepare_archive.sh [options]

Options:
  --input PATH               Raw archive input          (default: data/archive/archive.bin)
  --output PATH              Prepared archive output    (default: data/archive/archive_ready.bin)
  --work-dir PATH            Work directory             (default: .archive_prep)
  --dedup                    Deduplicate positions (RAM-bound; unsafe >~1B unique positions)
  --no-dedup                 Disable deduplication (default)
  --draw-ratio F|none        Target draw ratio          (default: 0.60)
  --drop-king-only true|false  Drop king-only positions (default: true)
  --drop-high-abs-draws N    Drop draws with |score|>=N cp (default: 0 — disabled)
  --mem-mb N                 Shuffle memory budget in MB (default: $MEM_MB)
  --no-cleanup               Keep work directory after completion
EOF
      exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ── Validate input ────────────────────────────────────────────────────────────
[[ -f "$INPUT" ]] || { echo "Missing input: $INPUT" >&2; exit 1; }
sz=$(stat -c%s "$INPUT")
(( sz % 32 == 0 )) || { echo "Input not 32-byte aligned: $INPUT" >&2; exit 1; }
input_pos=$(( sz / 32 ))

BULLET_UTILS="$REPO_ROOT/../bullet/target/release/bullet-utils"
if [[ ! -x "$BULLET_UTILS" ]]; then
  cargo build --release -p bullet-utils --manifest-path "$REPO_ROOT/../bullet/Cargo.toml"
fi

mkdir -p "$WORK_DIR"
mkdir -p "$(dirname "$OUTPUT")"

TOTAL_PHASES=3

phase_start=0
phase_begin() {
  phase_start=$SECONDS
  echo "[${1}/$TOTAL_PHASES] ${2}  ($(date '+%H:%M:%S'))"
}
phase_end() {
  local elapsed=$(( SECONDS - phase_start ))
  printf "  done in %dm%02ds\n" $(( elapsed / 60 )) $(( elapsed % 60 ))
}

echo "=== Archive preparation ==="
printf "  input      : %s (%d positions)\n" "$INPUT" "$input_pos"
printf "  output     : %s\n" "$OUTPUT"
printf "  dedup      : %s\n" "$DEDUP"
printf "  draw-ratio           : %s\n" "$DRAW_RATIO"
printf "  drop-king-only       : %s\n" "$DROP_KING_ONLY"
printf "  drop-high-abs-draws  : %s\n" "$DROP_HIGH_ABS_DRAWS"
echo "==========================="

# ── Phase 1: Clean + dedup ────────────────────────────────────────────────────
phase_begin 1 "Cleaning and deduplicating"
needs_clean=false
[[ "$DEDUP" == true ]]             && needs_clean=true
[[ "$DRAW_RATIO" != "none" ]]      && needs_clean=true
[[ "$DROP_KING_ONLY" != "false" ]] && needs_clean=true
(( DROP_HIGH_ABS_DRAWS > 0 ))      && needs_clean=true

TO_SHUFFLE="$INPUT"
if [[ "$needs_clean" == true ]]; then
  clean_args=(
    --input                     "$INPUT"
    --output                    "$WORK_DIR/cleaned.bin"
    --target-draw-ratio         "$DRAW_RATIO"
    --drop-king-only            "$DROP_KING_ONLY"
    --drop-high-abs-score-draws "$DROP_HIGH_ABS_DRAWS"
  )
  [[ "$DEDUP" == true ]] && clean_args+=(--dedup)
  cargo run --release --manifest-path "$REPO_ROOT/train/Cargo.toml" --bin clean_data -- "${clean_args[@]}"
  TO_SHUFFLE="$WORK_DIR/cleaned.bin"
else
  echo "  no filters active — skipping clean step"
fi
phase_end

# ── Phase 2: Shuffle ──────────────────────────────────────────────────────────
phase_begin 2 "Shuffling"
"$BULLET_UTILS" shuffle \
  --input       "$TO_SHUFFLE" \
  --mem-used-mb "$MEM_MB" \
  --output      "$OUTPUT"
out_pos=$(( $(stat -c%s "$OUTPUT") / 32 ))
echo "  output: $OUTPUT ($out_pos positions)"
phase_end

# ── Phase 3: Cleanup ──────────────────────────────────────────────────────────
phase_begin 3 "Cleanup"
if [[ "$NO_CLEANUP" == true ]]; then
  echo "  skipped (--no-cleanup)."
else
  rm -rf "$WORK_DIR"
  echo "  removed: $WORK_DIR"
fi
phase_end

echo "Done. Archive ready at: $OUTPUT"
