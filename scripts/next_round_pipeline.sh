#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── File paths ────────────────────────────────────────────────────────────────
NEW_IN="$REPO_ROOT/data/new.bin"
ARCHIVE_IN="$REPO_ROOT/data/archive/archive_ready.bin"
OUT="$REPO_ROOT/data/data_train_next.bin"
NEW_EXPLICIT=false
ARCHIVE_EXPLICIT=false
WORK_DIR="$REPO_ROOT/.round_pipeline"
MEM_MB=50000

# ── Datagen defaults ──────────────────────────────────────────────────────────
DG_GAMES=2000000
DG_DEPTH=16
DG_THREADS=32
DG_HASH=16
DG_SYZYGY="$REPO_ROOT/assets/syzygy"

# ── Mix defaults (ratios computed from file sizes after arg parsing) ──────────
NEW_RATIO=""
ARCHIVE_RATIO=""
NEW_RATIO_EXPLICIT=false
ARCHIVE_RATIO_EXPLICIT=false
MAX_POSITIONS=9999999999
DRAW_RATIO=0.60
DROP_KING_ONLY=true
DROP_HIGH_ABS_DRAWS=0
DEDUP=true

# ── Training defaults ─────────────────────────────────────────────────────────
# Fine-tune mode (default): warm-start from default.nnue, low LR, few SBs.
# From-scratch mode (--from-scratch): random init, standard LR, many SBs.
TRAIN_DATA="$REPO_ROOT/data/new.bin"
TRAIN_LR=0.000003
TRAIN_WDL=0.15
TRAIN_SUPERBATCHES=250
TRAIN_SAVE_RATE=1
TRAIN_EPOCHS=1
TRAIN_NAME=safeft
TRAIN_INIT_NET="$REPO_ROOT/crates/chess-nnue/nets/default.nnue"
TRAIN_BPS=""
FROM_SCRATCH=false
TRAIN_DATA_EXPLICIT=false
LR_EXPLICIT=false
WDL_EXPLICIT=false
SB_EXPLICIT=false
SR_EXPLICIT=false
EP_EXPLICIT=false
NAME_EXPLICIT=false

# ── Phase flags ───────────────────────────────────────────────────────────────
SKIP_DATAGEN=true
SKIP_MIX=true
SKIP_CLEAN=true
SKIP_TRAIN=false
RUN_QUALITY_CHECK=true
QUALITY_STRICT=false
QUALITY_SAMPLE=0
NO_CLEANUP=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    # datagen
    --games)               DG_GAMES="$2";             shift 2 ;;
    --depth)               DG_DEPTH="$2";             shift 2 ;;
    --dg-threads)          DG_THREADS="$2";           shift 2 ;;
    --hash)                DG_HASH="$2";              shift 2 ;;
    --syzygy)              DG_SYZYGY="$2";            shift 2 ;;
    # paths
    --new)                 NEW_IN="$2";     NEW_EXPLICIT=true;      shift 2 ;;
    --archive)             ARCHIVE_IN="$2"; ARCHIVE_EXPLICIT=true;  shift 2 ;;
    --out)                 OUT="$2";                  shift 2 ;;
    --work-dir)            WORK_DIR="$2";             shift 2 ;;
    # mix
    --new-ratio)           NEW_RATIO="$2";     NEW_RATIO_EXPLICIT=true;     shift 2 ;;
    --archive-ratio)       ARCHIVE_RATIO="$2"; ARCHIVE_RATIO_EXPLICIT=true; shift 2 ;;
    --max-positions)       MAX_POSITIONS="$2";        shift 2 ;;
    --draw-ratio)          DRAW_RATIO="$2";           shift 2 ;;
    --drop-king-only)      DROP_KING_ONLY="$2";       shift 2 ;;
    --drop-high-abs-draws) DROP_HIGH_ABS_DRAWS="$2";  shift 2 ;;
    --dedup)               DEDUP=true;                shift ;;
    --mem-mb)              MEM_MB="$2";               shift 2 ;;
    # training
    --train-data)          TRAIN_DATA="$2"; TRAIN_DATA_EXPLICIT=true; shift 2 ;;
    --lr)                  TRAIN_LR="$2";   LR_EXPLICIT=true;         shift 2 ;;
    --wdl)                 TRAIN_WDL="$2";  WDL_EXPLICIT=true;        shift 2 ;;
    --superbatches)        TRAIN_SUPERBATCHES="$2"; SB_EXPLICIT=true;  shift 2 ;;
    --save-rate)           TRAIN_SAVE_RATE="$2"; SR_EXPLICIT=true;    shift 2 ;;
    --epochs)              TRAIN_EPOCHS="$2"; EP_EXPLICIT=true;       shift 2 ;;
    --train-name)          TRAIN_NAME="$2"; NAME_EXPLICIT=true;       shift 2 ;;
    --init-net)            TRAIN_INIT_NET="$2";                        shift 2 ;;
    --bps)                 TRAIN_BPS="$2";                             shift 2 ;;
    --from-scratch)        FROM_SCRATCH=true;                          shift ;;
    # phase control
    --skip-datagen)        SKIP_DATAGEN=true;         shift ;;
    --skip-mix)            SKIP_MIX=true;             shift ;;
    --skip-clean)          SKIP_CLEAN=true;           shift ;;
    --skip-train)          SKIP_TRAIN=true;           shift ;;
    --run-datagen)         SKIP_DATAGEN=false;        shift ;;
    --run-mix)             SKIP_MIX=false;            shift ;;
    --run-clean)           SKIP_CLEAN=false;          shift ;;
    --skip-quality-check)  RUN_QUALITY_CHECK=false;   shift ;;
    --quality-strict)      QUALITY_STRICT=true;       shift ;;
    --quality-sample)      QUALITY_SAMPLE="$2";      shift 2 ;;
    --no-cleanup)          NO_CLEANUP=true;           shift ;;
    -h|--help)
      cat <<EOF
Training pipeline: datagen → clean+dedup fresh → mix → quality-check → train → cleanup

The archive must be prepared once with prepare_archive.sh (dedup/clean/shuffle) before
use here. This script only prepares the fresh data and mixes it with the ready archive.

Two training modes:
  Fine-tune (default): warm-start from default.nnue, low LR, few SBs.
    Caution: self-play Elo gains from fine-tuning do not reliably reflect
    absolute strength. Validate against a fixed external opponent (e.g. SF).
  From-scratch (--from-scratch): random init, LR=0.001, 250 SBs, 3 epochs.
    The only confirmed path to genuine improvement. Use with --run-mix to
    train on all available data.

Usage:
  bash scripts/next_round_pipeline.sh [options]

  Recommended from-scratch run:
    bash scripts/next_round_pipeline.sh --from-scratch --run-mix

  Inputs: data/new.bin (fresh self-play) and data/archive/archive_ready.bin (prepared archive).
  Mix ratios default to proportional file sizes; override with --new-ratio / --archive-ratio.

  Prepare archive (run once when archive changes):
    bash scripts/prepare_archive.sh --input data/archive/archive.bin --dedup

Phase control:
  --run-datagen         Run data generation (default: skipped)
  --run-mix             Run mix pipeline (default: skipped)
  --run-clean           Clean fresh data before shuffling (default: skipped; same filters as archive)
  --skip-quality-check  Skip pre-train data quality check
  --quality-strict      Fail pipeline if quality check reports warnings
  --quality-sample N    Check at most N positions (default: 0 = all)
  --skip-train          Skip training
  --no-cleanup          Keep work dir and intermediate checkpoints

Datagen:
  --games N             Games to generate           (default: $DG_GAMES)
  --depth D             Search depth per move        (default: $DG_DEPTH)
  --dg-threads T        Datagen threads              (default: $DG_THREADS)
  --hash MB             TT size per thread in MB     (default: $DG_HASH)
  --syzygy DIR          Syzygy tablebase directory   (default: assets/syzygy)

Data paths:
  --new PATH            New self-play data           (default: data/new.bin)
                        Missing file → new-ratio forced to 0 (warn, not error)
  --archive PATH        Prepared archive data        (default: data/archive/archive_ready.bin)
                        Must be pre-processed with prepare_archive.sh (deduped, cleaned, shuffled).
                        Missing file → archive-ratio forced to 0 (warn, not error)
  --out PATH            Mixed training output        (default: data/data_train_next.bin)

Mix:
  --new-ratio F         New data sampling weight     (default: proportional to file size)
  --archive-ratio F     Archive sampling weight      (default: proportional to file size)
  --max-positions N     Cap total positions          (default: all)
  --draw-ratio F        Target draw ratio for fresh data (default: $DRAW_RATIO)
  --dedup               Deduplicate fresh data before mixing (archive dedup is done in prepare_archive.sh)
  --mem-mb N            Memory budget for fresh-data shuffle (default: $MEM_MB)

Training:
  --from-scratch        Train from random init (lr=0.001, wdl=0.75, 250 SBs, 3 epochs)
                        Auto-uses mixed data when --run-mix is also passed.
  --train-data PATH     Training data file           (overrides auto-selection)
  --lr F                Learning rate
  --wdl F               WDL blending
  --superbatches N      Superbatches to run
  --save-rate N         Save every N superbatches
  --epochs N            Passes over data
  --train-name ID       Checkpoint name prefix
  --init-net PATH       Warm-start net (fine-tune mode only)
  --bps N               Batches per superbatch (default: auto-computed)
EOF
      exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ── Apply from-scratch preset (only for flags not explicitly set) ─────────────
if [[ "$FROM_SCRATCH" == true ]]; then
  [[ "$LR_EXPLICIT"   == false ]] && TRAIN_LR=0.001
  [[ "$WDL_EXPLICIT"  == false ]] && TRAIN_WDL=0.75
  [[ "$SB_EXPLICIT"   == false ]] && TRAIN_SUPERBATCHES=250
  [[ "$SR_EXPLICIT"   == false ]] && TRAIN_SAVE_RATE=20
  [[ "$EP_EXPLICIT"   == false ]] && TRAIN_EPOCHS=3
  [[ "$NAME_EXPLICIT" == false ]] && TRAIN_NAME=scratch
fi

# ── Compute default mix ratios from file sizes ────────────────────────────────
if [[ "$NEW_RATIO_EXPLICIT" == false && "$ARCHIVE_RATIO_EXPLICIT" == false ]]; then
  new_sz=0;     [[ -f "$NEW_IN"      ]] && new_sz=$(stat -c%s "$NEW_IN")
  archive_sz=0; [[ -f "$ARCHIVE_IN"  ]] && archive_sz=$(stat -c%s "$ARCHIVE_IN")
  total_sz=$(( new_sz + archive_sz ))
  if (( total_sz > 0 )); then
    read -r NEW_RATIO ARCHIVE_RATIO < <(python3 -c "
n=$new_sz; a=$archive_sz; t=n+a
print(f'{n/t:.4f}', f'{a/t:.4f}')")
  else
    NEW_RATIO=0.5
    ARCHIVE_RATIO=0.5
  fi
elif [[ "$NEW_RATIO_EXPLICIT" == false ]]; then
  NEW_RATIO=$(python3 -c "print(f'{max(0.0, 1.0 - float($ARCHIVE_RATIO)):.4f}')")
elif [[ "$ARCHIVE_RATIO_EXPLICIT" == false ]]; then
  ARCHIVE_RATIO=$(python3 -c "print(f'{max(0.0, 1.0 - float($NEW_RATIO)):.4f}')")
fi

# ── Print startup config ──────────────────────────────────────────────────────
echo "=== Pipeline config ==="
new_pos=0;     [[ -f "$NEW_IN"     ]] && new_pos=$(( $(stat -c%s "$NEW_IN") / 32 ))
archive_pos_p=0; [[ -f "$ARCHIVE_IN" ]] && archive_pos_p=$(( $(stat -c%s "$ARCHIVE_IN") / 32 ))
printf "  new data    : %s (%d positions)\n" "$NEW_IN" "$new_pos"
printf "  archive     : %s (%d positions)\n" "$ARCHIVE_IN" "$archive_pos_p"
printf "  mix ratios  : new=%-6s  archive=%s\n" "$NEW_RATIO" "$ARCHIVE_RATIO"
printf "  mix output  : %s\n" "$OUT"
if [[ "$SKIP_TRAIN" == false ]]; then
  if [[ "$FROM_SCRATCH" == true ]]; then
    printf "  training    : from-scratch  lr=%s  wdl=%s  superbatches=%s  epochs=%s\n" \
      "$TRAIN_LR" "$TRAIN_WDL" "$TRAIN_SUPERBATCHES" "$TRAIN_EPOCHS"
  else
    printf "  training    : fine-tune  lr=%s  wdl=%s  superbatches=%s  init-net=%s\n" \
      "$TRAIN_LR" "$TRAIN_WDL" "$TRAIN_SUPERBATCHES" "$(basename "$TRAIN_INIT_NET")"
  fi
fi
echo "======================="

# ── Auto-use mixed output as training data when mix ran and data not set ──────
if [[ "$SKIP_MIX" == false && "$TRAIN_DATA_EXPLICIT" == false ]]; then
  TRAIN_DATA="$OUT"
fi

# If mix is skipped but a mixed dataset already exists, prefer it by default.
# This avoids accidentally training on raw fresh data (usually draw-heavy).
if [[ "$SKIP_MIX" == true && "$TRAIN_DATA_EXPLICIT" == false && -f "$OUT" ]]; then
  TRAIN_DATA="$OUT"
  echo "[info] Using existing mixed dataset for training: $OUT"
fi

# Resolve relative paths to absolute (training runs from train/ subdir)
[[ "$TRAIN_DATA"     = /* ]] || TRAIN_DATA="$REPO_ROOT/$TRAIN_DATA"
[[ "$TRAIN_INIT_NET" = /* ]] || TRAIN_INIT_NET="$REPO_ROOT/$TRAIN_INIT_NET"

# ── Input validation ──────────────────────────────────────────────────────────
check_bin() {
  [[ -f "$1" ]] || { echo "Missing input: $1" >&2; exit 1; }
  local sz; sz=$(stat -c%s "$1")
  (( sz % 32 == 0 )) || { echo "Input is not 32-byte aligned: $1" >&2; exit 1; }
}

if [[ "$SKIP_MIX" == false ]]; then
  if [[ "$SKIP_DATAGEN" == true ]]; then
    if [[ -f "$NEW_IN" ]]; then
      check_bin "$NEW_IN"
    else
      echo "[warn] New data not found: $NEW_IN — new-ratio forced to 0"
      NEW_RATIO=0
    fi
  fi
  if [[ -f "$ARCHIVE_IN" ]]; then
    check_bin "$ARCHIVE_IN"
  else
    echo "[warn] Archive not found: $ARCHIVE_IN — archive-ratio forced to 0"
    ARCHIVE_RATIO=0
  fi
fi

if [[ "$SKIP_TRAIN" == false && "$FROM_SCRATCH" == false ]]; then
  [[ -f "$TRAIN_INIT_NET" ]] || { echo "Missing init-net: $TRAIN_INIT_NET" >&2; exit 1; }
fi

BULLET_UTILS="$REPO_ROOT/../bullet/target/release/bullet-utils"
if [[ "$SKIP_MIX" == false ]] && [[ ! -x "$BULLET_UTILS" ]]; then
  cargo build --release -p bullet-utils --manifest-path "$REPO_ROOT/../bullet/Cargo.toml"
fi

mkdir -p "$WORK_DIR"

TOTAL_PHASES=9

# ── Phase 1: Datagen ──────────────────────────────────────────────────────────
echo "[1/$TOTAL_PHASES] Datagen"
if [[ "$SKIP_DATAGEN" == true ]]; then
  echo "  skipped."
else
  echo "  $DG_GAMES games, depth $DG_DEPTH, $DG_THREADS threads → $NEW_IN"
  dg_args=(
    --games   "$DG_GAMES"
    --depth   "$DG_DEPTH"
    --threads "$DG_THREADS"
    --hash    "$DG_HASH"
    --output  "$NEW_IN"
  )
  [[ -d "$DG_SYZYGY" ]] && dg_args+=(--syzygy "$DG_SYZYGY")
  "$REPO_ROOT/target/release/chess-datagen" "${dg_args[@]}"
fi

# ── Phase 2: Clean fresh data ─────────────────────────────────────────────────
echo "[2/$TOTAL_PHASES] Cleaning fresh data"
FRESH_TO_SHUFFLE="$NEW_IN"
if [[ "$SKIP_MIX" == true ]]; then
  echo "  skipped (--skip-mix)."
elif [[ "$SKIP_CLEAN" == true && "$DEDUP" == false ]]; then
  echo "  skipped (--skip-clean and --dedup not set)."
elif [[ ! -f "$NEW_IN" ]]; then
  echo "  skipped (no new data)."
else
  clean_args=(
    --input                     "$NEW_IN"
    --output                    "$WORK_DIR/fresh_clean.bin"
    --target-draw-ratio         "$DRAW_RATIO"
    --drop-king-only            "$DROP_KING_ONLY"
    --drop-high-abs-score-draws "$DROP_HIGH_ABS_DRAWS"
  )
  [[ "$DEDUP" == true ]] && clean_args+=(--dedup)
  cargo run --release --manifest-path "$REPO_ROOT/train/Cargo.toml" --bin clean_data -- "${clean_args[@]}"
  FRESH_TO_SHUFFLE="$WORK_DIR/fresh_clean.bin"
fi

# ── Phase 3: Shuffle fresh data ───────────────────────────────────────────────
echo "[3/$TOTAL_PHASES] Shuffling fresh data"
if [[ "$SKIP_MIX" == true ]]; then
  echo "  skipped (--skip-mix)."
elif [[ ! -f "$FRESH_TO_SHUFFLE" ]]; then
  echo "  skipped (no new data)."
else
  "$BULLET_UTILS" shuffle \
    --input      "$FRESH_TO_SHUFFLE" \
    --mem-used-mb "$MEM_MB" \
    --output     "$WORK_DIR/fresh_shuf.bin"
fi

# ── Phase 4: Sample inputs (skip dd copy when using all positions) ────────────
echo "[4/$TOTAL_PHASES] Sampling inputs"
input_new=""
input_archive=""
if [[ "$SKIP_MIX" == true ]]; then
  echo "  skipped (--skip-mix)."
else
  new_pos=0
  [[ -f "$WORK_DIR/fresh_shuf.bin" ]] && new_pos=$(( $(stat -c%s "$WORK_DIR/fresh_shuf.bin") / 32 ))
  archive_pos=0
  [[ -f "$ARCHIVE_IN" ]] && archive_pos=$(( $(stat -c%s "$ARCHIVE_IN") / 32 ))

  max_avail=$(( new_pos + archive_pos ))
  (( MAX_POSITIONS > max_avail )) && MAX_POSITIONS=$max_avail

  read -r target_new target_archive < <(
  python3 - <<PY
max_pos   = int($MAX_POSITIONS)
ratios    = [float($NEW_RATIO), float($ARCHIVE_RATIO)]
ratio_sum = sum(ratios)
if ratio_sum <= 0:
    raise SystemExit("Ratio sum must be > 0")
ratios = [r / ratio_sum for r in ratios]
vals   = [int(max_pos * r) for r in ratios]
vals[0] += max_pos - sum(vals)
print(*vals)
PY
  )

  use_new=$(( target_new < new_pos ? target_new : new_pos ))
  use_archive=$(( target_archive < archive_pos ? target_archive : archive_pos ))

  used=$(( use_new + use_archive ))
  remaining=$(( MAX_POSITIONS - used ))
  # Only top up from a source if its target ratio was non-zero; an explicit
  # ratio=0 means "never use this source", not "use it as overflow filler".
  if (( remaining > 0 && target_new > 0 )); then
    add=$(( new_pos - use_new )); take=$(( add < remaining ? add : remaining ))
    use_new=$(( use_new + take )); remaining=$(( remaining - take ))
  fi
  if (( remaining > 0 && target_archive > 0 )); then
    add=$(( archive_pos - use_archive )); take=$(( add < remaining ? add : remaining ))
    use_archive=$(( use_archive + take ))
  fi

  (( use_new + use_archive > 0 )) || { echo "No positions selected from any source." >&2; exit 1; }

  echo "  new=$use_new  archive=$use_archive"

  rm -f "$WORK_DIR/part_new.bin" "$WORK_DIR/part_archive.bin"

  # Use source files directly when taking all positions; dd only when subsampling.
  if (( use_new > 0 )); then
    if (( use_new < new_pos )); then
      dd if="$WORK_DIR/fresh_shuf.bin" of="$WORK_DIR/part_new.bin" bs=32 count="$use_new" status=none
      input_new="$WORK_DIR/part_new.bin"
    else
      input_new="$WORK_DIR/fresh_shuf.bin"
    fi
  fi

  if (( use_archive > 0 )); then
    if (( use_archive < archive_pos )); then
      dd if="$ARCHIVE_IN" of="$WORK_DIR/part_archive.bin" bs=32 count="$use_archive" status=none
      input_archive="$WORK_DIR/part_archive.bin"
    else
      input_archive="$ARCHIVE_IN"
    fi
  fi
fi

# ── Phase 5: Interleave ───────────────────────────────────────────────────────
# Both sources are already shuffled (fresh in Phase 3, archive via prepare_archive.sh).
# Single streaming pass: reads each source once, writes output once.
echo "[5/$TOTAL_PHASES] Interleaving"
if [[ "$SKIP_MIX" == true ]]; then
  echo "  skipped (--skip-mix)."
else
  interleave_inputs=()
  [[ -n "$input_new"     ]] && interleave_inputs+=("$input_new")
  [[ -n "$input_archive" ]] && interleave_inputs+=("$input_archive")

  (( ${#interleave_inputs[@]} > 0 )) || { echo "No inputs for interleave." >&2; exit 1; }

  if (( ${#interleave_inputs[@]} == 1 )); then
    # Single source — already shuffled; copy straight to output.
    echo "  single source, copying to output"
    cp "${interleave_inputs[0]}" "$OUT"
  else
    "$BULLET_UTILS" interleave "${interleave_inputs[@]}" --output "$OUT"
  fi
fi

# ── Phase 6: Data quality check ───────────────────────────────────────────────
echo "[6/$TOTAL_PHASES] Data quality check"
if [[ "$SKIP_TRAIN" == true || "$RUN_QUALITY_CHECK" == false ]]; then
  if [[ "$SKIP_TRAIN" == true ]]; then
    echo "  skipped (--skip-train)."
  else
    echo "  skipped (--skip-quality-check)."
  fi
else
  [[ -f "$TRAIN_DATA" ]] || { echo "Training data not found: $TRAIN_DATA" >&2; exit 1; }
  qc_args=(
    --input "$TRAIN_DATA"
    --sample "$QUALITY_SAMPLE"
  )
  [[ "$QUALITY_STRICT" == true ]] && qc_args+=(--strict)
  cargo run --release --manifest-path "$REPO_ROOT/train/Cargo.toml" --bin quality_check -- "${qc_args[@]}"
fi

# ── Phase 7: Train ────────────────────────────────────────────────────────────
echo "[7/$TOTAL_PHASES] Training"
if [[ "$SKIP_TRAIN" == true ]]; then
  echo "  skipped."
else
  [[ -f "$TRAIN_DATA" ]] || { echo "Training data not found: $TRAIN_DATA" >&2; exit 1; }
  if [[ "$TRAIN_DATA_EXPLICIT" == false && "$SKIP_MIX" == true && "$TRAIN_DATA" == "$REPO_ROOT/data/new.bin" ]]; then
    echo "[warn] Training on raw new data: $TRAIN_DATA"
    echo "[warn] This data is typically draw-heavy; run with --run-mix or pass --train-data for better signal."
  fi
  ckpt_dir="$REPO_ROOT/train/checkpoints"
  # Wipe any stale checkpoints from a previous run of this name so only the
  # current run's checkpoints are present afterwards.
  for d in "$ckpt_dir/$TRAIN_NAME"-*/; do
    [[ -d "$d" ]] && { rm -rf "$d"; echo "  removed stale checkpoint: $d"; }
  done
  if [[ "$FROM_SCRATCH" == true ]]; then
    echo "  mode=from-scratch  lr=$TRAIN_LR  wdl=$TRAIN_WDL  superbatches=$TRAIN_SUPERBATCHES  epochs=$TRAIN_EPOCHS${TRAIN_BPS:+  bps=$TRAIN_BPS}"
  else
    echo "  mode=fine-tune  lr=$TRAIN_LR  wdl=$TRAIN_WDL  superbatches=$TRAIN_SUPERBATCHES  init-net=$(basename "$TRAIN_INIT_NET")${TRAIN_BPS:+  bps=$TRAIN_BPS}"
  fi
  train_args=(
    --data         "$TRAIN_DATA"
    --lr           "$TRAIN_LR"
    --wdl          "$TRAIN_WDL"
    --superbatches "$TRAIN_SUPERBATCHES"
    --save-rate    "$TRAIN_SAVE_RATE"
    --epochs       "$TRAIN_EPOCHS"
    --name         "$TRAIN_NAME"
  )
  [[ "$FROM_SCRATCH" == false ]] && train_args+=(--init-net "$TRAIN_INIT_NET")
  [[ -n "$TRAIN_BPS" ]] && train_args+=(--bps "$TRAIN_BPS")
  (cd "$REPO_ROOT/train" && cargo run --release --bin train -- "${train_args[@]}")
fi

# ── Phase 8: Summary ──────────────────────────────────────────────────────────
echo "[8/$TOTAL_PHASES] Summary"
if [[ "$SKIP_MIX" == false && -f "$OUT" ]]; then
  out_pos=$(( $(stat -c%s "$OUT") / 32 ))
  [[ -f "$NEW_IN"     ]] && echo "  new data      : $NEW_IN ($(( $(stat -c%s "$NEW_IN") / 32 )) positions)"
  [[ -f "$ARCHIVE_IN" ]] && echo "  archive       : $ARCHIVE_IN ($(( $(stat -c%s "$ARCHIVE_IN") / 32 )) positions)"
  echo "  training data : $OUT ($out_pos positions)"
fi
if [[ "$SKIP_TRAIN" == false ]]; then
  log_file="$REPO_ROOT/train/checkpoints/$TRAIN_NAME-$TRAIN_SUPERBATCHES/log.txt"
  if [[ -f "$log_file" ]]; then
    echo ""
    (cd "$REPO_ROOT/train" && python3 best_checkpoint.py "checkpoints/$TRAIN_NAME-$TRAIN_SUPERBATCHES/log.txt")
    echo ""
  fi

  final_ckpt="$REPO_ROOT/train/checkpoints/$TRAIN_NAME-$TRAIN_SUPERBATCHES"
  if [[ -f "$final_ckpt/quantised.bin" ]]; then
    echo "  Test vs SF3190 (authoritative):"
    echo "    fastchess \\"
    echo "      -engine cmd=target/release/endspiel option.EvalFile=$final_ckpt/quantised.bin option.Hash=64 option.Threads=1 name=candidate \\"
    echo "      -engine cmd=/usr/games/stockfish option.UCI_LimitStrength=true option.UCI_Elo=3190 option.Hash=64 option.Threads=1 name=sf3190 \\"
    echo "      -openings file=$REPO_ROOT/assets/openings.epd format=epd order=random \\"
    echo "      -each tc=10+0.1 -rounds 100 -concurrency 8 -recover"
    echo ""
    echo "  Promote if stronger (rebuild after copy):"
    echo "    cargo build --release"
  fi
fi

# ── Phase 9: Cleanup ──────────────────────────────────────────────────────────
echo "[9/$TOTAL_PHASES] Cleanup"
if [[ "$NO_CLEANUP" == true ]]; then
  echo "  skipped (--no-cleanup)."
else
  rm -rf "$WORK_DIR"
  echo "  removed: $WORK_DIR"
fi

echo "Done."
