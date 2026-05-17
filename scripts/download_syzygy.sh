#!/usr/bin/env bash
# Download Syzygy endgame tablebases (WDL + DTZ) to assets/syzygy/.
#
# The tablebases are not stored in git; run this script once to populate the
# local directory.  The engine is then pointed to that directory via the UCI
# option "SyzygyPath" (see README).
#
# Usage:
#   bash scripts/download_syzygy.sh [--wdl-only] [--dtz-only] [--dir <path>] [--pieces 3-4-5|6|all]
#
# Defaults:
#   - Downloads both WDL (.rtbw) and DTZ (.rtbz) files
#   - Target directory: assets/syzygy  (relative to the repository root)
#   - Pieces: 3-4-5 (small, ~350 MB). Use --pieces 6 for 6-man (~150 GB) or
#     --pieces all for both sets.
#   - Mirror: rsync://tablebase.sesse.net/tablebase/syzygy/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST_DIR="$REPO_ROOT/assets/syzygy"
INCLUDE_WDL=true
INCLUDE_DTZ=true
PIECES="3-4-5"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --wdl-only) INCLUDE_DTZ=false ; shift ;;
        --dtz-only) INCLUDE_WDL=false ; shift ;;
        --dir)      DEST_DIR="$2" ; shift 2 ;;
        --pieces)   PIECES="$2" ; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2 ; exit 1 ;;
    esac
done

case "$PIECES" in
    3-4-5|6|all) ;;
    *) echo "Invalid --pieces: $PIECES (expected 3-4-5, 6, or all)" >&2; exit 1 ;;
esac

mkdir -p "$DEST_DIR"

# ---------------------------------------------------------------------------
# Build rsync filter
# ---------------------------------------------------------------------------
FILTER_ARGS=()
if $INCLUDE_WDL; then
    FILTER_ARGS+=(--include="*.rtbw")
fi
if $INCLUDE_DTZ; then
    FILTER_ARGS+=(--include="*.rtbz")
fi
FILTER_ARGS+=(--exclude="*")

# ---------------------------------------------------------------------------
# Primary mirror: Sesse's tablebase server
# ---------------------------------------------------------------------------
PRIMARY_BASE="rsync://tablebase.sesse.net/tablebase/syzygy"

# Build the list of rsync source paths based on --pieces.
# Sesse splits 6-man into separate dirs: 6-WDL (.rtbw) and 6-DTZ (.rtbz).
# 3-4-5 is a single dir containing both file types.
RSYNC_SOURCES=()
case "$PIECES" in
    3-4-5)
        RSYNC_SOURCES=("$PRIMARY_BASE/3-4-5/")
        ;;
    6)
        $INCLUDE_WDL && RSYNC_SOURCES+=("$PRIMARY_BASE/6-WDL/")
        $INCLUDE_DTZ && RSYNC_SOURCES+=("$PRIMARY_BASE/6-DTZ/")
        ;;
    all)
        RSYNC_SOURCES+=("$PRIMARY_BASE/3-4-5/")
        $INCLUDE_WDL && RSYNC_SOURCES+=("$PRIMARY_BASE/6-WDL/")
        $INCLUDE_DTZ && RSYNC_SOURCES+=("$PRIMARY_BASE/6-DTZ/")
        ;;
esac

echo "============================================================"
echo " Syzygy tablebase downloader"
echo "============================================================"
echo " Target : $DEST_DIR"
echo " Pieces : $PIECES"
echo " WDL    : $INCLUDE_WDL"
echo " DTZ    : $INCLUDE_DTZ"
echo " Sources:"
for src in "${RSYNC_SOURCES[@]}"; do
    echo "          $src"
done
echo ""
echo " Approximate sizes:"
echo "   3-4-5 WDL only : ~150 MB   (both: ~350 MB)"
echo "   6     WDL only : ~70 GB    (both: ~150 GB)"
echo "============================================================"
echo ""

# Check for rsync
if ! command -v rsync &>/dev/null; then
    echo "ERROR: rsync is not installed. Install it with your package manager." >&2
    echo "  Ubuntu/Debian : sudo apt install rsync" >&2
    echo "  macOS         : brew install rsync" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Download – only 3/4/5-man files (filter by piece-count prefix pattern)
# ---------------------------------------------------------------------------
# Syzygy file names encode piece counts as K*K* where * is any piece letter.
# 3-man: KXK   (3 chars between K…K pattern, e.g. KQK, KRK, KBK, KNK, KPK)
# 4-man: KXXK or KXKX
# 5-man: KXXXK or KXXKX or KXKXX
#
# We use rsync's --filter rules to include only files whose base name has at
# most 7 characters before .rtb{w,z} (the longest 5-man name is 7 chars,
# e.g. KQPKQP would be 6-man; 5-man max is KQQPKP = 6 chars).
# Practically: include all and let TB_LARGEST do the right thing at runtime.
# The extra 6-7 man files are large (many GB) and Sesse only hosts them on
# separate paths, so the filter below is sufficient.

echo "Starting rsync download (this may take several minutes to many hours)..."
echo ""

for src in "${RSYNC_SOURCES[@]}"; do
    echo ">>> Fetching from $src"
    rsync -avz --no-compress --progress \
          "${FILTER_ARGS[@]}" \
          "$src" \
          "$DEST_DIR/" \
        || {
            echo "" >&2
            echo "rsync from $src failed." >&2
            echo "Fallback mirror: https://syzygy-tables.info/" >&2
            exit 1
        }
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Download complete!"
echo " Location : $DEST_DIR"
WDL_COUNT=$(find "$DEST_DIR" -name "*.rtbw" 2>/dev/null | wc -l)
DTZ_COUNT=$(find "$DEST_DIR" -name "*.rtbz" 2>/dev/null | wc -l)
echo " WDL files (.rtbw) : $WDL_COUNT"
echo " DTZ files (.rtbz) : $DTZ_COUNT"
TOTAL_MB=$(du -sm "$DEST_DIR" 2>/dev/null | cut -f1 || echo "?")
echo " Total size        : ${TOTAL_MB} MB"
echo "============================================================"
echo ""
echo "To use in the engine, set the UCI option:"
echo "  setoption name SyzygyPath value $DEST_DIR"
