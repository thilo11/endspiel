#!/usr/bin/env python3
"""
Summarise a Bullet training run and identify the best checkpoint.

Usage:
    python3 best_checkpoint.py                        # auto-find latest log
    python3 best_checkpoint.py checkpoints/endspiel-300/log.txt
"""

import sys
import os
import re
import glob
from collections import defaultdict


def find_latest_log(checkpoints_dir="checkpoints"):
    """Return the log.txt from the highest-numbered checkpoint directory."""
    pattern = os.path.join(checkpoints_dir, "*", "log.txt")
    logs = glob.glob(pattern)
    if not logs:
        return None

    def checkpoint_number(path):
        m = re.search(r"-(\d+)[/\\]log\.txt$", path)
        return int(m.group(1)) if m else -1

    return max(logs, key=checkpoint_number)


def parse_log(path):
    """
    Parse 'superbatch,batch,loss' lines.
    Returns dict: superbatch -> list of loss values.
    """
    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 3:
                continue
            try:
                sb = int(parts[0])
                loss = float(parts[2])
                data[sb].append(loss)
            except ValueError:
                continue
    return data


def find_saved_checkpoints(log_path):
    """Return a set of superbatch numbers that have a quantised.bin on disk."""
    checkpoints_dir = os.path.dirname(os.path.dirname(log_path))
    name_match = re.search(r"([^/\\]+)-\d+[/\\]log\.txt$", log_path)
    prefix = name_match.group(1) if name_match else None

    saved = set()
    if prefix:
        for d in glob.glob(os.path.join(checkpoints_dir, f"{prefix}-*")):
            m = re.search(rf"{re.escape(prefix)}-(\d+)$", d)
            if m and os.path.exists(os.path.join(d, "quantised.bin")):
                saved.add(int(m.group(1)))
    return saved


def summarise(log_path, top_n=5):
    print(f"Log: {log_path}\n")

    data = parse_log(log_path)
    if not data:
        print("No data found in log.")
        return

    avg = {sb: sum(losses) / len(losses) for sb, losses in data.items()}
    saved = find_saved_checkpoints(log_path)
    max_sb = max(avg)

    # Best superbatch by loss, regardless of whether a checkpoint was saved
    best_any_sb, best_any_loss = min(avg.items(), key=lambda x: x[1])

    # Best superbatch among those with a saved checkpoint on disk
    saved_in_log = {sb: loss for sb, loss in avg.items() if sb in saved}
    if saved_in_log:
        best_saved_sb, best_saved_loss = min(saved_in_log.items(), key=lambda x: x[1])
    else:
        best_saved_sb, best_saved_loss = best_any_sb, best_any_loss

    # Last N saved checkpoints in superbatch order (most relevant for cosine decay)
    recent_saved = sorted(saved_in_log.items(), key=lambda x: x[0])

    print(f"{'Superbatch':<14} {'Avg loss':<12} {'Note'}")
    print("-" * 46)
    # Determine recommended sb for table annotations
    _rec_for_table = (min(saved, key=lambda s: abs(s - best_any_sb))
                      if best_any_sb not in saved else best_saved_sb) if saved else max_sb

    # Always show the recommended checkpoint, even if it falls outside the last top_n
    tail = recent_saved[-(top_n):]
    tail_sbs = {sb for sb, _ in tail}
    if _rec_for_table not in tail_sbs and any(sb == _rec_for_table for sb, _ in recent_saved):
        rec_row = [(sb, loss) for sb, loss in recent_saved if sb == _rec_for_table]
        rows = sorted(rec_row + list(tail), key=lambda x: x[0])
    else:
        rows = tail
    shown_sbs = {sb for sb, _ in rows}
    skipped = sum(1 for sb, _ in recent_saved if sb not in shown_sbs)

    for sb, loss in rows:
        tag = ""
        if sb == _rec_for_table:
            tag = "<-- RECOMMENDED"
            if sb == max_sb:
                tag += " (final)"
        elif sb == max_sb:
            tag = "(final)"
        elif sb == best_saved_sb and sb != _rec_for_table:
            tag = "(lowest saved loss)"
        print(f"  {sb:<12} {loss:<12.6f} {tag}")

    if skipped > 0:
        print(f"  ... ({skipped} earlier checkpoints not shown)")

    print()

    # Summary
    dir_prefix = re.sub(r"-\d+[/\\]log\.txt$", "", log_path)

    # Recommend the lowest-loss saved checkpoint.
    # If the best-any was not saved, the nearest saved checkpoint is a better
    # choice than the final — empirically, the final can be worse when the net
    # peaks mid-run (common with WDL=0 or aggressive LR schedules).
    if best_any_sb not in saved:
        nearest = min(saved, key=lambda s: abs(s - best_any_sb)) if saved else None
        rec_sb = nearest
        print(f"True best (SB {best_any_sb}, loss {best_any_loss:.6f}) was not saved.")
        print(f"Recommended:   SB {rec_sb} (nearest saved to best)  —  avg loss {avg[rec_sb]:.6f}")
    else:
        rec_sb = best_saved_sb
        print(f"Recommended:   SB {rec_sb}  —  avg loss {best_saved_loss:.6f}")

    if rec_sb != max_sb:
        final_loss = avg[max_sb]
        delta = final_loss - avg[rec_sb]
        sign = "+" if delta >= 0 else ""
        print(f"Final (SB {max_sb}):  avg loss {final_loss:.6f}  ({sign}{delta:.6f} vs recommended)")
        print()
        print(f"The final checkpoint is NOT recommended here: its loss is higher than")
        print(f"the mid-run best, which indicates the net peaked before the end of")
        print(f"training (typical with WDL=0 or when LR decays past the noise floor).")
    else:
        print(f"(Final checkpoint — loss kept improving until the end.)")

    rec_path = f"{dir_prefix}-{rec_sb}/quantised.bin"
    print()
    print(f"To install:")
    print(f"  cp {rec_path} ../crates/chess-nnue/nets/default.nnue")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_path = find_latest_log()
        if not log_path:
            print("No log.txt found in checkpoints/. Pass a path explicitly.")
            sys.exit(1)

    if not os.path.exists(log_path):
        print(f"File not found: {log_path}")
        sys.exit(1)

    summarise(log_path)
