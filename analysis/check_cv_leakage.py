"""Check that cross-validation folds are contiguous in time and buffered against the window.

Overlapping input windows make a random split look far better than it is: with a 12-hour
window at 30-minute cadence, two anchors an hour apart share 22 of 24 input steps, so a
validation anchor drawn next to a training anchor is very nearly a training sample.
Ahmadzadeh et al. (2021) measured a 53% inflation from exactly this (mean TSS 0.92 on a
random split against 0.60 on a time-segmented one).

Two things have to hold, and they are different. The folds must be contiguous blocks in
time -- a fold whose validation anchors are scattered through the record is a random split
by another name. And there must be a gap between the two sets at least as wide as the input
window, or an anchor just outside a training block still shares most of its input with one
inside it.

Usage:
    python analysis/check_cv_leakage.py --datasets-root /path/to/datasets --cv-dir cv5_ap
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd


def fold_report(train_csv: str, valid_csv: str, window_hours: float, cadence_min: float) -> dict:
    """Summarise one fold's temporal structure and its overlap with the training set."""
    tr = pd.read_csv(train_csv, parse_dates=["datetime"]).datetime.sort_values()
    va = pd.read_csv(valid_csv, parse_dates=["datetime"]).datetime.sort_values()

    step = pd.Timedelta(minutes=cadence_min)
    gaps = va.diff().dropna()
    n_blocks = int((gaps > step * 1.5).sum()) + 1

    shared = len(set(tr).intersection(set(va)))
    t = tr.values.astype("datetime64[s]").astype(np.int64)
    v = va.values.astype("datetime64[s]").astype(np.int64)
    limit = window_hours * 3600
    pos = np.searchsorted(t, v)
    near = 0
    for k, x in zip(pos, v):
        for j in (k - 1, k):
            if 0 <= j < len(t) and abs(t[j] - x) <= limit:
                near += 1
                break
    return {"n_valid": len(va), "first": va.iloc[0], "last": va.iloc[-1],
            "n_blocks": n_blocks, "shared_anchors": shared,
            "within_window": near / len(va) if len(va) else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description="Check CV folds for window-overlap leakage")
    ap.add_argument("--datasets-root", required=True)
    ap.add_argument("--cv-dir", required=True, help="e.g. cv5_ap")
    ap.add_argument("--window-hours", type=float, default=12.0,
                    help="input window length; the required buffer between the two sets")
    ap.add_argument("--cadence-min", type=float, default=30.0)
    args = ap.parse_args()

    folds = sorted(glob.glob(os.path.join(args.datasets_root, args.cv_dir, "fold*")))
    if not folds:
        raise SystemExit(f"no folds under {os.path.join(args.datasets_root, args.cv_dir)}")

    print(f"{args.cv_dir}: buffer required between train and validation = "
          f"{args.window_hours:g} h (the input window)\n")
    print(f"{'fold':8s} {'n':>7s} {'validation span':>22s} {'blocks':>7s} {'shared':>7s} "
          f"{'within window':>14s}")
    worst = 0.0
    for d in folds:
        r = fold_report(os.path.join(d, "train_index.csv"),
                        os.path.join(d, "validation_index.csv"),
                        args.window_hours, args.cadence_min)
        worst = max(worst, r["within_window"])
        span = f"{r['first']:%Y-%m} - {r['last']:%Y-%m}"
        print(f"{os.path.basename(d):8s} {r['n_valid']:7d} {span:>22s} {r['n_blocks']:7d} "
              f"{r['shared_anchors']:7d} {100*r['within_window']:13.1f}%")

    print("\nA fold spanning one continuous date range is a time split even when the block "
          "count is\nhigh -- gaps in the source data break a contiguous period into many "
          "runs. What matters is\nthe last column: validation anchors that sit within one "
          "input window of a training anchor.")
    if worst > 0.05:
        print(f"\nLEAKY: up to {100*worst:.1f}% of validation anchors share most of their "
              f"input window with\na training anchor. Scores from these folds are inflated; "
              f"widen the gap between the sets.")
    else:
        print(f"\nCLEAN: at most {100*worst:.1f}% of validation anchors fall within an input "
              f"window of a\ntraining anchor, so the fold scores are not inflated by window "
              f"overlap.")


if __name__ == "__main__":
    main()
