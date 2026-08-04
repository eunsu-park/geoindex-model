"""Split the training anchors into a quiet regime and a storm regime at ap30 >= 48.

The two-model idea needs no code change at all. A regime is a property of the target, known at
training time, and table mode already takes the training anchors from a CSV -- so the whole
split is two index files and a Hydra override. `src/`, `configs/` and `scripts/` are untouched,
which is the point: the tree was deliberately restored to the manuscript state.

Why 48. ap30 is not continuous -- it is a 36-level ladder, and between 48 and 56 there is
nothing, so a threshold written as "50" silently means 56 and "100" means 111. 48 is on the
ladder (Kp 5o). Measured with ridges over 1998-2021 / 2022-2025, it is also where the split
behaves best:

                              >= 39      >= 48      >= 56
  storm training anchors     79,246     56,571     40,806
  P(storm|x) vs observed  0.251/0.254  0.187/0.187  0.142/0.136
  rho within the storm branch  0.619      0.599      0.581   (pooled model: 0.594)
  reproduction on storms       0.619      0.649      0.683

48 is the only one where the branch probability is exactly calibrated while the storm branch
still discriminates better than the pooled model on the same rows.

Note it is "Kp 5o", not "the G1 threshold" -- G1 is Kp 5, which spans ap 39-56. This project has
mislabelled the G scale once already; the docs should say Kp 5o.

Both regimes share the pooled normalization statistics (`table_stats_ap.pkl`), so the two
models see identically scaled inputs and their outputs are directly comparable. Validation stays
the full index for both, so each model is scored on everything and the branches are sliced
afterwards.

Usage:
    python build_regime_indices.py                       # writes to the data root
    python build_regime_indices.py --threshold 39        # a different ladder level
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

LADDER = [0, 2, 3, 4, 5, 6, 7, 9, 12, 15, 18, 22, 27, 32, 39, 48, 56, 67, 80, 94, 111, 132,
          154, 179, 207, 236, 265, 294, 324, 355, 388, 421, 456, 494, 534, 617]


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap_.add_argument("--data-root", default="~/Projects/GeoIndex/datasets")
    ap_.add_argument("--table", default="data.parquet")
    ap_.add_argument("--source-index", default="total_ap/train_index.csv")
    ap_.add_argument("--out-dir", default="regime48_ap")
    ap_.add_argument("--threshold", type=float, default=48.0)
    ap_.add_argument("--target", default="ap30")
    ap_.add_argument("--out-steps", type=int, default=24,
                     help="forecast window length; must match the +io group")
    args = ap_.parse_args()

    if args.threshold not in LADDER:
        nearest = min(LADDER, key=lambda v: abs(v - args.threshold))
        raise SystemExit(
            f"{args.threshold:g} is not on the ap ladder, so the effective threshold would "
            f"silently be {min(v for v in LADDER if v >= args.threshold)}. "
            f"Use {nearest} or another ladder level: {LADDER[10:22]}")

    root = os.path.expanduser(args.data_root)
    out_dir = os.path.join(root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    table = pd.read_parquet(os.path.join(root, args.table)).set_index("datetime").sort_index()
    grid = pd.date_range(table.index[0], table.index[-1], freq="30min")
    series = table[args.target].reindex(grid).ffill()
    values = series.to_numpy(float)

    # The regime of an anchor is the maximum over ITS forecast window, [T, T+out_steps),
    # which is the convention the stored validation archives use.
    n_a = len(grid) - args.out_steps
    peak = np.lib.stride_tricks.sliding_window_view(values, args.out_steps)[:n_a].max(axis=1)
    peak_of = pd.Series(peak, index=grid[:n_a])

    src = pd.read_csv(os.path.join(root, args.source_index))
    src["datetime"] = pd.to_datetime(src["datetime"])
    src["peak"] = src["datetime"].map(peak_of)
    missing = src["peak"].isna().sum()
    if missing:
        print(f"note: {missing} anchors have no full window in the table and are dropped")
        src = src.dropna(subset=["peak"])

    storm = src["peak"] >= args.threshold
    for name, sel in (("storm", storm), ("quiet", ~storm)):
        path = os.path.join(out_dir, f"train_index_{name}.csv")
        src.loc[sel, ["datetime", "label"]].to_csv(path, index=False)
        print(f"{name:6s} {int(sel.sum()):8,d} anchors ({100*sel.mean():5.1f} %)  -> {path}")

    print(f"\nsource index {len(src):,} anchors, threshold ap30 >= {args.threshold:g} "
          f"(Kp 5o) over a {args.out_steps*0.5:g} h window")
    print(f"validation is unchanged for both regimes; normalization statistics are shared.")


if __name__ == "__main__":
    main()
