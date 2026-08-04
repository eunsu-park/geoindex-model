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
models see identically scaled inputs and their outputs are directly comparable.

Validation is split too, and that split matters: training uses its validation set for early
stopping and checkpoint selection, so a branch model told to validate on the full set is
selected on a distribution it was never trained for. Train on the branch validation index, then
run validate.py again with the FULL index to score.

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
    ap_.add_argument("--source-validation-index", default="total_ap/validation_index.csv")
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

    def split(source, kind):
        df_ = pd.read_csv(os.path.join(root, source))
        df_["datetime"] = pd.to_datetime(df_["datetime"])
        df_["peak"] = df_["datetime"].map(peak_of)
        missing = int(df_["peak"].isna().sum())
        if missing:
            print(f"note: {missing} {kind} anchors have no full window and are dropped")
            df_ = df_.dropna(subset=["peak"])
        storm = df_["peak"] >= args.threshold
        for name, sel in (("storm", storm), ("quiet", ~storm)):
            path = os.path.join(out_dir, f"{kind}_index_{name}.csv")
            df_.loc[sel, ["datetime", "label"]].to_csv(path, index=False)
            print(f"  {kind:10s} {name:6s} {int(sel.sum()):8,d} "
                  f"({100*sel.mean():5.1f} %)  -> {os.path.basename(path)}")
        return len(df_)

    n_train = split(args.source_index, "train")
    n_val = split(args.source_validation_index, "validation")

    print(f"\nthreshold ap30 >= {args.threshold:g} (Kp 5o) over a "
          f"{args.out_steps*0.5:g} h window; {n_train:,} training and {n_val:,} validation "
          f"anchors classified")
    print("Normalization statistics stay pooled (table_stats_ap.pkl) so the two models see")
    print("identically scaled inputs.")
    print()
    print("The BRANCH validation index is for training only. Early stopping and checkpoint")
    print("selection use whatever validation set training is given, so a branch model told to")
    print("validate on the full set is selected on a distribution it was never trained for --")
    print("the storm model then stops after a few epochs and stays conservative. Train on the")
    print("branch index, then run validate.py again with the FULL index to score.")


if __name__ == "__main__":
    main()
