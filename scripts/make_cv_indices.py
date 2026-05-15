#!/usr/bin/env python
"""Generate cross-validation fold index files with a 24-hour embargo.

Splits the full event index by calendar year into five expanding-window
folds. For each fold, the training index excludes the 24-hour window
immediately preceding the validation start, so a training sample's
target window cannot overlap with a validation sample's input window
(input is 12 h before T=0, target is 12 h after T=0; required gap is
24 h = input_span + target_span).

Expanding window folds:
    Fold 1: train 1995..2005-12-30, val 2006-01-01..2009-12-31
    Fold 2: train 1995..2009-12-30, val 2010-01-01..2013-12-31
    Fold 3: train 1995..2013-12-30, val 2014-01-01..2017-12-31
    Fold 4: train 1995..2017-12-30, val 2018-01-01..2021-12-31
    Fold 5: train 1995..2021-12-30, val 2022-01-01..2025-12-31

Source: union of {train_index.csv, validation_index.csv}.  The original
label column is preserved unchanged; rows are partitioned by datetime
only.

Usage:
    python scripts/make_cv_indices.py \\
        --source-train ${DATA_ROOT}/total/train_index.csv \\
        --source-val   ${DATA_ROOT}/total/validation_index.csv \\
        --output-dir   ${DATA_ROOT}/cv5
"""

import argparse
import os
import sys
from typing import List, Tuple

import pandas as pd


# (fold_id, val_start_inclusive, val_end_exclusive)
FOLDS: List[Tuple[int, str, str]] = [
    (1, "2006-01-01", "2010-01-01"),
    (2, "2010-01-01", "2014-01-01"),
    (3, "2014-01-01", "2018-01-01"),
    (4, "2018-01-01", "2022-01-01"),
    (5, "2022-01-01", "2026-01-01"),
]

EMBARGO = pd.Timedelta(hours=24)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-train", required=True, help="Path to source train_index.csv")
    p.add_argument("--source-val", required=True, help="Path to source validation_index.csv")
    p.add_argument("--output-dir", required=True, help="Output directory (will create foldN/ subdirs)")
    return p.parse_args()


def load_full_index(train_path: str, val_path: str) -> pd.DataFrame:
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    for name, df in (("train", train_df), ("val", val_df)):
        if "datetime" not in df.columns or "label" not in df.columns:
            sys.exit(f"Error: {name} index is missing required columns (datetime, label).")

    df = pd.concat([train_df, val_df], ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return df


def write_fold(df: pd.DataFrame, fold_id: int, val_start: pd.Timestamp,
               val_end: pd.Timestamp, output_dir: str) -> None:
    train_cutoff = val_start - EMBARGO
    train_mask = df["datetime"] < train_cutoff
    val_mask = (df["datetime"] >= val_start) & (df["datetime"] < val_end)

    train_split = df.loc[train_mask, ["datetime", "label"]]
    val_split = df.loc[val_mask, ["datetime", "label"]]

    fold_dir = os.path.join(output_dir, f"fold{fold_id}")
    os.makedirs(fold_dir, exist_ok=True)
    train_split.to_csv(os.path.join(fold_dir, "train_index.csv"), index=False)
    val_split.to_csv(os.path.join(fold_dir, "validation_index.csv"), index=False)

    train_pos = int(train_split["label"].sum())
    val_pos = int(val_split["label"].sum())
    print(
        f"Fold {fold_id}: "
        f"train={len(train_split):>6} ({train_pos} pos) cutoff<{train_cutoff:%Y-%m-%d %H:%M}, "
        f"val={len(val_split):>6} ({val_pos} pos) [{val_start:%Y-%m-%d}, {val_end:%Y-%m-%d})"
    )


def main() -> None:
    args = parse_args()

    df = load_full_index(args.source_train, args.source_val)
    print(f"Loaded {len(df)} unique events from {df['datetime'].min()} to {df['datetime'].max()}")

    os.makedirs(args.output_dir, exist_ok=True)
    for fold_id, val_start, val_end in FOLDS:
        write_fold(df, fold_id, pd.Timestamp(val_start), pd.Timestamp(val_end), args.output_dir)

    print(f"\nWrote 5 fold index pairs to: {args.output_dir}")


if __name__ == "__main__":
    main()
