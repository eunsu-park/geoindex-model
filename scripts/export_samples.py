#!/usr/bin/env python
"""Materialize per-sample CSV files from the table-mode dataset.

Reads the single Parquet table + train/validation index CSVs that the
table-mode loader uses, and writes one CSV file per anchor time.  Each
output file contains the full 384-row window the model sees: 240 input
rows (T-5d to T-0) followed by 144 target rows (T-0 to T+3d), with the
datetime column preserved for traceability.

Values are written raw (un-normalized); recipients should compute their
own normalization statistics from the training portion only.

Usage:
    python scripts/export_samples.py \\
        --parquet   /Users/eunsupark/tmp/dataset/data.parquet \\
        --train-index /Users/eunsupark/tmp/dataset/total/train_index.csv \\
        --val-index   /Users/eunsupark/tmp/dataset/total/validation_index.csv \\
        --out-dir     /Users/eunsupark/AP
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


WINDOW_INPUT_START = -240   # T-5d (5 days * 48 points/day)
WINDOW_TARGET_END = 144     # T+3d (3 days * 48 points/day)
WINDOW_LEN = WINDOW_TARGET_END - WINDOW_INPUT_START  # 384


def export_split(
    df: pd.DataFrame,
    dt_to_row: dict,
    index_csv: Path,
    out_dir: Path,
    split_name: str,
) -> tuple[int, int]:
    """Write one CSV per anchor time in ``index_csv`` into ``out_dir``.

    Returns:
        (written, skipped) counts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_df = pd.read_csv(index_csv)
    idx_df["datetime"] = pd.to_datetime(idx_df["datetime"])
    anchors = idx_df["datetime"].values

    n_rows = len(df)
    written, skipped = 0, 0
    log_every = max(1, len(anchors) // 50)  # ~50 progress lines per split

    for i, anchor in enumerate(anchors):
        anchor64 = np.datetime64(anchor)
        ref_row = dt_to_row.get(anchor64)
        if ref_row is None:
            skipped += 1
            continue

        start = ref_row + WINDOW_INPUT_START
        end = ref_row + WINDOW_TARGET_END
        if start < 0 or end > n_rows:
            skipped += 1
            continue

        window = df.iloc[start:end]
        ts = pd.Timestamp(anchor)
        fname = ts.strftime("%Y%m%d%H%M%S") + ".csv"
        window.to_csv(out_dir / fname, index=False, float_format="%.6g")
        written += 1

        if (i + 1) % log_every == 0:
            print(
                f"  [{split_name}] {i + 1}/{len(anchors)} "
                f"({(i + 1) / len(anchors):.1%}) — "
                f"written={written}, skipped={skipped}",
                flush=True,
            )

    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--val-index", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    print(f"Loading parquet: {args.parquet}", flush=True)
    df = pd.read_parquet(args.parquet)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"  -> {len(df):,} rows, columns: {list(df.columns)}", flush=True)

    # Map datetime64 -> positional row index (matches TableBaseDataset.dt_to_row)
    dt_values = df["datetime"].values  # numpy datetime64[ns]
    dt_to_row = {ts: i for i, ts in enumerate(dt_values)}

    for split_name, idx_path in (
        ("train", args.train_index),
        ("validation", args.val_index),
    ):
        out_dir = args.out_dir / split_name
        print(f"\n=== Exporting {split_name} -> {out_dir} ===", flush=True)
        if not idx_path.exists():
            print(f"  !! index not found: {idx_path}", flush=True)
            sys.exit(1)
        written, skipped = export_split(
            df=df,
            dt_to_row=dt_to_row,
            index_csv=idx_path,
            out_dir=out_dir,
            split_name=split_name,
        )
        print(
            f"  -> done: written={written:,}, skipped={skipped:,}",
            flush=True,
        )

    print("\nAll splits exported.", flush=True)


if __name__ == "__main__":
    main()
