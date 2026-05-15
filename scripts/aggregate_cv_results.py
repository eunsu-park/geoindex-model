#!/usr/bin/env python
"""Aggregate cross-validation validation results into summary tables.

Walks ${save_root}/${io}_${model}_fold${N}/validation/${epoch}/validation_results.txt
for every (model, fold) pair, parses the overall metrics, and writes
wide-format CSV plus a Markdown report (rows=model, fold1..foldN +
mean / std for each metric).

Usage:
    python scripts/aggregate_cv_results.py \\
        --save-root /opt/nas/ap_share/results \\
        --io in12h_out12h \\
        --epoch best \\
        --output-dir /opt/nas/ap_share/results/_cv_summary/in12h_out12h
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


METRICS = ["loss", "mae", "rmse", "r2"]

# Lines of the form: "  Average MAE:  1.2345 (+/-0.1234)"
PATTERN = re.compile(
    r"^\s*Average\s+(Loss|MAE|RMSE|R2)\s*:\s*([-+0-9.eE]+)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save-root", required=True, help="Root containing experiment dirs")
    p.add_argument("--io", default="in12h_out12h", help="I/O config name (default: in12h_out12h)")
    p.add_argument("--epoch", default="best", help="Validation epoch dir name (default: best)")
    p.add_argument("--folds", default="1,2,3,4,5", help="Comma-separated fold IDs (default: 1,2,3,4,5)")
    p.add_argument("--models", default=None, help="Comma-separated model list. Default: auto-discover from save_root")
    p.add_argument("--output-dir", required=True, help="Where to write summary CSV/Markdown")
    return p.parse_args()


def parse_summary(path: Path) -> Optional[Dict[str, float]]:
    """Parse a validation_results.txt and return {metric_lower: value}."""
    if not path.exists():
        return None
    metrics: Dict[str, float] = {}
    with path.open() as f:
        for line in f:
            m = PATTERN.match(line)
            if m:
                key = m.group(1).lower()
                metrics[key] = float(m.group(2))
                if len(metrics) == len(METRICS):
                    break
    if not metrics:
        return None
    return metrics


def epoch_dirname(epoch: str) -> str:
    """Map epoch identifier to validation output directory name."""
    e = epoch.strip()
    if e.lower() in ("best", "final"):
        return e.lower()
    try:
        return f"epoch_{int(e):04d}"
    except ValueError:
        return e


def discover_models(save_root: Path, io: str, folds: List[int]) -> List[str]:
    """List models present under save_root with the given io prefix."""
    pattern = re.compile(rf"^{re.escape(io)}_(.+)_fold(\d+)$")
    found: Dict[str, set] = {}
    if not save_root.exists():
        return []
    for entry in save_root.iterdir():
        if not entry.is_dir():
            continue
        m = pattern.match(entry.name)
        if not m:
            continue
        model, fold = m.group(1), int(m.group(2))
        if fold in folds:
            found.setdefault(model, set()).add(fold)
    return sorted(found.keys())


def aggregate(save_root: Path, io: str, models: List[str], folds: List[int],
              epoch_dir: str) -> Dict[str, Dict[str, List[Optional[float]]]]:
    """Return {model: {metric: [val_fold1, val_fold2, ...]}} with None for missing."""
    results: Dict[str, Dict[str, List[Optional[float]]]] = {}
    for model in models:
        per_metric: Dict[str, List[Optional[float]]] = {m: [] for m in METRICS}
        for fold in folds:
            exp = f"{io}_{model}_fold{fold}"
            txt = save_root / exp / "validation" / epoch_dir / "validation_results.txt"
            parsed = parse_summary(txt)
            if parsed is None:
                print(f"[WARN] missing or unparseable: {txt}", file=sys.stderr)
                for m in METRICS:
                    per_metric[m].append(None)
            else:
                for m in METRICS:
                    per_metric[m].append(parsed.get(m))
        results[model] = per_metric
    return results


def summarize(values: List[Optional[float]]) -> Dict[str, Optional[float]]:
    """Compute mean/std over non-None entries."""
    finite = [v for v in values if v is not None and not np.isnan(v)]
    if not finite:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(finite)), "std": float(np.std(finite, ddof=0))}


def write_wide_csv(results, models, folds, output_path: Path) -> None:
    header = ["model"]
    for metric in METRICS:
        header += [f"{metric}_fold{f}" for f in folds]
        header += [f"{metric}_mean", f"{metric}_std"]

    rows = []
    for model in models:
        row = [model]
        for metric in METRICS:
            vals = results[model][metric]
            row += ["" if v is None else f"{v:.6f}" for v in vals]
            s = summarize(vals)
            row += ["" if s["mean"] is None else f"{s['mean']:.6f}",
                    "" if s["std"] is None else f"{s['std']:.6f}"]
        rows.append(row)

    with output_path.open("w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")
    print(f"Wrote: {output_path}")


def write_markdown(results, models, folds, output_path: Path, io: str, epoch_dir: str) -> None:
    lines: List[str] = []
    lines.append(f"# CV Summary - {io} ({epoch_dir})\n")
    lines.append(f"Models: {len(models)} | Folds: {folds}\n")

    for metric in METRICS:
        lines.append(f"\n## {metric.upper()}\n")
        header = ["Model"] + [f"Fold {f}" for f in folds] + ["Mean", "Std"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for model in models:
            vals = results[model][metric]
            s = summarize(vals)
            cells = [model]
            cells += ["-" if v is None else f"{v:.4f}" for v in vals]
            cells += ["-" if s["mean"] is None else f"{s['mean']:.4f}",
                      "-" if s["std"] is None else f"{s['std']:.4f}"]
            lines.append("| " + " | ".join(cells) + " |")

    with output_path.open("w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote: {output_path}")


def main() -> None:
    args = parse_args()
    save_root = Path(args.save_root)
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    epoch_dir = epoch_dirname(args.epoch)

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = discover_models(save_root, args.io, folds)
        if not models:
            sys.exit(f"No experiments found under {save_root} matching {args.io}_*_fold[1..5]")

    print(f"Aggregating {len(models)} models x {len(folds)} folds for {args.io} ({epoch_dir})")
    results = aggregate(save_root, args.io, models, folds, epoch_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.io}_{epoch_dir}"
    write_wide_csv(results, models, folds, output_dir / f"cv_summary_{tag}.csv")
    write_markdown(results, models, folds, output_dir / f"cv_summary_{tag}.md",
                   args.io, epoch_dir)


if __name__ == "__main__":
    main()
