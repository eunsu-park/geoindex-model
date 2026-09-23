"""Aggregate the 5-fold CV of the co-author's focal recipe against our standard-loss folds.

Reads, per model and fold, the forecast npz written by `analysis/infer_checkpoint.py`
(deterministic `pred`, `target`, `input_ap30`, `anchors`, all in raw ap30 units) for

* `focal/{model}_fold{k}.npz`  — the co-author's recipe trained by `coauthor_focal_cv.py`
* `ours_standard/{model}_fold{k}.npz` — our August sweep fold checkpoints (standard loss)

plus two reference forecasts built on the same anchors: persistence (last input value)
and 27-day recurrence (the ap30 value 27 days before each lead time, read from the table).

Writes `cv_summary.csv` + `cv_summary.md`: MAE, RMSE, bias, CC, MAE skill vs persistence
and storm-point MAE (target ≥ 50) per fold, with mean ± std over folds, for each
(recipe, model); paired per-fold differences (focal − ours); and a paper-format table
(mean ± std of MAE, bias and MAE skill, one row per method) matching Table 6 of the
DeepAP manuscript. Both recipes were early-stopped on the same fold validation split,
so the comparison is like-for-like (and equally optimistic).

    python analysis/aggregate_coauthor_focal_cv.py --root ~/Projects/GeoIndex/results/coauthor_focal_cv_2026-09
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["gnn_transformer", "tcn", "lstm"]
RECIPES = {"focal": "focal", "ours": "ours_standard"}
BASELINES = ["persistence", "recurrence27"]
FOLDS = [1, 2, 3, 4, 5]
RECURRENCE_STEPS = 27 * 48  # 27 days at 30-min cadence
LABELS = {
    ("baseline", "persistence"): "Persistence",
    ("baseline", "recurrence27"): "27-day Recurrence",
    ("focal", "lstm"): "Focal LSTM",
    ("focal", "tcn"): "Focal TCN",
    ("focal", "gnn_transformer"): "Focal GNN+Transformer",
    ("ours", "lstm"): "Standard LSTM (ours)",
    ("ours", "tcn"): "Standard TCN (ours)",
    ("ours", "gnn_transformer"): "Standard GNN+Transformer (ours)",
}


def table_series(path: Path) -> pd.Series:
    """Load the 30-min ap30 series from the shared table, indexed by datetime."""
    tab = pd.read_parquet(path, columns=["datetime", "ap30"])
    tab["datetime"] = pd.to_datetime(tab["datetime"])
    return tab.set_index("datetime")["ap30"].astype(float)


def score(pred: np.ndarray, target: np.ndarray, persistence: np.ndarray) -> dict:
    err = pred - target
    storm = target >= 50
    return {
        "n_anchors": int(len(target)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
        "cc": float(np.corrcoef(pred.ravel(), target.ravel())[0, 1]),
        "skill": float(1 - np.abs(err).mean() / np.abs(persistence - target).mean()),
        "mae_storm": float(np.abs(err[storm]).mean()) if storm.any() else float("nan"),
    }


def load(npz_path: Path) -> dict:
    d = np.load(npz_path)
    pred, target, inp = d["pred"].astype(float), d["target"].astype(float), d["input_ap30"].astype(float)
    return {
        "pred": pred,
        "target": target,
        "persistence": np.repeat(inp[:, -1:], target.shape[1], axis=1),
        "anchors": pd.to_datetime(d["anchors"]),  # anchor = first target time
    }


def recurrence(series: pd.Series, anchors: pd.DatetimeIndex, n_leads: int) -> np.ndarray:
    """ap30 at (anchor + lead) − 27 days, for lead = 0 … n_leads − 1 half-hours."""
    cols = []
    for j in range(n_leads):
        t = anchors + pd.Timedelta(minutes=30 * j) - pd.Timedelta(minutes=30 * RECURRENCE_STEPS)
        cols.append(series.reindex(t).to_numpy())
    rec = np.stack(cols, axis=1)
    if np.isnan(rec).any():
        raise ValueError(f"recurrence lookup has {int(np.isnan(rec).any(axis=1).sum())} anchors with missing table rows")
    return rec


def fmt_pm(mean: float, std: float, signed: bool = False) -> str:
    return f"{mean:+.2f} ± {std:.2f}" if signed else f"{mean:.2f} ± {std:.2f}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="~/Projects/GeoIndex/results/coauthor_focal_cv_2026-09")
    p.add_argument("--table", default="~/Projects/GeoIndex/datasets/data.parquet",
                   help="30-min ap30 table used for the 27-day recurrence reference")
    args = p.parse_args()
    root = Path(args.root).expanduser()
    series = table_series(Path(args.table).expanduser())

    rows = []
    for fold in FOLDS:
        # references: one per fold, on the anchors shared by every model of the fold
        ref = None
        for recipe, sub in RECIPES.items():
            for model in MODELS:
                path = root / sub / f"{model}_fold{fold}.npz"
                if not path.exists():
                    print(f"missing: {path}")
                    continue
                d = load(path)
                if ref is None:
                    ref = d
                    rec = recurrence(series, d["anchors"], d["target"].shape[1])
                    rows.append({"recipe": "baseline", "model": "persistence", "fold": fold,
                                 **score(d["persistence"], d["target"], d["persistence"])})
                    rows.append({"recipe": "baseline", "model": "recurrence27", "fold": fold,
                                 **score(rec, d["target"], d["persistence"])})
                elif not d["anchors"].equals(ref["anchors"]):
                    raise ValueError(f"{path}: anchor set differs within fold {fold}")
                rows.append({"recipe": recipe, "model": model, "fold": fold,
                             **score(d["pred"], d["target"], d["persistence"])})
    df = pd.DataFrame(rows)
    df.to_csv(root / "cv_summary.csv", index=False)

    metrics = ["mae", "rmse", "bias", "cc", "skill", "mae_storm"]
    lines = ["# 5-fold CV — co-author focal recipe vs our standard loss (in6h_out6h, raw ap30 units)", ""]
    lines.append("Folds: 1 = val 2006–09 · 2 = 2010–13 · 3 = 2014–17 · 4 = 2018–21 · 5 = 2022–25 "
                 "(chronological, 24 h embargo, per-fold statistics). Both recipes early-stopped on the fold's "
                 "validation split (focal: best raw MAE, 30 epochs / patience 10; ours: best normalized loss, "
                 "patience 10). Persistence = last input value held; 27-day recurrence = ap30 at each lead "
                 "time minus 27 days, from the shared table.")
    lines.append("")

    # paper-format table (DeepAP manuscript Table 6 layout)
    lines.append("## Paper-format table — mean ± std across the five folds")
    lines.append("")
    lines.append("MAE denotes the mean absolute error, Bias the mean signed error (forecast − observed), and "
                 "MAE Skill the improvement in MAE relative to persistence, 1 − MAE / MAE_persistence.")
    lines.append("")
    lines.append("| Method | MAE | Bias | MAE Skill |")
    lines.append("|---|---|---|---|")
    for key, label in LABELS.items():
        sub = df[(df.recipe == key[0]) & (df.model == key[1])]
        if sub.empty:
            continue
        lines.append(f"| {label} | {fmt_pm(sub.mae.mean(), sub.mae.std(ddof=1))} "
                     f"| {fmt_pm(sub.bias.mean(), sub.bias.std(ddof=1), signed=True)} "
                     f"| {fmt_pm(sub.skill.mean(), sub.skill.std(ddof=1))} |")
    lines.append("")

    for metric in metrics:
        lines.append(f"## {metric.upper()}")
        lines.append("")
        lines.append("| Recipe | Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Std |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        groups = [("baseline", b) for b in BASELINES] + [(r, m) for m in MODELS for r in RECIPES]
        for recipe, model in groups:
            sub = df[(df.recipe == recipe) & (df.model == model)].set_index("fold")
            vals = [sub[metric].get(f, np.nan) for f in FOLDS]
            arr = np.array(vals, dtype=float)
            cells = " | ".join("—" if np.isnan(v) else f"{v:.3f}" for v in vals)
            lines.append(f"| {recipe} | {model} | {cells} | {np.nanmean(arr):.3f} | {np.nanstd(arr, ddof=1):.3f} |")
        lines.append("")
    # paired differences (focal − ours) per fold
    lines.append("## Paired difference per fold (focal − ours)")
    lines.append("")
    lines.append("| Model | Metric | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Folds focal better |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for model in MODELS:
        f = df[(df.recipe == "focal") & (df.model == model)].set_index("fold")
        o = df[(df.recipe == "ours") & (df.model == model)].set_index("fold")
        for metric in ["mae", "cc", "mae_storm"]:
            diffs = [f[metric].get(k, np.nan) - o[metric].get(k, np.nan) for k in FOLDS]
            arr = np.array(diffs, dtype=float)
            better = int(np.sum(arr < 0)) if metric != "cc" else int(np.sum(arr > 0))
            cells = " | ".join("—" if np.isnan(v) else f"{v:+.3f}" for v in diffs)
            lines.append(f"| {model} | {metric} | {cells} | {np.nanmean(arr):+.3f} | {better}/5 |")
    (root / "cv_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
