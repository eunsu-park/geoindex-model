"""Aggregate the 5-fold CV of the co-author's focal recipe against our standard-loss folds.

Reads, per model and fold, the forecast npz written by `analysis/infer_checkpoint.py`
(deterministic `pred`, `target`, `input_ap30`, all in raw ap30 units) for

* `focal/{model}_fold{k}.npz`  — the co-author's recipe trained by `coauthor_focal_cv.py`
* `ours_standard/{model}_fold{k}.npz` — our August sweep fold checkpoints (standard loss)

and writes `cv_summary.csv` + `cv_summary.md`: MAE, RMSE, CC, MAE skill vs persistence
and storm-point MAE (target ≥ 50) per fold, with mean ± std over folds, for each
(recipe, model). Both recipes were early-stopped on the same fold validation split,
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
FOLDS = [1, 2, 3, 4, 5]


def score(npz_path: Path) -> dict:
    d = np.load(npz_path)
    pred, target, inp = d["pred"], d["target"], d["input_ap30"]
    err = pred - target
    persistence = np.abs(inp[:, -1:] - target).mean()
    storm = target >= 50
    return {
        "n_anchors": int(len(target)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
        "cc": float(np.corrcoef(pred.ravel(), target.ravel())[0, 1]),
        "skill": float(1 - np.abs(err).mean() / persistence),
        "mae_storm": float(np.abs(err[storm]).mean()) if storm.any() else float("nan"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="~/Projects/GeoIndex/results/coauthor_focal_cv_2026-09")
    args = p.parse_args()
    root = Path(args.root).expanduser()

    rows = []
    for recipe, sub in RECIPES.items():
        for model in MODELS:
            for fold in FOLDS:
                path = root / sub / f"{model}_fold{fold}.npz"
                if not path.exists():
                    print(f"missing: {path}")
                    continue
                rows.append({"recipe": recipe, "model": model, "fold": fold, **score(path)})
    df = pd.DataFrame(rows)
    df.to_csv(root / "cv_summary.csv", index=False)

    metrics = ["mae", "rmse", "cc", "skill", "mae_storm"]
    lines = ["# 5-fold CV — co-author focal recipe vs our standard loss (in6h_out6h, raw ap30 units)", ""]
    lines.append("Folds: 1 = val 2006–09 · 2 = 2010–13 · 3 = 2014–17 · 4 = 2018–21 · 5 = 2022–25 "
                 "(chronological, 24 h embargo, per-fold statistics). Both recipes early-stopped on the fold's "
                 "validation split (focal: best raw MAE, 30 epochs / patience 10; ours: best normalized loss, "
                 "patience 10).")
    lines.append("")
    for metric in metrics:
        lines.append(f"## {metric.upper()}")
        lines.append("")
        lines.append("| Recipe | Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Std |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for model in MODELS:
            for recipe in RECIPES:
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
