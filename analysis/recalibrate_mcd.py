"""Post-hoc recalibration of MC-dropout predictive intervals (one sigma scale per run).

MC-dropout intervals are badly under-dispersed: the dropout ensemble captures only a
slice of predictive uncertainty, so the reported +/- 2 sigma band covers far less than
the nominal 95%. This script fits one positive scalar ``sigma_scale`` per run so that
``std' = sigma_scale * mcd_std`` restores ~95% coverage at 2 sigma (coverage-matching),
estimated leakage-free by temporal K-fold cross-fitting over the validation anchors. The
deploy scale (fit on all anchors) is what a live forecast multiplies its band by.

Reads the folded-in MC-dropout stored by the validation pass
(``<experiment>/validation/<epoch>/npz.zip``; one ``.npz`` per anchor with keys
``mcd_mean``/``mcd_std``/``targets`` and an ``anchor`` field, the timestamp also being the
file stem ``YYYYMMDDHHMMSS``). Pure numpy -- no model, no GPU. Point-forecast metrics are
untouched; only the uncertainty band is rescaled. Writes ``validation/<epoch>/calibration.json``.

Usage:
    # one run
    python analysis/recalibrate_mcd.py --results-dir /path/to/results \
        --experiment in12h_out12h_gnn_patchtst

    # every run under results-dir (optionally filtered)
    python analysis/recalibrate_mcd.py --results-dir /path/to/results --filter out12h
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.uncertainty import fit_sigma_scale, recalibrate_cv, uncertainty_metrics  # noqa: E402,F401


def load_run_long(results_dir: str, experiment: str, epoch: str = "best",
                  target_index: int = 0) -> dict:
    """Read a run's folded-in MC-dropout npz archive into long-form arrays.

    Expects ``<results_dir>/<experiment>/validation/<epoch>/npz.zip`` with one ``.npz`` per
    anchor (keys ``mcd_mean``/``mcd_std``/``targets``, each shape (target_len, n_target_vars);
    optional scalar ``anchor``). The anchor timestamp is the ``anchor`` field if present, else
    the file stem ``YYYYMMDDHHMMSS``.

    Args:
        results_dir: Root results directory.
        experiment: Experiment (run) name.
        epoch: Checkpoint epoch subdir (default 'best').
        target_index: Which target channel to recalibrate (default 0, the primary target).

    Returns:
        Dict of numpy arrays: anchor (datetime64[ns]), horizon (int), true, mean, std.
        Empty arrays if the archive is missing.
    """
    zip_path = os.path.join(results_dir, experiment, "validation", epoch, "npz.zip")
    empty = {k: np.array([]) for k in ("anchor", "horizon", "true", "mean", "std")}
    if not os.path.exists(zip_path):
        return empty

    anchors, horizons, trues, means, stds = [], [], [], [], []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".npz"):
                continue
            data = np.load(io.BytesIO(z.read(name)), allow_pickle=True)
            if "mcd_mean" not in data or "mcd_std" not in data:
                continue  # a validation npz without folded MCD; skip
            stem = str(data["anchor"]) if "anchor" in data \
                else os.path.splitext(os.path.basename(name))[0]
            try:
                anchor_dt = pd.to_datetime(stem, format="%Y%m%d%H%M%S")
            except (ValueError, TypeError):
                continue
            m = np.asarray(data["mcd_mean"])[:, target_index].ravel()
            s = np.asarray(data["mcd_std"])[:, target_index].ravel()
            t = np.asarray(data["targets"])[:, target_index].ravel()
            h = np.arange(len(m))
            anchors.append(np.full(len(m), np.datetime64(anchor_dt), dtype="datetime64[ns]"))
            horizons.append(h)
            trues.append(t)
            means.append(m)
            stds.append(s)

    if not means:
        return empty

    # Sort by (anchor, horizon) so contiguous temporal folds are well-defined.
    anchor = np.concatenate(anchors)
    horizon = np.concatenate(horizons)
    order = np.lexsort((horizon, anchor))
    return {
        "anchor": anchor[order],
        "horizon": horizon[order],
        "true": np.concatenate(trues)[order],
        "mean": np.concatenate(means)[order],
        "std": np.concatenate(stds)[order],
    }


def recalibrate_run(results_dir: str, experiment: str, epoch: str = "best",
                    n_folds: int = 5, coverage: float = 0.95, k: float = 2.0,
                    write: bool = True) -> dict | None:
    """Recalibrate one run and (optionally) write ``validation/<epoch>/calibration.json``.

    Returns the recalibration dict, or None if the run has no folded-in MC-dropout.
    """
    long = load_run_long(results_dir, experiment, epoch)
    if long["true"].size == 0:
        return None
    result = recalibrate_cv(long["anchor"], long["horizon"], long["true"],
                            long["mean"], long["std"],
                            n_folds=n_folds, coverage=coverage, k=k)
    result["experiment"] = experiment
    result["epoch"] = epoch
    if write:
        out = os.path.join(results_dir, experiment, "validation", epoch, "calibration.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description="Post-hoc MC-dropout interval recalibration")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--experiment", default="",
                        help="Single run name; omit to sweep every run under results-dir.")
    parser.add_argument("--filter", default="", help="Substring filter when sweeping.")
    parser.add_argument("--epoch", default="best")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--k", type=float, default=2.0)
    parser.add_argument("--no-write", action="store_true",
                        help="Print only; do not write calibration.json.")
    args = parser.parse_args()

    if args.experiment:
        experiments = [args.experiment]
    else:
        experiments = sorted(
            d for d in os.listdir(args.results_dir)
            if d.startswith("in") and os.path.isdir(os.path.join(args.results_dir, d))
        )
        if args.filter:
            experiments = [e for e in experiments if args.filter in e]

    print(f"Recalibrating {len(experiments)} run(s) at {args.k:g}-sigma "
          f"(target coverage {args.coverage:.2f}, {args.n_folds}-fold temporal CV)\n")
    print(f"{'experiment':45s}  {'scale':>6s}  {'picp2_raw':>9s} -> {'picp2_recal':>11s}  "
          f"{'crps_raw':>8s} -> {'crps_recal':>10s}")
    n_done = 0
    for exp in experiments:
        result = recalibrate_run(args.results_dir, exp, args.epoch, args.n_folds,
                                 args.coverage, args.k, write=not args.no_write)
        if result is None:
            print(f"  [SKIP] {exp} -- no folded-in MC-dropout")
            continue
        n_done += 1
        print(f"{exp:45s}  {result['sigma_scale']:6.2f}  "
              f"{result['picp_2sigma_raw']:9.3f} -> {result['picp_2sigma_recal']:11.3f}  "
              f"{result['crps_gaussian_raw']:8.3f} -> {result['crps_gaussian_recal']:10.3f}")

    dest = "printed (not written)" if args.no_write else "validation/<epoch>/calibration.json"
    print(f"\nDone: {n_done}/{len(experiments)} recalibrated -> {dest}")


if __name__ == "__main__":
    main()
