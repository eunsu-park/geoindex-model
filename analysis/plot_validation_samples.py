"""Render per-sample prediction plots for every validation anchor of a run.

Batch sweeps run with ``validation.save_plots: false`` (plots are heavy and
pollute the cloud-synced results tree), so this script regenerates the plots
on demand from the validation archive
(``<experiment>/validation/<epoch>/npz.zip``; one ``.npz`` per anchor with
denormalized ``inputs``/``targets``/``predictions`` and the variable name
lists). Pure numpy + matplotlib -- no model, no GPU, no dataset.

Multi-channel runs (the recursive sweep) automatically get the grouped
envelope-band layout from ``src.plotting``; single-target runs keep the
classic per-variable layout.

Usage:
    # one run, all validation samples
    python analysis/plot_validation_samples.py \
        --results-dir /path/to/results --experiment ap_recursive_in12h_out6h_linear

    # every run under results-dir whose name contains "recursive", 8 processes
    python analysis/plot_validation_samples.py \
        --results-dir /path/to/results --filter recursive --workers 8

    # quick look: first 20 anchors only
    python analysis/plot_validation_samples.py \
        --results-dir /path/to/results --experiment ap_in12h_out6h_linear --limit 20
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.plotting import plot_prediction_timeseries  # noqa: E402


class IdentityNormalizer:
    """Normalizer stand-in for already-denormalized npz arrays.

    ``plot_prediction_timeseries`` uses its normalizer for two things:
    denormalization (a no-op here -- the validation npz stores original-scale
    arrays) and per-variable normalization methods (used to pick log y-axes).
    The methods are read from configs/base.yaml, which needs no statistics.
    """

    def __init__(self):
        with open(os.path.join(REPO_ROOT, "configs", "base.yaml")) as f:
            cfg = yaml.safe_load(f)
        norm = cfg["data"]["timeseries"]["normalization"]
        self._methods = norm.get("methods", {}) or {}
        self._default = norm.get("default", "zscore")

    def denormalize_omni(self, data, variable):
        return data

    def get_method(self, variable):
        return self._methods.get(variable, self._default)


def plot_member(args_tuple) -> str | None:
    """Render one npz member to a PNG. Returns the anchor on failure, else None."""
    zip_path, member, out_dir, experiment = args_tuple
    stem = os.path.splitext(os.path.basename(member))[0]
    try:
        with zipfile.ZipFile(zip_path) as z:
            data = np.load(io.BytesIO(z.read(member)), allow_pickle=True)
        inputs = np.asarray(data["inputs"])
        predictions = np.asarray(data["predictions"])
        targets = np.asarray(data["targets"]) if "targets" in data else None
        input_variables = [str(v) for v in data["input_variables"]]
        target_variables = [str(v) for v in data["target_variables"]]
        anchor = str(data["anchor"]) if "anchor" in data else stem

        plot_prediction_timeseries(
            inputs=inputs,
            predictions=predictions,
            input_variables=input_variables,
            target_variables=target_variables,
            save_path=Path(out_dir) / f"{stem}.png",
            targets=targets,
            title=f"{experiment} @ {anchor}",
            normalizer=IdentityNormalizer(),
        )
        return None
    except Exception as e:  # keep the sweep going; report at the end
        print(f"  FAIL {stem}: {e!r}")
        return stem


def plot_run(results_dir: str, experiment: str, epoch: str = "best",
             output_dir: str | None = None, limit: int = 0,
             workers: int = 1) -> bool:
    """Render plots for every validation anchor of one run.

    Args:
        results_dir: Root results directory.
        experiment: Experiment (run) name.
        epoch: Checkpoint epoch subdir (default 'best').
        output_dir: Where to write PNGs (default
            ``<experiment>/validation/<epoch>/plots``).
        limit: Render only the first N anchors (0 = all).
        workers: Parallel rendering processes.

    Returns:
        True if the archive existed and every rendered plot succeeded.
    """
    zip_path = os.path.join(results_dir, experiment, "validation", epoch, "npz.zip")
    if not os.path.exists(zip_path):
        print(f"SKIP {experiment}: no archive at {zip_path}")
        return False

    with zipfile.ZipFile(zip_path) as z:
        members = sorted(n for n in z.namelist() if n.endswith(".npz"))
    if limit > 0:
        members = members[:limit]
    if not members:
        print(f"SKIP {experiment}: archive holds no .npz members")
        return False

    out_dir = output_dir or os.path.join(
        results_dir, experiment, "validation", epoch, "plots")
    os.makedirs(out_dir, exist_ok=True)

    print(f"{experiment}: rendering {len(members)} plots -> {out_dir}")
    tasks = [(zip_path, m, out_dir, experiment) for m in members]
    if workers > 1:
        with Pool(processes=workers) as pool:
            failures = [r for r in pool.imap_unordered(plot_member, tasks, chunksize=8) if r]
    else:
        failures = [r for r in map(plot_member, tasks) if r]

    done = len(members) - len(failures)
    print(f"{experiment}: {done}/{len(members)} plots written"
          + (f", {len(failures)} failed" if failures else ""))
    return not failures


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate per-sample validation plots from npz.zip")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--experiment", default="",
                        help="Single run name. Empty: sweep every run under results-dir.")
    parser.add_argument("--filter", default="",
                        help="Substring filter on run names when sweeping.")
    parser.add_argument("--epoch", default="best")
    parser.add_argument("--output-dir", default="",
                        help="PNG output dir (single-run mode only; default "
                             "<experiment>/validation/<epoch>/plots).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Render only the first N anchors per run (0 = all).")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if args.experiment:
        experiments = [args.experiment]
    else:
        experiments = sorted(
            d for d in os.listdir(args.results_dir)
            if os.path.isdir(os.path.join(args.results_dir, d))
            and (not args.filter or args.filter in d)
        )
        if not experiments:
            print(f"No runs under {args.results_dir} match '{args.filter}'")
            sys.exit(1)
        if args.output_dir:
            print("--output-dir applies to single-run mode only; ignoring it.")
            args.output_dir = ""

    ok = True
    for exp in experiments:
        ok = plot_run(args.results_dir, exp, epoch=args.epoch,
                      output_dir=args.output_dir or None,
                      limit=args.limit, workers=args.workers) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
