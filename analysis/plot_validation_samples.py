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

Full-sweep renders are large (a full-period validation index is ~23k anchors
per run, tens of GB of PNGs across a sweep): run them where the results are
LOCAL (the GPU server), and point ``--output-root`` OUTSIDE any cloud-synced
tree. Existing PNGs are skipped, so an interrupted sweep resumes where it
stopped (``--overwrite`` forces a re-render).

Usage:
    # one run, all validation samples
    python analysis/plot_validation_samples.py \
        --results-dir /path/to/results --experiment ap_in12h_out6h_linear \
        --output-root ~/plots_sweepA

    # the whole 2026-08 short-horizon direct sweep, 12 processes
    python analysis/plot_validation_samples.py \
        --results-dir /path/to/results \
        --filter '^ap_in(6h|12h|18h|1d)_out[1-6]h_' \
        --output-root ~/plots_sweepA --workers 12

    # quick look: first 20 anchors only
    python analysis/plot_validation_samples.py \
        --results-dir /path/to/results --experiment ap_in12h_out6h_linear \
        --output-root /tmp/plots --limit 20
"""

from __future__ import annotations

import argparse
import io
import os
import re
import shutil
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


# Per-process state (set by _init_worker): the archive is opened once per
# worker instead of once per plot -- re-parsing a ~23k-entry central
# directory for every member dominates the runtime otherwise.
_ZF: zipfile.ZipFile | None = None
_NORM: IdentityNormalizer | None = None


def _init_worker(zip_path: str):
    global _ZF, _NORM
    _ZF = zipfile.ZipFile(zip_path)
    _NORM = IdentityNormalizer()


def plot_member(args_tuple) -> tuple[str, str | None]:
    """Render one npz member to a PNG.

    Returns:
        ('done'|'skipped'|'failed', anchor stem or None).
    """
    member, out_dir, experiment, overwrite = args_tuple
    stem = os.path.splitext(os.path.basename(member))[0]
    save_path = Path(out_dir) / f"{stem}.png"
    if not overwrite and save_path.exists():
        return ("skipped", None)
    try:
        data = np.load(io.BytesIO(_ZF.read(member)), allow_pickle=True)
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
            save_path=save_path,
            targets=targets,
            title=f"{experiment} @ {anchor}",
            normalizer=_NORM,
        )
        return ("done", None)
    except Exception as e:  # keep the sweep going; report at the end
        print(f"  FAIL {stem}: {e!r}")
        return ("failed", stem)


def plot_run(results_dir: str, experiment: str, epoch: str = "best",
             output_dir: str | None = None, limit: int = 0,
             workers: int = 1, overwrite: bool = False) -> bool:
    """Render plots for every validation anchor of one run.

    Args:
        results_dir: Root results directory.
        experiment: Experiment (run) name.
        epoch: Checkpoint epoch subdir (default 'best').
        output_dir: Where to write PNGs (default
            ``<experiment>/validation/<epoch>/plots``).
        limit: Render only the first N anchors (0 = all).
        workers: Parallel rendering processes.
        overwrite: Re-render PNGs that already exist.

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

    print(f"{experiment}: rendering {len(members)} plots -> {out_dir}", flush=True)
    tasks = [(m, out_dir, experiment, overwrite) for m in members]
    if workers > 1:
        with Pool(processes=workers, initializer=_init_worker,
                  initargs=(zip_path,)) as pool:
            outcomes = list(pool.imap_unordered(plot_member, tasks, chunksize=32))
    else:
        _init_worker(zip_path)
        outcomes = [plot_member(t) for t in tasks]

    done = sum(1 for s, _ in outcomes if s == "done")
    skipped = sum(1 for s, _ in outcomes if s == "skipped")
    failed = sum(1 for s, _ in outcomes if s == "failed")
    print(f"{experiment}: {done} written, {skipped} skipped (existing), "
          f"{failed} failed", flush=True)
    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate per-sample validation plots from npz.zip")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--experiment", default="",
                        help="Single run name. Empty: sweep every run under results-dir.")
    parser.add_argument("--filter", default="",
                        help="Regex (re.search) on run names when sweeping, "
                             "e.g. '^ap_in(6h|12h|18h|1d)_out[1-6]h_'.")
    parser.add_argument("--epoch", default="best")
    parser.add_argument("--output-root", default="",
                        help="Root dir for PNGs; each run writes to "
                             "<output-root>/<experiment>/. Point this OUTSIDE "
                             "any cloud-synced tree for sweep-scale renders.")
    parser.add_argument("--output-dir", default="",
                        help="Exact PNG dir (single-run mode only; overrides "
                             "--output-root).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Render only the first N anchors per run (0 = all).")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-render PNGs that already exist.")
    parser.add_argument("--min-free-gb", type=float, default=40.0,
                        help="Stop (resumably) before starting a run when the "
                             "output filesystem has less free space than this. "
                             "0 disables the guard.")
    args = parser.parse_args()

    if args.experiment:
        experiments = [args.experiment]
    else:
        pattern = re.compile(args.filter) if args.filter else None
        experiments = sorted(
            d for d in os.listdir(args.results_dir)
            if os.path.isdir(os.path.join(args.results_dir, d))
            and (pattern is None or pattern.search(d))
        )
        if not experiments:
            print(f"No runs under {args.results_dir} match '{args.filter}'")
            sys.exit(1)
        if args.output_dir:
            print("--output-dir applies to single-run mode only; ignoring it.")
            args.output_dir = ""

    def out_dir_for(exp: str) -> str | None:
        if args.output_dir:
            return args.output_dir
        if args.output_root:
            return os.path.join(args.output_root, exp)
        return None  # default: <experiment>/validation/<epoch>/plots

    ok = True
    for i, exp in enumerate(experiments, 1):
        if args.min_free_gb > 0:
            probe = out_dir_for(exp) or args.results_dir
            while not os.path.isdir(probe):
                probe = os.path.dirname(probe) or "/"
            free_gb = shutil.disk_usage(probe).free / 1e9
            if free_gb < args.min_free_gb:
                print(f"STOP: only {free_gb:.1f} GB free (< {args.min_free_gb} GB) "
                      f"before run {i}/{len(experiments)}. Free up space and "
                      f"re-run the same command — finished PNGs are skipped.",
                      flush=True)
                sys.exit(3)
        if len(experiments) > 1:
            print(f"[{i}/{len(experiments)}]", end=" ")
        ok = plot_run(args.results_dir, exp, epoch=args.epoch,
                      output_dir=out_dir_for(exp),
                      limit=args.limit, workers=args.workers,
                      overwrite=args.overwrite) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
