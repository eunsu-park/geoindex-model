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


def _local_score(results_dir: str, exp: str, epoch: str) -> float:
    """Fraction of a run's loose PNGs that are locally hydrated (sampled).

    Cloud-only placeholders report st_blocks == 0. Used to order zip-mode
    sweeps so net-zero-disk conversions (delete ~ as much as the zip adds)
    run before conversions that must download their sources.
    """
    pd = os.path.join(results_dir, exp, "validation", epoch, "plots")
    if not os.path.isdir(pd):
        return 0.0
    files = [f for f in os.listdir(pd) if f.endswith(".png")]
    if not files:
        return 0.0
    sample = files[::max(1, len(files) // 20)][:20]
    local = 0
    for f in sample:
        try:
            local += os.stat(os.path.join(pd, f)).st_blocks > 0
        except OSError:
            pass
    return local / len(sample)


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


def _write_plots_zip(plots_zip: str, src_dir: str, expected: set,
                     experiment: str, loose_dir: str | None = None,
                     scratch_dir: str | None = None) -> bool:
    """Write plots.zip atomically from ``src_dir`` and clean up sources.

    Writes ``plots.zip.tmp``, verifies the member count against ``expected``,
    then renames into place; only after that are the loose dir and scratch
    dir removed.

    Args:
        plots_zip: Final zip path.
        src_dir: Directory holding the PNGs to zip.
        expected: Exact set of PNG basenames the zip must contain.
        experiment: Run name (log prefix).
        loose_dir: Loose plots dir to delete on success (if it exists).
        scratch_dir: Scratch dir to delete on success (if distinct).

    Returns:
        True on verified success.
    """
    tmp_zip = plots_zip + ".tmp"
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for f in sorted(expected):
            zf.write(os.path.join(src_dir, f), arcname=f)
    with zipfile.ZipFile(tmp_zip) as zf:
        n_zipped = sum(1 for n in zf.namelist() if n.endswith(".png"))
    if n_zipped != len(expected):
        print(f"{experiment}: zip verification failed ({n_zipped} != "
              f"{len(expected)}); sources kept", flush=True)
        os.remove(tmp_zip)
        return False
    os.replace(tmp_zip, plots_zip)
    for d in (loose_dir, scratch_dir):
        if d and os.path.isdir(d):
            shutil.rmtree(d)
    print(f"{experiment}: plots.zip written ({n_zipped} PNGs), sources removed",
          flush=True)
    return True


def plot_run(results_dir: str, experiment: str, epoch: str = "best",
             output_dir: str | None = None, limit: int = 0,
             workers: int = 1, overwrite: bool = False,
             zip_mode: bool = False, scratch_root: str | None = None) -> bool:
    """Render plots for every validation anchor of one run.

    In zip mode the run's deliverable is a single
    ``validation/<epoch>/plots.zip``: PNGs are rendered into a local scratch
    directory (seeded from any pre-existing loose ``plots/`` directory, which
    hydrates cloud-only files), zipped (stored, PNGs are already compressed),
    and the loose directory is deleted only after the zip verifies complete.
    An existing plots.zip marks the run done.

    Args:
        results_dir: Root results directory.
        experiment: Experiment (run) name.
        epoch: Checkpoint epoch subdir (default 'best').
        output_dir: Where to write PNGs (loose mode only; default
            ``<experiment>/validation/<epoch>/plots``).
        limit: Render only the first N anchors (0 = all; ignored in zip mode).
        workers: Parallel rendering processes.
        overwrite: Re-render PNGs (loose) / rebuild plots.zip (zip mode).
        zip_mode: Produce plots.zip instead of loose PNGs.
        scratch_root: Local scratch root for zip mode (must be OUTSIDE any
            cloud-synced tree).

    Returns:
        True if the archive existed and the run's plots completed.
    """
    val_dir = os.path.join(results_dir, experiment, "validation", epoch)
    zip_path = os.path.join(val_dir, "npz.zip")
    if not os.path.exists(zip_path):
        print(f"SKIP {experiment}: no archive at {zip_path}")
        return False

    with zipfile.ZipFile(zip_path) as z:
        members = sorted(n for n in z.namelist() if n.endswith(".npz"))
    if not members:
        print(f"SKIP {experiment}: archive holds no .npz members")
        return False

    loose_dir = os.path.join(val_dir, "plots")
    if zip_mode:
        plots_zip = os.path.join(val_dir, "plots.zip")
        if os.path.exists(plots_zip) and not overwrite:
            # Done earlier; clear any leftover loose dir it superseded.
            if os.path.isdir(loose_dir):
                shutil.rmtree(loose_dir)
                print(f"{experiment}: plots.zip present, removed stale loose dir",
                      flush=True)
            return True
        if limit > 0:
            print(f"{experiment}: --limit is ignored in zip mode "
                  f"(a partial plots.zip would read as complete)", flush=True)

        # Fast path: a COMPLETE loose dir is zipped in place — no scratch
        # copy, no render, half the transient disk (reading a cloud-only
        # placeholder still hydrates it for the duration of the zip).
        expected = {os.path.splitext(os.path.basename(m))[0] + ".png"
                    for m in members}
        loose_have = ({f for f in os.listdir(loose_dir) if f.endswith(".png")}
                      if os.path.isdir(loose_dir) else set())
        if expected <= loose_have:
            print(f"{experiment}: complete loose dir, zipping in place "
                  f"({len(expected)} PNGs)", flush=True)
            return _write_plots_zip(plots_zip, loose_dir, expected,
                                    experiment, loose_dir=loose_dir)

        out_dir = os.path.join(scratch_root, experiment)
        os.makedirs(out_dir, exist_ok=True)
        # Seed scratch from a pre-existing partial loose render.
        seeded = 0
        for f in loose_have:
            if not os.path.exists(os.path.join(out_dir, f)):
                shutil.copy2(os.path.join(loose_dir, f), os.path.join(out_dir, f))
                seeded += 1
        if seeded:
            print(f"{experiment}: seeded {seeded} PNGs from loose dir", flush=True)
    else:
        if limit > 0:
            members = members[:limit]
        out_dir = output_dir or loose_dir
        os.makedirs(out_dir, exist_ok=True)

    print(f"{experiment}: rendering {len(members)} plots -> {out_dir}", flush=True)
    tasks = [(m, out_dir, experiment, overwrite and not zip_mode) for m in members]
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

    if not zip_mode:
        return failed == 0

    if failed > 0:
        print(f"{experiment}: keeping scratch for resume ({out_dir})", flush=True)
        return False

    # Zip the scratch, then drop the loose dir and scratch.
    have = {f for f in os.listdir(out_dir) if f.endswith(".png")}
    missing = expected - have
    if missing:
        print(f"{experiment}: {len(missing)} PNGs missing after render "
              f"(e.g. {sorted(missing)[:3]}); keeping scratch, no zip", flush=True)
        return False
    return _write_plots_zip(plots_zip, out_dir, expected, experiment,
                            loose_dir=loose_dir, scratch_dir=out_dir)


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
    parser.add_argument("--zip", action="store_true", dest="zip_mode",
                        help="Produce one validation/<epoch>/plots.zip per run "
                             "(rendered via a local scratch dir; any loose "
                             "plots/ dir is absorbed and deleted).")
    parser.add_argument("--scratch-dir", default="",
                        help="Scratch root for --zip (default: system temp). "
                             "Must be OUTSIDE any cloud-synced tree.")
    args = parser.parse_args()

    scratch_root = None
    if args.zip_mode:
        import tempfile
        scratch_root = args.scratch_dir or os.path.join(
            tempfile.gettempdir(), "plot_zip_scratch")
        if "CloudStorage" in os.path.abspath(scratch_root):
            sys.exit("--scratch-dir must not be inside a cloud-synced tree")
        os.makedirs(scratch_root, exist_ok=True)

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

    if args.zip_mode and len(experiments) > 1:
        print(f"Ordering {len(experiments)} runs: locally-hydrated loose dirs "
              f"first (net-zero disk conversions before download-heavy ones)",
              flush=True)
        score = {e: _local_score(args.results_dir, e, args.epoch)
                 for e in experiments}
        experiments.sort(key=lambda e: -score[e])

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
                      output_dir=None if args.zip_mode else out_dir_for(exp),
                      limit=args.limit, workers=args.workers,
                      overwrite=args.overwrite,
                      zip_mode=args.zip_mode, scratch_root=scratch_root) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
