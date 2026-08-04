"""Render one forecast plot per validation anchor, for visual inspection of every sample.

The validation archives already hold everything a plot needs -- input history, observations and
predictions, all denormalized -- so this needs neither the checkpoint nor a GPU. It also lets
several runs be overlaid on the same axes, which the pipeline's own per-sample plots cannot do.

Three outputs, because 23,514 plots are only useful if they can be navigated:

  plots/YYYYMM/<anchor>.png   one panel per anchor, foldered by month
  index.csv                   per-anchor metrics, sorted by observed peak (storms first)
  sheets/sheet_NNN.png        contact sheets of 24 panels, in the same order as index.csv

Usage:
    python analysis/plot_all_anchors.py --results-dir ~/Projects/GeoIndex/results \\
        --runs probe_ap_in12h_out12h_gnn_transformer_baseline:baseline \\
               probe_ap_in12h_out12h_gnn_transformer_obs:obs \\
        --out ~/Projects/GeoIndex/results/plots_all_obs
    python analysis/plot_all_anchors.py ... --limit 200      # the 200 largest storms only
    python analysis/plot_all_anchors.py ... --sheets-only    # skip the individual panels
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

# Colours match the verification pages: observed is ink, the runs take the categorical slots.
INK = "#14171b"
RUN_COLOURS = ["#2a78d6", "#1baf7a", "#eb6834", "#4a3aa7"]
PERS = "#878d97"
STEP_H = 0.5


def load_run(results_dir: str, run: str) -> dict:
    """Read every stored sample of one validation archive."""
    path = os.path.join(results_dir, run, "validation", "best", "npz.zip")
    anchors, true, pred, inputs, ivars, tvar = [], [], [], [], None, None
    with zipfile.ZipFile(path) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".npz")):
            d = np.load(io.BytesIO(z.read(name)), allow_pickle=True)
            if ivars is None:
                ivars = [str(x) for x in np.asarray(d["input_variables"]).ravel()]
                tvar = str(np.asarray(d["target_variables"]).ravel()[0])
            anchors.append(str(np.asarray(d["anchor"])))
            true.append(np.asarray(d["targets"])[:, 0])
            pred.append(np.asarray(d["predictions"])[:, 0])
            inputs.append(np.asarray(d["inputs"])[:, ivars.index(tvar)])
    return {"anchor": np.asarray(anchors), "true": np.asarray(true),
            "pred": np.asarray(pred), "hist": np.asarray(inputs), "tvar": tvar}


def _fmt_anchor(stamp: str) -> str:
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[8:10]}:{stamp[10:12]} UTC"


def draw(ax, hist, true, preds, labels, tvar, title, compact=False):
    """One anchor: input history, observation, and every run's forecast on shared axes."""
    n_in, n_out = len(hist), len(true)
    t_hist = np.arange(-n_in, 0) * STEP_H + STEP_H
    t_out = np.arange(1, n_out + 1) * STEP_H

    ax.plot(t_hist, hist, color=INK, lw=1.4, alpha=.55)
    ax.plot(np.r_[t_hist[-1], t_out], np.r_[hist[-1], true], color=INK, lw=1.8,
            label="observed")
    # persistence: the anchor value held across the window, the baseline every model must beat
    ax.plot(t_out, np.full(n_out, hist[-1]), color=PERS, lw=1.2, ls=":", label="persistence")
    for i, (p, lab) in enumerate(zip(preds, labels)):
        ax.plot(np.r_[t_hist[-1], t_out], np.r_[hist[-1], p],
                color=RUN_COLOURS[i % len(RUN_COLOURS)], lw=1.6, label=lab)

    ax.axvline(0, color=INK, lw=.8, alpha=.35)
    top = max(true.max(), hist.max(), max(p.max() for p in preds)) if preds else true.max()
    for level, style in ((50, (0, (3, 3))), (100, (0, (5, 2)))):
        if level < top * 1.15:
            ax.axhline(level, color=PERS, lw=.7, ls=style, alpha=.7)
    ax.set_xlim(t_hist[0], t_out[-1])
    ax.set_ylim(0, max(top * 1.12, 12))
    ax.set_title(title, fontsize=8 if compact else 9.5, loc="left", color=INK)
    ax.tick_params(labelsize=6.5 if compact else 8, colors="#4e545e")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#e2e3e4")
    if not compact:
        ax.set_xlabel("hours from the anchor", fontsize=8, color="#4e545e")
        ax.set_ylabel(tvar, fontsize=8, color="#4e545e")


def _title(stamp, true, preds, labels):
    obs_max = true.max()
    at = (int(true.argmax()) + 1) * STEP_H
    parts = [f"{_fmt_anchor(stamp)}", f"obs peak {obs_max:.0f} @+{at:.1f}h"]
    parts += [f"{lab} {p.max():.0f}" for p, lab in zip(preds, labels)]
    return "  ·  ".join(parts)


def render_one(args):
    """Worker: draw and save a single anchor panel."""
    (stamp, hist, true, preds, labels, tvar, out_dir, dpi) = args
    month_dir = os.path.join(out_dir, "plots", stamp[:6])
    os.makedirs(month_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    draw(ax, hist, true, preds, labels, tvar, _title(stamp, true, preds, labels))
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(month_dir, f"{stamp}.png"), dpi=dpi)
    plt.close(fig)
    return stamp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run_dir:label pairs; the first is drawn first")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="render only the N largest observed peaks (index.csv still covers all)")
    ap.add_argument("--dpi", type=int, default=85)
    ap.add_argument("--sheets-only", action="store_true")
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() - 2, 1))
    args = ap.parse_args()

    results_dir = os.path.expanduser(args.results_dir)
    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)

    specs = [r.split(":") if ":" in r else (r, r) for r in args.runs]
    runs = [(lab, load_run(results_dir, name)) for name, lab in specs]
    base = runs[0][1]
    for lab, r in runs[1:]:
        assert np.array_equal(r["anchor"], base["anchor"]), f"{lab}: anchors differ"
    anchors, true, hist = base["anchor"], base["true"], base["hist"]
    labels = [lab for lab, _ in runs]
    preds = [r["pred"] for _, r in runs]
    tvar = base["tvar"]
    print(f"{len(anchors)} anchors, {len(runs)} runs: {', '.join(labels)}")

    # index, ordered so the interesting samples are first
    obs_max = true.max(axis=1)
    order = np.argsort(-obs_max)
    idx_path = os.path.join(out_dir, "index.csv")
    with open(idx_path, "w") as fp:
        # "observed_" prefix, not "obs_": a run may be labelled "obs" and the columns collide
        head = ["rank", "anchor", "observed_peak", "observed_peak_at_h", "anchor_value"]
        head += [f"{lab}_peak" for lab in labels] + [f"{lab}_mae" for lab in labels]
        fp.write(",".join(head) + "\n")
        for rank, i in enumerate(order, 1):
            row = [rank, anchors[i], f"{obs_max[i]:.0f}",
                   f"{(int(true[i].argmax())+1)*STEP_H:.1f}", f"{hist[i][-1]:.0f}"]
            row += [f"{p[i].max():.1f}" for p in preds]
            row += [f"{np.abs(p[i]-true[i]).mean():.2f}" for p in preds]
            fp.write(",".join(str(v) for v in row) + "\n")
    print(f"wrote {idx_path}")

    selected = order[:args.limit] if args.limit else order
    if not args.sheets_only:
        jobs = [(anchors[i], hist[i], true[i], [p[i] for p in preds], labels, tvar,
                 out_dir, args.dpi) for i in selected]
        print(f"rendering {len(jobs)} panels with {args.workers} workers ...")
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for _ in ex.map(render_one, jobs, chunksize=32):
                done += 1
                if done % 1000 == 0:
                    print(f"  {done}/{len(jobs)}", flush=True)
        print(f"  {done}/{len(jobs)} done -> {os.path.join(out_dir, 'plots')}")

    if not args.no_sheets:
        sheet_dir = os.path.join(out_dir, "sheets")
        os.makedirs(sheet_dir, exist_ok=True)
        per, cols = 24, 3
        n_sheets = (len(selected) + per - 1) // per
        print(f"contact sheets: {n_sheets} of {per} panels each ...")
        for s in range(n_sheets):
            chunk = selected[s * per:(s + 1) * per]
            rows = (len(chunk) + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.6, rows * 2.0))
            axes = np.atleast_1d(axes).ravel()
            for ax in axes[len(chunk):]:
                ax.axis("off")
            for ax, i in zip(axes, chunk):
                draw(ax, hist[i], true[i], [p[i] for p in preds], labels, tvar,
                     _title(anchors[i], true[i], [p[i] for p in preds], labels), compact=True)
            fig.tight_layout()
            fig.savefig(os.path.join(sheet_dir, f"sheet_{s:04d}.png"), dpi=80)
            plt.close(fig)
            if (s + 1) % 20 == 0:
                print(f"  {s+1}/{n_sheets}", flush=True)
        print(f"  {n_sheets} sheets -> {sheet_dir}")


if __name__ == "__main__":
    main()
