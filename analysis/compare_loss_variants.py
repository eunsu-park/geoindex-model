"""Compare loss / target-transform probe variants on the diagnostics that matter.

Reads each variant's ``validation/<epoch>/npz.zip`` (deterministic predictions; the probe runs
with ``validation.mcd_samples=0``) and reports, in original units:

- MAE and skill against persistence, pooled and at lead 1
- storm reproduction: mean prediction / mean observation for obs >= 30 / 50 / 100
- dispersion: std(prediction) / std(observation), and overall bias

The persistence anchor is the last observed target value in the input window. Variants that
drop the target input channel have no anchor of their own, so the anchor series is taken from
whichever variant still carries it and matched by anchor timestamp.

Usage:
    python analysis/compare_loss_variants.py --results-dir /path/to/results \
        --prefix probe_ap_in12h_out12h_gnn_transformer
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import zipfile

import numpy as np

THRESHOLDS = (30.0, 50.0, 100.0)


def load_variant(zip_path: str) -> dict:
    """Read one run's per-event arrays.

    Args:
        zip_path: Path to ``validation/<epoch>/npz.zip``.

    Returns:
        Dict with anchor (str array), true / pred (n_events, n_leads) and pers (n_events,)
        or None when the target channel is absent from the inputs.
    """
    anchors, true, pred, pers = [], [], [], []
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".npz")):
            d = np.load(io.BytesIO(z.read(name)), allow_pickle=True)
            tvar = str(np.asarray(d["target_variables"]).ravel()[0])
            ivars = [str(x) for x in np.asarray(d["input_variables"]).ravel()]
            anchors.append(str(np.asarray(d["anchor"])))
            true.append(np.asarray(d["targets"])[:, 0])
            pred.append(np.asarray(d["predictions"])[:, 0])
            pers.append(np.asarray(d["inputs"])[-1, ivars.index(tvar)] if tvar in ivars else np.nan)
    pers_arr = np.asarray(pers)
    return {"anchor": np.asarray(anchors), "true": np.asarray(true),
            "pred": np.asarray(pred), "pers": None if np.isnan(pers_arr).all() else pers_arr}


def summarize(name: str, v: dict, pers_map: dict) -> dict:
    """Compute the comparison row for one variant."""
    true, pred = v["true"], v["pred"]
    pers = v["pers"]
    if pers is None:
        pers = np.array([pers_map.get(a, np.nan) for a in v["anchor"]])
    keep = ~np.isnan(pers)
    true, pred, pers = true[keep], pred[keep], pers[keep]
    pers2 = np.repeat(pers[:, None], true.shape[1], axis=1)

    mae = np.abs(pred - true).mean()
    mae_p = np.abs(pers2 - true).mean()
    mae1 = np.abs(pred[:, 0] - true[:, 0]).mean()
    mae_p1 = np.abs(pers2[:, 0] - true[:, 0]).mean()

    row = {"variant": name, "n": len(true), "MAE": mae, "skill": 1 - mae / mae_p,
           "skill_l1": 1 - mae1 / mae_p1, "bias": (pred - true).mean(),
           "std_ratio": pred.std() / true.std()}
    for t in THRESHOLDS:
        m = true >= t
        row[f"repro_{int(t)}"] = (pred[m].mean() / true[m].mean()) if m.sum() > 50 else np.nan
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare loss-probe variants")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--prefix", required=True, help="e.g. probe_ap_in12h_out12h_gnn_transformer")
    ap.add_argument("--epoch", default="best")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(
        args.results_dir, f"{args.prefix}_*", "validation", args.epoch, "npz.zip")))
    if not paths:
        raise SystemExit(f"no runs matched {args.prefix}_* under {args.results_dir}")

    loaded = {}
    for p in paths:
        run = p.split(os.sep)[-4]
        loaded[run[len(args.prefix) + 1:]] = load_variant(p)
        print(f"loaded {run}")

    pers_map: dict = {}
    for v in loaded.values():
        if v["pers"] is not None:
            pers_map = dict(zip(v["anchor"], v["pers"]))
            break

    rows = [summarize(name, v, pers_map) for name, v in loaded.items()]
    rows.sort(key=lambda r: -r["skill"])

    print(f"\n{'variant':20s} {'MAE':>7s} {'skill':>7s} {'skill@l1':>9s} {'bias':>7s} "
          f"{'std_r':>6s} {'>=30':>6s} {'>=50':>6s} {'>=100':>6s}")
    for r in rows:
        print(f"{r['variant']:20s} {r['MAE']:7.3f} {r['skill']:7.3f} {r['skill_l1']:9.3f} "
              f"{r['bias']:+7.3f} {r['std_ratio']:6.3f} "
              f"{100*r['repro_30']:5.1f}% {100*r['repro_50']:5.1f}% {100*r['repro_100']:5.1f}%")
    print("\nrepro = mean(prediction) / mean(observation) on events above the threshold; "
          "100% = unbiased in the tail.")


if __name__ == "__main__":
    main()
