"""Compare loss / target-transform probe variants on the diagnostics that matter.

Reads each variant's ``validation/<epoch>/npz.zip`` (deterministic predictions; the probe runs
with ``validation.mcd_samples=0``) and reports, in original units:

- MAE and skill against persistence, pooled and at lead 1
- storm reproduction: mean prediction / mean observation for obs >= 30 / 50 / 100
- dispersion: std(prediction) / std(observation), and overall bias

The persistence anchor is the last observed target value in the input window. Variants that
drop the target input channel have no anchor of their own, so the anchor series is taken from
whichever variant still carries it and matched by anchor timestamp.

With ``--ridge`` a reference row is added: a plain per-lead ridge regression on the same
input window, fitted on the same training index and scored on the same anchors. It is the
line a trained model has to clear -- a variant that only wins on MAE while falling behind
ridge on skill@l1 / std_r / tail reproduction has bought its MAE by damping the forecast.

With ``--categorical`` each variant is also scored as a yes/no storm warning at the NOAA
thresholds. The two rankings are close to opposite: the MAE-optimal variants under-forecast
storms by three to eight times (frequency BIAS 0.12-0.41) and score below plain persistence
on HSS, while the tier-weighted ones sit near BIAS 1.0 and lead. Accuracy reporting and
warning operation are different products and cannot be ranked by one number.

Usage:
    python analysis/compare_loss_variants.py --results-dir /path/to/results \
        --prefix probe_ap_in12h_out12h_gnn_transformer --ridge --categorical
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
            "pred": np.asarray(pred), "pers": None if np.isnan(pers_arr).all() else pers_arr,
            "input_variables": ivars, "target_variable": tvar,
            "n_in": int(np.asarray(d["inputs"]).shape[0])}


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


def _windows(frame, anchors, columns, n_in, n_out, target):
    """Build (input window, target window) pairs at the given anchor timestamps.

    Follows the dataset's half-open convention (``input_start=-N, input_end=0,
    target_start=0``): the input window is rows ``[i-n_in, i)`` -- it ends one step BEFORE
    the anchor row -- and the target window is rows ``[i, i+n_out)``, starting AT the anchor
    row. Including row ``i`` in the input would leak the lead-1 target.

    Args:
        frame: Source dataframe indexed by datetime, 30-min cadence.
        anchors: Anchor timestamps (pandas DatetimeIndex or Series).
        columns: Input variable names, in model order.
        n_in: Input window length in steps.
        n_out: Number of forecast steps.
        target: Target column name.

    Returns:
        (X, Y) arrays of shape (n, n_in, n_vars) and (n, n_out).
    """
    pos = {t: i for i, t in enumerate(frame.index)}
    values = frame[columns].to_numpy(float)
    tgt = frame[target].to_numpy(float)
    X, Y = [], []
    for t in anchors:
        i = pos.get(t)
        if i is None or i - n_in < 0 or i + n_out > len(values):
            continue
        w, y = values[i - n_in:i], tgt[i:i + n_out]
        if np.isfinite(w).all() and np.isfinite(y).all():
            X.append(w)
            Y.append(y)
    return np.asarray(X), np.asarray(Y)


def ridge_reference(sample: dict, anchors, datasets_root: str, parquet: str,
                    train_index: str, alpha: float) -> np.ndarray:
    """Fit one ridge per lead on the training index and predict at ``anchors``.

    Features are the flattened input window; columns that are non-negative throughout the
    training data get a log1p compression first, then everything is standardised. The
    target is fitted in log1p space and mapped back with expm1, matching how the models
    normalise ``ap30``/``hp30``.

    Args:
        sample: One loaded variant, used only for the window geometry and variable names.
        anchors: Anchor timestamps to predict at, in the loaded runs' order.
        datasets_root: Directory holding the parquet and the index CSVs.
        parquet: Parquet filename relative to ``datasets_root``.
        train_index: Training index CSV relative to ``datasets_root``.
        alpha: Ridge penalty.

    Returns:
        Predictions of shape (len(anchors), n_out); rows with an unusable window are NaN.
    """
    import pandas as pd

    frame = pd.read_parquet(os.path.join(datasets_root, parquet))
    frame = frame.set_index("datetime").sort_index()
    tr = pd.read_csv(os.path.join(datasets_root, train_index), parse_dates=["datetime"])

    cols, target = sample["input_variables"], sample["target_variable"]
    n_in, n_out = sample["n_in"], sample["true"].shape[1]
    Xtr, Ytr = _windows(frame, tr.datetime, cols, n_in, n_out, target)
    ts = pd.to_datetime(anchors, format="%Y%m%d%H%M%S")
    Xte, _ = _windows(frame, ts, cols, n_in, n_out, target)
    if len(Xtr) == 0 or len(Xte) != len(anchors):
        raise SystemExit(f"ridge: window build failed (train {len(Xtr)}, "
                         f"test {len(Xte)} vs {len(anchors)} anchors)")

    logmask = Xtr.reshape(-1, Xtr.shape[-1]).min(axis=0) >= 0.0  # sign-free columns only
    def feats(X, mu=None, sd=None):
        Z = X.copy()
        Z[:, :, logmask] = np.log1p(np.clip(Z[:, :, logmask], 0, None))
        F = np.nan_to_num(Z.reshape(len(X), -1))
        if mu is None:
            mu, sd = F.mean(0), F.std(0) + 1e-8
        return (F - mu) / sd, mu, sd

    Ftr, mu, sd = feats(Xtr)
    Fte, _, _ = feats(Xte, mu, sd)
    Ltr = np.log1p(np.clip(Ytr, 0, None))
    out = np.empty((len(Fte), n_out))
    for j in range(n_out):
        xm, ym = Ftr.mean(0), Ltr[:, j].mean()
        Xc = Ftr - xm
        w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(Xc.shape[1]), Xc.T @ (Ltr[:, j] - ym))
        out[:, j] = np.clip(np.expm1((Fte - xm) @ w + ym), 0, None)
    return out


def contingency(pred: np.ndarray, obs: np.ndarray, threshold: float) -> dict:
    """Score a forecast as a yes/no event at one ap threshold.

    The 2x2 table is built over every (event, lead) pair. BIAS is the frequency bias --
    forecast events divided by observed events -- so 1.0 means the forecast calls the event
    as often as it happens, below 1.0 means it under-forecasts. It is the metric that
    separates a damped forecast from a usable warning: an MAE-optimal model can score well
    on MAE while calling storms three to eight times too rarely.

    Args:
        pred: Predicted values.
        obs: Observed values (same shape).
        threshold: Event threshold in ap units.

    Returns:
        Dict with hits/false_alarms/misses/correct_negatives and pod, far, csi, bias, hss.
    """
    hit = int(((obs >= threshold) & (pred >= threshold)).sum())
    false_alarm = int(((obs < threshold) & (pred >= threshold)).sum())
    miss = int(((obs >= threshold) & (pred < threshold)).sum())
    correct_neg = int(((obs < threshold) & (pred < threshold)).sum())

    observed_yes = hit + miss
    forecast_yes = hit + false_alarm
    denom = (observed_yes * (miss + correct_neg) + forecast_yes * (false_alarm + correct_neg))
    return {
        "hits": hit, "false_alarms": false_alarm, "misses": miss,
        "correct_negatives": correct_neg,
        "pod": hit / observed_yes if observed_yes else float("nan"),
        "far": false_alarm / forecast_yes if forecast_yes else float("nan"),
        "csi": hit / (hit + false_alarm + miss) if (hit + false_alarm + miss) else float("nan"),
        "bias": forecast_yes / observed_yes if observed_yes else float("nan"),
        "hss": 2 * (hit * correct_neg - false_alarm * miss) / denom if denom else 0.0,
    }


def report_categorical(preds: dict, obs: np.ndarray, thresholds) -> None:
    """Print one contingency table per threshold, ranked by HSS."""
    for threshold in thresholds:
        n_events = int((obs >= threshold).sum())
        print(f"\n=== threshold {threshold:g} ap  "
              f"(observed {n_events} / {obs.size} = {100 * n_events / obs.size:.2f}%) ===")
        print(f"{'variant':20s} {'POD':>6s} {'FAR':>6s} {'CSI':>6s} {'BIAS':>6s} {'HSS':>7s}")
        scored = [(name, contingency(p, obs, threshold)) for name, p in preds.items()]
        for name, c in sorted(scored, key=lambda t: -t[1]["hss"]):
            print(f"{name:20s} {c['pod']:6.3f} {c['far']:6.3f} {c['csi']:6.3f} "
                  f"{c['bias']:6.3f} {c['hss']:7.3f}")
    print("\nPOD = hits/observed, FAR = false alarms/forecast, CSI = hits/(hits+FA+misses),")
    print("BIAS = forecast events/observed events (1.0 = right frequency), HSS = Heidke skill.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare loss-probe variants")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--prefix", required=True, help="e.g. probe_ap_in12h_out12h_gnn_transformer")
    ap.add_argument("--epoch", default="best")
    ap.add_argument("--ridge", action="store_true",
                    help="add a per-lead ridge regression reference row")
    ap.add_argument("--datasets-root", default=os.path.expanduser("~/Projects/GeoIndex/datasets"))
    ap.add_argument("--parquet", default="data.parquet")
    ap.add_argument("--train-index", default="total_ap/train_index.csv")
    ap.add_argument("--ridge-alpha", type=float, default=100.0)
    ap.add_argument("--categorical", action="store_true",
                    help="also score each variant as a yes/no storm warning (POD/FAR/CSI/BIAS/HSS)")
    ap.add_argument("--thresholds", default="30,50,100",
                    help="comma-separated ap thresholds for --categorical (default NOAA G1/G2/G3+)")
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
    ridge_pred = None

    if args.ridge:
        sample = next(iter(loaded.values()))
        ridge_pred = ridge_reference(sample, sample["anchor"], args.datasets_root,
                                     args.parquet, args.train_index, args.ridge_alpha)
        rows.append(summarize("RIDGE (reference)",
                              {**sample, "pred": ridge_pred}, pers_map))

    rows.sort(key=lambda r: -r["skill"])

    print(f"\n{'variant':20s} {'MAE':>7s} {'skill':>7s} {'skill@l1':>9s} {'bias':>7s} "
          f"{'std_r':>6s} {'>=30':>6s} {'>=50':>6s} {'>=100':>6s}")
    for r in rows:
        print(f"{r['variant']:20s} {r['MAE']:7.3f} {r['skill']:7.3f} {r['skill_l1']:9.3f} "
              f"{r['bias']:+7.3f} {r['std_ratio']:6.3f} "
              f"{100*r['repro_30']:5.1f}% {100*r['repro_50']:5.1f}% {100*r['repro_100']:5.1f}%")
    print("\nrepro = mean(prediction) / mean(observation) on events above the threshold; "
          "100% = unbiased in the tail.")

    if args.categorical:
        sample = next(iter(loaded.values()))
        preds = {name: v["pred"] for name, v in loaded.items()}
        if ridge_pred is not None:
            preds["RIDGE (reference)"] = ridge_pred
        preds["persistence"] = np.repeat(sample["pers"][:, None], sample["true"].shape[1], axis=1)
        report_categorical(preds, sample["true"],
                           [float(t) for t in args.thresholds.split(",")])


if __name__ == "__main__":
    main()
