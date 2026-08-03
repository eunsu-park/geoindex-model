"""Calibrate a trained run into a storm-warning product without retraining it.

The models rank storm risk well (AUC ~0.87) but their predictions are damped -- the
MSE/MAE-optimal amount of shrinkage for a predictor whose correlation with truth is ~0.55 --
so a forecast value rarely reaches the ap threshold it is compared against. Warning on
``max(prediction) >= 50`` therefore discards most of what the model knows: on the held-out
split of ``ap_in12h_out12h_gnn_transformer`` it detects 4.9% of rapid-rise events. Warning on
a threshold fitted to a false-alarm budget detects 25-46% of them, from the same stored
predictions, with no change to the model.

Two products come out of this:

- a decision threshold per false-alarm budget, so the operating point is chosen from the
  cost of a miss versus a false alarm rather than inherited from the index scale;
- an isotonic map from score to probability, so the forecast can be issued as "probability
  of ap >= X within the next N hours" and the consumer picks their own threshold.

Everything is fitted on the earlier part of the record and scored on the later part, which
is also how it behaves in deployment: the fit ages, and the event rate moves with the solar
cycle, so refit periodically.

Usage:
    python analysis/calibrate_warning.py --results-dir /path/to/results \
        --run ap_in12h_out12h_gnn_transformer --write-json warning_calibration.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import zipfile

import numpy as np

FAR_BUDGETS = (0.20, 0.30, 0.40, 0.50)
RELIABILITY_BINS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.01)


def load_run(zip_path: str, n_leads: int) -> dict:
    """Read per-anchor score, observed maximum and anchor value from a validation archive.

    Args:
        zip_path: Path to ``validation/<epoch>/npz.zip``.
        n_leads: Number of forecast steps to score over (the warning horizon).

    Returns:
        Dict with anchor, score (max prediction), obs_max and anchor_value arrays.
    """
    anchors, score, obs_max, anchor_val = [], [], [], []
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".npz")):
            d = np.load(io.BytesIO(z.read(name)), allow_pickle=True)
            ivars = [str(x) for x in np.asarray(d["input_variables"]).ravel()]
            tvar = str(np.asarray(d["target_variables"]).ravel()[0])
            k = min(n_leads, np.asarray(d["predictions"]).shape[0])
            anchors.append(str(np.asarray(d["anchor"])))
            score.append(float(np.asarray(d["predictions"])[:k, 0].max()))
            obs_max.append(float(np.asarray(d["targets"])[:k, 0].max()))
            anchor_val.append(float(np.asarray(d["inputs"])[-1, ivars.index(tvar)]))
    return {"anchor": np.asarray(anchors), "score": np.asarray(score),
            "obs_max": np.asarray(obs_max), "anchor_value": np.asarray(anchor_val)}


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U). 0.5 is no discrimination."""
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    n_pos = int(label.sum())
    n_neg = len(label) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def contingency(pred_yes: np.ndarray, obs_yes: np.ndarray) -> dict:
    """POD / FAR / HSS / frequency bias for a boolean forecast."""
    a = int((pred_yes & obs_yes).sum())
    b = int((pred_yes & ~obs_yes).sum())
    c = int((~pred_yes & obs_yes).sum())
    d = int((~pred_yes & ~obs_yes).sum())
    den = (a + c) * (c + d) + (a + b) * (b + d)
    return {"pod": a / max(a + c, 1), "far": b / max(a + b, 1),
            "bias": (a + b) / max(a + c, 1),
            "hss": 2 * (a * d - b * c) / den if den else 0.0}


def fit_threshold(score: np.ndarray, label: np.ndarray, budget: float | None) -> float:
    """Lowest threshold whose false-alarm ratio stays within ``budget``.

    Thresholds are taken as quantiles of the training scores rather than a fixed grid in ap
    units, so the same routine works whatever scale the score is on. With ``budget=None``
    the HSS-maximising threshold is returned instead.

    Args:
        score: Training-split scores.
        label: Training-split event labels.
        budget: Maximum acceptable false-alarm ratio, or None for HSS-optimal.

    Returns:
        The chosen threshold.
    """
    grid = np.quantile(score, np.arange(0.50, 0.999, 0.002))
    if budget is None:
        return float(max(grid, key=lambda t: contingency(score >= t, label)["hss"]))
    ok = [t for t in grid if contingency(score >= t, label)["far"] <= budget]
    return float(min(ok)) if ok else float(grid[-1])


def isotonic_fit(score: np.ndarray, label: np.ndarray) -> tuple:
    """Pool-adjacent-violators fit of P(event | score); returns (knot_x, knot_y).

    Isotonic rather than a sigmoid because the score-to-probability relation only has to be
    monotone -- it is not logistic in shape, and forcing that shape mis-calibrates the tail
    where the warnings actually matter.
    """
    order = np.argsort(score)
    x = score[order].astype(float)
    y = label[order].astype(float)

    values, weights, starts = [], [], []
    for i, yi in enumerate(y):
        values.append(yi)
        weights.append(1.0)
        starts.append(i)
        while len(values) > 1 and values[-2] > values[-1]:
            v2, w2 = values.pop(), weights.pop()
            starts.pop()
            v1, w1 = values.pop(), weights.pop()
            values.append((v1 * w1 + v2 * w2) / (w1 + w2))
            weights.append(w1 + w2)
    fitted = np.empty(len(y))
    pos = 0
    for v, w in zip(values, weights):
        fitted[pos:pos + int(w)] = v
        pos += int(w)
    return x, fitted


def isotonic_apply(knot_x: np.ndarray, knot_y: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Evaluate the fitted step function, clamped to the fitted range."""
    return np.clip(np.interp(score, knot_x, knot_y), 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate a run into a warning product")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--run", required=True, help="experiment directory name")
    ap.add_argument("--epoch", default="best")
    ap.add_argument("--leads", type=int, default=24, help="forecast steps in the warning horizon")
    ap.add_argument("--event-threshold", type=float, default=50.0, help="ap level defining the event")
    ap.add_argument("--onset-rise", type=float, default=30.0,
                    help="rise over the anchor value that marks a rapid-rise event")
    ap.add_argument("--fit-fraction", type=float, default=0.6,
                    help="earlier fraction of anchors used to fit; the rest is held out")
    ap.add_argument("--write-json", help="write the deploy calibration to this path")
    args = ap.parse_args()

    path = os.path.join(args.results_dir, args.run, "validation", args.epoch, "npz.zip")
    data = load_run(path, args.leads)
    score, obs, a0 = data["score"], data["obs_max"], data["anchor_value"]
    label = obs >= args.event_threshold
    onset = ((obs - a0) >= args.onset_rise) & (a0 < args.event_threshold * 0.6)

    n = len(label)
    cut = int(n * args.fit_fraction)
    tr, te = slice(0, cut), slice(cut, n)
    hours = args.leads * 0.5

    print(f"run              {args.run}")
    print(f"event            ap >= {args.event_threshold:g} within {hours:g} h "
          f"({args.leads} steps)")
    print(f"anchors          fit {cut} ({data['anchor'][0]}–{data['anchor'][cut-1]}), "
          f"held out {n-cut} ({data['anchor'][cut]}–{data['anchor'][-1]})")
    print(f"event rate       fit {100*label[tr].mean():.1f}%  held out {100*label[te].mean():.1f}%")
    print(f"AUC (all)        {auc(score, label):.3f}   rapid-rise events held out: {int(onset[te].sum())}")

    print(f"\n{'decision rule':34s} {'thr':>7s} {'POD':>6s} {'FAR':>6s} {'BIAS':>6s} {'HSS':>6s} {'rise POD':>9s}")

    def row(label_text, thr):
        c = contingency(score[te] >= thr, label[te])
        rise = float((score[te][onset[te]] >= thr).mean()) if onset[te].sum() else float("nan")
        print(f"{label_text:34s} {thr:7.2f} {c['pod']:6.3f} {c['far']:6.3f} {c['bias']:6.3f} "
              f"{c['hss']:6.3f} {rise:9.3f}")
        return {"threshold": thr, **c, "rise_pod": rise}

    rows = {"raw": row(f"raw scale (score >= {args.event_threshold:g})", args.event_threshold)}
    rows["hss_optimal"] = row("HSS-optimal", fit_threshold(score[tr], label[tr], None))
    for budget in FAR_BUDGETS:
        rows[f"far_{budget:g}"] = row(f"false-alarm budget <= {budget:.0%}",
                                      fit_threshold(score[tr], label[tr], budget))

    knot_x, knot_y = isotonic_fit(score[tr], label[tr])
    prob = isotonic_apply(knot_x, knot_y, score[te])
    base = float(label[tr].mean())
    brier = float(np.mean((prob - label[te]) ** 2))
    brier_ref = float(np.mean((base - label[te]) ** 2))
    print(f"\nisotonic probability   Brier {brier:.4f} vs climatology {brier_ref:.4f} "
          f"(skill {1 - brier/brier_ref:+.3f})")
    print(f"{'forecast probability':22s} {'n':>7s} {'mean fc':>8s} {'observed':>9s}")
    for lo, hi in zip(RELIABILITY_BINS[:-1], RELIABILITY_BINS[1:]):
        m = (prob >= lo) & (prob < hi)
        if m.sum():
            print(f"{f'{lo:.2f} – {min(hi,1.0):.2f}':22s} {int(m.sum()):7d} "
                  f"{prob[m].mean():8.3f} {label[te][m].mean():9.3f}")

    if args.write_json:
        # Thin the step function to its jumps: the knots are what deployment needs.
        keep = np.concatenate(([0], np.where(np.diff(knot_y) > 0)[0] + 1, [len(knot_y) - 1]))
        payload = {
            "run": args.run, "epoch": args.epoch,
            "event": {"threshold_ap": args.event_threshold, "horizon_steps": args.leads,
                      "horizon_hours": hours},
            "score": "max of the deterministic forecast over the horizon",
            "fit": {"anchors": cut, "first": data["anchor"][0], "last": data["anchor"][cut - 1],
                    "event_rate": float(label[tr].mean())},
            "thresholds": {k: v["threshold"] for k, v in rows.items()},
            "held_out": {k: {m: v[m] for m in ("pod", "far", "bias", "hss", "rise_pod")}
                         for k, v in rows.items()},
            "isotonic": {"score": [float(v) for v in knot_x[keep]],
                         "probability": [float(v) for v in knot_y[keep]]},
        }
        with open(args.write_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.write_json}  ({len(keep)} isotonic knots)")


if __name__ == "__main__":
    main()
