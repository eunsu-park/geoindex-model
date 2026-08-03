"""Report a run against the community verification guidelines for geomagnetic index models.

Liemohn et al. (2018), "Model Evaluation Guidelines for Geomagnetic Index Predictions",
Space Weather 16, 2079-2102, doi:10.1029/2018SW002067, prescribes what a paper reporting an
index forecast has to show. Three of its requirements are the ones that matter here.

The regression slope B is the damping metric. A perfect forecast has zero offset and unit
slope; B below one means the forecast does not keep pace with the observed increase, which is
the conditional bias in Murphy's decomposition. Reporting it per lead makes the damping a
number instead of an impression.

Event scores must be computed over a range of thresholds, not one. The guidelines ask for at
least ten thresholds and a ROC curve, with at least ten hits and ten correct negatives at each
so the contingency table is not built on a handful of cases. A single threshold hides exactly
the behaviour that matters, which is how warning on the index scale came to detect 4.9% of
rapid-rise events here.

Frequency bias belongs next to POD. Values below one mean misses dominate -- the model
under-forecasts the event at that threshold -- and it is the quantitative fingerprint of a
damped forecast in event space, where the slope B is its fingerprint in regression space.

Errors are also binned by observed activity, because quiet intervals dominate the record and
a pooled error is a statement about quiet time. Liemohn et al. note that storm intervals are a
small part of the database and so quiet conditions dominate any pooled data-model metric.

Usage:
    python analysis/liemohn_metrics.py --results-dir /path/to/results \
        --run ap_in12h_out12h_gnn_transformer
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile

import numpy as np

MIN_CASES = 10          # Liemohn et al.: >= 10 hits and >= 10 correct negatives per threshold
ACTIVITY_BINS = ((0, 30, "quiet"), (30, 50, "G1-ish"), (50, 100, "moderate"), (100, 1e9, "large"))


def load(zip_path: str) -> dict:
    """Read predictions, observations and the anchor value from a validation archive."""
    pred, true, anchor = [], [], []
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".npz")):
            d = np.load(io.BytesIO(z.read(name)), allow_pickle=True)
            ivars = [str(x) for x in np.asarray(d["input_variables"]).ravel()]
            tvar = str(np.asarray(d["target_variables"]).ravel()[0])
            pred.append(np.asarray(d["predictions"])[:, 0])
            true.append(np.asarray(d["targets"])[:, 0])
            anchor.append(np.asarray(d["inputs"])[-1, ivars.index(tvar)])
    return {"pred": np.asarray(pred), "true": np.asarray(true), "anchor": np.asarray(anchor)}


def fit_line(obs: np.ndarray, pred: np.ndarray) -> tuple:
    """Least-squares fit of prediction on observation; returns (intercept A, slope B, r)."""
    slope, intercept = np.polyfit(obs, pred, 1)
    return float(intercept), float(slope), float(np.corrcoef(obs, pred)[0, 1])


def contingency(pred_yes: np.ndarray, obs_yes: np.ndarray) -> dict:
    """2x2 table with the scores the guidelines ask for."""
    hit = int((pred_yes & obs_yes).sum())
    fa = int((pred_yes & ~obs_yes).sum())
    miss = int((~pred_yes & obs_yes).sum())
    corr_neg = int((~pred_yes & ~obs_yes).sum())
    den = (hit + miss) * (miss + corr_neg) + (hit + fa) * (fa + corr_neg)
    pod = hit / max(hit + miss, 1)
    pofd = fa / max(fa + corr_neg, 1)
    return {"hits": hit, "false_alarms": fa, "misses": miss, "correct_negatives": corr_neg,
            "pod": pod, "pofd": pofd, "far": fa / max(hit + fa, 1),
            "csi": hit / max(hit + fa + miss, 1),
            "freq_bias": (hit + fa) / max(hit + miss, 1),
            "hss": 2 * (hit * corr_neg - fa * miss) / den if den else 0.0,
            "tss": pod - pofd,
            "usable": hit >= MIN_CASES and corr_neg >= MIN_CASES}


def main() -> None:
    ap = argparse.ArgumentParser(description="Liemohn et al. (2018) verification set")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--epoch", default="best")
    ap.add_argument("--thresholds", default="10,20,30,40,50,60,80,100,120,150,200",
                    help="event thresholds in ap units; the guidelines ask for at least ten")
    args = ap.parse_args()

    d = load(os.path.join(args.results_dir, args.run, "validation", args.epoch, "npz.zip"))
    pred, true, anchor = d["pred"], d["true"], d["anchor"]
    n_leads = true.shape[1]
    pers = np.repeat(anchor[:, None], n_leads, axis=1)

    print(f"run     {args.run}")
    print(f"anchors {len(true)}, leads {n_leads}, pairs {true.size}\n")

    print("Regression fit per lead (prediction on observation). A perfect forecast has A = 0 "
          "and B = 1;\nB < 1 is the conditional bias -- the forecast not keeping pace with the "
          "observed increase.")
    print(f"{'lead':>6s} {'A':>8s} {'B':>7s} {'r':>7s} | {'B (persistence)':>16s}")
    for j in range(n_leads):
        if j not in (0, 1, 3, 5, 11, 17, n_leads - 1):
            continue
        a, b, r = fit_line(true[:, j], pred[:, j])
        _, b_pers, _ = fit_line(true[:, j], pers[:, j])
        print(f"{(j+1)*0.5:5.1f}h {a:8.2f} {b:7.3f} {r:7.3f} | {b_pers:16.3f}")

    thresholds = [float(t) for t in args.thresholds.split(",")]
    score = pred.max(axis=1)
    obs_max = true.max(axis=1)
    print(f"\nEvent scores over {len(thresholds)} thresholds, scoring the whole forecast window "
          f"(does the\nindex reach the level within {n_leads*0.5:g} h). Rows marked 'thin' fall "
          f"below the {MIN_CASES}-case floor.")
    print(f"{'thr':>6s} {'base rate':>10s} {'POD':>6s} {'POFD':>6s} {'FAR':>6s} {'CSI':>6s} "
          f"{'FB':>6s} {'HSS':>6s} {'TSS':>6s}")
    for t in thresholds:
        c = contingency(score >= t, obs_max >= t)
        flag = "" if c["usable"] else "   thin"
        print(f"{t:6.0f} {100*(obs_max >= t).mean():9.2f}% {c['pod']:6.3f} {c['pofd']:6.3f} "
              f"{c['far']:6.3f} {c['csi']:6.3f} {c['freq_bias']:6.3f} {c['hss']:6.3f} "
              f"{c['tss']:6.3f}{flag}")
    print("\nFB below 1 means misses dominate: the forecast calls the event less often than it "
          "happens.\nThat is the damping seen in event space, as B < 1 is the same damping in "
          "regression space.")

    print("\nROC on the same score, sweeping the decision threshold rather than the index "
          "scale.")
    print(f"{'quantile':>9s} {'thr':>8s} {'POFD':>7s} {'POD':>7s}")
    for q in (0.50, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999):
        t = float(np.quantile(score, q))
        c = contingency(score >= t, obs_max >= 50.0)
        print(f"{q:9.3f} {t:8.2f} {c['pofd']:7.3f} {c['pod']:7.3f}")
    print("event for the ROC: ap >= 50 within the window")

    print("\nError binned by observed activity. A pooled error is a statement about quiet time,"
          "\nsince quiet intervals dominate the record.")
    print(f"{'bin':>22s} {'n pairs':>10s} {'share':>7s} {'MAE':>8s} {'RMSE':>8s} {'bias':>8s} "
          f"{'MAE (pers)':>11s}")
    flat_t, flat_p, flat_q = true.ravel(), pred.ravel(), pers.ravel()
    for lo, hi, name in ACTIVITY_BINS:
        m = (flat_t >= lo) & (flat_t < hi)
        if not m.any():
            continue
        err = flat_p[m] - flat_t[m]
        label = f"{name} [{lo:g},{hi:g})" if hi < 1e8 else f"{name} [{lo:g},inf)"
        print(f"{label:>22s} {int(m.sum()):10d} {100*m.mean():6.2f}% "
              f"{np.abs(err).mean():8.3f} {np.sqrt((err**2).mean()):8.3f} {err.mean():+8.3f} "
              f"{np.abs(flat_q[m] - flat_t[m]).mean():11.3f}")


if __name__ == "__main__":
    main()
