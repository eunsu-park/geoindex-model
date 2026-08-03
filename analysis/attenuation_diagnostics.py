"""Diagnose whether a forecast's damping is optimal, and whether a probability head can help.

A forecast that shrinks toward the mean is not automatically defective. For any calibrated
conditional mean, two identities follow from the tower property alone, with no distributional
assumption:

    sigma_pred / sigma_obs = rho
    slope of E[prediction | observation] = rho^2

So a model correlated rho with the truth *should* have a std ratio of rho, and *should*
reproduce only rho^2 of an observed peak when you stratify by the observed value. Reporting
"the model only reproduces 30% of large events" as a defect is meaningless until it is
compared against rho^2. This script makes that comparison, and adds the MSE skill ceiling
(also rho^2 -- Murphy 1988's decomposition) so a run can be scored against what is attainable
rather than against 1.

It then answers the question that decides whether a probability head is worth training. If
the conditional law is homoscedastic, P(Y >= t | X) is a monotone function of the predicted
mean, so thresholding the mean is Bayes-optimal at some shifted threshold and no second head
can beat a recalibrated threshold. If it is heteroscedastic, monotonicity fails and a head
that models the exceedance probability directly has principled headroom. The residual-spread
scan settles that empirically.

Finally it separates the two things a loss change can do: move the operating point along a
fixed ROC curve (which a recalibrated threshold gets for free), or actually reorder the
predictions. Only the second is new information. AUC and POD at matched false-alarm ratio
tell them apart.

Usage:
    python analysis/attenuation_diagnostics.py --results-dir /path/to/results \
        --prefix probe_ap_in12h_out12h_gnn_transformer --reference tier_off
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from compare_loss_variants import load_variant


def attenuation_row(name: str, pred: np.ndarray, true: np.ndarray, pers: np.ndarray,
                    tail: float) -> dict:
    """Compare a run's dispersion and tail reproduction against the rho / rho^2 ideal."""
    rho = float(np.corrcoef(pred.ravel(), true.ravel())[0, 1])
    std_ratio = float(pred.std() / true.std())
    hi = true >= tail
    repro = float(pred[hi].mean() / true[hi].mean()) if hi.sum() > 30 else float("nan")
    mae = float(np.abs(pred - true).mean())
    skill = 1.0 - mae / float(np.abs(pers - true).mean())
    return {"variant": name, "rho": rho, "std_ratio": std_ratio,
            "std_over_rho": std_ratio / rho, "rho2": rho ** 2, "tail_repro": repro,
            "repro_over_rho2": repro / rho ** 2, "skill": skill,
            "ceiling_frac": skill / rho ** 2}


def heteroscedasticity(pred: np.ndarray, true: np.ndarray, n_bins: int = 10) -> dict:
    """Residual spread as a function of the prediction, plus its rank correlation.

    A flat profile means a threshold on the mean is Bayes-optimal at some level; a rising one
    means it is not, and modelling the exceedance probability directly has headroom.
    """
    p, t = pred.ravel(), true.ravel()
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    rows = []
    for i in range(n_bins):
        upper = p <= edges[i + 1] if i == n_bins - 1 else p < edges[i + 1]
        m = (p >= edges[i]) & upper
        if m.sum() < 50:
            continue
        rows.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": int(m.sum()),
                     "resid_std": float((t[m] - p[m]).std()),
                     "p_exceed_50": float((t[m] >= 50).mean())})
    spread = np.array([r["resid_std"] for r in rows])
    rank = lambda a: np.argsort(np.argsort(a))  # noqa: E731
    order = np.arange(len(spread))
    return {"bins": rows, "ratio": float(spread[-1] / spread[0]),
            "rank_corr": float(np.corrcoef(rank(order), rank(spread))[0, 1])}


def pod_at_far(score: np.ndarray, label: np.ndarray, budgets) -> list:
    """Best detection rate achievable within each false-alarm budget."""
    grid = np.quantile(score, np.arange(0.50, 0.9999, 0.0005))
    out = []
    for budget in budgets:
        best = 0.0
        for t in grid:
            p = score >= t
            hit = int((p & label).sum())
            fa = int((p & ~label).sum())
            miss = int((~p & label).sum())
            if fa / max(hit + fa, 1) <= budget:
                best = max(best, hit / max(hit + miss, 1))
        out.append(best)
    return out


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U)."""
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    n_pos = int(label.sum())
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (len(label) - n_pos)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Attenuation and heteroscedasticity diagnostics")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--epoch", default="best")
    ap.add_argument("--tail", type=float, default=100.0, help="ap level defining a large event")
    ap.add_argument("--event-threshold", type=float, default=50.0)
    ap.add_argument("--leads", type=int, default=24)
    ap.add_argument("--reference", default=None,
                    help="variant to compare ROC against (default: best MSE skill)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(
        args.results_dir, f"{args.prefix}_*", "validation", args.epoch, "npz.zip")))
    if not paths:
        raise SystemExit(f"no runs matched {args.prefix}_* under {args.results_dir}")
    loaded = {p.split(os.sep)[-4][len(args.prefix) + 1:]: load_variant(p) for p in paths}

    sample = next(iter(loaded.values()))
    true = sample["true"][:, :args.leads]
    pers = np.repeat(sample["pers"][:, None], true.shape[1], axis=1)

    print("Attenuation: a calibrated conditional mean has std ratio rho and tail "
          "reproduction rho^2,\nand its MSE skill cannot exceed rho^2. Ratios near 1.00 are "
          "the ideal, not a defect.\n")
    print(f"{'variant':16s} {'rho':>6s} {'std_r':>6s} {'/rho':>6s} | {'rho^2':>6s} "
          f"{'tail':>6s} {'/rho^2':>7s} | {'skill':>7s} {'ceiling':>8s} {'of ceil':>8s}")
    rows = [attenuation_row(k, v["pred"][:, :args.leads], true, pers, args.tail)
            for k, v in loaded.items()]
    for r in sorted(rows, key=lambda x: -x["skill"]):
        print(f"{r['variant']:16s} {r['rho']:6.3f} {r['std_ratio']:6.3f} "
              f"{r['std_over_rho']:6.2f} | {r['rho2']:6.3f} {r['tail_repro']:6.3f} "
              f"{r['repro_over_rho2']:7.2f} | {r['skill']:7.3f} {r['rho2']:8.3f} "
              f"{100*r['ceiling_frac']:7.1f}%")

    ref = args.reference or max(rows, key=lambda r: r["skill"])["variant"]
    label = true.max(axis=1) >= args.event_threshold
    budgets = (0.1, 0.2, 0.3, 0.4, 0.5)
    scores = {k: v["pred"][:, :args.leads].max(axis=1) for k, v in loaded.items()}
    ref_pod = pod_at_far(scores[ref], label, budgets)
    ref_auc = auc(scores[ref], label)

    print(f"\nDiscrimination against '{ref}'. A loss that only moves the operating point "
          f"leaves these\nunchanged -- a recalibrated threshold buys that for free. Only a "
          f"reordering is new information.")
    print(f"{'variant':16s} {'AUC':>7s} {'dAUC':>8s} " +
          "".join(f"{'POD@'+str(int(b*100))+'%':>10s}" for b in budgets))
    for k in sorted(scores, key=lambda k: -auc(scores[k], label)):
        a = auc(scores[k], label)
        pods = pod_at_far(scores[k], label, budgets)
        print(f"{k:16s} {a:7.4f} {a-ref_auc:+8.4f} " +
              "".join(f"{p:10.3f}" for p in pods) +
              ("" if k == ref else "   d " + " ".join(f"{p-q:+.3f}" for p, q in zip(pods, ref_pod))))

    het = heteroscedasticity(loaded[ref]["pred"][:, :args.leads], true)
    print(f"\nHeteroscedasticity of '{ref}': if the residual spread rises with the prediction,"
          f"\na threshold on the mean is not Bayes-optimal and an exceedance head has headroom.")
    print(f"{'prediction bin':>22s} {'n':>8s} {'resid std':>10s} {'P(obs>=50)':>11s}")
    for b in het["bins"]:
        print(f"{b['lo']:10.1f}~{b['hi']:10.1f} {b['n']:8d} {b['resid_std']:10.2f} "
              f"{100*b['p_exceed_50']:10.2f}%")
    print(f"\nspread ratio top/bottom bin {het['ratio']:.1f}x, rank correlation "
          f"{het['rank_corr']:+.3f}")
    verdict = ("heteroscedastic -- an exceedance-probability head has principled headroom"
               if het["ratio"] > 2 and het["rank_corr"] > 0.8 else
               "near-homoscedastic -- a recalibrated threshold on the mean is close to optimal")
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
