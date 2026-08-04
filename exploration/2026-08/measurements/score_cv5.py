"""Does the observation-time gain hold across folds, or was it one period's luck?

The single-split result was AUC +0.0109 and per-lead rho 0.567 -> 0.600. Both are paired
comparisons here: every fold trains and validates on the same anchors under both clocks, so
the difference is attributable to the clock and nothing else. A gain that survives five
disjoint validation periods spanning 2006-2025 is a property of the change; one that appears
in one or two folds is a property of the period.

The folds are of very different size and activity (fold1 validates 2006-2009, deep solar
minimum), so the per-fold spread is expected to be wide. What matters is the sign and whether
the paired difference is consistent.
"""

import io
import os
import zipfile

import numpy as np

ROOT = os.path.expanduser("~/Projects/GeoIndex/results")
OBS = "cv5obs_ap_in12h_out12h_gnn_transformer_fold{}"
BSN = "ap_in12h_out12h_gnn_transformer_fold{}"
TAIL, EVENT = 100.0, 50.0


def load(run):
    path = os.path.join(ROOT, run, "validation", "best", "npz.zip")
    anchors, true, pred, pers = [], [], [], []
    with zipfile.ZipFile(path) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".npz")):
            d = np.load(io.BytesIO(z.read(name)), allow_pickle=True)
            ivars = [str(x) for x in np.asarray(d["input_variables"]).ravel()]
            tvar = str(np.asarray(d["target_variables"]).ravel()[0])
            anchors.append(str(np.asarray(d["anchor"])))
            true.append(np.asarray(d["targets"])[:, 0])
            pred.append(np.asarray(d["predictions"])[:, 0])
            pers.append(np.asarray(d["inputs"])[-1, ivars.index(tvar)])
    return (np.asarray(anchors), np.asarray(true), np.asarray(pred), np.asarray(pers))


def auc(score, label):
    ranks = np.empty(len(score), float)
    ranks[np.argsort(score)] = np.arange(1, len(score) + 1)
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_pos, n_neg = label.sum(), (~label).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pod_at_far(score, label, budget=0.30):
    order = np.argsort(-score)
    lab = label[order]
    hits, fa = np.cumsum(lab), np.cumsum(~lab)
    ok = fa / np.maximum(hits + fa, 1) <= budget
    return float(hits[ok].max() / label.sum()) if ok.any() and label.sum() else float("nan")


rows = []
print(f"{'fold':>5s} {'period':>21s} {'n':>7s} {'ev>=50':>7s} | "
      f"{'AUC bsn':>8s} {'AUC obs':>8s} {'dAUC':>8s} | "
      f"{'lead bsn':>8s} {'lead obs':>8s} {'d':>7s} | "
      f"{'peak bsn':>8s} {'peak obs':>8s} {'d':>7s}")
for f in range(1, 6):
    a_o, t_o, p_o, pe_o = load(OBS.format(f))
    a_b, t_b, p_b, pe_b = load(BSN.format(f))
    common = np.intersect1d(a_o, a_b)
    so, sb = np.isin(a_o, common), np.isin(a_b, common)
    oo, ob = np.argsort(a_o[so]), np.argsort(a_b[sb])
    true = t_b[sb][ob]
    assert np.allclose(true, t_o[so][oo], atol=1e-3), f"fold{f}: targets differ"
    po, pb = p_o[so][oo], p_b[sb][ob]
    obs = true.max(axis=1)
    label = obs >= EVENT
    r = {
        "fold": f, "n": len(obs), "ev": int(label.sum()),
        "t0": a_b[sb][ob][0][:6], "t1": a_b[sb][ob][-1][:6],
        "auc_b": auc(pb.max(1), label), "auc_o": auc(po.max(1), label),
        "lead_b": float(np.corrcoef(pb.ravel(), true.ravel())[0, 1]),
        "lead_o": float(np.corrcoef(po.ravel(), true.ravel())[0, 1]),
        "peak_b": float(np.corrcoef(pb.max(1), obs)[0, 1]),
        "peak_o": float(np.corrcoef(po.max(1), obs)[0, 1]),
        "mae_b": float(np.abs(pb - true).mean()), "mae_o": float(np.abs(po - true).mean()),
        "pod_b": pod_at_far(pb.max(1), label), "pod_o": pod_at_far(po.max(1), label),
        "am_b": float(np.bincount(pb.argmax(1), minlength=true.shape[1]).max() / len(obs)),
        "am_o": float(np.bincount(po.argmax(1), minlength=true.shape[1]).max() / len(obs)),
    }
    rows.append(r)
    print(f"{f:5d} {r['t0']+'–'+r['t1']:>21s} {r['n']:7d} {r['ev']:7d} | "
          f"{r['auc_b']:8.4f} {r['auc_o']:8.4f} {r['auc_o']-r['auc_b']:+8.4f} | "
          f"{r['lead_b']:8.3f} {r['lead_o']:8.3f} {r['lead_o']-r['lead_b']:+7.3f} | "
          f"{r['peak_b']:8.3f} {r['peak_o']:8.3f} {r['peak_o']-r['peak_b']:+7.3f}")

print("\nPaired differences across the five folds (obs − bow-shock):")
print(f"{'metric':16s} {'mean':>8s} {'min':>8s} {'max':>8s} {'folds up':>9s}")
for key, lab in (("auc", "AUC >= 50"), ("lead", "per-lead rho"), ("peak", "peak rho"),
                 ("pod", "POD @ FAR 30%")):
    d = np.array([r[f"{key}_o"] - r[f"{key}_b"] for r in rows])
    print(f"{lab:16s} {d.mean():+8.4f} {d.min():+8.4f} {d.max():+8.4f} "
          f"{int((d > 0).sum())}/5")
d = np.array([r["mae_o"] - r["mae_b"] for r in rows])
print(f"{'per-lead MAE':16s} {d.mean():+8.4f} {d.min():+8.4f} {d.max():+8.4f} "
      f"{int((d < 0).sum())}/5 better")

print("\nShape: concentration of the predicted maximum (lower is better; baseline single-split "
      "was 26.2 %)")
print(f"{'fold':>5s} {'bow-shock':>10s} {'obs':>10s}")
for r in rows:
    print(f"{r['fold']:5d} {100*r['am_b']:9.1f}% {100*r['am_o']:9.1f}%")

print("\nWeighted by validation size (the folds differ ~3x in n):")
w = np.array([r["n"] for r in rows], float)
w /= w.sum()
for key, lab in (("auc", "AUC"), ("lead", "per-lead rho"), ("peak", "peak rho")):
    b = np.array([r[f"{key}_b"] for r in rows])
    o = np.array([r[f"{key}_o"] for r in rows])
    print(f"  {lab:14s} {(w*b).sum():.4f} -> {(w*o).sum():.4f}  ({(w*(o-b)).sum():+.4f})")
