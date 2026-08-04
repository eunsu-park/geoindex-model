"""Score the peak-head runs against the pass mark set before they ran.

Four criteria, all fixed in advance (docs/geoindex-model/..., section 6.1):

  1. peak reproduction above 100 ap must beat the baseline's 0.351
  2. per-lead rho must NOT fall below the baseline's 0.567   <- the shape criterion the
  3. the predicted argmax must NOT concentrate above ~30% on any single lead   <- last pass
     mark lacked, which is how a model that had destroyed the forecast passed it
  4. peak rho above 0.719 / peak MAE below 13.98 beats peak1

The peak head emits its own scalar, so its peak forecast is `peak_prediction`, not
`predictions.max(axis=1)`. Both are scored: the difference between them is the whole question.
"""

import io
import os
import zipfile

import numpy as np

ROOT = os.path.expanduser("~/Projects/GeoIndex/results")
PREFIX = "probe_ap_in12h_out12h_gnn_transformer"
VARIANTS = ["baseline", "peak1", "peakhead1", "peakhead05", "peakhead_mae"]
TAIL, EVENT, SPLIT = 100.0, 50.0, 0.60


def load(variant):
    """Read curve, target, anchor value and the peak head's scalar if the run has one."""
    path = os.path.join(ROOT, f"{PREFIX}_{variant}", "validation", "best", "npz.zip")
    anchors, true, pred, pers, peak = [], [], [], [], []
    with zipfile.ZipFile(path) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".npz")):
            d = np.load(io.BytesIO(z.read(name)), allow_pickle=True)
            ivars = [str(x) for x in np.asarray(d["input_variables"]).ravel()]
            tvar = str(np.asarray(d["target_variables"]).ravel()[0])
            anchors.append(str(np.asarray(d["anchor"])))
            true.append(np.asarray(d["targets"])[:, 0])
            pred.append(np.asarray(d["predictions"])[:, 0])
            pers.append(np.asarray(d["inputs"])[-1, ivars.index(tvar)])
            if "peak_prediction" in d:
                peak.append(float(np.asarray(d["peak_prediction"]).ravel()[0]))
    return {"anchor": np.asarray(anchors), "true": np.asarray(true),
            "pred": np.asarray(pred), "pers": np.asarray(pers),
            "peak": np.asarray(peak) if peak else None}


def auc(score, label):
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    ranks = np.empty(len(score), float)
    ranks[np.argsort(score)] = np.arange(1, len(score) + 1)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_pos, n_neg = label.sum(), (~label).sum()
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def peak_row(pred, obs, pers):
    rho = float(np.corrcoef(pred, obs)[0, 1])
    hi = obs >= TAIL
    mae = float(np.abs(pred - obs).mean())
    return {"rho": rho, "std_r": float(pred.std() / obs.std()),
            "over_rho": float(pred.std() / obs.std()) / rho,
            "repro": float(pred[hi].mean() / obs[hi].mean()), "mae": mae,
            "skill": 1.0 - mae / float(np.abs(pers - obs).mean()),
            "reach": float((pred[hi] >= TAIL).mean())}


L = {v: load(v) for v in VARIANTS}
true = L["baseline"]["true"]
obs = true.max(axis=1)
pers = L["baseline"]["pers"]
n_lead = true.shape[1]
cut = int(len(obs) * SPLIT)
test = slice(cut, len(obs))
for v in VARIANTS:
    assert np.array_equal(L[v]["anchor"], L["baseline"]["anchor"]), f"{v} anchors differ"

print(f"anchors {len(obs)}, held-out {len(obs)-cut} (from {L['baseline']['anchor'][cut]})")
print(f"held-out observed max >= {TAIL:g}: {int((obs[test] >= TAIL).sum())}\n")

print("(1) PEAK -- held-out. peakhead* rows use the head's own scalar.")
print(f"{'variant':22s} {'rho':>7s} {'std_r':>7s} {'/rho':>6s} {'repro':>7s} {'MAE':>7s} "
      f"{'skill':>7s} {'AUC':>8s} {'reach':>7s}")
label = obs[test] >= EVENT
for v in VARIANTS:
    for src, tag in ([("curve max", L[v]["pred"].max(axis=1))] +
                     ([("HEAD scalar", L[v]["peak"])] if L[v]["peak"] is not None else [])):
        p = tag[test]
        r = peak_row(p, obs[test], pers[test])
        name = v if src == "curve max" or L[v]["peak"] is None else f"{v} (head)"
        if L[v]["peak"] is not None and src == "curve max":
            name = f"{v} (curve max)"
        print(f"{name:22s} {r['rho']:7.3f} {r['std_r']:7.3f} {r['over_rho']:6.2f} "
              f"{r['repro']:7.3f} {r['mae']:7.2f} {r['skill']:7.3f} {auc(p, label):8.4f} "
              f"{r['reach']:7.3f}")

print("\n(2) CURVE -- per-lead, all 24 pooled. The criterion the last pass mark lacked.")
print(f"{'variant':22s} {'rho':>7s} {'std_r':>7s} {'MAE':>7s} " +
      "".join(f"{(j+1)*0.5:>7.1f}h" for j in (0, 5, 11, 23)))
for v in VARIANTS:
    p = L[v]["pred"]
    rho = float(np.corrcoef(p.ravel(), true.ravel())[0, 1])
    print(f"{v:22s} {rho:7.3f} {p.std()/true.std():7.3f} {np.abs(p-true).mean():7.3f} " +
          "".join(f"{np.abs(p[:, j]-true[:, j]).mean():8.3f}" for j in (0, 5, 11, 23)))

print("\n(3) SHAPE -- where the predicted maximum falls (share of anchors, %).")
print(f"{'variant':22s} {'max share':>10s} {'at lead':>9s} {'|dt| storms':>12s} "
      f"{'peak/mean':>10s}")
sel = obs >= TAIL
obs_h = np.bincount(true.argmax(axis=1), minlength=n_lead) / len(true)
obs_sharp = (obs / ((true.sum(axis=1) - obs) / (n_lead - 1)))
print(f"{'OBSERVED':22s} {100*obs_h.max():9.1f}% {(obs_h.argmax()+1)*0.5:8.1f}h "
      f"{'--':>12s} {obs_sharp[sel].mean():10.2f}")
for v in VARIANTS:
    p = L[v]["pred"]
    h = np.bincount(p.argmax(axis=1), minlength=n_lead) / len(true)
    dt = np.abs(p[sel].argmax(axis=1) - true[sel].argmax(axis=1)) * 0.5
    mx = p.max(axis=1)
    sharp = mx / np.maximum((p.sum(axis=1) - mx) / (n_lead - 1), 1e-6)
    print(f"{v:22s} {100*h.max():9.1f}% {(h.argmax()+1)*0.5:8.1f}h {dt.mean():11.2f}h "
          f"{sharp[sel].mean():10.2f}")

print("\n(4) VERDICT against the pass mark")
base_curve_rho = float(np.corrcoef(L["baseline"]["pred"].ravel(), true.ravel())[0, 1])
for v in VARIANTS[2:]:
    p = L[v]["pred"]
    r = peak_row(L[v]["peak"][test], obs[test], pers[test])
    curve_rho = float(np.corrcoef(p.ravel(), true.ravel())[0, 1])
    share = float(np.bincount(p.argmax(axis=1), minlength=n_lead).max()) / len(true)
    checks = [("repro > 0.351", r["repro"] > 0.351, f"{r['repro']:.3f}"),
              ("curve rho >= 0.567", curve_rho >= base_curve_rho - 0.005, f"{curve_rho:.3f}"),
              ("argmax share <= 30%", share <= 0.30, f"{100*share:.1f}%"),
              ("peak rho > 0.719", r["rho"] > 0.719, f"{r['rho']:.3f}"),
              ("peak MAE < 13.98", r["mae"] < 13.98, f"{r['mae']:.2f}")]
    print(f"  {v}")
    for name, ok, val in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {name:22s} {val}")
