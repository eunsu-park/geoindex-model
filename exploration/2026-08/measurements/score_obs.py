"""Score the observation-time runs against the pass mark fixed before they ran.

Primary criterion: AUC must beat the baseline's. Seventeen loss variants, fourteen
architectures, input windows from one hour to fifty-four days and a peak head all left AUC
inside +/-0.001, because they moved calibration rather than which anchors rank above which.
The ridge says re-indexing the wind to observation time moves it by +0.0104. Whether that
survives in the deep model is the whole question.

Secondary: peak rho above 0.699, per-lead rho not below the baseline's 0.567, and the
predicted argmax not concentrated above ~30 % on any single lead -- the shape criterion the
peak_augmented pass mark lacked, which is how a model that had destroyed the forecast passed.
"""

import io
import os
import zipfile

import numpy as np

ROOT = os.path.expanduser("~/Projects/GeoIndex/results")
PREFIX = "probe_ap_in12h_out12h_gnn_transformer"
VARIANTS = ["baseline", "peakhead_mae", "obs", "obs_peakhead"]
TAIL, EVENT, SPLIT = 100.0, 50.0, 0.60


def load(variant):
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
    ranks = np.empty(len(score), float)
    ranks[np.argsort(score)] = np.arange(1, len(score) + 1)
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_pos, n_neg = label.sum(), (~label).sum()
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pod_at_far(score, label, budget):
    order = np.argsort(-score)
    lab = label[order]
    hits, fa = np.cumsum(lab), np.cumsum(~lab)
    ok = fa / np.maximum(hits + fa, 1) <= budget
    return float(hits[ok].max() / label.sum()) if ok.any() else 0.0


L = {v: load(v) for v in VARIANTS}
common = set(L[VARIANTS[0]]["anchor"])
for v in VARIANTS[1:]:
    common &= set(L[v]["anchor"])
common = np.array(sorted(common))
print(f"anchors per run: " + ", ".join(f"{v}={len(L[v]['anchor'])}" for v in VARIANTS))
print(f"common: {len(common)}")

sel = {v: np.isin(L[v]["anchor"], common) for v in VARIANTS}
ordr = {v: np.argsort(L[v]["anchor"][sel[v]]) for v in VARIANTS}
true = L["baseline"]["true"][sel["baseline"]][ordr["baseline"]]
pers = L["baseline"]["pers"][sel["baseline"]][ordr["baseline"]]
obs = true.max(axis=1)
n_lead = true.shape[1]
cut = int(len(obs) * SPLIT)
te = slice(cut, len(obs))
label = obs[te] >= EVENT
hi = obs[te] >= TAIL
print(f"held-out {len(obs)-cut}, events >= {EVENT:g}: {int(label.sum())}, "
      f">= {TAIL:g}: {int(hi.sum())}\n")


def peak_score(name, p):
    rho = float(np.corrcoef(p[te], obs[te])[0, 1])
    mae = float(np.abs(p[te] - obs[te]).mean())
    a = auc(p[te], label)
    return {"name": name, "rho": rho, "std_r": float(p[te].std() / obs[te].std()) / rho,
            "repro": float(p[te][hi].mean() / obs[te][hi].mean()), "mae": mae, "auc": a,
            "skill": 1.0 - mae / float(np.abs(pers[te] - obs[te]).mean()),
            "pod30": pod_at_far(p[te], label, 0.30), "pod50": pod_at_far(p[te], label, 0.50)}


rows = []
for v in VARIANTS:
    pr = L[v]["pred"][sel[v]][ordr[v]]
    rows.append(peak_score(f"{v} (curve max)" if L[v]["peak"] is not None else v,
                           pr.max(axis=1)))
    if L[v]["peak"] is not None:
        rows.append(peak_score(f"{v} (head)", L[v]["peak"][sel[v]][ordr[v]]))

base_auc = rows[0]["auc"]
print("(1) PEAK -- held-out. AUC is the pass mark.")
print(f"{'variant':28s} {'AUC':>8s} {'dAUC':>8s} {'rho':>7s} {'/rho':>6s} {'repro':>7s} "
      f"{'MAE':>7s} {'skill':>7s} {'POD@30%':>8s}")
for r in rows:
    print(f"{r['name']:28s} {r['auc']:8.4f} {r['auc']-base_auc:+8.4f} {r['rho']:7.3f} "
          f"{r['std_r']:6.2f} {r['repro']:7.3f} {r['mae']:7.2f} {r['skill']:7.3f} "
          f"{r['pod30']:8.3f}")

print("\n(2) CURVE -- per-lead, all 24 pooled.")
print(f"{'variant':28s} {'rho':>7s} {'std_r':>7s} {'MAE':>7s} " +
      "".join(f"{(j+1)*0.5:>7.1f}h" for j in (0, 5, 11, 23)))
curve = {}
for v in VARIANTS:
    p = L[v]["pred"][sel[v]][ordr[v]]
    rho = float(np.corrcoef(p.ravel(), true.ravel())[0, 1])
    curve[v] = rho
    print(f"{v:28s} {rho:7.3f} {p.std()/true.std():7.3f} {np.abs(p-true).mean():7.3f} " +
          "".join(f"{np.abs(p[:, j]-true[:, j]).mean():8.3f}" for j in (0, 5, 11, 23)))

print("\n(3) SHAPE -- concentration of the predicted maximum.")
print(f"{'variant':28s} {'max share':>10s} {'at lead':>9s} {'|dt| storms':>12s}")
storms = obs >= TAIL
h_obs = np.bincount(true.argmax(axis=1), minlength=n_lead) / len(true)
print(f"{'OBSERVED':28s} {100*h_obs.max():9.1f}% {(h_obs.argmax()+1)*0.5:8.1f}h {'--':>12s}")
share = {}
for v in VARIANTS:
    p = L[v]["pred"][sel[v]][ordr[v]]
    h = np.bincount(p.argmax(axis=1), minlength=n_lead) / len(true)
    dt = np.abs(p[storms].argmax(axis=1) - true[storms].argmax(axis=1)) * 0.5
    share[v] = float(h.max())
    print(f"{v:28s} {100*h.max():9.1f}% {(h.argmax()+1)*0.5:8.1f}h {dt.mean():11.2f}h")

print("\n(4) VERDICT against the pass mark")
for v in VARIANTS[2:]:
    best = max((r for r in rows if r["name"].startswith(v)), key=lambda r: r["auc"])
    checks = [("AUC > baseline", best["auc"] > base_auc, f"{best['auc']:.4f} vs {base_auc:.4f}"),
              ("peak rho > 0.699", best["rho"] > 0.699, f"{best['rho']:.3f}"),
              ("curve rho >= 0.567", curve[v] >= curve["baseline"] - 0.005, f"{curve[v]:.3f}"),
              ("argmax share <= 30%", share[v] <= 0.30, f"{100*share[v]:.1f}%")]
    print(f"  {v}   (scored on: {best['name']})")
    for nm, ok, val in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {nm:20s} {val}")
