"""Full-sample evaluation of the observation-time model against the baseline.

All 23,514 validation anchors (2022-01-03 .. 2025-12-26) are out-of-sample: both models train
on 1995-2021, so no held-out subsetting is needed here. Earlier comparisons used the last 40 %
only because a recalibration was being fitted on the earlier part; nothing is fitted here.

Produces the figure data for the verification set Liemohn et al. (2018) asks for -- regression
slope per lead, error by lead, event scores over a threshold sweep, ROC, activity-binned error,
peak reliability -- plus the dispersion diagnostics the damping analysis needs.
"""

import io
import json
import os
import zipfile

import numpy as np

ROOT = os.path.expanduser("~/Projects/GeoIndex/results")
PREFIX = "probe_ap_in12h_out12h_gnn_transformer"
RUNS = {"baseline": "baseline", "obs": "obs"}
THRESHOLDS = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200]
ACTIVITY = [(0, 15, "very quiet"), (15, 30, "quiet"), (30, 50, "unsettled"),
            (50, 100, "active"), (100, 1e9, "storm")]


def load(variant):
    path = os.path.join(ROOT, f"{PREFIX}_{variant}", "validation", "best", "npz.zip")
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


def contingency(pred_yes, obs_yes):
    hit = int((pred_yes & obs_yes).sum())
    fa = int((pred_yes & ~obs_yes).sum())
    miss = int((~pred_yes & obs_yes).sum())
    cn = int((~pred_yes & ~obs_yes).sum())
    den = (hit + miss) * (miss + cn) + (hit + fa) * (fa + cn)
    pod = hit / max(hit + miss, 1)
    pofd = fa / max(fa + cn, 1)
    return {"hits": hit, "fa": fa, "miss": miss, "cn": cn, "pod": pod, "pofd": pofd,
            "far": fa / max(hit + fa, 1), "csi": hit / max(hit + fa + miss, 1),
            "fb": (hit + fa) / max(hit + miss, 1),
            "hss": 2 * (hit * cn - fa * miss) / den if den else 0.0,
            "usable": hit >= 10 and cn >= 10}


def auc(score, label):
    ranks = np.empty(len(score), float)
    ranks[np.argsort(score)] = np.arange(1, len(score) + 1)
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    p, n = label.sum(), (~label).sum()
    return float((ranks[label].sum() - p * (p + 1) / 2) / (p * n))


L = {k: load(v) for k, v in RUNS.items()}
anchors, true, _, pers = L["baseline"]
for k, (a, t, _, _) in L.items():
    assert np.array_equal(a, anchors) and np.allclose(t, true, atol=1e-3), k
n_lead = true.shape[1]
obs_max = true.max(axis=1)
out = {"n_anchors": int(len(anchors)), "n_leads": int(n_lead),
       "period": [anchors[0], anchors[-1]],
       "leads": [round((j + 1) * 0.5, 1) for j in range(n_lead)]}
print(f"anchors {len(anchors)}  {anchors[0]} .. {anchors[-1]}  pairs {true.size}")

# ── per lead: rho, MAE, slope, dispersion ─────────────────────────────────────
out["per_lead"] = {}
for k, (_, _, p, _) in L.items():
    rho, mae, slope, disp = [], [], [], []
    for j in range(n_lead):
        rho.append(float(np.corrcoef(p[:, j], true[:, j])[0, 1]))
        mae.append(float(np.abs(p[:, j] - true[:, j]).mean()))
        b, _a = np.polyfit(true[:, j], p[:, j], 1)
        slope.append(float(b))
        disp.append(float(p[:, j].std() / true[:, j].std()))
    out["per_lead"][k] = {"rho": [round(v, 4) for v in rho],
                          "mae": [round(v, 3) for v in mae],
                          "slope": [round(v, 4) for v in slope],
                          "disp": [round(v, 4) for v in disp]}
pj = np.repeat(pers[:, None], n_lead, axis=1)
out["per_lead"]["persistence"] = {
    "mae": [round(float(np.abs(pj[:, j] - true[:, j]).mean()), 3) for j in range(n_lead)],
    "rho": [round(float(np.corrcoef(pj[:, j], true[:, j])[0, 1]), 4) for j in range(n_lead)]}

# ── event scores over the threshold sweep, on the window maximum ──────────────
out["events"] = {}
for k, (_, _, p, _) in L.items():
    rows = []
    score = p.max(axis=1)
    for t in THRESHOLDS:
        c = contingency(score >= t, obs_max >= t)
        rows.append({"thr": t, "base_rate": round(float((obs_max >= t).mean()), 5),
                     "pod": round(c["pod"], 4), "far": round(c["far"], 4),
                     "csi": round(c["csi"], 4), "hss": round(c["hss"], 4),
                     "fb": round(c["fb"], 4), "usable": bool(c["usable"]),
                     "hits": c["hits"], "miss": c["miss"]})
    out["events"][k] = rows

# ── ROC on the window maximum, event = ap >= 50 ──────────────────────────────
label = obs_max >= 50
out["roc"] = {"event": "max ap30 >= 50 within 12 h", "n_events": int(label.sum())}
for k, (_, _, p, _) in L.items():
    score = p.max(axis=1)
    qs = np.linspace(0, 1, 101)
    pts = []
    for q in qs:
        t = float(np.quantile(score, q))
        c = contingency(score >= t, label)
        pts.append([round(c["pofd"], 5), round(c["pod"], 5)])
    out["roc"][k] = {"points": pts, "auc": round(auc(score, label), 4)}

# ── activity-binned error ────────────────────────────────────────────────────
flat_t = true.ravel()
out["activity"] = {"bins": [l for _, _, l in ACTIVITY]}
for k, (_, _, p, _) in L.items():
    rows = []
    for lo, hi, lab in ACTIVITY:
        m = (flat_t >= lo) & (flat_t < hi)
        e = p.ravel()[m] - flat_t[m]
        rows.append({"label": lab, "n": int(m.sum()), "share": round(float(m.mean()), 5),
                     "mae": round(float(np.abs(e).mean()), 3),
                     "bias": round(float(e.mean()), 3)})
    out["activity"][k] = rows
out["activity"]["persistence"] = [
    {"label": lab, "mae": round(float(np.abs(pj.ravel()[(flat_t >= lo) & (flat_t < hi)]
                                             - flat_t[(flat_t >= lo) & (flat_t < hi)]).mean()), 3)}
    for lo, hi, lab in ACTIVITY]

# ── peak reliability, by decile of the prediction ────────────────────────────
out["reliability"] = {}
for k, (_, _, p, _) in L.items():
    s = p.max(axis=1)
    q = np.quantile(s, np.linspace(0, 1, 11))
    q[-1] += 1e-6
    b = np.clip(np.digitize(s, q[1:-1]), 0, 9)
    out["reliability"][k] = [[round(float(s[b == i].mean()), 2),
                              round(float(obs_max[b == i].mean()), 2), int((b == i).sum())]
                             for i in range(10)]

# ── headline numbers ─────────────────────────────────────────────────────────
out["headline"] = {}
for k, (_, _, p, _) in L.items():
    s = p.max(axis=1)
    hi = obs_max >= 100
    out["headline"][k] = {
        "lead_rho": round(float(np.corrcoef(p.ravel(), true.ravel())[0, 1]), 4),
        "lead_mae": round(float(np.abs(p - true).mean()), 3),
        "peak_rho": round(float(np.corrcoef(s, obs_max)[0, 1]), 4),
        "peak_mae": round(float(np.abs(s - obs_max).mean()), 3),
        "repro": round(float(s[hi].mean() / obs_max[hi].mean()), 4),
        "auc": out["roc"][k]["auc"],
        "disp": round(float(p.std() / true.std()), 4)}
out["headline"]["persistence"] = {
    "lead_mae": round(float(np.abs(pj - true).mean()), 3),
    "lead_rho": round(float(np.corrcoef(pj.ravel(), true.ravel())[0, 1]), 4)}

json.dump(out, open("figdata_eval.json", "w"), allow_nan=False)
print("\nheadline (all samples):")
for k, v in out["headline"].items():
    print(" ", k, v)
print("\nbytes:", len(open("figdata_eval.json").read()))
