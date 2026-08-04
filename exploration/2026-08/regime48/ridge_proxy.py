"""Ridge pre-test for "regime split + peak head" -- with the proxy itself validated first.

The previous ridge pre-test predicted the storm branch would reproduce 0.649 of the observed
peak; the trained model returned 0.360. The reason was a design error in the proxy, not in the
idea: the ridge fitted the window MAXIMUM directly as a scalar, while the deep model fits 24
per-lead values and the maximum is taken afterwards. Those are different targets, and the
difference is exactly the double shrinkage of section 2.7 -- the maximum of 24 separately shrunk
conditional means is not the shrunk maximum.

So this script fits BOTH structures, per regime:

    curve   24 outputs, maximum taken afterwards   <- what the deep model does today
    peak    1 output, the window maximum           <- what PeakHeadLoss does

and it checks step 1 before reporting step 2:

    STEP 1  does the curve ridge reproduce the deep model's measured numbers?
            (deep, on storm rows: curve level 109 % of observed, peak 52 %; pooled 93 % / 43 %)
    STEP 2  only if step 1 holds, what does the separate peak output add?

Everything matches the deep runs: `data.parquet` (bow-shock time, which is what server_ap uses),
the same `total_ap` train index split at ap30 >= 48, and the same 23,514 validation anchors read
from the stored archive.

Usage:
    python ridge_proxy.py
"""

from __future__ import annotations

import io
import os
import zipfile

import numpy as np
import pandas as pd

D = os.path.expanduser("~/Projects/GeoIndex/datasets")
R = os.path.expanduser("~/Projects/GeoIndex/results")
DEEP = {"pooled": "probe_ap_in12h_out12h_gnn_transformer_baseline",
        "quiet": "regime48_ap_in12h_out12h_gnn_transformer_quiet",
        "storm": "regime48_ap_in12h_out12h_gnn_transformer_storm"}
WIND = ["v_avg", "np_avg", "t_avg", "bx_avg", "by_avg", "bz_avg", "bt_avg"]
PAST = OUT = 24
THR = 48.0


def ridge(X, Y, alpha=100.0):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    Y2 = Y if Y.ndim == 2 else Y[:, None]
    yb = Y2.mean(0)
    W = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (Y2 - yb))
    return lambda Xn: ((Xn - mu) / sd) @ W + yb


def load_deep(run):
    path = os.path.join(R, run, "validation", "best", "npz.zip")
    a, t, p = [], [], []
    with zipfile.ZipFile(path) as z:
        for n in sorted(x for x in z.namelist() if x.endswith(".npz")):
            d = np.load(io.BytesIO(z.read(n)), allow_pickle=True)
            a.append(str(np.asarray(d["anchor"])))
            t.append(np.asarray(d["targets"])[:, 0])
            p.append(np.asarray(d["predictions"])[:, 0])
    return np.asarray(a), np.asarray(t), np.asarray(p)


# ── data, matched to the deep runs ───────────────────────────────────────────
tbl = pd.read_parquet(os.path.join(D, "data.parquet")).set_index("datetime").sort_index()
grid = pd.date_range(tbl.index[0], tbl.index[-1], freq="30min")
tbl = tbl.reindex(grid)
cols = WIND + ["ap30"]
tbl[cols] = tbl[cols].interpolate(limit=6).ffill().bfill()
ap = tbl["ap30"].to_numpy(float)
pos = {t: i for i, t in enumerate(grid)}


def features_and_targets(stamps):
    """Past window and the 24-step future for a list of anchor timestamps."""
    idx = np.array([pos[t] for t in stamps])
    X = np.column_stack(
        [tbl[w].to_numpy(float)[idx[:, None] + np.arange(-PAST, 0)] for w in WIND]
        + [ap[idx[:, None] + np.arange(-PAST, 0)]])
    Y = ap[idx[:, None] + np.arange(0, OUT)]
    return X, Y


train_idx = pd.read_csv(os.path.join(D, "total_ap/train_index.csv"))
train_idx["datetime"] = pd.to_datetime(train_idx["datetime"])
tr_stamps = [t for t in train_idx["datetime"]
             if PAST <= pos.get(t, -1) < len(grid) - OUT]
Xtr, Ytr = features_and_targets(tr_stamps)
tr_peak = Ytr.max(1)
tr_storm = tr_peak >= THR

anchors, true, _ = load_deep(DEEP["pooled"])
te_stamps = pd.to_datetime(pd.Series(anchors), format="%Y%m%d%H%M%S")
Xte, Yte = features_and_targets(list(te_stamps))
assert np.allclose(Yte, true, atol=1e-3), "reconstructed targets differ from the archive"
te_peak = Yte.max(1)
te_storm = te_peak >= THR
print(f"train {len(tr_stamps):,} anchors (storm {int(tr_storm.sum()):,}); "
      f"test {len(te_stamps):,} (storm {int(te_storm.sum()):,})\n")

# ── fit both structures, per regime ─────────────────────────────────────────
fits = {}
for name, sel in (("pooled", np.ones(len(Xtr), bool)), ("quiet", ~tr_storm), ("storm", tr_storm)):
    fits[(name, "curve")] = ridge(Xtr[sel], Ytr[sel])
    fits[(name, "peak")] = ridge(Xtr[sel], tr_peak[sel])

curve = {k: fits[(k, "curve")](Xte) for k in ("pooled", "quiet", "storm")}
peak = {k: fits[(k, "peak")](Xte).ravel() for k in ("pooled", "quiet", "storm")}

# ── STEP 1: is the proxy faithful? ──────────────────────────────────────────
m = te_storm
obs_level, obs_peak = Yte[m].mean(), te_peak[m].mean()
print("STEP 1 — does the curve ridge reproduce what the trained model actually did?")
print(f"  {'':22s} {'curve level':>12s} {'peak':>10s}    (share of observed)")
print(f"  {'OBSERVED (storm rows)':22s} {obs_level:12.1f} {obs_peak:10.1f}")
deep_ref = {"pooled": (0.93, 0.43), "storm": (1.09, 0.52)}
ok = True
for k in ("pooled", "storm"):
    lv = curve[k][m].mean() / obs_level
    pk = curve[k][m].max(1).mean() / obs_peak
    dl, dp = deep_ref[k]
    good = abs(lv - dl) < 0.12 and abs(pk - dp) < 0.12
    ok &= good
    print(f"  {'ridge ' + k:22s} {100*lv:11.0f}% {100*pk:9.0f}%    "
          f"deep was {100*dl:.0f}% / {100*dp:.0f}%   {'match' if good else 'MISMATCH'}")
if not ok:
    print("\n  The proxy does not reproduce the trained model. Anything below is not a")
    print("  prediction about the deep model -- read it as a statement about ridges only.")
else:
    print("\n  The proxy tracks the trained model, so step 2 is a usable prediction.")

# ── STEP 2: what does a separate peak output add? ───────────────────────────
print("\nSTEP 2 — separate peak output vs taking the maximum of the curve")
print(f"  {'':34s} {'peak':>8s} {'share':>7s} {'rho':>7s} {'MAE':>8s}")


def line(lab, p):
    rho = float(np.corrcoef(p, te_peak[m])[0, 1])
    print(f"  {lab:34s} {p.mean():8.1f} {100*p.mean()/obs_peak:6.0f}% {rho:7.3f} "
          f"{float(np.abs(p - te_peak[m]).mean()):8.2f}")


print(f"  {'OBSERVED':34s} {obs_peak:8.1f} {100:6.0f}%")
line("pooled, max of the curve", curve["pooled"][m].max(1))
line("storm branch, max of the curve", curve["storm"][m].max(1))
line("pooled, separate peak output", peak["pooled"][m])
line("storm branch, separate peak output", peak["storm"][m])

print("\n  quiet rows")
mq = ~te_storm
print(f"  {'OBSERVED':34s} {te_peak[mq].mean():8.1f}")
for lab, p in (("pooled, max of the curve", curve["pooled"][mq].max(1)),
               ("quiet branch, max of the curve", curve["quiet"][mq].max(1)),
               ("quiet branch, separate peak output", peak["quiet"][mq])):
    print(f"  {lab:34s} {p.mean():8.1f} "
          f"{100*p.mean()/te_peak[mq].mean():6.0f}% "
          f"{float(np.corrcoef(p, te_peak[mq])[0,1]):7.3f} "
          f"{float(np.abs(p - te_peak[mq]).mean()):8.2f}")

print("\n  sharpness of the predicted curve on storm rows (observed "
      f"{float((te_peak[m] / Yte[m].mean(1)).mean()):.2f})")
for k in ("pooled", "storm"):
    c = curve[k][m]
    print(f"    {k:12s} {float((c.max(1) / np.maximum(c.mean(1), 1e-6)).mean()):.2f}")
