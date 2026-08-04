"""Is the peak term's gain new information, or a rescaling a post-hoc fit buys for free?

The peak runs raise tail reproduction from 0.354 to ~0.49 and MSE skill from 0.126 to 0.373
while AUC on the same event moves by +0.0004. That pattern is the signature of a monotone
rescaling: the ranking of anchors is unchanged, only the numbers attached to them grew.

Variance inflation and distribution matching have been rejected in print repeatedly because
they buy exactly this and cost accuracy. So the control has to be run: fit an isotonic map
from the BASELINE's peak prediction to the observed peak on the earlier 60 % of anchors,
apply it to the rest, and see whether the baseline lands where the peak runs land. If it
does, the peak loss adds nothing a calibration curve does not. If the peak runs stay ahead
on the held-out part, the gain is in the model.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/GitHub/njit-geoindex/geoindex-model/analysis"))
from compare_loss_variants import load_variant  # noqa: E402

ROOT = os.path.expanduser("~/Projects/GeoIndex/results")
PREFIX = "probe_ap_in12h_out12h_gnn_transformer"
VARIANTS = ["baseline", "peak1", "peak05", "peak_mae"]
TAIL, EVENT, SPLIT = 100.0, 50.0, 0.60


def pava(x, y):
    """Isotonic regression of y on x (pool-adjacent-violators); returns (knots_x, knots_y)."""
    order = np.argsort(x)
    xs, ys = x[order], y[order].astype(float)
    w = np.ones_like(ys)
    vals, wts, idx = [], [], []
    for i in range(len(ys)):
        vals.append(ys[i]); wts.append(w[i]); idx.append(i)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2 = vals.pop(), wts.pop(); i2 = idx.pop()
            v1, w1 = vals.pop(), wts.pop(); idx.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2)); wts.append(w1 + w2); idx.append(i2)
    fit = np.empty(len(ys))
    start = 0
    for v, ww in zip(vals, wts):
        n = int(round(ww))
        fit[start:start + n] = v
        start += n
    return xs, fit


def apply_iso(knots_x, knots_y, x):
    return np.interp(x, knots_x, knots_y)


def score(pred, true, pers):
    rho = float(np.corrcoef(pred, true)[0, 1])
    hi = true >= TAIL
    mae = float(np.abs(pred - true).mean())
    return {"rho": rho, "std_r": float(pred.std() / true.std()),
            "repro": float(pred[hi].mean() / true[hi].mean()),
            "mae": mae, "skill": 1.0 - mae / float(np.abs(pers - true).mean()),
            "reach": float((pred[hi] >= TAIL).mean())}


loaded = {v: load_variant(os.path.join(ROOT, f"{PREFIX}_{v}", "validation", "best", "npz.zip"))
          for v in VARIANTS}
true = loaded["baseline"]["true"]
obs = true.max(axis=1)
pers = loaded["baseline"]["pers"]

n = len(obs)
cut = int(n * SPLIT)
fit_s, test_s = slice(0, cut), slice(cut, n)
print(f"anchors {n}: fit on the first {cut} (chronological), score on the last {n-cut}")
print(f"held-out observed max >= {TAIL:g}: {int((obs[test_s] >= TAIL).sum())}\n")

# Spearman between the baseline's peak score and each peak run's -- how much reordering
# actually happened.
def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


base_peak = loaded["baseline"]["pred"].max(axis=1)
print("Reordering against the baseline (Spearman on the peak score). 1.000 would mean the")
print("peak run is a pure monotone rescaling of the baseline and carries no new ordering.")
for v in VARIANTS[1:]:
    print(f"  {v:10s} {spearman(base_peak, loaded[v]['pred'].max(axis=1)):.4f}")

print(f"\n{'variant':22s} {'rho':>7s} {'std_r':>7s} {'repro':>7s} {'MAE':>7s} "
      f"{'skill':>7s} {'reach>=100':>11s}   (held-out only)")
rows = []
for v in VARIANTS:
    p = loaded[v]["pred"].max(axis=1)
    rows.append((v, score(p[test_s], obs[test_s], pers[test_s])))

# the control: baseline peak, isotonically recalibrated on the fit part
kx, ky = pava(base_peak[fit_s], obs[fit_s])
recal = apply_iso(kx, ky, base_peak[test_s])
rows.insert(1, ("baseline + isotonic", score(recal, obs[test_s], pers[test_s])))

# and the cheaper control: a single linear rescale, also fitted on the fit part
slope = float(np.polyfit(base_peak[fit_s], obs[fit_s], 1)[0])
inter = float(np.polyfit(base_peak[fit_s], obs[fit_s], 1)[1])
lin = slope * base_peak[test_s] + inter
rows.insert(2, (f"baseline x{slope:.2f}{inter:+.1f}", score(lin, obs[test_s], pers[test_s])))

# and each peak run with the same linear rescale on top -- is the peak loss already calibrated,
# or does it still leave amplitude on the table?
for v in VARIANTS[1:]:
    pk = loaded[v]["pred"].max(axis=1)
    sl, ic = np.polyfit(pk[fit_s], obs[fit_s], 1)
    rows.append((f"{v} x{sl:.2f}{ic:+.1f}", score(sl * pk[test_s] + ic, obs[test_s], pers[test_s])))

for name, r in rows:
    print(f"{name:22s} {r['rho']:7.3f} {r['std_r']:7.3f} {r['repro']:7.3f} {r['mae']:7.2f} "
          f"{r['skill']:7.3f} {r['reach']:11.3f}")

print("\nIf 'baseline + isotonic' matches the peak runs, the peak loss bought a calibration")
print("curve. If the peak runs stay ahead, it bought something the curve cannot.")
