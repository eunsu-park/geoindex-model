"""Score the peak-augmented runs against the pass mark set before they ran.

The pass mark is on the 12-hour block maximum, not on the per-lead values: the peak
correlation must beat the baseline's, tail reproduction above 100 ap must beat it too, and
the dispersion ratio sigma_r / rho must not rise much above the baseline -- a rise there
would mean the gain came from shouting louder rather than from discrimination.

Per-lead rows are printed alongside because a loss that trades per-lead accuracy for peak
accuracy is supposed to lose a little there; the point is how much.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/GitHub/njit-geoindex/geoindex-model/analysis"))
from compare_loss_variants import load_variant  # noqa: E402

ROOT = os.path.expanduser("~/Projects/GeoIndex/results")
PREFIX = "probe_ap_in12h_out12h_gnn_transformer"
VARIANTS = ["baseline", "peak1", "peak05", "peak_soft", "peak_mae"]
TAIL = 100.0
EVENT = 50.0


def auc(score, label):
    """Rank-based area under the ROC curve."""
    order = np.argsort(score)
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks over ties so a constant score scores 0.5, not 1.0
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_pos, n_neg = label.sum(), (~label).sum()
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pod_at_far(score, label, budget):
    """Highest POD reachable while the false-alarm ratio stays within the budget."""
    order = np.argsort(-score)
    lab = label[order]
    hits = np.cumsum(lab)
    fa = np.cumsum(~lab)
    far = fa / np.maximum(hits + fa, 1)
    ok = far <= budget
    return float(hits[ok].max() / label.sum()) if ok.any() else 0.0


def row(pred, true, pers=None):
    """Dispersion / reproduction / skill for one (pred, true) pair."""
    rho = float(np.corrcoef(pred.ravel(), true.ravel())[0, 1])
    std_ratio = float(pred.std() / true.std())
    hi = true >= TAIL
    repro = float(pred[hi].mean() / true[hi].mean()) if hi.sum() > 30 else float("nan")
    mae = float(np.abs(pred - true).mean())
    out = {"rho": rho, "std_r": std_ratio, "over_rho": std_ratio / rho,
           "repro": repro, "mae": mae, "n_hi": int(hi.sum())}
    if pers is not None:
        out["skill"] = 1.0 - mae / float(np.abs(pers - true).mean())
    return out


loaded = {}
for v in VARIANTS:
    p = os.path.join(ROOT, f"{PREFIX}_{v}", "validation", "best", "npz.zip")
    loaded[v] = load_variant(p)
    print(f"loaded {v}: {loaded[v]['true'].shape}", flush=True)

true = loaded["baseline"]["true"]
pers = loaded["baseline"]["pers"]
obs_max = true.max(axis=1)
pers_max = pers  # persistence over the window is the anchor repeated, so its max is the anchor

for v in VARIANTS:
    assert np.array_equal(loaded[v]["anchor"], loaded["baseline"]["anchor"]), f"{v} anchors differ"

print(f"\nanchors {len(obs_max)}, leads {true.shape[1]}, "
      f"observed max >= {TAIL:g}: {int((obs_max >= TAIL).sum())}, "
      f">= {EVENT:g}: {int((obs_max >= EVENT).sum())}\n")

print("PEAK LEVEL -- the 12 h block maximum (this is what the pass mark is on)")
print(f"{'variant':10s} {'rho':>7s} {'std_r':>7s} {'/rho':>6s} {'repro>=100':>11s} "
      f"{'MAE':>7s} {'skill':>7s} {'AUC>=50':>8s} {'POD@30%':>8s} {'POD@50%':>8s}")
label = obs_max >= EVENT
base_auc = None
for v in VARIANTS:
    pm = loaded[v]["pred"].max(axis=1)
    r = row(pm, obs_max, pers_max)
    a = auc(pm, label)
    base_auc = base_auc if base_auc is not None else a
    print(f"{v:10s} {r['rho']:7.3f} {r['std_r']:7.3f} {r['over_rho']:6.2f} "
          f"{r['repro']:11.3f} {r['mae']:7.2f} {r['skill']:7.3f} "
          f"{a:8.4f} {pod_at_far(pm, label, 0.30):8.3f} {pod_at_far(pm, label, 0.50):8.3f}")

print("\nPER LEAD -- all 24 values pooled (a peak term is allowed to cost a little here)")
print(f"{'variant':10s} {'rho':>7s} {'std_r':>7s} {'/rho':>6s} {'repro>=100':>11s} "
      f"{'MAE':>7s} {'skill':>7s}")
pers_grid = np.repeat(pers[:, None], true.shape[1], axis=1)
for v in VARIANTS:
    r = row(loaded[v]["pred"], true, pers_grid)
    print(f"{v:10s} {r['rho']:7.3f} {r['std_r']:7.3f} {r['over_rho']:6.2f} "
          f"{r['repro']:11.3f} {r['mae']:7.3f} {r['skill']:7.3f}")

print("\nPER LEAD MAE by lead (h)")
leads = [0, 1, 3, 5, 11, 17, true.shape[1] - 1]
print(f"{'variant':10s} " + "".join(f"{(j+1)*0.5:>8.1f}h" for j in leads))
for v in VARIANTS:
    p = loaded[v]["pred"]
    print(f"{v:10s} " + "".join(f"{np.abs(p[:, j]-true[:, j]).mean():9.3f}" for j in leads))

print("\nWHERE THE PEAK LANDS -- storms only (observed max >= 100)")
sel = obs_max >= TAIL
print(f"{'variant':10s} {'mean pred max':>14s} {'mean obs max':>13s} "
      f"{'timing |dt| h':>14s} {'reached >=100':>14s}")
for v in VARIANTS:
    p = loaded[v]["pred"][sel]
    t = true[sel]
    dt = np.abs(p.argmax(axis=1) - t.argmax(axis=1)) * 0.5
    print(f"{v:10s} {p.max(axis=1).mean():14.2f} {t.max(axis=1).mean():13.2f} "
          f"{dt.mean():14.2f} {(p.max(axis=1) >= TAIL).mean():14.3f}")
