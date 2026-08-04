"""Can trajectories fix the curve's shape? A pre-test that needs no new model.

The curve is flat because it is a conditional mean, and a conditional mean is required to be
flat: sigma_pred/sigma_obs = rho. So the shape cannot be fixed inside that framing. The
proposal is to emit sampled trajectories instead -- each one spiky, their mean still the flat
curve -- and the question is whether the shape statistics can be made right at all.

That is testable without training anything. Dress the existing point forecast in a residual
process estimated from the data: if trajectories built that way reproduce the observed
spikiness and a calibrated maximum, a properly trained generative model can do at least as
well. If they cannot, the premise is wrong and autoregressive rollout would fail for the same
reason.

Nothing here adds information. The ensemble mean is the same damped curve; only the
distribution around it is new.

The residual process is fitted on the earlier 60 % of anchors and applied to the rest.
Everything is done in log1p space, where the model was trained, and snapped back onto the
36-level ap ladder -- ap30 is not continuous, and the ladder is part of the shape.
"""

import io
import json
import os
import zipfile

import numpy as np

ROOT = os.path.expanduser("~/Projects/GeoIndex/results")
RUN = "probe_ap_in12h_out12h_gnn_transformer_obs"
SPLIT = 0.60
N_TRAJ = 200
SEED = 11
LADDER = np.array([0, 2, 3, 4, 5, 6, 7, 9, 12, 15, 18, 22, 27, 32, 39, 48, 56, 67, 80, 94,
                   111, 132, 154, 179, 207, 236, 265, 294, 324, 355, 388, 421, 456, 494,
                   534, 617], float)


def load(run):
    path = os.path.join(ROOT, run, "validation", "best", "npz.zip")
    true, pred = [], []
    with zipfile.ZipFile(path) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".npz")):
            d = np.load(io.BytesIO(z.read(name)), allow_pickle=True)
            true.append(np.asarray(d["targets"])[:, 0])
            pred.append(np.asarray(d["predictions"])[:, 0])
    return np.asarray(true), np.asarray(pred)


def snap(x):
    """Nearest level on the ap ladder."""
    idx = np.abs(x[..., None] - LADDER).argmin(axis=-1)
    return LADDER[idx]


def sharpness(curves):
    """Peak divided by the mean of the other steps -- 1.0 is a plateau."""
    mx = curves.max(axis=-1)
    rest = (curves.sum(axis=-1) - mx) / (curves.shape[-1] - 1)
    return mx / np.maximum(rest, 1e-6)


true, pred = load(RUN)
n, n_lead = true.shape
cut = int(n * SPLIT)
fit, test = slice(0, cut), slice(cut, n)
print(f"anchors {n}, leads {n_lead}; residual process fitted on {cut}, applied to {n - cut}")

lt, lp = np.log1p(true), np.log1p(np.maximum(pred, 0))
resid = lt - lp

# ── the residual process ─────────────────────────────────────────────────────
# Scale depends on how disturbed the forecast says it is (heteroscedasticity is real and is
# almost entirely explained by the predicted mean), so bin by the anchor's predicted maximum.
level = lp.max(axis=1)
edges = np.quantile(level[fit], np.linspace(0, 1, 11))
edges[-1] += 1e-6
bin_of = np.clip(np.digitize(level, edges[1:-1]), 0, 9)

scale = np.zeros((10, n_lead))
for b in range(10):
    m = (bin_of[fit] == b)
    scale[b] = resid[fit][m].std(axis=0)
print(f"residual sd in log1p space: quietest bin {scale[0].mean():.3f}, "
      f"most disturbed {scale[9].mean():.3f}")

# Correlation across leads, on standardized residuals so the scale model is not double counted.
z = resid[fit] / scale[bin_of[fit]]
corr = np.corrcoef(z, rowvar=False)
corr = (corr + corr.T) / 2
print(f"lead-to-lead residual correlation: adjacent {np.mean(np.diag(corr, 1)):.3f}, "
      f"6 h apart {np.mean(np.diag(corr, 12)):.3f}, 12 h apart {corr[0, -1]:.3f}")

# A correlated draw needs a factorization; nudge to positive definite if needed.
w, V = np.linalg.eigh(corr)
w = np.maximum(w, 1e-6)
L = V @ np.diag(np.sqrt(w))

# ── generate trajectories on the held-out part ───────────────────────────────
rng = np.random.default_rng(SEED)
te_pred, te_true = pred[test], true[test]
te_bin = bin_of[test]
m = len(te_true)
print(f"\ndrawing {N_TRAJ} trajectories for each of {m} held-out anchors ...")
traj = np.empty((m, N_TRAJ, n_lead), np.float32)
for k in range(N_TRAJ):
    e = rng.standard_normal((m, n_lead)) @ L.T
    traj[:, k, :] = snap(np.expm1(np.log1p(np.maximum(te_pred, 0)) + e * scale[te_bin]))

obs_max = te_true.max(axis=1)
tr_max = traj.max(axis=2)
point_max = te_pred.max(axis=1)
ens_mean = traj.mean(axis=1)

# ── does the shape come out right? ───────────────────────────────────────────
print("\n(1) SHAPE — peak divided by the mean of the rest of the window")
storm = obs_max >= 100
rows = [("observed", sharpness(te_true)),
        ("point forecast", sharpness(te_pred)),
        ("ensemble mean", sharpness(ens_mean)),
        ("single trajectory", sharpness(traj[:, 0, :]))]
print(f"{'':22s} {'all':>8s} {'storms':>8s}")
for lab, v in rows:
    print(f"{lab:22s} {v.mean():8.2f} {v[storm].mean():8.2f}")

print("\n(2) THE MAXIMUM — distribution of the 12 h peak")
qs = [50, 75, 90, 95, 99, 99.9]
print(f"{'':22s} " + "".join(f"{'p'+str(q):>9s}" for q in qs) + f"{'mean':>9s}")
for lab, v in [("observed", obs_max), ("point forecast", point_max),
               ("trajectories (pooled)", tr_max.ravel())]:
    print(f"{lab:22s} " + "".join(f"{np.percentile(v, q):9.0f}" for q in qs) +
          f"{v.mean():9.1f}")

print("\n(3) CALIBRATION — is the observed peak a plausible draw from the ensemble?")
pit = (tr_max < obs_max[:, None]).mean(axis=1)
print(f"PIT of the observed maximum: mean {pit.mean():.3f} (0.500 ideal), "
      f"below 0.05 {100*(pit < .05).mean():.1f} %, above 0.95 {100*(pit > .95).mean():.1f} %")
print("   (a calibrated ensemble puts ~5 % in each tail; more above 0.95 means the")
print("    ensemble is still too low)")
for lo, hi, lab in [(0, 30, "quiet"), (30, 100, "active"), (100, 1e9, "storm")]:
    m2 = (obs_max >= lo) & (obs_max < hi)
    if m2.sum() < 20:
        continue
    print(f"   {lab:8s} n={int(m2.sum()):6d}  PIT mean {pit[m2].mean():.3f}  "
          f"above 0.95 {100*(pit[m2] > .95).mean():5.1f} %")

print("\n(4) THE QUESTION THAT MATTERS — P(peak >= threshold), forecast vs observed")
print(f"{'threshold':>10s} {'observed rate':>14s} {'mean forecast p':>16s} {'reliability':>12s}")
for t in (50, 100, 150):
    p_fc = (tr_max >= t).mean(axis=1)
    obs_rate = float((obs_max >= t).mean())
    print(f"{t:10.0f} {obs_rate:13.4f} {p_fc.mean():16.4f} "
          f"{('over' if p_fc.mean() > obs_rate else 'under'):>12s}")

out = {"sharpness": {lab: [round(float(v.mean()), 3), round(float(v[storm].mean()), 3)]
                     for lab, v in rows},
       "maxdist": {lab: [round(float(np.percentile(v, q)), 1) for q in qs]
                   for lab, v in [("observed", obs_max), ("point", point_max),
                                  ("traj", tr_max.ravel())]},
       "pit_hist": np.histogram(pit, bins=20, range=(0, 1))[0].tolist(),
       "n_test": int(m), "n_traj": N_TRAJ}
json.dump(out, open("figdata_traj.json", "w"), allow_nan=False)
print("\nwrote figdata_traj.json")
