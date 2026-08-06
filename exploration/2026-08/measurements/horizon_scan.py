"""Is the ceiling a property of the problem, or only of the 12-hour horizon?

Every bound in this investigation was measured on a 12-hour window, which is also where Bz is
least predictable (lead correlation 0.832 at 0.5 h, 0.533 at 2 h, 0.098 at 12 h). If the short
leads turn out to be far from their own ceiling, a 0-3 h specialist is the one modelling
experiment left worth building.

Method, after two false starts that are worth recording so nobody repeats them:

  * Refitting the ridge at a shorter OUTPUT WINDOW does not work. On the raw scale the fit is
    dominated by the heavy tail and loses to persistence by a mile at 1 h; refitting on log1p and
    inverting with expm1 is worse still -- adding TRUE future Bz then makes correlation go DOWN,
    which is impossible for a correct fit and betrays the bias of exp(E[log(1+y)]).
  * The fix is not to refit at all. A multi-output ridge fits every output column independently,
    so column k of the 12-hour fit IS the fit for lead k. Decomposing the validated 12-hour
    ablation by lead gives the per-lead headroom with no new estimator and no transform.

Persistence is the last observed value, `ap[T-1]`. Note the window convention: the target runs
[T, T+24) and the input [T-24, T), so `ap[T]` is the first TARGET, not the last observation.
Using it as persistence hands the baseline the answer at the first lead.

Usage:
    python horizon_scan.py
"""

from __future__ import annotations

import io
import os
import zipfile

import numpy as np
import pandas as pd

D = os.path.expanduser("~/Projects/GeoIndex/datasets")
R = os.path.expanduser("~/Projects/GeoIndex/results")
POOLED = "probe_ap_in12h_out12h_gnn_transformer_baseline"
WIND = ["v_avg", "np_avg", "t_avg", "bx_avg", "by_avg", "bz_avg", "bt_avg"]
PAST = OUT = 24


def ridge(X, Y, alpha=100.0):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    yb = Y.mean(0)
    W = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (Y - yb))
    return lambda Xn: ((Xn - mu) / sd) @ W + yb


tbl = pd.read_parquet(os.path.join(D, "data.parquet")).set_index("datetime").sort_index()
grid = pd.date_range(tbl.index[0], tbl.index[-1], freq="30min")
tbl = tbl.reindex(grid)
cols = WIND + ["ap30"]
tbl[cols] = tbl[cols].interpolate(limit=6).ffill().bfill()
ap = tbl["ap30"].to_numpy(float)
wind = {w: tbl[w].to_numpy(float) for w in WIND}
pos = {t: i for i, t in enumerate(grid)}

tr = pd.read_csv(os.path.join(D, "total_ap/train_index.csv"))
tr["datetime"] = pd.to_datetime(tr["datetime"])
tr_idx = np.array([pos[t] for t in tr["datetime"] if PAST <= pos.get(t, -1) < len(grid) - OUT])

deep_t, deep_p, anchors = [], [], []
with zipfile.ZipFile(os.path.join(R, POOLED, "validation", "best", "npz.zip")) as z:
    for n in sorted(x for x in z.namelist() if x.endswith(".npz")):
        d = np.load(io.BytesIO(z.read(n)), allow_pickle=True)
        anchors.append(str(np.asarray(d["anchor"])))
        deep_t.append(np.asarray(d["targets"])[:, 0])
        deep_p.append(np.asarray(d["predictions"])[:, 0])
deep_t, deep_p = np.asarray(deep_t), np.asarray(deep_p)
te_idx = np.array([pos[t] for t in pd.to_datetime(pd.Series(anchors), format="%Y%m%d%H%M%S")])


def build(idx):
    fw = idx[:, None] + np.arange(0, OUT)
    pw = idx[:, None] + np.arange(-PAST, 0)
    return dict(a=ap[pw],
                w=np.column_stack([wind[w][pw] for w in WIND]),
                env=np.column_stack([wind[w][fw] for w in WIND if w != "bz_avg"]),
                bz=wind["bz_avg"][fw],
                Y=ap[fw],
                pers=np.repeat(ap[idx - 1][:, None], OUT, axis=1))


TR, TE = build(tr_idx), build(te_idx)
Y = TE["Y"]
assert np.allclose(Y, deep_t, atol=1e-3), "targets differ from the archive"


def fit(keys):
    return ridge(np.hstack([TR[k] for k in keys]), TR["Y"])(np.hstack([TE[k] for k in keys]))


P = {"2": fit(["a", "w"]), "3": fit(["a", "w", "env"]),
     "4": fit(["a", "w", "bz"]), "5": fit(["a", "w", "env", "bz"])}


def rho(A, k):
    return float(np.corrcoef(A[:, k], Y[:, k])[0, 1])


print(f"{len(te_idx):,} validation anchors, {len(tr_idx):,} training\n")
print("  The ridge is only a usable instrument where it tracks the trained model. It does,")
print("  at every lead, so the headroom columns can be read as the deep model's headroom.\n")
print(f"  {'lead':>6s} {'deep':>7s} {'ridge':>7s} {'pers':>7s} |"
      f" {'+env':>8s} {'+bz':>8s} {'+all':>8s}")
for k in range(OUT):
    if k not in (0, 1, 3, 5, 11, 17, 23):
        continue
    print(f"  {(k+1)*0.5:5.1f}h {rho(deep_p, k):7.3f} {rho(P['2'], k):7.3f} "
          f"{rho(TE['pers'], k):7.3f} | "
          f"{rho(P['3'], k) - rho(P['2'], k):+8.3f} "
          f"{rho(P['4'], k) - rho(P['2'], k):+8.3f} "
          f"{rho(P['5'], k) - rho(P['2'], k):+8.3f}")

print("\n  Headroom is what perfect knowledge of the future wind would add at that lead.")
print("  It grows monotonically with lead. At 0.5-1 h there is nothing to extract, so a")
print("  short-horizon specialist has no information prize; whatever it could win there is")
print("  a fitting question against persistence, not a modelling one.")
