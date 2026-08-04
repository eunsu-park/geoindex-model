"""What would forecasting the wind alongside the index actually buy?

Three things are already measured and bound the answer:

  * the wind -> index mapping is nearly solved -- a ridge given the true future wind reaches
    peak rho 0.858 against 0.677 from the past alone;
  * almost all of the information about ap sits in the last hour of wind;
  * architecture is irrelevant at this scale (a ridge beats 12 of 14 trained networks).

So the question reduces to: how far ahead is the WIND itself predictable, and what does the
mapping return when fed a forecast of it rather than the truth? That is a two-stage pipeline
and it can be built out of ridges today.

  stage 1   past wind            -> future wind        (the hard part)
  stage 2   past + future wind   -> max ap over 12 h   (the easy part, rho 0.86 with truth)

Feeding stage 1's output into stage 2 gives the number a jointly-trained model would be
competing against, without training anything. Chained ridges are a lower bound on what a
network could do -- but this session has shown the gap between the two is around 0.004, so it
is a tight one.

Trained on 1998-2021 (the L1 era), scored on 2022-2025.
"""

import os
import subprocess

import numpy as np
import pandas as pd

D = os.path.expanduser("~/Projects/GeoIndex/datasets")
WIND = ["v_avg", "np_avg", "t_avg", "bx_avg", "by_avg", "bz_avg", "bt_avg"]
PAST, OUT = 24, 24
TRAIN_FROM, TEST_FROM = "1998-01-01", "2022-01-01"


def ridge(Xtr, Ytr, alpha=100.0):
    """Multi-output ridge; returns a predictor."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (Xtr - mu) / sd
    yb = Ytr.mean(0)
    W = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (Ytr - yb))
    return lambda Xn: ((Xn - mu) / sd) @ W + yb


df = pd.read_parquet(os.path.join(D, "data_obs.parquet")).set_index("datetime").sort_index()
grid = pd.date_range(df.index[0], df.index[-1], freq="30min")
df = df.reindex(grid)
df[WIND + ["ap30"]] = df[WIND + ["ap30"]].interpolate(limit=6).ffill().bfill()
ap = df["ap30"].to_numpy(float)
n = len(grid)
n_a = n - OUT
idx = np.arange(n_a)
times = grid[:n_a]
tgt_peak = np.lib.stride_tricks.sliding_window_view(ap, OUT)[:n_a].max(axis=1)
pers = ap[:n_a]

is_tr = (times >= pd.Timestamp(TRAIN_FROM)) & (times < pd.Timestamp(TEST_FROM))
is_te = times >= pd.Timestamp(TEST_FROM)


def lags(series, k=PAST):
    out = []
    for b in range(k):
        lo = idx - k + b
        c = np.full(n_a, np.nan)
        ok = lo >= 0
        c[ok] = series[lo[ok]]
        out.append(c)
    return out


past_cols, future_cols = [], []
for w in WIND:
    a = df[w].to_numpy(float)
    past_cols += lags(a)
    future_cols += [a[np.minimum(idx + k, n - 1)] for k in range(OUT)]
Xpast = np.column_stack(past_cols + lags(ap))
Yfut = np.column_stack(future_cols)                       # 7 vars x 24 leads = 168
ok = ~np.isnan(Xpast).any(1)
tr, te = ok & is_tr, ok & is_te
print(f"train {tr.sum()}, test {te.sum()}; past features {Xpast.shape[1]}, "
      f"future wind targets {Yfut.shape[1]}")

# ── how far ahead is the wind predictable at all? ────────────────────────────
f1 = ridge(Xpast[tr], Yfut[tr])
Yhat = f1(Xpast[te])
print("\n(1) STAGE 1 — forecasting the wind from its own past (correlation, held out)")
print(f"{'lead':>6s} " + "".join(f"{w.split('_')[0]:>8s}" for w in WIND))
for k in (0, 1, 3, 5, 11, 17, 23):
    row = []
    for j, w in enumerate(WIND):
        c = j * OUT + k
        row.append(np.corrcoef(Yhat[:, c], Yfut[te][:, c])[0, 1])
    print(f"{(k+1)*0.5:5.1f}h " + "".join(f"{v:8.3f}" for v in row))
bz = WIND.index("bz_avg") * OUT
v_i = WIND.index("v_avg") * OUT
print(f"\n   Bz decorrelates fastest: {np.corrcoef(Yhat[:, bz], Yfut[te][:, bz])[0,1]:.3f} at "
      f"0.5 h, {np.corrcoef(Yhat[:, bz+3], Yfut[te][:, bz+3])[0,1]:.3f} at 2 h, "
      f"{np.corrcoef(Yhat[:, bz+23], Yfut[te][:, bz+23])[0,1]:.3f} at 12 h")
print(f"   Speed persists:          {np.corrcoef(Yhat[:, v_i], Yfut[te][:, v_i])[0,1]:.3f} at "
      f"0.5 h, {np.corrcoef(Yhat[:, v_i+23], Yfut[te][:, v_i+23])[0,1]:.3f} at 12 h")


def score(p, label):
    o = tgt_peak[te]
    hi = o >= 100
    rho = float(np.corrcoef(p, o)[0, 1])
    mae = float(np.abs(p - o).mean())
    return {"label": label, "rho": rho, "mae": mae,
            "repro": float(p[hi].mean() / o[hi].mean()),
            "skill": 1.0 - mae / float(np.abs(pers[te] - o).mean())}


# ── stage 2, three ways ──────────────────────────────────────────────────────
rows = []
f_past = ridge(Xpast[tr], tgt_peak[tr][:, None])
rows.append(score(f_past(Xpast[te]).ravel(), "past wind only (today)"))

Xfull = np.column_stack([Xpast, Yfut])
f_true = ridge(Xfull[tr], tgt_peak[tr][:, None])
rows.append(score(f_true(np.column_stack([Xpast[te], Yfut[te]])).ravel(),
                  "+ TRUE future wind (ceiling)"))

# the joint model's realistic case: stage 2 trained on truth, served stage 1's forecast
rows.append(score(f_true(np.column_stack([Xpast[te], Yhat])).ravel(),
                  "+ FORECAST future wind"))

# and trained on the forecast, which is what a jointly-trained model would effectively do
Yhat_tr = f1(Xpast[tr])
f_cons = ridge(np.column_stack([Xpast[tr], Yhat_tr])[tr[tr]] if False else
               np.column_stack([Xpast[tr], Yhat_tr]), tgt_peak[tr][:, None])
rows.append(score(f_cons(np.column_stack([Xpast[te], Yhat])).ravel(),
                  "+ forecast wind, trained on it"))

# how much future wind is actually worth having: truncate to the first k leads
for k in (2, 6, 12):
    cols = np.concatenate([np.arange(j * OUT, j * OUT + k) for j in range(len(WIND))])
    Xk_tr = np.column_stack([Xpast[tr], Yfut[tr][:, cols]])
    Xk_te = np.column_stack([Xpast[te], Yfut[te][:, cols]])
    fk = ridge(Xk_tr, tgt_peak[tr][:, None])
    rows.append(score(fk(Xk_te).ravel(), f"+ TRUE wind, first {k*0.5:.0f} h only"))

print("\n(2) STAGE 2 — max ap over the next 12 h")
print(f"{'variant':34s} {'rho':>7s} {'MAE':>7s} {'repro':>7s} {'skill':>7s}")
for r in rows:
    print(f"{r['label']:34s} {r['rho']:7.3f} {r['mae']:7.2f} {r['repro']:7.3f} {r['skill']:7.3f}")

base = rows[0]["rho"]
ceil = rows[1]["rho"]
print(f"\n(3) HOW MUCH OF THE GAP EACH ROUTE CLOSES  ({base:.3f} -> {ceil:.3f})")
for r in rows[2:]:
    print(f"   {r['label']:34s} {100*(r['rho']-base)/(ceil-base):6.1f} %")
