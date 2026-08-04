"""Phase 0 validation: is training in L1-observation time worth what the ceiling test implied?

OMNI HRO is time-shifted from the monitor to the bow shock nose, so a row stamped t is wind
that has ALREADY arrived. Training on it hands the model zero head start. Operationally the
monitor sees that wind timeshift_sec earlier, so at wall-clock t the coming bow-shock wind is
known out to t + shift -- and the current pipeline throws that interval away.

Two ways to capture it:
  (d) keep bow-shock time and add the future wind as extra inputs -- needs a propagation step
      at serve time, and that step is the crude part
  (b) re-index the wind to the time it was OBSERVED -- no serve-time propagation at all,
      because NOAA RTSW timestamps are already observation time

(b) is the one worth having if it works. (d) is measured alongside as the upper bound it should
be approaching.

Only the seven wind channels move. ap30 is a ground index measured at real time and is not
propagated -- shifting it would be a bug, not a feature.

Training starts in 1998: before that the OMNI source sits at 65-85 Re (near-Earth monitors,
11-21 min of shift) rather than at L1 (220-235 Re, 38-64 min), so the earlier era teaches a
lead the operational system does not have. The shift is also offered as a feature, since a
model that knows how much warning it has can use it.
"""

import os

import numpy as np
import pandas as pd

PARQUET = os.path.expanduser("~/Projects/GeoIndex/datasets/data.parquet")
AUX = "omni_aux_30min.csv"
WIND = ["v_avg", "np_avg", "t_avg", "bx_avg", "by_avg", "bz_avg", "bt_avg"]
PAST, OUT = 24, 24                       # 12 h of history, 12 h forecast window
TRAIN_FROM, TEST_FROM = "1998-01-01", "2022-01-01"
STEP = pd.Timedelta("30min")


def ridge(Xtr, ytr, alpha=100.0):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (Xtr - mu) / sd
    yb = ytr.mean()
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (ytr - yb))
    return lambda Xn: ((Xn - mu) / sd) @ w + yb


df = pd.read_parquet(PARQUET).set_index("datetime").sort_index()
grid = pd.date_range(df.index[0], df.index[-1], freq="30min")
df = df.reindex(grid)
aux = pd.read_csv(AUX, parse_dates=["dt30"]).set_index("dt30").reindex(grid)

cols = WIND + ["ap30"]
df[cols] = df[cols].interpolate(limit=6).ffill().bfill()
shift_s = aux["timeshift_sec"].interpolate(limit=12).ffill().bfill().to_numpy()

# ---- the re-indexing -------------------------------------------------------------------
# Row i is wind that reached the bow shock at grid[i]; the monitor saw it shift_s[i] earlier.
obs_time = grid - pd.to_timedelta(shift_s, unit="s")
wind_obs = pd.DataFrame({c: df[c].to_numpy() for c in WIND}, index=obs_time)
wind_obs = wind_obs[~wind_obs.index.duplicated(keep="last")].sort_index()
# onto the regular grid: position s now holds the wind OBSERVED at s
wind_l1 = wind_obs.reindex(grid, method="nearest", tolerance=STEP)
print(f"re-indexed to observation time; unmatched grid points "
      f"{wind_l1[WIND[0]].isna().mean():.2%}")
wind_l1 = wind_l1.interpolate(limit=4).ffill().bfill()

ap = df["ap30"].to_numpy(float)
n = len(grid)
n_a = n - OUT
idx = np.arange(n_a)
times = grid[:n_a]
tgt = np.lib.stride_tricks.sliding_window_view(ap, OUT)[:n_a].max(axis=1)   # ap[i : i+24)
pers = ap[:n_a]


def lags(series, n_lag=PAST):
    """n_lag trailing values of one channel, ending at (not including) the anchor."""
    out = []
    for b in range(n_lag):
        lo = idx - n_lag + b
        col = np.full(n_a, np.nan)
        ok = lo >= 0
        col[ok] = series[lo[ok]]
        out.append(col)
    return out


ap_lags = lags(ap)
bsn = np.column_stack([c for w in WIND for c in lags(df[w].to_numpy(float))] + ap_lags)
l1 = np.column_stack([c for w in WIND for c in lags(wind_l1[w].to_numpy(float))] + ap_lags)
# (d) bow-shock time plus the next hour of bow-shock wind -- the upper bound
fut = []
for w in WIND:
    a = df[w].to_numpy(float)
    fut += [a[np.minimum(idx + k, n - 1)] for k in (0, 1)]
bsn_fut = np.column_stack([bsn] + fut)
sh = shift_s[:n_a] / 60.0
l1_shift = np.column_stack([l1, sh])

is_train = (times >= pd.Timestamp(TRAIN_FROM)) & (times < pd.Timestamp(TEST_FROM))
is_test = times >= pd.Timestamp(TEST_FROM)
print(f"train {int(is_train.sum())} ({TRAIN_FROM}..{TEST_FROM}), test {int(is_test.sum())}")
print(f"test storms (max ap >= 100): {int((tgt[is_test] >= 100).sum())}")
print(f"median shift in test: {np.median(sh[is_test]):.0f} min\n")

VARIANTS = [
    ("(a) bow-shock time  [current]", bsn),
    ("(b) L1 observation time", l1),
    ("(c) (b) + shift as a feature", l1_shift),
    ("(d) bow-shock + next 1 h wind", bsn_fut),
]
print(f"{'variant':32s} {'cols':>5s} {'peak rho':>9s} {'peak MAE':>9s} {'repro':>7s} "
      f"{'skill':>7s} {'AUC>=50':>8s}")


def auc(s, l):
    r = np.empty(len(s))
    r[np.argsort(s)] = np.arange(1, len(s) + 1)
    _, inv, c = np.unique(s, return_inverse=True, return_counts=True)
    sm = np.zeros(len(c))
    np.add.at(sm, inv, r)
    r = (sm / c)[inv]
    P, N = l.sum(), (~l).sum()
    return float((r[l].sum() - P * (P + 1) / 2) / (P * N))


base = None
for name, X in VARIANTS:
    ok = ~np.isnan(X).any(1)
    tr, te = ok & is_train, ok & is_test
    p = ridge(X[tr], tgt[tr])(X[te])
    o, hi = tgt[te], tgt[te] >= 100
    rho = float(np.corrcoef(p, o)[0, 1])
    mae = float(np.abs(p - o).mean())
    base = rho if base is None else base
    print(f"{name:32s} {X.shape[1]:5d} {rho:9.3f} {mae:9.2f} "
          f"{p[hi].mean()/o[hi].mean():7.3f} {1-mae/np.abs(pers[te]-o).mean():7.3f} "
          f"{auc(p, o >= 50):8.4f}   {'' if base==rho else f'{rho-base:+.3f}'}")

print("\nStorm-time only (the lead is shortest exactly when it matters):")
print(f"{'variant':32s} {'repro':>7s} {'MAE':>8s}")
for name, X in VARIANTS:
    ok = ~np.isnan(X).any(1)
    tr, te = ok & is_train, ok & is_test
    p = ridge(X[tr], tgt[tr])(X[te])
    o = tgt[te]
    hi = o >= 100
    print(f"{name:32s} {p[hi].mean()/o[hi].mean():7.3f} {np.abs(p[hi]-o[hi]).mean():8.2f}")
