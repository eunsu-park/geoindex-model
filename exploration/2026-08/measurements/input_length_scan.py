"""How far back is worth looking, and does anything else in the record help?

The trained models span 6 h to 3 days of input and sit at lead rho 0.568-0.582 with no trend,
so more history bought nothing over a 12x range. That could still be an artefact of the deep
models -- more input means more parameters and a harder optimisation, which can hide a real
signal. A ridge cannot hide it: the feature count is held FIXED at 24 bins per channel
whatever the span, so only the reach of the window changes, never the model's capacity.

Windows run to 54 days so the 27-day solar rotation is inside the reach. Coronal-hole high
speed streams recur with that period, which is the one physical reason to expect long memory
in this target.

Then three additions that are not window length at all:
  recurrence  -- ap27 = the target's own value one and two solar rotations ago
  cycle       -- where we are in the ~11-year cycle (a running 365-day mean of ap)
  precond     -- how disturbed the magnetosphere already is (running means at 1/3/7 days)

Everything is scored on the same held-out period as the deep models (2022 onward) and against
the same quantity they are judged on: the maximum ap over the next 12 hours.
"""

import os

import numpy as np
import pandas as pd

PARQUET = os.path.expanduser("~/Projects/GeoIndex/datasets/data.parquet")
CH = ["v_avg", "np_avg", "t_avg", "bx_avg", "by_avg", "bz_avg", "bt_avg", "ap30"]
STEP_H = 0.5                 # the record is half-hourly
N_BINS = 24                  # fixed feature budget per channel, whatever the span
OUT_STEPS = 24               # 12 h forecast window
TEST_FROM = "2022-01-01"
WINDOWS_H = [1, 2, 3, 6, 12, 24, 48, 72, 24 * 7, 24 * 14, 24 * 27, 24 * 54]


def ridge_fit(X, y, alpha):
    """Closed-form ridge on standardized features with an unpenalized intercept."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    ybar = y.mean()
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (y - ybar))
    return lambda Xn: ((Xn - mu) / sd) @ w + ybar


def score(pred, obs, pers):
    rho = float(np.corrcoef(pred, obs)[0, 1])
    hi = obs >= 100
    mae = float(np.abs(pred - obs).mean())
    return {"rho": rho, "mae": mae, "repro": float(pred[hi].mean() / obs[hi].mean()),
            "skill": 1.0 - mae / float(np.abs(pers - obs).mean()), "n_hi": int(hi.sum())}


df = pd.read_parquet(PARQUET).set_index("datetime").sort_index()
full = pd.date_range(df.index[0], df.index[-1], freq="30min")
df = df.reindex(full)
print(f"record {df.index[0]} .. {df.index[-1]}  rows {len(df)}")
print(f"missing ap30 {df['ap30'].isna().mean():5.2%}, bz_avg {df['bz_avg'].isna().mean():5.2%}")
df[CH] = df[CH].interpolate(limit=6).ffill().bfill()

ap = df["ap30"].to_numpy(float)
chan = {c: df[c].to_numpy(float) for c in CH}
n = len(df)

# target: the maximum over the next 12 h, at every half-hourly anchor
tgt = np.lib.stride_tricks.sliding_window_view(ap, OUT_STEPS)[1:].max(axis=1)
n_anchor = len(tgt)
idx = np.arange(n_anchor)
is_test = df.index[:n_anchor] >= pd.Timestamp(TEST_FROM)
pers = ap[:n_anchor]                       # persistence = the anchor value

# running means for the extra features, causal (no leakage past the anchor)
csum = np.concatenate([[0.0], np.cumsum(ap)])
def running_mean(hours):
    k = int(hours / STEP_H)
    out = np.full(n, np.nan)
    out[k:] = (csum[k + 1:n + 1] - csum[1:n - k + 1]) / k
    return out
run = {h: running_mean(h) for h in (24, 24 * 3, 24 * 7, 24 * 365)}


def binned(window_h):
    """Means over equal slices of the trailing window; at most N_BINS slices per channel.

    A window shorter than N_BINS half-hours cannot be cut into N_BINS bins, so it uses one bin
    per available sample instead. Reporting the column count keeps that visible -- otherwise a
    3 h window would silently be scored on 12 h of data.
    """
    k = int(window_h / STEP_H)
    n_bins = min(N_BINS, k)
    per = max(k // n_bins, 1)
    used = per * n_bins
    cols = []
    for c in CH:
        cs = np.concatenate([[0.0], np.cumsum(chan[c])])
        for b in range(n_bins):
            # bin b covers [anchor-used+b*per, anchor-used+(b+1)*per)
            lo = idx - used + b * per
            hi = lo + per
            v = np.full(n_anchor, np.nan)
            ok = lo >= 0
            v[ok] = (cs[hi[ok]] - cs[lo[ok]]) / per
            cols.append(v)
    return np.column_stack(cols), used


print(f"\nanchors {n_anchor}, test from {TEST_FROM}: {int(is_test.sum())}, "
      f"observed max >= 100 in test: {int((tgt[is_test] >= 100).sum())}\n")
print("(A) INPUT WINDOW LENGTH -- feature count fixed at "
      f"{N_BINS} bins x {len(CH)} channels = {N_BINS*len(CH)}")
print(f"{'window':>10s} {'cols':>5s} {'train n':>9s} {'rho':>7s} {'MAE':>7s} {'repro':>7s} {'skill':>7s}")
best = None
for wh in WINDOWS_H:
    X, used = binned(wh)
    ok = ~np.isnan(X).any(1)
    tr, te = ok & ~is_test, ok & is_test
    if tr.sum() < 5000:
        print(f"{wh/24:9.1f}d  too few training rows"); continue
    f = ridge_fit(X[tr], tgt[tr], alpha=100.0)
    s = score(f(X[te]), tgt[te], pers[te])
    label = f"{wh}h" if wh < 48 else f"{wh/24:.0f}d"
    print(f"{label:>10s} {X.shape[1]:5d} {int(tr.sum()):9d} {s['rho']:7.3f} {s['mae']:7.2f} {s['repro']:7.3f} "
          f"{s['skill']:7.3f}")
    if best is None or wh == 24:
        best = (X, ok)

print("\n(B) EXTRA INFORMATION on top of a 24 h window (same 192 features + the additions)")
X24, _ = binned(24)
extras = {
    "recurrence (ap at -27d, -54d)": [np.roll(ap[:n_anchor], int(24*27/STEP_H)),
                                      np.roll(ap[:n_anchor], int(24*54/STEP_H))],
    "cycle (365 d running mean)": [run[24*365][:n_anchor]],
    "preconditioning (1/3/7 d means)": [run[24][:n_anchor], run[24*3][:n_anchor],
                                        run[24*7][:n_anchor]],
}
rows = [("24 h window alone", X24)]
for name, cols in extras.items():
    rows.append((f"+ {name}", np.column_stack([X24] + cols)))
rows.append(("+ all three", np.column_stack([X24] + [c for v in extras.values() for c in v])))
print(f"{'features':34s} {'cols':>5s} {'rho':>7s} {'MAE':>7s} {'repro':>7s} {'skill':>7s}")
for name, X in rows:
    ok = ~np.isnan(X).any(1)
    ok[:int(24*54/STEP_H)] = False          # the longest lag needs a full run-up
    tr, te = ok & ~is_test, ok & is_test
    f = ridge_fit(X[tr], tgt[tr], alpha=100.0)
    s = score(f(X[te]), tgt[te], pers[te])
    print(f"{name:34s} {X.shape[1]:5d} {s['rho']:7.3f} {s['mae']:7.2f} {s['repro']:7.3f} "
          f"{s['skill']:7.3f}")

print("\n(C) CEILING CHECK -- the same ridge given the FUTURE solar wind it is trying to "
      "anticipate.\nNot achievable; it is what the input set is worth when the driver is "
      "observed rather than forecast.")
fut = []
for c in CH[:-1]:
    cs = np.concatenate([[0.0], np.cumsum(chan[c])])
    per = OUT_STEPS // 6
    for b in range(6):
        lo = idx + b * per; hi = np.minimum(lo + per, n)
        v = np.full(n_anchor, np.nan); good = hi > lo
        v[good] = (cs[hi[good]] - cs[lo[good]]) / (hi[good] - lo[good])
        fut.append(v)
Xf = np.column_stack([X24] + fut)
ok = ~np.isnan(Xf).any(1)
tr, te = ok & ~is_test, ok & is_test
f = ridge_fit(Xf[tr], tgt[tr], alpha=100.0)
s = score(f(Xf[te]), tgt[te], pers[te])
print(f"{'24 h past + 12 h FUTURE wind':34s} {Xf.shape[1]:5d} {s['rho']:7.3f} {s['mae']:7.2f} "
      f"{s['repro']:7.3f} {s['skill']:7.3f}")
