"""Does a longer input window recover any of what imagery would provide?

The corrected imagery bound says a coronal hole's whole message -- the stream speed and the
compression at the interface -- is worth +0.107, and that about 64 % of the ceiling needs no
knowledge of the field direction. A coronal hole is also a recurrent structure, so the obvious
question is whether the same information is already sitting in a longer in-situ window: if a
stream hit Earth one rotation ago it may hit again, and no image is needed to know that.

Three earlier tests said no -- the ap input sweep (one hour reaches 99 % of the best window), the
bt sweep (one hour is everything and a rotation is far worse), and a recurrence block added to the
model (+0.000). All three share two defects with the imagery measurement that turned out to be
wrong:

  * they used 30-minute AVERAGE columns only, and the average cancels the fluctuation that
    drives the index;
  * the recurrence block carried ap30 and speed but not bt, which is half of what a coronal hole
    announces.

So the sweep is redone on the full average/minimum/maximum set, the recurrence block gains bt and
window summaries, and the lag is searched over 26 to 29 days rather than assumed to be 27 --
the synodic rotation is 27.2753 days and differential rotation moves it with latitude.

Usage:
    python input_length_v2.py
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
VARS = ["v", "np", "t", "bx", "by", "bz", "bt"]
ALL = [f"{v}_{s}" for v in VARS for s in ("avg", "min", "max")]
OUT = 24
MAXHIST = 30 * 48
# (history steps, stride, label)
WINDOWS = [(2, 1, "1 h"), (12, 1, "6 h"), (24, 1, "12 h"), (48, 2, "1 d"),
           (144, 6, "3 d"), (336, 12, "7 d"), (672, 24, "14 d"), (1296, 48, "27 d")]


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
cols = ALL + ["ap30"]
tbl[cols] = tbl[cols].interpolate(limit=6).ffill().bfill()
S = {c: tbl[c].to_numpy(float) for c in cols}
pos = {t: i for i, t in enumerate(grid)}

tr = pd.read_csv(os.path.join(D, "total_ap/train_index.csv"))
tr["datetime"] = pd.to_datetime(tr["datetime"])
tr_idx = np.array([pos[t] for t in tr["datetime"]
                   if MAXHIST <= pos.get(t, -1) < len(grid) - OUT])
with zipfile.ZipFile(os.path.join(R, POOLED, "validation", "best", "npz.zip")) as z:
    anchors = [str(np.asarray(np.load(io.BytesIO(z.read(n)), allow_pickle=True)["anchor"]))
               for n in sorted(x for x in z.namelist() if x.endswith(".npz"))]
te_all = np.array([pos[t] for t in pd.to_datetime(pd.Series(anchors), format="%Y%m%d%H%M%S")])
te_idx = te_all[te_all >= MAXHIST]
print(f"train {len(tr_idx):,}, test {len(te_idx):,} "
      f"({len(te_all) - len(te_idx)} dropped for lacking 30 days of history)\n")

Ytr = S["ap30"][tr_idx[:, None] + np.arange(0, OUT)]
Yte = S["ap30"][te_idx[:, None] + np.arange(0, OUT)]


def hist(idx, steps, stride, keys):
    off = np.arange(-steps, 0, stride)
    return np.column_stack([S[c][idx[:, None] + off] for c in keys])


def rho(P):
    return float(np.mean([np.corrcoef(P[:, k], Yte[:, k])[0, 1] for k in range(OUT)]))


def score(ftr, fte):
    return rho(ridge(ftr, Ytr)(fte))


print("  ── input length, full avg/min/max set ────────────────────────")
print(f"  {'window':>10s} {'features':>9s} {'per-lead rho':>13s}")
best = None
for steps, stride, name in WINDOWS:
    keys = ALL + ["ap30"]
    a, b = hist(tr_idx, steps, stride, keys), hist(te_idx, steps, stride, keys)
    r = score(a, b)
    if name == "12 h":
        best = r
    print(f"  {name:>10s} {a.shape[1]:9d} {r:13.3f}"
          + ("   <- current" if name == "12 h" else ""))

# ── a recurrence block that carries what a coronal hole actually announces ──
print("\n  ── 12 h of recent history PLUS one rotation back ─────────────")
print("     the recurrence block now carries bt as well as ap30 and speed,")
print("     with the lag searched rather than assumed\n")
base_keys = ALL + ["ap30"]
Atr, Ate = hist(tr_idx, 24, 1, base_keys), hist(te_idx, 24, 1, base_keys)
print(f"  {'':34s} {'per-lead rho':>13s} {'gain':>8s}")
print(f"  {'12 h recent only':34s} {best:13.3f}")

REC_KEYS = ["ap30", "v_avg", "v_max", "bt_avg", "bt_max", "np_avg"]


def rec_block(idx, lags):
    out = []
    for lag_d in lags:
        off = np.arange(0, OUT, 4) - int(round(lag_d * 48))
        blk = np.column_stack([S[c][idx[:, None] + off] for c in REC_KEYS])
        # summaries: a ridge reads a level more easily than a raw series
        wide = np.arange(-24, 49, 8) - int(round(lag_d * 48))
        summ = np.column_stack([S[c][idx[:, None] + wide].max(1) for c in REC_KEYS]
                               + [S[c][idx[:, None] + wide].mean(1) for c in REC_KEYS])
        out += [blk, summ]
    return np.hstack(out)


for label, lags in (("+ recurrence at 27 d", [27.0]),
                    ("+ recurrence, lags 26-29 d", [26.0, 27.0, 27.5, 28.0, 29.0])):
    ftr = np.hstack([Atr, rec_block(tr_idx, lags)])
    fte = np.hstack([Ate, rec_block(te_idx, lags)])
    r = score(ftr, fte)
    print(f"  {label:34s} {r:13.3f} {r - best:+8.3f}")

print("\n  For reference, what the information itself is worth if it were known perfectly:")
fw = te_idx[:, None] + np.arange(0, OUT)
fwt = tr_idx[:, None] + np.arange(0, OUT)
vb = ["v_avg", "v_min", "v_max", "bt_avg", "bt_min", "bt_max"]
r = score(np.hstack([Atr, np.column_stack([S[c][fwt] for c in vb])]),
          np.hstack([Ate, np.column_stack([S[c][fw] for c in vb])]))
print(f"  {'true future v and bt':34s} {r:13.3f} {r - best:+8.3f}")
print("\n  If a longer window reaches none of that, the information a coronal-hole image would")
print("  supply is not sitting in the in-situ record, and imagery is the only route to it.")
