"""Can the field magnitude be predicted better than it currently is?

The headroom above the model splits into a phase part (true future Bz, +0.140) and an envelope
part (+0.104). Inside the envelope the largest single component is not speed but bt, the field
magnitude, worth +0.077 -- and bt is only 0.565 predictable at 12 h. That combination is the one
crack left in the envelope: a quantity that matters and that we are demonstrably bad at.

It also has a physical reason to be predictable at longer range than 12 hours. |B| at Earth is
elevated in ICME sheaths and in CIR compressions, and both are multi-day structures. The input
window was swept for the ap target and 12 h won, but it has never been swept for bt.

So: predict bt at a range of leads from input windows from one hour to one rotation, and see
whether any of them beats what the model is given now.

Usage:
    python bt_scan.py
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
LEADS = [(1, "0.5 h"), (4, "2 h"), (12, "6 h"), (24, "12 h"), (48, "24 h")]
# (steps of history, stride) -- coarser sampling for long windows keeps the design matrix sane
WINDOWS = [(2, 1, "1 h"), (12, 1, "6 h"), (24, 1, "12 h"),
           (48, 2, "1 d"), (144, 6, "3 d"), (336, 12, "7 d"), (1296, 48, "27 d")]
MAXHIST = 1296


def ridge(X, y, alpha=100.0):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    yb = y.mean()
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - yb))
    return lambda Xn: ((Xn - mu) / sd) @ w + yb


tbl = pd.read_parquet(os.path.join(D, "data.parquet")).set_index("datetime").sort_index()
grid = pd.date_range(tbl.index[0], tbl.index[-1], freq="30min")
tbl = tbl.reindex(grid)
cols = WIND + ["ap30"]
tbl[cols] = tbl[cols].interpolate(limit=6).ffill().bfill()
series = {c: tbl[c].to_numpy(float) for c in cols}
pos = {t: i for i, t in enumerate(grid)}

tr = pd.read_csv(os.path.join(D, "total_ap/train_index.csv"))
tr["datetime"] = pd.to_datetime(tr["datetime"])
tr_idx = np.array([pos[t] for t in tr["datetime"]
                   if MAXHIST <= pos.get(t, -1) < len(grid) - max(l for l, _ in LEADS)])
with zipfile.ZipFile(os.path.join(R, POOLED, "validation", "best", "npz.zip")) as z:
    anchors = [str(np.asarray(np.load(io.BytesIO(z.read(n)), allow_pickle=True)["anchor"]))
               for n in sorted(x for x in z.namelist() if x.endswith(".npz"))]
te_all = np.array([pos[t] for t in pd.to_datetime(pd.Series(anchors), format="%Y%m%d%H%M%S")])
te_idx = te_all[te_all >= MAXHIST]
print(f"train {len(tr_idx):,}, test {len(te_idx):,} "
      f"({len(te_all) - len(te_idx)} test anchors lack a full rotation of history)\n")


def features(idx, hist, stride):
    off = np.arange(-hist, 0, stride)
    return np.column_stack([series[c][idx[:, None] + off] for c in cols])


print(f"  correlation of predicted bt with observed bt\n")
print("  " + f"{'input window':>14s}" + "".join(f"{n:>10s}" for _, n in LEADS)
      + f"{'  features':>11s}")
for hist, stride, wname in WINDOWS:
    Xtr, Xte = features(tr_idx, hist, stride), features(te_idx, hist, stride)
    row = []
    for lead, _ in LEADS:
        ytr = series["bt_avg"][tr_idx + lead]
        yte = series["bt_avg"][te_idx + lead]
        p = ridge(Xtr, ytr)(Xte)
        row.append(float(np.corrcoef(p, yte)[0, 1]))
    mark = "  <- current" if wname == "12 h" else ""
    print(f"  {wname:>14s}" + "".join(f"{v:10.3f}" for v in row)
          + f"{Xtr.shape[1]:9d}{mark}")

print("\n  persistence of bt for reference")
row = []
for lead, _ in LEADS:
    p = series["bt_avg"][te_idx - 1]
    yte = series["bt_avg"][te_idx + lead]
    row.append(float(np.corrcoef(p, yte)[0, 1]))
print(f"  {'':>14s}" + "".join(f"{v:10.3f}" for v in row))

print("\n  A longer window winning at 12 h would be a crack in the envelope worth up to the")
print("  +0.077 that perfect bt is worth. A flat row means bt is as predictable as it gets")
print("  from in-situ history, and the envelope is closed from this direction too.")
