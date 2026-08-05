"""An upper bound on the LASCO channel, without ingesting a single image.

The recurrence test closed the EUV coronal-hole channel: knowing what the Sun showed us one
rotation ago adds nothing the in-situ wind does not already carry. But coronal holes recur and
eruptions do not, so that test says nothing about a coronagraph, whose subject is CMEs.

A coronagraph cannot measure the magnetic field -- Thomson-scattered white light gives electron
density. So the most a perfect LASCO pipeline could ever deliver is: a disturbance is coming, and
here is when it gets here. That is an arrival-time forecast, and its value can be bounded from
above without any image, by simply telling the model the answer:

    ORACLE   for every anchor, the exact step at which a disturbance front arrives inside the
             forecast window, and how large the jump is

Nothing built from LASCO can beat the oracle, because the oracle already has perfect arrival
timing and perfect amplitude, which is strictly more than a coronagraph plus a propagation model
can produce. If the oracle is worth little, LASCO is worth less.

Fronts are detected in the in-situ series as a sharp rise in speed with a simultaneous rise in
field magnitude -- the shock/sheath signature. The threshold is set by percentile so the rate is
reported rather than assumed.

Usage:
    python arrival_oracle.py
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

# ── front detection: speed jump over one hour, with the field rising too ────
v, bt = wind["v_avg"], wind["bt_avg"]
dv = np.zeros_like(v)
dbt = np.zeros_like(bt)
dv[2:] = v[2:] - v[:-2]
dbt[2:] = bt[2:] - bt[:-2]
thr_v = float(np.percentile(dv, 99.5))
front = ((dv >= thr_v) & (dbt > 0)).astype(float)
amp = np.where(front > 0, dv, 0.0)
print(f"front detector: dv/1h >= {thr_v:.1f} km/s with the field rising -> "
      f"{int(front.sum()):,} fronts in {len(grid):,} steps "
      f"({100*front.mean():.2f} % of steps, about {front.sum()/31:.0f} per year)\n")


def blocks(stamps):
    idx = np.array([pos[t] for t in stamps])
    past_a = ap[idx[:, None] + np.arange(-PAST, 0)]
    past_w = np.column_stack([wind[w][idx[:, None] + np.arange(-PAST, 0)] for w in WIND])
    fw = idx[:, None] + np.arange(0, OUT)
    oracle = np.hstack([front[fw], amp[fw]])
    fut_nobz = np.column_stack([wind[w][fw] for w in WIND if w != "bz_avg"])
    fut_bz = wind["bz_avg"][fw]
    return dict(a=past_a, w=past_w, orc=oracle, env=fut_nobz, bz=fut_bz,
                Y=ap[fw], hasfront=front[fw].max(1))


tr = pd.read_csv(os.path.join(D, "total_ap/train_index.csv"))
tr["datetime"] = pd.to_datetime(tr["datetime"])
tr_stamps = [t for t in tr["datetime"] if PAST <= pos.get(t, -1) < len(grid) - OUT]

with zipfile.ZipFile(os.path.join(R, POOLED, "validation", "best", "npz.zip")) as z:
    anchors = [str(np.asarray(np.load(io.BytesIO(z.read(n)), allow_pickle=True)["anchor"]))
               for n in sorted(x for x in z.namelist() if x.endswith(".npz"))]
te_stamps = list(pd.to_datetime(pd.Series(anchors), format="%Y%m%d%H%M%S"))

TR, TE = blocks(tr_stamps), blocks(te_stamps)
Yte = TE["Y"]
peak = Yte.max(1)
storm = peak >= 47.5
hasf = TE["hasfront"] > 0
print(f"train {len(TR['Y']):,}, test {len(TE['Y']):,}; "
      f"{int(hasf.sum()):,} test windows contain a front ({100*hasf.mean():.1f} %), "
      f"{int((hasf & storm).sum()):,} of the {int(storm.sum()):,} storm windows "
      f"({100*(hasf & storm).sum()/storm.sum():.1f} %)\n")

VARIANTS = [
    ("2  ap + past wind  <- the model",      ["a", "w"]),
    ("   + ORACLE arrival time & amplitude", ["a", "w", "orc"]),
    ("3  + TRUE future envelope",            ["a", "w", "env"]),
    ("4  + TRUE future bz",                  ["a", "w", "bz"]),
    ("5  + TRUE future wind, all of it",     ["a", "w", "env", "bz"]),
]

print(f"  {'':38s} {'per-lead':>9s} {'peak':>7s} {'storm':>7s}  {'vs model':>10s}")
base = None
for label, keys in VARIANTS:
    f = ridge(np.hstack([TR[k] for k in keys]), TR["Y"])
    P = f(np.hstack([TE[k] for k in keys]))
    pl = float(np.mean([np.corrcoef(P[:, k], Yte[:, k])[0, 1] for k in range(OUT)]))
    pk = P.max(1)
    rho = float(np.corrcoef(pk, peak)[0, 1])
    rho_s = float(np.corrcoef(pk[storm], peak[storm])[0, 1])
    if base is None:
        base, mark = pl, ""
    else:
        mark = f"{pl - base:+.3f}"
    print(f"  {label:38s} {pl:9.3f} {rho:7.3f} {rho_s:7.3f}  {mark:>10s}")

print("\n  Nothing built from a coronagraph can exceed the ORACLE row: it already has")
print("  exact arrival timing and exact amplitude, which LASCO plus a propagation")
print("  model cannot produce. Read it as the ceiling on the whole LASCO channel.")
