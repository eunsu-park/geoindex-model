"""Can free proxies for solar imagery reach the envelope bucket?

`wind_ablation.py` split the headroom above the current model into two parts: perfect knowledge
of the future Bz is worth +0.139 of per-lead correlation (the phase), and perfect knowledge of
every other future wind component is worth +0.104 (the envelope). Imaging measures density and
emission, not magnetic field, so imagery can only ever aim at the envelope.

This asks whether the envelope is reachable from proxies we already have, before anyone ingests
an image archive. The logic is that a solar EUV image tells Earth exactly one thing about the
next few days -- a coronal hole is facing us and a high-speed stream will follow -- and the same
solar feature already announced itself 27 days ago, when the Sun last rotated it past us. The
recurrence is a FREE, EXACT record of what the image can only estimate. So:

    if the 27-day recurrence cannot reach the envelope, 193 A cannot either.

It is a one-sided test. Failing it closes the EUV coronal-hole channel; passing it does not prove
imagery works, only that the information exists and imagery is worth trying for.

Proxies, all built from the same table:

    REC_ap   ap30 across the window the Sun showed us one rotation ago
    REC_v    speed across the same window -- the stream itself, not its effect
    ACT      trailing 3-day and 27-day level, standing in for F10.7 / sunspot number

Usage:
    python recurrence_proxy.py
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
ROT = 27 * 48                      # one solar rotation in 30-minute steps
REC_OFF = np.arange(-24, 49, 2)    # -12 h to +24 h around it, hourly: slack for lag error


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

# trailing levels, computed once over the whole series
csum = np.concatenate([[0.0], np.cumsum(ap)])


def trailing(idx, days):
    n = days * 48
    return (csum[idx + 1] - csum[np.maximum(idx + 1 - n, 0)]) / n


def blocks(stamps):
    idx = np.array([pos[t] for t in stamps])
    ok = idx >= ROT + 24 + PAST
    idx = idx[ok]
    past_a = ap[idx[:, None] + np.arange(-PAST, 0)]
    past_w = np.column_stack([wind[w][idx[:, None] + np.arange(-PAST, 0)] for w in WIND])
    rec_a = ap[idx[:, None] - ROT + REC_OFF]
    rec_v = wind["v_avg"][idx[:, None] - ROT + REC_OFF]
    act = np.column_stack([trailing(idx, 3), trailing(idx, 27)])
    fut_nobz = np.column_stack([wind[w][idx[:, None] + np.arange(0, OUT)]
                                for w in WIND if w != "bz_avg"])
    Y = ap[idx[:, None] + np.arange(0, OUT)]
    return dict(a=past_a, w=past_w, rec_a=rec_a, rec_v=rec_v, act=act,
                env=fut_nobz, Y=Y, ok=ok)


tr = pd.read_csv(os.path.join(D, "total_ap/train_index.csv"))
tr["datetime"] = pd.to_datetime(tr["datetime"])
tr_stamps = [t for t in tr["datetime"] if PAST <= pos.get(t, -1) < len(grid) - OUT]

with zipfile.ZipFile(os.path.join(R, POOLED, "validation", "best", "npz.zip")) as z:
    anchors = [str(np.asarray(np.load(io.BytesIO(z.read(n)), allow_pickle=True)["anchor"]))
               for n in sorted(x for x in z.namelist() if x.endswith(".npz"))]
te_stamps = list(pd.to_datetime(pd.Series(anchors), format="%Y%m%d%H%M%S"))

TR, TE = blocks(tr_stamps), blocks(te_stamps)
print(f"train {len(TR['Y']):,} anchors, test {len(TE['Y']):,} "
      f"({int((~TE['ok']).sum())} test anchors dropped for lacking a full rotation of history)\n")

Yte = TE["Y"]
peak = Yte.max(1)
storm = peak >= 47.5

VARIANTS = [
    ("2  ap + past wind  <- the model",  ["a", "w"]),
    ("   + REC_ap  (ap 27 d ago)",       ["a", "w", "rec_a"]),
    ("   + REC_v   (speed 27 d ago)",    ["a", "w", "rec_v"]),
    ("   + ACT     (trailing level)",    ["a", "w", "act"]),
    ("   + all three proxies",           ["a", "w", "rec_a", "rec_v", "act"]),
    ("3  + TRUE future envelope",        ["a", "w", "env"]),
]

print(f"  {'':36s} {'per-lead':>9s} {'peak':>7s} {'storm':>7s}")
print(f"  {'':36s} {'rho':>9s} {'rho':>7s} {'rho':>7s}")
base = None
for label, keys in VARIANTS:
    f = ridge(np.hstack([TR[k] for k in keys]), TR["Y"])
    P = f(np.hstack([TE[k] for k in keys]))
    pl = float(np.mean([np.corrcoef(P[:, k], Yte[:, k])[0, 1] for k in range(OUT)]))
    pk = P.max(1)
    rho = float(np.corrcoef(pk, peak)[0, 1])
    rho_s = float(np.corrcoef(pk[storm], peak[storm])[0, 1])
    mark = ""
    if base is None:
        base = (pl, rho)
    else:
        mark = f"   {pl - base[0]:+.3f} per-lead"
    print(f"  {label:36s} {pl:9.3f} {rho:7.3f} {rho_s:7.3f}{mark}")

print(f"\n  storm anchors n={int(storm.sum())}")
print("  the envelope row is the target the proxies are aiming at; a proxy that reaches")
print("  none of it says the EUV coronal-hole channel has nothing for this horizon")
