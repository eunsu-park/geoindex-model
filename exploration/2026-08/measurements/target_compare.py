"""Is the ceiling a property of ap30, or of the problem?

Everything measured so far was measured on ap30. If hp30 splits its headroom differently -- less
in the Bz phase, more in the envelope -- then the conclusions about what remote sensing could
ever reach are ap-specific and have to be redone per index. If it splits the same way, the
ceiling is a property of the wind and transfers.

Same anchors, same features, same ridge; only the target changes, and each target carries its own
history as an input. Keeping the split fixed isolates the target rather than confounding it with
a different validation set.

Usage:
    python target_compare.py
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
for c in WIND + ["ap30", "hp30"]:
    miss = float(tbl[c].isna().mean())
    tbl[c] = tbl[c].interpolate(limit=6).ffill().bfill()
    if miss > 0.02:
        print(f"note: {c} was {100*miss:.1f} % missing before fill")
wind = {w: tbl[w].to_numpy(float) for w in WIND}
targets = {"ap30": tbl["ap30"].to_numpy(float), "hp30": tbl["hp30"].to_numpy(float)}
pos = {t: i for i, t in enumerate(grid)}

tr = pd.read_csv(os.path.join(D, "total_ap/train_index.csv"))
tr["datetime"] = pd.to_datetime(tr["datetime"])
tr_idx = np.array([pos[t] for t in tr["datetime"] if PAST <= pos.get(t, -1) < len(grid) - OUT])
with zipfile.ZipFile(os.path.join(R, POOLED, "validation", "best", "npz.zip")) as z:
    anchors = [str(np.asarray(np.load(io.BytesIO(z.read(n)), allow_pickle=True)["anchor"]))
               for n in sorted(x for x in z.namelist() if x.endswith(".npz"))]
te_idx = np.array([pos[t] for t in pd.to_datetime(pd.Series(anchors), format="%Y%m%d%H%M%S")])
print(f"{len(tr_idx):,} training / {len(te_idx):,} validation anchors, identical for both "
      f"targets\n")

for name, series in targets.items():
    def build(idx):
        fw = idx[:, None] + np.arange(0, OUT)
        pw = idx[:, None] + np.arange(-PAST, 0)
        return dict(a=series[pw],
                    w=np.column_stack([wind[w][pw] for w in WIND]),
                    env=np.column_stack([wind[w][fw] for w in WIND if w != "bz_avg"]),
                    bz=wind["bz_avg"][fw],
                    Y=series[fw],
                    pers=np.repeat(series[idx - 1][:, None], OUT, axis=1))

    TR, TE = build(tr_idx), build(te_idx)
    Y = TE["Y"]

    def fit(keys):
        return ridge(np.hstack([TR[k] for k in keys]), TR["Y"])(np.hstack([TE[k] for k in keys]))

    def rho(A):
        return float(np.mean([np.corrcoef(A[:, k], Y[:, k])[0, 1] for k in range(OUT)]))

    base = rho(fit(["a", "w"]))
    env = rho(fit(["a", "w", "env"])) - base
    bz = rho(fit(["a", "w", "bz"])) - base
    allw = rho(fit(["a", "w", "env", "bz"])) - base
    only = {c: rho(fit(["a", "w", "f_" + c])) - base for c in ()}   # placeholder, see below
    per_comp = {}
    for c in WIND:
        TRc = dict(TR); TEc = dict(TE)
        TRc["one"] = wind[c][tr_idx[:, None] + np.arange(0, OUT)]
        TEc["one"] = wind[c][te_idx[:, None] + np.arange(0, OUT)]
        P = ridge(np.hstack([TRc[k] for k in ("a", "w", "one")]), TRc["Y"])(
            np.hstack([TEc[k] for k in ("a", "w", "one")]))
        per_comp[c] = rho(P) - base

    print(f"  ── target {name} " + "─" * 40)
    print(f"     range {Y.min():.1f} to {Y.max():.1f}, window mean {Y.mean():.2f}")
    print(f"     model (own history + past wind)   per-lead rho {base:.3f}   "
          f"persistence {rho(TE['pers']):.3f}")
    print(f"     headroom over the model           {allw:+.3f}")
    print(f"       phase    (true future bz)       {bz:+.3f}   "
          f"{100*bz/allw:4.0f} % of it")
    print(f"       envelope (everything else)      {env:+.3f}   "
          f"{100*env/allw:4.0f} % of it")
    print("     single components: " + ",  ".join(
        f"{c.replace('_avg','')} {per_comp[c]:+.3f}" for c in WIND) + "\n")

print("  If the phase share matches across the two indices, the ceiling belongs to the solar")
print("  wind rather than to ap30, and every bound in this investigation transfers.")
