"""Generate the spread where the uncertainty actually lives: in the wind, not in the index.

The tower property closes deterministic joint forecasting -- for any deterministic g,
E[Y | X, g(X)] = E[Y | X], so predicting the wind and feeding it forward adds exactly nothing.
It does not close a GENERATIVE version. If g is a sample rather than a function, the ensemble
spread is produced where the physics puts it, and the ensemble is a distribution rather than a
damped mean.

An earlier test dressed the ap output in a residual process and got the shape right (sharpness
2.73 against an observed 2.59) while the level stayed wrong (storm PIT 0.815). That is the
signature of resampling on the wrong side of the mapping. This resamples the WIND:

    1  W_wind   (past ap, past wind) -> the next 12 h of wind
    2  residual trajectories from training, kept whole so the cross-variable and temporal
       correlation survives -- this is ensemble copula coupling, not a new method
    3  W_map    (past ap, past wind, future wind) -> the next 12 h of ap
    4  the mapping has its own irreducible error -- even given the TRUE wind it only reaches
       rho 0.822 -- so each member also carries a resampled trajectory of W_map's residual.
       Sampling only the wind leaves that variance out and the ensemble comes back
       under-dispersed, which is what the first version of this script measured.
    5  for each anchor draw K wind residuals and K mapping residuals, add, push through W_map

Residuals are drawn within a bin of predicted field magnitude. Storm-time wind residuals are far
larger than quiet-time ones, and unconditional resampling is exactly what under-disperses storms.

W_map also carries a rectified coupling term. Only southward Bz reconnects, so a linear map in bz
treats +5 nT and -5 nT alike and no ensemble built on it can be physically right. Adding
Bs = max(0, -bz) and v*Bs costs two columns per step.

What to read: the ensemble MEAN should reproduce the plain forecast and buy nothing -- that is the
theorem, and a mean that moves means the plumbing is wrong. The value, if any, is in the spread:
member sharpness against the observed, PIT calibration, and the Brier score of P(peak >= t).

Usage:
    python wind_ensemble.py [--members 50]
"""

from __future__ import annotations

import argparse
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
SEED = 20260806


def ridge(X, Y, alpha=100.0):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    yb = Y.mean(0)
    W = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (Y - yb))
    return lambda Xn: ((Xn - mu) / sd) @ W + yb


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap_.add_argument("--members", type=int, default=50)
    ap_.add_argument("--bins", type=int, default=10)
    args = ap_.parse_args()
    rng = np.random.default_rng(SEED)

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
    tr_idx = np.array([pos[t] for t in tr["datetime"]
                       if PAST <= pos.get(t, -1) < len(grid) - OUT])
    with zipfile.ZipFile(os.path.join(R, POOLED, "validation", "best", "npz.zip")) as z:
        anchors = [str(np.asarray(np.load(io.BytesIO(z.read(n)), allow_pickle=True)["anchor"]))
                   for n in sorted(x for x in z.namelist() if x.endswith(".npz"))]
    te_idx = np.array([pos[t]
                       for t in pd.to_datetime(pd.Series(anchors), format="%Y%m%d%H%M%S")])

    def past(idx):
        pw = idx[:, None] + np.arange(-PAST, 0)
        return np.hstack([ap[pw]] + [wind[w][pw] for w in WIND])

    def future_wind(idx):
        fw = idx[:, None] + np.arange(0, OUT)
        return np.hstack([wind[w][fw] for w in WIND])

    def target(idx):
        return ap[idx[:, None] + np.arange(0, OUT)]

    Xtr, Xte = past(tr_idx), past(te_idx)
    Wtr, Wte = future_wind(tr_idx), future_wind(te_idx)
    Ytr, Yte = target(tr_idx), target(te_idx)
    print(f"train {len(tr_idx):,}, test {len(te_idx):,}, {args.members} members\n")

    # ── 1-2. the wind forecast and its residual trajectories ────────────────
    f_wind = ridge(Xtr, Wtr)
    Rtr = Wtr - f_wind(Xtr)
    What_te = f_wind(Xte)

    bz_col, v_col = WIND.index("bz_avg") * OUT, WIND.index("v_avg") * OUT
    bt_col = WIND.index("bt_avg") * OUT

    def couple(Wblock):
        """Bs = max(0, -bz) and v*Bs -- only southward Bz reconnects."""
        bs = np.maximum(0.0, -Wblock[:, bz_col:bz_col + OUT])
        return np.hstack([bs, bs * Wblock[:, v_col:v_col + OUT]])

    # residual pool binned by predicted field magnitude, so storm spread stays storm-sized
    lvl_tr = f_wind(Xtr)[:, bt_col:bt_col + OUT].mean(1)
    edges = np.quantile(lvl_tr, np.linspace(0, 1, args.bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    pool = [np.flatnonzero((lvl_tr >= edges[b]) & (lvl_tr < edges[b + 1]))
            for b in range(args.bins)]
    lvl_te = What_te[:, bt_col:bt_col + OUT].mean(1)
    bin_te = np.clip(np.searchsorted(edges, lvl_te, side="right") - 1, 0, args.bins - 1)

    # ── 3. the mapping, with and without the rectified term ─────────────────
    print("  the wind -> index mapping (true future wind given)")
    plain = ridge(np.hstack([Xtr, Wtr]), Ytr)
    rect = ridge(np.hstack([Xtr, Wtr, couple(Wtr)]), Ytr)

    def per_lead(P, Y=Yte):
        return float(np.mean([np.corrcoef(P[:, k], Y[:, k])[0, 1] for k in range(OUT)]))

    direct = np.maximum(ridge(Xtr, Ytr)(Xte), 0.0)   # clipped: the raw ridge goes negative
                                                    # on quiet anchors and the sharpness ratio
                                                    # then explodes
    print(f"    direct forecast, no future wind        {per_lead(direct):.3f}")
    print(f"    + true future wind, linear             "
          f"{per_lead(plain(np.hstack([Xte, Wte]))):.3f}")
    print(f"    + true future wind, rectified coupling "
          f"{per_lead(rect(np.hstack([Xte, Wte, couple(Wte)]))):.3f}\n")
    f_map = rect
    Emap = Ytr - f_map(np.hstack([Xtr, Wtr, couple(Wtr)]))
    lvl_ap = f_map(np.hstack([Xtr, Wtr, couple(Wtr)])).mean(1)
    e_edges = np.quantile(lvl_ap, np.linspace(0, 1, args.bins + 1))
    e_edges[0], e_edges[-1] = -np.inf, np.inf
    e_pool = [np.flatnonzero((lvl_ap >= e_edges[b]) & (lvl_ap < e_edges[b + 1]))
              for b in range(args.bins)]

    # ── 4. the ensemble ─────────────────────────────────────────────────────
    members = np.empty((args.members, len(te_idx), OUT))
    for m in range(args.members):
        draw = np.empty(len(te_idx), dtype=int)
        for b in range(args.bins):
            sel = bin_te == b
            draw[sel] = rng.choice(pool[b], size=int(sel.sum()), replace=True)
        Wk = What_te + Rtr[draw]
        yk = f_map(np.hstack([Xte, Wk, couple(Wk)]))
        # the mapping's own residual, drawn within a bin of predicted level
        eb = np.clip(np.searchsorted(e_edges, yk.mean(1), side="right") - 1, 0, args.bins - 1)
        pick = np.empty(len(te_idx), dtype=int)
        for b in range(args.bins):
            sel = eb == b
            pick[sel] = rng.choice(e_pool[b], size=int(sel.sum()), replace=True)
        members[m] = yk + Emap[pick]
    members = np.maximum(members, 0.0)

    mean = members.mean(0)
    obs_peak = Yte.max(1)
    mem_peak = members.max(2)                     # (members, anchors)

    print("  THEOREM CHECK -- the ensemble mean must not beat the plain forecast")
    print(f"    direct forecast    per-lead rho {per_lead(direct):.3f}")
    print(f"    ensemble mean      per-lead rho {per_lead(mean):.3f}\n")

    def sharp(A):
        return float(np.mean(A.max(-1) / np.maximum(A.mean(-1), 1e-6)))

    storm = obs_peak >= 47.5
    print("  SHAPE -- is a member shaped like an observation?")
    print(f"    observed                 sharpness {sharp(Yte):.2f}   "
          f"(storms only {sharp(Yte[storm]):.2f})")
    print(f"    direct forecast          sharpness {sharp(direct):.2f}")
    print(f"    ensemble mean            sharpness {sharp(mean):.2f}")
    print(f"    a single member          sharpness "
          f"{float(np.mean([sharp(members[m]) for m in range(args.members)])):.2f}\n")

    print("  CALIBRATION -- where does the observed peak fall inside the ensemble?")
    for lab, m_ in (("all anchors", np.ones_like(storm)), ("storm anchors", storm)):
        pit = (mem_peak[:, m_] < obs_peak[m_]).mean(0)
        below = float((mem_peak[:, m_].max(0) < obs_peak[m_]).mean())
        print(f"    {lab:16s} mean PIT {pit.mean():.3f}  (0.500 is calibrated), "
              f"observed above every member {100*below:5.1f} %")

    print("\n  PROBABILITY -- P(peak >= t), Brier against climatology and a point forecast")
    tr_peak = Ytr.max(1)
    pk_direct_tr = ridge(Xtr, Ytr)(Xtr).max(1)
    pk_direct_te = direct.max(1)
    for thr in (48.0, 100.0):
        y = (obs_peak >= thr - 0.5).astype(float)
        clim = float((tr_peak >= thr - 0.5).mean())
        p_ens = (mem_peak >= thr - 0.5).mean(0)
        # isotonic-lite: empirical exceedance rate by decile of the point forecast
        qs = np.quantile(pk_direct_tr, np.linspace(0, 1, 21))
        qs[0], qs[-1] = -np.inf, np.inf
        rate = np.array([float((tr_peak[(pk_direct_tr >= qs[i]) & (pk_direct_tr < qs[i+1])]
                                >= thr - 0.5).mean() if ((pk_direct_tr >= qs[i]) &
                               (pk_direct_tr < qs[i+1])).sum() else clim) for i in range(20)])
        p_pt = rate[np.clip(np.searchsorted(qs, pk_direct_te, side="right") - 1, 0, 19)]
        b = lambda p: float(np.mean((p - y) ** 2))
        print(f"    ap >= {thr:5.0f}  base rate {y.mean():.3f}   "
              f"climatology {b(np.full_like(y, clim)):.4f}   "
              f"point forecast {b(p_pt):.4f}   ensemble {b(p_ens):.4f}")


if __name__ == "__main__":
    main()
