"""A one-to-three day storm probability, built from OMNI alone.

The 12-hour curve is bounded and the bound is understood. This is a different product on the same
data: not how large the index will be, but how likely a storm is on each of the next three days.

    day 1   P( max ap30 over [T,      T+24 h] >= 48 )
    day 2   P( max ap30 over [T+24 h, T+48 h] >= 48 )
    day 3   P( max ap30 over [T+48 h, T+72 h] >= 48 )

Why this escapes what closed the curve: a probability is not a conditional mean of a magnitude,
so `sigma_pred/sigma_obs = rho` does not apply and there is no damping to fight. The product is
honest about what is unknown instead of averaging it away.

The bar is not climatology. At this horizon the cheap baseline that actually competes is the
27-day recurrence -- the Sun turns, and whatever produced a stream last rotation may produce one
again -- together with persistence of the current activity level. A model that beats climatology
but not recurrence has learned the solar cycle and nothing else. All three are scored.

Inputs are OMNI only: recent wind and index, a coarse multi-day history, the recurrence window
aligned to each target day, and trailing activity levels standing in for F10.7.

A last row bounds what a coronagraph could add at this horizon, without ingesting an image. The
most a perfect LASCO pipeline delivers is that a disturbance is coming and when -- it cannot
measure the magnetic field. So the model is handed exactly that: whether a front arrives inside
the target day, and how large the jump is. Nothing built from images can beat it, and days 2 and
3 are the horizon where a coronagraph is supposed to earn its keep.

Usage:
    python storm_probability.py [--threshold 48]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

D = os.path.expanduser("~/Projects/GeoIndex/datasets")
WIND = ["v_avg", "np_avg", "t_avg", "bx_avg", "by_avg", "bz_avg", "bt_avg"]
ROT = 27 * 48
DAY = 48
LADDER = [0, 2, 3, 4, 5, 6, 7, 9, 12, 15, 18, 22, 27, 32, 39, 48, 56, 67, 80, 94, 111, 132,
          154, 179, 207, 236, 265, 294, 324, 355, 388, 421, 456, 494, 534, 617]


def ridge(X, y, alpha=100.0):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    yb = y.mean()
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - yb))
    return lambda Xn: ((Xn - mu) / sd) @ w + yb


def calibrate(score_tr, y_tr, score_te, bins=20):
    """Turn any score into a probability by its empirical exceedance rate on training."""
    qs = np.quantile(score_tr, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    base = float(y_tr.mean())
    rate = np.empty(bins)
    for b in range(bins):
        m = (score_tr >= qs[b]) & (score_tr < qs[b + 1])
        rate[b] = float(y_tr[m].mean()) if m.sum() >= 30 else base
    return rate[np.clip(np.searchsorted(qs, score_te, side="right") - 1, 0, bins - 1)]


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap_.add_argument("--threshold", type=float, default=48.0)
    ap_.add_argument("--jitter", action="store_true",
                     help="sweep the oracle's arrival-time error instead of the model table")
    args = ap_.parse_args()
    if args.threshold not in LADDER:
        raise SystemExit(f"{args.threshold:g} is not an ap ladder value; 50 silently means 56")
    thr = args.threshold - 0.5

    tbl = pd.read_parquet(os.path.join(D, "data.parquet")).set_index("datetime").sort_index()
    grid = pd.date_range(tbl.index[0], tbl.index[-1], freq="30min")
    tbl = tbl.reindex(grid)
    cols = WIND + ["ap30"]
    tbl[cols] = tbl[cols].interpolate(limit=6).ffill().bfill()
    ap = tbl["ap30"].to_numpy(float)
    wind = {w: tbl[w].to_numpy(float) for w in WIND}
    pos = {t: i for i, t in enumerate(grid)}
    csum = np.concatenate([[0.0], np.cumsum(ap)])

    lo, hi = ROT + 24, len(grid) - 3 * DAY - 1
    tr = pd.read_csv(os.path.join(D, "total_ap/train_index.csv"))
    va = pd.read_csv(os.path.join(D, "total_ap/validation_index.csv"))
    for f in (tr, va):
        f["datetime"] = pd.to_datetime(f["datetime"])
    tr_idx = np.array([pos[t] for t in tr["datetime"] if lo <= pos.get(t, -1) <= hi])
    te_idx = np.array([pos[t] for t in va["datetime"] if lo <= pos.get(t, -1) <= hi])
    print(f"train {len(tr_idx):,} anchors ({grid[tr_idx.min()].date()} to "
          f"{grid[tr_idx.max()].date()}), test {len(te_idx):,} "
          f"({grid[te_idx.min()].date()} to {grid[te_idx.max()].date()})")
    print(f"target: a storm reaching ap30 >= {args.threshold:g} inside each 24 h day\n")

    def trailing(idx, days):
        n = days * DAY
        return (csum[idx + 1] - csum[np.maximum(idx + 1 - n, 0)]) / n

    def feats(idx, day):
        """day is 0, 1 or 2; the recurrence window is aligned to THAT day's target."""
        recent = np.hstack([ap[idx[:, None] + np.arange(-24, 0)]]
                           + [wind[w][idx[:, None] + np.arange(-24, 0)] for w in WIND])
        coarse = np.hstack([ap[idx[:, None] + np.arange(-7 * DAY, 0, 12)]]
                           + [wind[w][idx[:, None] + np.arange(-7 * DAY, 0, 12)]
                              for w in ("v_avg", "bt_avg", "np_avg")])
        rec_off = np.arange(day * DAY, (day + 1) * DAY, 6) - ROT
        rec = np.hstack([ap[idx[:, None] + rec_off], wind["v_avg"][idx[:, None] + rec_off]])
        lvl = np.column_stack([trailing(idx, 1), trailing(idx, 3), trailing(idx, 27)])
        return dict(recent=recent, coarse=coarse, rec=rec, lvl=lvl)

    # front detection: a one-hour speed rise with the field rising too -- the shock signature
    v, bt = wind["v_avg"], wind["bt_avg"]
    dv, dbt = np.zeros_like(v), np.zeros_like(bt)
    dv[2:], dbt[2:] = v[2:] - v[:-2], bt[2:] - bt[:-2]
    front = ((dv >= np.percentile(dv, 99.5)) & (dbt > 0)).astype(float)
    amp = np.where(front > 0, dv, 0.0)

    def oracle(idx, day):
        w = idx[:, None] + np.arange(day * DAY, (day + 1) * DAY, 2)
        return np.hstack([front[w], amp[w]])

    def jitter(O, sigma_h, seed):
        """Slide each front by a random error. One column is one hour at this stride."""
        r = np.random.default_rng(seed)
        half = O.shape[1] // 2
        ind, a = O[:, :half], O[:, half:]
        oi, oa = np.zeros_like(ind), np.zeros_like(a)
        shift = np.rint(r.normal(0.0, sigma_h, size=len(O))).astype(int)
        for i in range(len(O)):
            for k in np.flatnonzero(ind[i]):
                k2 = k + shift[i]
                if 0 <= k2 < half:
                    oi[i, k2], oa[i, k2] = 1.0, a[i, k]
        return np.hstack([oi, oa])

    def label(idx, day):
        w = idx[:, None] + np.arange(day * DAY, (day + 1) * DAY)
        return (ap[w].max(1) >= thr).astype(float)

    if args.jitter:
        base = ["recent", "coarse", "rec", "lvl"]
        print("  Brier skill against climatology, by the arrival-time error a coronagraph")
        print("  pipeline would actually achieve. A 24 h window tolerates timing error far")
        print("  better than the 12 h curve does -- an error of a few hours usually still")
        print("  lands the front on the right day.\n")
        print(f"  {'':24s} {'day 1':>9s} {'day 2':>9s} {'day 3':>9s}")
        rows = {}
        for day in (0, 1, 2):
            F, G = feats(tr_idx, day), feats(te_idx, day)
            Otr, Ote = oracle(tr_idx, day), oracle(te_idx, day)
            ytr, yte = label(tr_idx, day), label(te_idx, day)
            bc = float(np.mean((ytr.mean() - yte) ** 2))

            def sk(otr, ote):
                Xtr = np.hstack([F[k] for k in base] + ([otr] if otr is not None else []))
                Xte = np.hstack([G[k] for k in base] + ([ote] if ote is not None else []))
                f = ridge(Xtr, ytr)
                p = calibrate(f(Xtr), ytr, f(Xte))
                return 1 - float(np.mean((p - yte) ** 2)) / bc

            rows.setdefault("OMNI only", []).append(sk(None, None))
            for sg in (0.0, 3.0, 6.0, 10.0, 18.0):
                lab = "perfect arrival" if sg == 0 else f"arrival +/- {sg:g} h"
                rows.setdefault(lab, []).append(
                    sk(Otr if sg == 0 else jitter(Otr, sg, 31),
                       Ote if sg == 0 else jitter(Ote, sg, 32)))
        for k, v in rows.items():
            print(f"  {k:24s}" + "".join(f"{x:9.3f}" for x in v))
        return

    MODELS = [
        ("climatology", None),
        ("persistence of level", ["lvl"]),
        ("27-day recurrence", ["rec"]),
        ("recurrence + level", ["rec", "lvl"]),
        ("OMNI wind + index", ["recent", "coarse", "lvl"]),
        ("everything", ["recent", "coarse", "rec", "lvl"]),
        ("+ ORACLE front arrival", ["recent", "coarse", "rec", "lvl", "orc"]),
    ]
    for day in (0, 1, 2):
        Ftr, Fte = feats(tr_idx, day), feats(te_idx, day)
        Ftr["orc"], Fte["orc"] = oracle(tr_idx, day), oracle(te_idx, day)
        ytr, yte = label(tr_idx, day), label(te_idx, day)
        clim = float(ytr.mean())
        bs_clim = float(np.mean((clim - yte) ** 2))
        print(f"  ── day {day+1}:  T+{day*24} h to T+{(day+1)*24} h "
              f"─────────────────────────")
        print(f"     base rate  train {clim:.3f}   test {yte.mean():.3f}"
              f"     climatology Brier {bs_clim:.4f}")
        print(f"     {'':24s} {'Brier':>8s} {'skill':>8s} {'AUC':>7s}")
        for name, keys in MODELS:
            if keys is None:
                p = np.full_like(yte, clim)
            else:
                s_tr = ridge(np.hstack([Ftr[k] for k in keys]), ytr)
                p = calibrate(s_tr(np.hstack([Ftr[k] for k in keys])), ytr,
                              s_tr(np.hstack([Fte[k] for k in keys])))
            bs = float(np.mean((p - yte) ** 2))
            o = np.argsort(p)
            r = np.empty_like(o, dtype=float)
            r[o] = np.arange(len(p))
            n1, n0 = yte.sum(), (1 - yte).sum()
            auc = float((r[yte == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))
            print(f"     {name:24s} {bs:8.4f} {1-bs/bs_clim:8.3f} {auc:7.3f}")
        print()

    print("  A model that beats climatology but not recurrence has learned the solar cycle.")
    print("  The ORACLE row is the ceiling on any coronagraph pipeline at this horizon: it")
    print("  already has perfect arrival timing and amplitude, which LASCO cannot produce.")


if __name__ == "__main__":
    main()
