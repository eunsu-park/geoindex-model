"""Does an exogenous solar observable beat the L1 information limit for ap30?

This is the one experiment the literature review turned up that has a measured positive result.
Chakraborty & Morley (2020), JSWSC 10, 36, doi:10.1051/swsc/2020037 added GOES X-ray background
flux and the flux ratio, lagged 48 hours, to an OMNI-driven Kp model and moved the STORM-INTERVAL
numbers:

    correlation  0.69 -> 0.75          RMSE  1.48 -> 0.90 Kp
    AUC at Kp >= 6 significantly improved (DeLong Z = -8.27, p < 2.2e-16)

It matters because it is the contrapositive of the tower property. A deterministic forecast of the
future wind, computed from data the model already has, adds exactly zero -- but X-ray flux is not
a function of the L1 wind, so nothing forbids it from adding information. And it added it exactly
where every intervention this August failed: in storms.

Data: the NCEI daily X-ray background product, `sci_xrsf-l2-bkd1d`, one file per satellite,
1995-01-03 to 2026-08-04 across GOES 08/10/12/13/14/15/16/17/18/19. Two traps in it.

  * The time epoch differs by generation. The v1-0-0 files (g08, g10, g12) are seconds since
    1970-01-01; the rest are seconds since 2000-01-01 12:00. Assuming one epoch silently shifts
    the old satellites by thirty years, which is what the first version of this script did.
  * Absolute calibration differs across generations. GOES-R (16/17/18/19) agree with each other to
    within 3 %, and the whole validation period sits inside that family; but g15/g16 differ by a
    median factor 0.892. Scale factors are chained from the measured overlaps and the residual
    step at each handover is printed rather than assumed away.

RESULT: null. Per-lead correlation moves -0.001 overall, +0.001 on storm anchors and -0.005 on
quiet ones, all inside the +/-0.015 precision floor.

The data is sound -- the merged background tracks the solar cycle exactly (log10 median -8.13 at
the 1996 minimum, -5.95 at the 2002 maximum, -8.27 in 2008, -5.97 in 2014, -8.11 in 2020, -5.97
in 2023) and correlates with daily-mean ap30 at r = +0.20, rising to +0.62 under 365-day
smoothing. That is the point: the background is a SOLAR-CYCLE proxy, and the model already has
the cycle from its own ap30 and wind history. At the 12-hour scale it carries almost nothing.

WHAT THIS DOES NOT TEST. Chakraborty & Morley's mechanism is flare -> CME -> storm, and their
48-hour lag is an EVENT signal: an X-ray flare two days ago means a CME may be arriving. The
daily background is the wrong observable for that -- it measures the quiescent level, not
eruptions. The flare channel (`xrsf-l2-flsum`, peak flux and class per event) is the one that
carries the CME proxy and it is untested here. Treat this as a null on the background, not on
X-ray data.

Usage:
    python goes_xray.py [--goes-dir ...] [--lag-hours 48]
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import re
import zipfile

import numpy as np
import pandas as pd

D = os.path.expanduser("~/Projects/GeoIndex/datasets")
R = os.path.expanduser("~/Projects/GeoIndex/results")
POOLED = "probe_ap_in12h_out12h_gnn_transformer_baseline"
WIND = [f"{v}_{s}" for v in ("v", "np", "t", "bx", "by", "bz", "bt") for s in ("avg", "min", "max")]
PAST = OUT = 24
# best-calibrated first; g14 last, its background record is only 34 % flagged good
PRIORITY = ["16", "18", "19", "17", "15", "13", "10", "08", "12", "14"]


def ridge(X, Y, alpha=100.0):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    yb = Y.mean(0)
    W = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (Y - yb))
    return lambda Xn: ((Xn - mu) / sd) @ W + yb


def load_goes(d):
    import netCDF4 as nc
    frames = {}
    for f in sorted(glob.glob(os.path.join(d, "sci_xrsf-l2-bkd1d_*.nc"))):
        sat = re.search(r"_g(\d+)_", f).group(1)
        ds = nc.Dataset(f)
        t = pd.DatetimeIndex(pd.to_datetime(nc.num2date(
            np.asarray(ds["time"][:], float), ds["time"].units,
            only_use_cftime_datetimes=False))).normalize()
        a = np.asarray(ds["bkd1d_xrsa_flux"][:], float)
        b = np.asarray(ds["bkd1d_xrsb_flux"][:], float)
        fa = np.asarray(ds["bkd1d_xrsa_flag"][:])
        fb = np.asarray(ds["bkd1d_xrsb_flag"][:])
        ds.close()
        a[(fa != 0) | ~np.isfinite(a) | (a <= 0)] = np.nan
        b[(fb != 0) | ~np.isfinite(b) | (b <= 0)] = np.nan
        df = pd.DataFrame({"a": a, "b": b}, index=t)
        frames[sat] = df[~df.index.duplicated()].sort_index()
    return frames


def merge(frames, ref="16"):
    """Chain multiplicative scale factors onto the reference satellite's calibration."""
    scale = {ref: 1.0}
    order = [ref] + [s for s in PRIORITY if s != ref]
    for _ in range(len(order)):
        for s in order:
            if s in scale:
                continue
            for t, k in scale.items():
                j = frames[s].join(frames[t], lsuffix="1", rsuffix="2",
                                   how="inner").dropna(subset=["b1", "b2"])
                if len(j) < 200:
                    continue
                r = float((j.b2 / j.b1).replace([np.inf, -np.inf], np.nan).median())
                scale[s] = k * r
                break
    unscaled = [s for s in frames if s not in scale]
    if unscaled:
        print(f"  note: no overlap chain for g{', g'.join(unscaled)}; left on native scale")
        for s in unscaled:
            scale[s] = 1.0
    print("  scale onto the g%s calibration: " % ref
          + ", ".join(f"g{s} x{scale[s]:.3f}" for s in PRIORITY if s in scale))

    idx = pd.date_range(min(f.index.min() for f in frames.values()),
                        max(f.index.max() for f in frames.values()), freq="D")
    out = pd.DataFrame(index=idx, columns=["a", "b", "sat"], dtype=object)
    for s in PRIORITY:
        if s not in frames:
            continue
        f = frames[s].reindex(idx)
        take = out.b.isna() & f.b.notna()
        out.loc[take, "a"] = f.a[take] * scale[s]
        out.loc[take, "b"] = f.b[take] * scale[s]
        out.loc[take, "sat"] = s
    out["a"] = pd.to_numeric(out.a)
    out["b"] = pd.to_numeric(out.b)
    hand = out.sat.ne(out.sat.shift()) & out.sat.notna() & out.sat.shift().notna()
    steps = []
    for i in np.flatnonzero(hand.to_numpy()):
        w = slice(max(0, i - 15), min(len(out), i + 15))
        pre, post = out.b.iloc[w][:15].median(), out.b.iloc[w][15:].median()
        if np.isfinite(pre) and np.isfinite(post) and pre > 0:
            steps.append(post / pre)
    print(f"  {len(steps)} satellite handovers; median step in B across them "
          f"{np.median(steps):.3f} (IQR {np.percentile(steps,25):.2f}-{np.percentile(steps,75):.2f})")
    print(f"  merged daily series {out.index.min().date()} .. {out.index.max().date()}, "
          f"{100*out.b.notna().mean():.1f} % of days have a background value")
    return out


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap_.add_argument("--goes-dir", default="/Users/eunsupark/.claude/jobs/b613c17a/tmp/goes")
    ap_.add_argument("--lag-hours", type=int, default=48)
    args = ap_.parse_args()

    daily = merge(load_goes(args.goes_dir))

    tbl = pd.read_parquet(os.path.join(D, "data.parquet")).set_index("datetime").sort_index()
    grid = pd.date_range(tbl.index[0], tbl.index[-1], freq="30min")
    tbl = tbl.reindex(grid)
    tbl[WIND + ["ap30"]] = tbl[WIND + ["ap30"]].interpolate(limit=6).ffill().bfill()

    # broadcast the daily series onto the 30-min grid, lagged, in log space
    lag = pd.Timedelta(hours=args.lag_hours)
    xb = np.log10(daily.b.astype(float))
    xr = np.log10((daily.a / daily.b).astype(float).replace([np.inf, -np.inf], np.nan))
    for nm, ser in (("gx_b", xb), ("gx_r", xr)):
        s = ser.reindex(pd.date_range(daily.index.min(), grid.max(), freq="D")).ffill(limit=5)
        tbl[nm] = s.reindex(grid - lag, method="ffill").to_numpy()
        tbl[nm + "_d7"] = tbl[nm] - pd.Series(tbl[nm].to_numpy(), index=grid).shift(
            freq="7D").reindex(grid).to_numpy()
    GX = ["gx_b", "gx_r", "gx_b_d7", "gx_r_d7"]
    cov = tbl[GX].notna().mean().mean()
    tbl[GX] = tbl[GX].interpolate(limit=48).ffill().bfill()
    print(f"  X-ray channels on the 30-min grid, {100*cov:.1f} % populated before fill, "
          f"lagged {args.lag_hours} h\n")

    S = {c: tbl[c].to_numpy(float) for c in WIND + GX + ["ap30"]}
    pos = {t: i for i, t in enumerate(grid)}
    tr = pd.read_csv(os.path.join(D, "total_ap/train_index.csv"))
    tr["datetime"] = pd.to_datetime(tr["datetime"])
    tri = np.array([pos[t] for t in tr["datetime"] if PAST <= pos.get(t, -1) < len(grid) - OUT])
    with zipfile.ZipFile(os.path.join(R, POOLED, "validation", "best", "npz.zip")) as z:
        anchors = [str(np.asarray(np.load(io.BytesIO(z.read(n)), allow_pickle=True)["anchor"]))
                   for n in sorted(x for x in z.namelist() if x.endswith(".npz"))]
    tei = np.array([pos[t] for t in pd.to_datetime(pd.Series(anchors), format="%Y%m%d%H%M%S")])

    def past(idx, keys):
        return np.column_stack([S[c][idx[:, None] + np.arange(-PAST, 0)] for c in keys])

    def gx(idx):
        return np.column_stack([S[c][idx] for c in GX])

    Ytr = S["ap30"][tri[:, None] + np.arange(0, OUT)]
    Yte = S["ap30"][tei[:, None] + np.arange(0, OUT)]
    peak = Yte.max(1)
    storm = peak >= 47.5
    btr, bte = past(tri, ["ap30"] + WIND), past(tei, ["ap30"] + WIND)

    def rho(P, m):
        return float(np.mean([np.corrcoef(P[m, k], Yte[m, k])[0, 1] for k in range(OUT)]))

    allm = np.ones(len(tei), bool)
    print(f"  {'':44s} {'all':>7s} {'storm':>7s} {'quiet':>7s}")
    P0 = ridge(btr, Ytr)(bte)
    print(f"  {'OMNI only (the model as it stands)':44s} {rho(P0, allm):7.3f} "
          f"{rho(P0, storm):7.3f} {rho(P0, ~storm):7.3f}")
    P1 = ridge(np.hstack([btr, gx(tri)]), Ytr)(np.hstack([bte, gx(tei)]))
    print(f"  {'+ GOES X-ray background and ratio':44s} {rho(P1, allm):7.3f} "
          f"{rho(P1, storm):7.3f} {rho(P1, ~storm):7.3f}")
    print(f"  {'gain':44s} {rho(P1, allm)-rho(P0, allm):+7.3f} "
          f"{rho(P1, storm)-rho(P0, storm):+7.3f} {rho(P1, ~storm)-rho(P0, ~storm):+7.3f}")

    print(f"\n  storm anchors n={int(storm.sum())}; the precision floor on any of these")
    print("  differences is about 0.015 (block bootstrap over monthly blocks).")


if __name__ == "__main__":
    main()
