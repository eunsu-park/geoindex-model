"""Do X-ray FLARES -- the eruption signal, not the background level -- add anything to ap30?

The companion script tested the GOES daily X-ray *background* and got a null. That was the wrong
observable for the mechanism. Chakraborty & Morley (2020) lag their X-ray inputs by 48 hours
because the chain is flare -> CME -> storm: an eruption two days ago means a disturbance may be
arriving now. The daily background measures the quiescent corona, not eruptions.

This uses the flare summary product, `sci_xrsf-l2-flsum`, which records one row per flare phase
(EVENT_START / EVENT_PEAK / EVENT_END / POST_EVENT) with the peak flux, the GOES class and the
time-integrated flux. Peaks are extracted and binned into lag windows chosen to match CME transit
rather than to match a statistical convenience:

    [t-24 h, t)    an eruption that would only just be arriving, if at all
    [t-48 h, t-24 h)
    [t-72 h, t-48 h)
    [t-120 h, t-72 h)   the slow tail of the transit-time distribution

For each window: the log of the summed integrated flux, the count of >= M and >= X class events,
and the log of the largest peak flux. Sixteen features.

If this is null too, then X-ray data as a whole carries nothing for a 12-hour ap30 forecast, and
the difference from Chakraborty & Morley is the horizon (they forecast Kp 3 hours ahead) or the
architecture (their X-ray gain appears inside a storm-routed branch, which a pooled linear model
cannot reproduce).

RESULT: null, and the decisive line is the control. Flare activity adds -0.006 to the full model
and -0.004 to a model given NO solar wind at all (ap30 history only, 0.512 -> 0.508). If the
eruption signal carried anything about the next twelve hours that the wind does not, it would
appear against the wind-free baseline. It does not, so the information is absent rather than
redundant.

Taken with the background null, X-ray data as a whole carries nothing for a 12-hour ap30 forecast.

The physical reason is visible in the feature itself: during solar maximum, 42 % of anchors have
an M-class flare somewhere in each 24-hour lag window. "A flare happened two days ago" is almost
always true and therefore almost uninformative. What would discriminate is whether a CME left and
whether it is aimed at Earth, and X-ray flux says neither -- that is a coronagraph observable,
which the imagery bound puts at +0.01 to +0.02 at this horizon after arrival-time error.

The standing caveat on the comparison with Chakraborty & Morley: they forecast Kp three hours
ahead, and their X-ray gain appears inside a storm-routed branch (an LSTM classifier feeding two
regime-specific deep Gaussian processes). A pooled linear model cannot reproduce a conditional
signal. This null is on the pooled linear form.

Usage:
    python goes_flares.py [--goes-dir ...]
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
WINDOWS = [(0, 24), (24, 48), (48, 72), (72, 120)]      # hours before the anchor


def ridge(X, Y, alpha=100.0):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    yb = Y.mean(0)
    W = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (Y - yb))
    return lambda Xn: ((Xn - mu) / sd) @ W + yb


def load_flares(d):
    """One row per flare peak, deduplicated across overlapping satellites."""
    import netCDF4 as nc
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "sci_xrsf-l2-flsum_*.nc"))):
        sat = re.search(r"_g(\d+)_", f).group(1)
        ds = nc.Dataset(f)
        st = np.asarray(ds["status"][:]).astype(str)
        keep = st == "EVENT_PEAK"
        t = pd.to_datetime(nc.num2date(np.asarray(ds["time"][:], float)[keep],
                                       ds["time"].units, only_use_cftime_datetimes=False))
        pk = np.asarray(ds["xrsb_flux"][:], float)[keep]
        ig = np.asarray(ds["integrated_flux"][:], float)[keep]
        cl = np.asarray(ds["flare_class"][:]).astype(str)[keep]
        ds.close()
        rows.append(pd.DataFrame({"t": pd.DatetimeIndex(t), "peak": pk, "integ": ig,
                                  "cls": np.char.upper(cl.astype(str)), "sat": sat}))
        print(f"    g{sat}  {len(t):6d} peaks  {pd.DatetimeIndex(t).min().date()} .. "
              f"{pd.DatetimeIndex(t).max().date()}")
    fl = pd.concat(rows).sort_values("t")
    fl = fl[np.isfinite(fl.peak) & (fl.peak > 0)]
    # two satellites see the same flare; keep one peak per 10-minute bucket
    fl["bucket"] = fl.t.dt.floor("10min")
    fl = fl.sort_values("peak", ascending=False).drop_duplicates("bucket").sort_values("t")
    fl["is_m"] = np.char.startswith(fl.cls.to_numpy().astype(str), "M") | \
        np.char.startswith(fl.cls.to_numpy().astype(str), "X")
    fl["is_x"] = np.char.startswith(fl.cls.to_numpy().astype(str), "X")
    fl["integ"] = np.where(np.isfinite(fl.integ) & (fl.integ > 0), fl.integ, 0.0)
    print(f"  {len(fl):,} distinct flare peaks after deduplication, "
          f"{int(fl.is_m.sum()):,} >= M, {int(fl.is_x.sum()):,} >= X")
    return fl


def flare_features(anchor_times, fl):
    """Summed and extreme flare activity in each transit-matched lag window."""
    t = fl.t.to_numpy()
    order = np.argsort(t)
    t = t[order]
    peak = fl.peak.to_numpy()[order]
    integ = fl.integ.to_numpy()[order]
    is_m = fl.is_m.to_numpy()[order].astype(float)
    is_x = fl.is_x.to_numpy()[order].astype(float)
    cum = {k: np.concatenate([[0.0], np.cumsum(v)]) for k, v in
           (("integ", integ), ("m", is_m), ("x", is_x))}
    at = anchor_times.to_numpy()
    out = []
    for lo, hi in WINDOWS:
        a = np.searchsorted(t, at - np.timedelta64(hi, "h"))
        b = np.searchsorted(t, at - np.timedelta64(lo, "h"))
        col = [np.log10(1.0 + (cum["integ"][b] - cum["integ"][a])),
               cum["m"][b] - cum["m"][a],
               cum["x"][b] - cum["x"][a]]
        mx = np.zeros(len(at))
        for i in range(len(at)):
            mx[i] = peak[a[i]:b[i]].max() if b[i] > a[i] else 0.0
        col.append(np.log10(1e-9 + mx))
        out.append(np.column_stack(col))
    return np.hstack(out)


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap_.add_argument("--goes-dir", default="/Users/eunsupark/.claude/jobs/b613c17a/tmp/goes")
    args = ap_.parse_args()

    print("  flare peaks per satellite:")
    fl = load_flares(args.goes_dir)

    tbl = pd.read_parquet(os.path.join(D, "data.parquet")).set_index("datetime").sort_index()
    grid = pd.date_range(tbl.index[0], tbl.index[-1], freq="30min")
    tbl = tbl.reindex(grid)
    tbl[WIND + ["ap30"]] = tbl[WIND + ["ap30"]].interpolate(limit=6).ffill().bfill()
    S = {c: tbl[c].to_numpy(float) for c in WIND + ["ap30"]}
    pos = {t: i for i, t in enumerate(grid)}

    tr = pd.read_csv(os.path.join(D, "total_ap/train_index.csv"))
    tr["datetime"] = pd.to_datetime(tr["datetime"])
    tri = np.array([pos[t] for t in tr["datetime"] if PAST <= pos.get(t, -1) < len(grid) - OUT])
    with zipfile.ZipFile(os.path.join(R, POOLED, "validation", "best", "npz.zip")) as z:
        anchors = [str(np.asarray(np.load(io.BytesIO(z.read(n)), allow_pickle=True)["anchor"]))
                   for n in sorted(x for x in z.namelist() if x.endswith(".npz"))]
    tei = np.array([pos[t] for t in pd.to_datetime(pd.Series(anchors), format="%Y%m%d%H%M%S")])

    Ftr = flare_features(pd.DatetimeIndex(grid[tri]), fl)
    Fte = flare_features(pd.DatetimeIndex(grid[tei]), fl)
    print(f"\n  {Ftr.shape[1]} flare features; share of anchors with at least one M-class flare")
    print(f"  in the window: " + ", ".join(
        f"{lo}-{hi}h {100*np.mean(Fte[:, 4*i+1] > 0):.0f} %" for i, (lo, hi) in enumerate(WINDOWS)))

    def past(idx):
        return np.column_stack([S[c][idx[:, None] + np.arange(-PAST, 0)]
                                for c in ["ap30"] + WIND])

    Ytr = S["ap30"][tri[:, None] + np.arange(0, OUT)]
    Yte = S["ap30"][tei[:, None] + np.arange(0, OUT)]
    storm = Yte.max(1) >= 47.5
    btr, bte = past(tri), past(tei)

    def rho(P, m):
        return float(np.mean([np.corrcoef(P[m, k], Yte[m, k])[0, 1] for k in range(OUT)]))

    allm = np.ones(len(tei), bool)
    print(f"\n  {'':44s} {'all':>7s} {'storm':>7s} {'quiet':>7s}")
    P0 = ridge(btr, Ytr)(bte)
    print(f"  {'OMNI only':44s} {rho(P0, allm):7.3f} {rho(P0, storm):7.3f} "
          f"{rho(P0, ~storm):7.3f}")
    P1 = ridge(np.hstack([btr, Ftr]), Ytr)(np.hstack([bte, Fte]))
    print(f"  {'+ X-ray flare activity, 1-5 days back':44s} {rho(P1, allm):7.3f} "
          f"{rho(P1, storm):7.3f} {rho(P1, ~storm):7.3f}")
    print(f"  {'gain':44s} {rho(P1, allm)-rho(P0, allm):+7.3f} "
          f"{rho(P1, storm)-rho(P0, storm):+7.3f} {rho(P1, ~storm)-rho(P0, ~storm):+7.3f}")

    print("\n  and flare activity ALONE, with no solar wind at all:")
    P2 = ridge(np.hstack([S["ap30"][tri[:, None] + np.arange(-PAST, 0)], Ftr]), Ytr)(
        np.hstack([S["ap30"][tei[:, None] + np.arange(-PAST, 0)], Fte]))
    P3 = ridge(S["ap30"][tri[:, None] + np.arange(-PAST, 0)], Ytr)(
        S["ap30"][tei[:, None] + np.arange(-PAST, 0)])
    print(f"  {'ap30 history only':44s} {rho(P3, allm):7.3f} {rho(P3, storm):7.3f} "
          f"{rho(P3, ~storm):7.3f}")
    print(f"  {'ap30 history + flare activity':44s} {rho(P2, allm):7.3f} "
          f"{rho(P2, storm):7.3f} {rho(P2, ~storm):7.3f}")
    print(f"  {'gain':44s} {rho(P2, allm)-rho(P3, allm):+7.3f} "
          f"{rho(P2, storm)-rho(P3, storm):+7.3f} {rho(P2, ~storm)-rho(P3, ~storm):+7.3f}")

    print(f"\n  storm anchors n={int(storm.sum())}; precision floor about 0.015.")
    print("  The second block matters: if flares add nothing even to a model with no wind at all,")
    print("  the information is absent rather than redundant.")


if __name__ == "__main__":
    main()
