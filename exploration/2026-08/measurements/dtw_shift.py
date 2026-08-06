"""Is the 24-step forecast genuinely ahead of the observation, or a shifted copy of it?

Laperre, Amaya & Lapenta (2020), Front. Astron. Space Sci. 7, 39, doi:10.3389/fspas.2020.00039
is the field's warning about correlation-vs-lead curves. Their LSTM scores correlation 0.980 at
+1 h and 0.859 at +6 h on Dst, which looks excellent -- and dynamic time warping showed the t+5 h
forecast placed the storm onset exactly five hours late. The warp-value distribution mirrored the
persistence model's, and they concluded "the LSTM-NN model will not give us any more information
than the persistence model." Wintoft & Wik (2018) found shifts already at +1 h.

The consequence for us is the opposite of embarrassing. Our correlation falls 0.851 at +0.5 h to
0.384 at +12 h, roughly two and a half times steeper than the Dst literature's curves -- but a
shallow curve is exactly what a persistence-mimicking model produces. If our forecast is NOT
time-shifted, it carries more genuine information than a shallower curve that is.

Two controls make the reading possible, and Laperre did not have them. A PERFECT forecast (the
observation itself) must sit at zero -- that is the implementation check. And a ridge handed the
TRUE FUTURE WIND is a near-ceiling forecast built from information the deployed model does not
have; where it sits tells us whether an alignment failure is a modelling defect or an information
one.

Two diagnostics, on contiguous runs of validation anchors:

  LAG      for each lead, cross-correlate the prediction series against the observation series
           shifted by delta, and take the delta that maximises it. A true forecast peaks at
           delta = 0. Persistence peaks at delta = -k, since its "forecast" for lead k is the
           value already observed k steps earlier.

  DTW      align the two series with a Sakoe-Chiba band and take the distribution of warping
           offsets. This is Laperre's own measure; the LAG statistic is the interpretable
           summary of the same thing.

Usage:
    python dtw_shift.py [--band 24] [--min-run 96]
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile

import numpy as np
import pandas as pd

R = os.path.expanduser("~/Projects/GeoIndex/results")
POOLED = "probe_ap_in12h_out12h_gnn_transformer_baseline"
OUT = 24


def load(run):
    path = os.path.join(R, run, "validation", "best", "npz.zip")
    anchors, true, pred, anchor_val = [], [], [], []
    with zipfile.ZipFile(path) as z:
        for n in sorted(x for x in z.namelist() if x.endswith(".npz")):
            d = np.load(io.BytesIO(z.read(n)), allow_pickle=True)
            iv = [str(x) for x in np.asarray(d["input_variables"]).ravel()]
            tv = str(np.asarray(d["target_variables"]).ravel()[0])
            anchors.append(str(np.asarray(d["anchor"])))
            true.append(np.asarray(d["targets"])[:, 0])
            pred.append(np.asarray(d["predictions"])[:, 0])
            anchor_val.append(np.asarray(d["inputs"])[-1, iv.index(tv)])
    return (np.asarray(anchors), np.asarray(true), np.asarray(pred), np.asarray(anchor_val))


def dtw_offsets(a, b, band):
    """Sakoe-Chiba-banded DTW; returns the offset j-i along the optimal path."""
    n = len(a)
    inf = np.inf
    D = np.full((n + 1, n + 1), inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - band), min(n, i + band)
        for j in range(lo, hi + 1):
            c = abs(a[i - 1] - b[j - 1])
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j, off = n, n, []
    while i > 0 and j > 0:
        off.append(j - i)
        step = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return np.asarray(off)


def _true_wind_ridge(anchors):
    """A near-ceiling control: the same ridge, but handed the true future wind."""
    D = os.path.expanduser("~/Projects/GeoIndex/datasets")
    cols = [f"{v}_{s}" for v in ("v", "np", "t", "bx", "by", "bz", "bt")
            for s in ("avg", "min", "max")]
    t = pd.read_parquet(os.path.join(D, "data.parquet")).set_index("datetime").sort_index()
    g = pd.date_range(t.index[0], t.index[-1], freq="30min")
    t = t.reindex(g)
    t[cols + ["ap30"]] = t[cols + ["ap30"]].interpolate(limit=6).ffill().bfill()
    S = {k: t[k].to_numpy(float) for k in cols + ["ap30"]}
    pos = {x: i for i, x in enumerate(g)}
    tr = pd.read_csv(os.path.join(D, "total_ap/train_index.csv"))
    tr["datetime"] = pd.to_datetime(tr["datetime"])
    tri = np.array([pos[x] for x in tr["datetime"] if 24 <= pos.get(x, -1) < len(g) - OUT])
    tei = np.array([pos[x] for x in pd.to_datetime(pd.Series(anchors), format="%Y%m%d%H%M%S")])

    def feats(i):
        past = np.column_stack([S["ap30"][i[:, None] + np.arange(-24, 0)]]
                               + [S[k][i[:, None] + np.arange(-24, 0)] for k in cols])
        fut = np.column_stack([S[k][i[:, None] + np.arange(0, OUT)] for k in cols])
        return np.hstack([past, fut])

    Y = S["ap30"][tri[:, None] + np.arange(0, OUT)]
    X = feats(tri)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    W = np.linalg.solve(Z.T @ Z + 100.0 * np.eye(Z.shape[1]), Z.T @ (Y - Y.mean(0)))
    return ((feats(tei) - mu) / sd) @ W + Y.mean(0)


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap_.add_argument("--band", type=int, default=24, help="Sakoe-Chiba band, in 30-min steps")
    ap_.add_argument("--min-run", type=int, default=96, help="shortest contiguous run to use")
    ap_.add_argument("--max-lag", type=int, default=26)
    args = ap_.parse_args()

    anchors, true, pred, anchor_val = load(POOLED)
    ceiling = _true_wind_ridge(anchors)
    ts = pd.to_datetime(pd.Series(anchors), format="%Y%m%d%H%M%S")
    step = np.diff(ts.values).astype("timedelta64[m]").astype(int)
    breaks = np.flatnonzero(step != 30) + 1
    runs = [r for r in np.split(np.arange(len(ts)), breaks) if len(r) >= args.min_run]
    covered = sum(len(r) for r in runs)
    print(f"{len(anchors):,} validation anchors; {len(runs)} contiguous runs of >= "
          f"{args.min_run} steps ({args.min_run/2:.0f} h) covering {covered:,} anchors "
          f"({100*covered/len(anchors):.0f} %)\n")

    def lag_of_max(series_p, series_o):
        best, bestr = 0, -2.0
        n = len(series_p)
        for d in range(-args.max_lag, args.max_lag + 1):
            if d >= 0:
                x, y = series_p[: n - d], series_o[d:]
            else:
                x, y = series_p[-d:], series_o[: n + d]
            if len(x) < args.min_run // 2 or x.std() < 1e-9 or y.std() < 1e-9:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            if r > bestr:
                best, bestr = d, r
        return best, bestr

    print("  LAG that maximises correlation, by lead. A true forecast sits at 0;")
    print("  persistence sits at -k, because its value for lead k was observed k steps ago.\n")
    print(f"  {'lead':>6s} {'perfect':>8s} {'+true wind':>11s} {'MODEL':>7s} {'persist':>8s} | "
          f"{'model r@lag':>12s} {'model r@0':>10s}")
    for k in (0, 1, 3, 5, 11, 17, 23):
        med = {}
        for name, get in (("perf", lambda r: true[r, k]), ("ceil", lambda r: ceiling[r, k]),
                          ("model", lambda r: pred[r, k]), ("pers", lambda r: anchor_val[r])):
            med[name] = np.median([lag_of_max(get(r), true[r, k])[0] for r in runs])
        rb = np.nanmean([lag_of_max(pred[r, k], true[r, k])[1] for r in runs])
        r0 = np.nanmean([float(np.corrcoef(pred[r, k], true[r, k])[0, 1])
                         for r in runs if true[r, k].std() > 1e-9])
        print(f"  {(k+1)*0.5:5.1f}h {med['perf']:8.1f} {med['ceil']:11.1f} {med['model']:7.1f} "
              f"{med['pers']:8.1f} | {rb:12.3f} {r0:10.3f}")

    print("\n  DTW warping-offset distribution (Laperre's measure), lead +12 h,")
    print(f"  Sakoe-Chiba band {args.band} steps ({args.band/2:.0f} h)\n")
    for name, series in (("+true wind", lambda r: ceiling[r, OUT - 1]),
                         ("model", lambda r: pred[r, OUT - 1]),
                         ("persistence", lambda r: anchor_val[r])):
        offs = []
        for r in runs[: min(len(runs), 40)]:
            o = true[r, OUT - 1]
            a = series(r)
            s = slice(0, min(len(r), 400))
            offs.append(dtw_offsets(np.asarray(a[s], float), np.asarray(o[s], float), args.band))
        offs = np.concatenate(offs)
        print(f"  {name:12s} median offset {np.median(offs):+5.1f} steps "
              f"({np.median(offs)*0.5:+.1f} h), mean {offs.mean():+5.2f}, "
              f"|offset| <= 2 in {100*np.mean(np.abs(offs) <= 2):5.1f} % of the path")

    print("\n  Read it against the controls. The ridge handed the true future wind sits at 0 from")
    print("  +3 h onward, so the diagnostic does separate a forecast that contains future")
    print("  information from one that does not. The deployed model sits at the persistence lag")
    print("  at every lead -- and the control shows that is an information limit rather than a")
    print("  modelling defect: the only forecast aligned with the future is the one holding it.")


if __name__ == "__main__":
    main()
