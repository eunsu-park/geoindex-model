"""Re-index the solar-wind columns from bow-shock arrival time to L1 observation time.

OMNI HRO is time-shifted from the monitoring spacecraft to the bow shock nose, so a row
stamped t holds wind that has ALREADY arrived. Training on it hands the model no head start:
forecasting ap at t+30 min is a genuine forecast rather than a delayed observation.

That is not the operational situation. The monitor sees that wind ``timeshift_sec`` earlier --
a median of 53 minutes in the validation period, 48 during storms -- so at wall-clock t the
coming bow-shock wind is already known out to t + shift. The pipeline throws that interval
away, and NOAA RTSW timestamps (which the realtime service consumes) are observation time
anyway, so training in bow-shock time also leaves the served model reading its inputs on a
different clock from the one it was fitted on.

Re-indexing fixes both at once. Measured with a ridge on the same features, held out on
2022-2025:

    bow-shock time (current)      peak rho 0.706   AUC 0.8574   reproduction 0.563
    L1 observation time           peak rho 0.724   AUC 0.8678   reproduction 0.580
    upper bound (+1 h true wind)  peak rho 0.729   AUC 0.8675   reproduction 0.588

So the re-index captures about 83 % of what the interval is worth, with no propagation step at
serve time. The AUC movement is the part worth noting: every other intervention tried in this
investigation left AUC inside +/-0.001, because they changed calibration rather than
discrimination. This one moves it.

Only the wind moves. ap30 and hp30 are ground indices measured at real time; shifting them
would be a bug rather than a feature.

The shift is read from ``omni_high_resolution.timeshift_sec``, cached to CSV next to the
output so a run without database access stays reproducible.

Usage:
    python analysis/make_observation_time_table.py --data-root ~/Projects/GeoIndex/datasets
    python analysis/make_observation_time_table.py --data-root ... --verify

If this is adopted, the logic belongs in geoindex-data's ``sw_30min`` build rather than here.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

# Every solar-wind column, all three statistics. These are propagated quantities.
WIND_VARS = ["v", "np", "t", "bx", "by", "bz", "bt"]
WIND_COLS = [f"{v}_{s}" for v in WIND_VARS for s in ("avg", "min", "max")]
# Ground indices, measured at real time. Never shifted.
GROUND_COLS = ["ap30", "hp30"]

SHIFT_SQL = """
SELECT date_trunc('hour', datetime)
         + interval '30 min' * floor(extract(minute from datetime)/30) AS dt30,
       avg(timeshift_sec) AS timeshift_sec
FROM omni_high_resolution
WHERE datetime >= '{start}' AND datetime < '{end}'
GROUP BY 1 ORDER BY 1
"""


def load_shift(cache_path: str, db: str, start: str, end: str) -> pd.Series:
    """Read the 30-min mean propagation delay, from cache if present else from the database.

    Args:
        cache_path: CSV cache written on first use.
        db: PostgreSQL database name.
        start: Inclusive lower bound, ``YYYY-MM-DD``.
        end: Exclusive upper bound.

    Returns:
        Series of seconds indexed by 30-minute bin start.

    Raises:
        RuntimeError: If the cache is absent and the database cannot be read.
    """
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["dt30"])
        print(f"shift: cache {cache_path} ({len(df)} rows)")
    else:
        try:
            import subprocess
            sql = SHIFT_SQL.format(start=start, end=end).replace("\n", " ")
            out = subprocess.run(
                ["psql", "-d", db, "-tAF,", "-c", sql],
                capture_output=True, text=True, check=True).stdout
        except Exception as exc:                                    # noqa: BLE001
            raise RuntimeError(
                f"no cache at {cache_path} and database '{db}' unreadable: {exc}") from exc
        rows = [r.split(",") for r in out.strip().splitlines() if r]
        df = pd.DataFrame(rows, columns=["dt30", "timeshift_sec"])
        df["dt30"] = pd.to_datetime(df["dt30"])
        df["timeshift_sec"] = pd.to_numeric(df["timeshift_sec"], errors="coerce")
        df.to_csv(cache_path, index=False)
        print(f"shift: queried {db}, cached to {cache_path} ({len(df)} rows)")
    return df.set_index("dt30")["timeshift_sec"]


def reindex_to_observation_time(table: pd.DataFrame, shift_s: np.ndarray) -> pd.DataFrame:
    """Move the wind columns onto the clock of the spacecraft that observed them.

    Row i holds wind that reached the bow shock at ``index[i]``; the monitor saw it
    ``shift_s[i]`` seconds earlier. Stamping each sample with that observation time and
    resampling back onto the regular grid puts, at every position, the wind a live monitor
    would have in hand at that moment.

    Args:
        table: 30-minute table indexed by a regular DatetimeIndex.
        shift_s: Propagation delay in seconds, aligned to ``table.index``.

    Returns:
        A copy with the wind columns re-indexed and the ground indices untouched.
    """
    grid = table.index
    observed_at = grid - pd.to_timedelta(shift_s, unit="s")
    wind = pd.DataFrame({c: table[c].to_numpy() for c in WIND_COLS}, index=observed_at)
    # A varying shift can map two arrivals onto one observation slot; the later sample is the
    # one a monitor would actually be reporting.
    wind = wind[~wind.index.duplicated(keep="last")].sort_index()
    moved = wind.reindex(grid, method="nearest", tolerance=pd.Timedelta("30min"))
    gap = float(moved[WIND_COLS[0]].isna().mean())
    print(f"re-index: {gap:.2%} of grid points had no observation within 30 min; "
          f"filled by interpolation")
    moved = moved.interpolate(limit=4).ffill().bfill()
    out = table.copy()
    out[WIND_COLS] = moved[WIND_COLS].to_numpy()
    return out


def verify(original: pd.DataFrame, moved: pd.DataFrame, test_from: str) -> None:
    """Refit the ridge on both tables and print the peak-forecast comparison."""
    feats = ["v_avg", "np_avg", "t_avg", "bx_avg", "by_avg", "bz_avg", "bt_avg"]
    ap = original["ap30"].to_numpy(float)
    n = len(original)
    n_a = n - 24
    idx = np.arange(n_a)
    times = original.index[:n_a]
    tgt = np.lib.stride_tricks.sliding_window_view(ap, 24)[:n_a].max(axis=1)

    def lags(series):
        out = []
        for b in range(24):
            lo = idx - 24 + b
            col = np.full(n_a, np.nan)
            ok = lo >= 0
            col[ok] = series[lo[ok]]
            out.append(col)
        return out

    ap_lags = lags(ap)
    print(f"\n{'table':28s} {'peak rho':>9s} {'peak MAE':>9s} {'repro>=100':>11s}")
    for name, tbl in (("bow-shock time (input)", original), ("observation time (output)", moved)):
        X = np.column_stack([c for f in feats for c in lags(tbl[f].to_numpy(float))] + ap_lags)
        ok = ~np.isnan(X).any(1)
        te = ok & (times >= pd.Timestamp(test_from))
        tr = ok & ~te
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd < 1e-9] = 1.0
        Z = (X[tr] - mu) / sd
        yb = tgt[tr].mean()
        w = np.linalg.solve(Z.T @ Z + 100 * np.eye(Z.shape[1]), Z.T @ (tgt[tr] - yb))
        p = ((X[te] - mu) / sd) @ w + yb
        o = tgt[te]
        hi = o >= 100
        print(f"{name:28s} {np.corrcoef(p, o)[0, 1]:9.3f} {np.abs(p - o).mean():9.2f} "
              f"{p[hi].mean() / o[hi].mean():11.3f}")


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap_.add_argument("--data-root", default="~/Projects/GeoIndex/datasets")
    ap_.add_argument("--table", default="data.parquet")
    ap_.add_argument("--out", default="data_obs.parquet")
    ap_.add_argument("--shift-cache", default="omni_timeshift_30min.csv")
    ap_.add_argument("--db", default="space_weather")
    ap_.add_argument("--verify", action="store_true",
                     help="refit the ridge on both tables and print the comparison")
    ap_.add_argument("--test-from", default="2022-01-01")
    args = ap_.parse_args()

    root = os.path.expanduser(args.data_root)
    src = pd.read_parquet(os.path.join(root, args.table))
    src = src.set_index("datetime").sort_index()
    grid = pd.date_range(src.index[0], src.index[-1], freq="30min")
    src = src.reindex(grid)
    print(f"table: {args.table}  {src.shape}  {grid[0]} .. {grid[-1]}")

    missing = [c for c in WIND_COLS + GROUND_COLS if c not in src.columns]
    if missing:
        raise SystemExit(f"table is missing expected columns: {missing}")

    shift = load_shift(os.path.join(root, args.shift_cache), args.db,
                       str(grid[0].date()), str((grid[-1] + pd.Timedelta("1D")).date()))
    shift = shift.reindex(grid).interpolate(limit=12).ffill().bfill()
    s = shift.to_numpy(float)
    print(f"shift: median {np.median(s)/60:.0f} min, "
          f"p10 {np.percentile(s, 10)/60:.0f}, p90 {np.percentile(s, 90)/60:.0f}")

    # Interpolate only for the re-index; the written table keeps the source's own gaps in the
    # ground indices so downstream missing-data handling is unchanged.
    filled = src.copy()
    filled[WIND_COLS] = filled[WIND_COLS].interpolate(limit=6).ffill().bfill()
    moved = reindex_to_observation_time(filled, s)
    moved[GROUND_COLS] = src[GROUND_COLS]

    out_path = os.path.join(root, args.out)
    moved.reset_index(names="datetime").to_parquet(out_path, index=False)
    print(f"\nwrote {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")
    print("Ground indices untouched:",
          all(src[c].equals(moved[c]) for c in GROUND_COLS))

    if args.verify:
        verify(filled, moved, args.test_from)


if __name__ == "__main__":
    main()
