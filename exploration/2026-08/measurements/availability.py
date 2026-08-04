"""Why does forecast availability collapse during storms?

The prior operational diagnosis records 12.5 % coverage at ap >= 50 as the top remaining bug,
but the figure came from a session record and its data is not in the repo, and the deployed
forecast archive lives on the Windows host. So this re-derives the question from what is here:
20 archived NOAA RTSW snapshots (2026-07-02 .. 07-22, ~24 h of 1-min data each) and the GFZ
index from the database over the same period, which contains a storm reaching ap30 207.

Two things are measured, using the production source-selection and alignment code:

  1. Feed completeness per 30-min bin, stratified by what the index actually did. If NOAA drops
     out during storms, availability collapses however good the alignment policy is.
  2. Whether align() would produce a window at each anchor -- before and after the 2026-08 fill
     fix. The old behaviour essentially never refused, so if availability was genuinely 12.5 %
     the cause has to lie outside align().
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser("~/GitHub/njit-geoindex/geoindex-realtime"))
from src.fetch.noaa_swpc import (_MAG_RENAME, _PLASMA_RENAME, _numeric,      # noqa: E402
                                 _records_to_dataframe, _select_primary_source)

RNT = os.path.expanduser("~/Projects/GeoIndex/datasets/realtime_bad/rnt")
ACTIVITY = [(0, 15, "very quiet"), (15, 30, "quiet"), (30, 50, "unsettled"),
            (50, 100, "active"), (100, 1e9, "storm")]


def load_snapshots(pattern, rename, numeric_cols):
    """Union of every archived snapshot, deduped to one row per minute as production does."""
    frames = []
    for path in sorted(glob.glob(os.path.join(RNT, pattern))):
        records = json.load(open(path))
        df = _records_to_dataframe(records)
        df = _select_primary_source(df)
        df = df.rename(columns=rename)
        df = _numeric(df, numeric_cols)
        frames.append(df[["datetime"] + numeric_cols])
    out = pd.concat(frames).drop_duplicates(subset="datetime", keep="last")
    return out.sort_values("datetime").reset_index(drop=True)


plasma = load_snapshots("rtsw_wind_1m_*.json", _PLASMA_RENAME, ["np", "v", "t"])
mag = load_snapshots("rtsw_mag_1m_*.json", _MAG_RENAME, ["bx", "by", "bz", "bt"])
swpc = plasma.merge(mag, on="datetime", how="outer").sort_values("datetime")
swpc = swpc.reset_index(drop=True)
print(f"RTSW archive: {len(swpc)} minutes, {swpc['datetime'].min()} .. {swpc['datetime'].max()}")

# Which minutes the archive actually covers (snapshots are irregular, so there are holes
# between them that say nothing about the feed).
covered = pd.Series(True, index=swpc["datetime"])
grid_1m = pd.date_range(swpc["datetime"].min(), swpc["datetime"].max(), freq="1min")
have = swpc.set_index("datetime").reindex(grid_1m)
gap = have["v"].isna()
print(f"minutes on the grid: {len(grid_1m)}, present: {(~gap).sum()} "
      f"({100*(~gap).mean():.1f} %)")

# 30-min bins: how many of the 30 minutes arrived, and how many are usable (v and bz both set)
bins = have.resample("30min")
present = bins["v"].count()
usable = bins.apply(lambda g: int((g["v"].notna() & g["bz"].notna()).sum()))
frame = pd.DataFrame({"minutes_present": present, "minutes_usable": usable})

# GFZ index for the same bins
import subprocess
sql = ("select datetime, ap30 from hpo_hp30 where datetime >= '%s' and datetime <= '%s' "
       "order by 1" % (frame.index.min(), frame.index.max()))
raw = subprocess.run(["psql", "-d", "space_weather", "-tAF,", "-c", sql],
                     capture_output=True, text=True, check=True).stdout
idx = pd.DataFrame([r.split(",") for r in raw.strip().splitlines()],
                   columns=["datetime", "ap30"])
idx["datetime"] = pd.to_datetime(idx["datetime"])
idx["ap30"] = pd.to_numeric(idx["ap30"], errors="coerce")
frame = frame.join(idx.set_index("datetime"))
print(f"bins: {len(frame)}, with an index value: {frame['ap30'].notna().sum()}, "
      f"ap30 >= 50: {(frame['ap30'] >= 50).sum()}, max {frame['ap30'].max():.0f}\n")

# Only bins the archive was meant to cover: at least one minute arrived somewhere in the
# surrounding hour, otherwise we are measuring the snapshot schedule, not the feed.
window = frame["minutes_present"].rolling(3, center=True, min_periods=1).max()
inside = window > 0
print("(1) FEED COMPLETENESS by what the index did, over bins the archive covers")
print(f"{'activity':>14s} {'bins':>6s} {'mean minutes/30':>16s} {'bins >=90% full':>16s} "
       f"{'bins empty':>11s}")
for lo, hi, label in ACTIVITY:
    m = inside & frame["ap30"].between(lo, hi, inclusive="left")
    if not m.any():
        continue
    sub = frame[m]
    print(f"{label:>14s} {len(sub):6d} {sub['minutes_usable'].mean():16.1f} "
          f"{100*(sub['minutes_usable'] >= 27).mean():15.1f}% "
          f"{100*(sub['minutes_usable'] == 0).mean():10.1f}%")

print("\n(2) WOULD align() PRODUCE A WINDOW? Anchors stepped every 30 min across the archive.")
from src.pipeline.aggregate import aggregate_30min                       # noqa: E402
from src.pipeline.align import align, InsufficientDataError              # noqa: E402
import logging                                                            # noqa: E402
logging.disable(logging.WARNING)

hpo = idx.rename(columns={"ap30": "ap30"}).copy()
hpo["hp30"] = np.cbrt(hpo["ap30"] / 2.5)
LOOKBACK, ROLLBACK = 24, 4
anchors = pd.date_range(frame.index.min() + pd.Timedelta(hours=14), frame.index.max(),
                        freq="30min")
# Restrict to anchors whose entire 12 h window lies inside one snapshot's coverage. The
# snapshots are irregular and 22 % of the archive span falls between them; those hours never
# existed in the live feed, and an anchor whose window straddles one is measuring the archive's
# sampling schedule rather than the service.
import glob as _glob, json as _json
_spans = []
for _p in sorted(_glob.glob(os.path.join(RNT, "rtsw_wind_1m_*.json"))):
    _t = pd.to_datetime([r["time_tag"] for r in _json.load(open(_p))])
    _spans.append((_t.min(), _t.max()))
def _inside(lo, hi):
    return any(a <= lo and hi <= b for a, b in _spans)

rows = []
for now in anchors:
    now_run = now + pd.Timedelta(minutes=3)     # the service fires a few minutes past the bin
    lo = now - pd.Timedelta(minutes=30 * (LOOKBACK + ROLLBACK + 2))
    if not _inside(lo, now_run):
        rows.append((now, "outside archive", np.nan))
        continue
    sw_slice = swpc[(swpc["datetime"] >= lo) & (swpc["datetime"] <= now_run)]
    if sw_slice.empty:
        rows.append((now, "no data", np.nan))
        continue
    try:
        sw30 = aggregate_30min(sw_slice, start=lo.to_pydatetime(),
                               end=(now + pd.Timedelta(minutes=30)).to_pydatetime())
        r = align(sw30, hpo[hpo["datetime"] <= now_run], now=now_run,
                  lookback_steps=LOOKBACK, boundary_offset_minutes=30,
                  anchor_rollback_max_attempts=ROLLBACK)
        rows.append((now, "issued", (now - r.t_end) / pd.Timedelta(minutes=30)))
    except InsufficientDataError:
        rows.append((now, "refused", np.nan))
    except Exception as exc:                                              # noqa: BLE001
        rows.append((now, f"error: {type(exc).__name__}", np.nan))

res = pd.DataFrame(rows, columns=["anchor", "outcome", "rollback_steps"]).set_index("anchor")
res = res.join(frame[["ap30", "minutes_usable"]])
print(f"{'activity':>14s} {'anchors':>8s} {'issued':>8s} {'refused':>8s} {'no data':>8s} "
       f"{'availability':>13s}")
for lo, hi, label in ACTIVITY:
    m = res["ap30"].between(lo, hi, inclusive="left")
    if not m.any():
        continue
    sub = res[m]
    iss = (sub["outcome"] == "issued").sum()
    n_in = (sub["outcome"] != "outside archive").sum()
    print(f"{label:>14s} {n_in:8d} {iss:8d} {(sub['outcome']=='refused').sum():8d} "
          f"{(sub['outcome']=='no data').sum():8d} "
          f"{(100*iss/n_in if n_in else float('nan')):12.1f}%")
n_in = (res["outcome"] != "outside archive").sum()
print(f"{'ALL':>14s} {n_in:8d} {(res['outcome']=='issued').sum():8d} "
      f"{(res['outcome']=='refused').sum():8d} {(res['outcome']=='no data').sum():8d} "
      f"{100*(res['outcome']=='issued').sum()/n_in:12.1f}%")
print(f"({(res['outcome']=='outside archive').sum()} anchors excluded: their window straddles "
      f"a hole between snapshots)")
other = res[~res["outcome"].isin(["issued", "refused", "no data", "outside archive"])]
if len(other):
    print("\nunexpected outcomes:", other["outcome"].value_counts().to_dict())
print("\nrollback steps used when issued:",
      res.loc[res["outcome"] == "issued", "rollback_steps"].value_counts().sort_index().to_dict())
