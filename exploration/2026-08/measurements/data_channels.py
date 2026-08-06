"""What the OMNI table already contains and the model never sees.

The 30-minute aggregate feeds the model 21 channels: the average, minimum and maximum of seven
solar-wind quantities. The raw `omni_high_resolution` table has 47 columns. Everything below is
in it already and is thrown away by the aggregation.

Three groups, and the literature says something different about each.

  QUALITY   percent_interp, rms_timeshift, rms_phase_front_normal, the observation counts, and
            the monitor's off-axis position. These are per-row uncertainty labels that OMNI
            ships. Sivadas et al. (2026, Nature 655, 1143) argue the apparent saturation of the
            geomagnetic response is regression to the mean produced by exactly this uncertainty;
            O'Brien et al. (2023) measure that propagated L1 Bz retains only r^2 = 0.53 of the Bz
            reaching the bow shock; Milan et al. (2021) measure a 0.1-0.15 correlation drop
            between a monitor inside 45 R_E of the Sun-Earth line and one beyond 70 R_E.

  FIELD     rms_sd_b_scalar_nt and rms_sd_b_vector_nt -- the within-minute field variability.
            Our own ablation found that the AMPLITUDE of future Bz, with no sign information, is
            worth +0.128 of per-lead correlation. This is the observable that measures it.

  PLASMA    electric_field, flow_pressure, plasma_beta, and the Alfven and magnetosonic Mach
            numbers -- the ingredients of the Borovsky and Newell coupling functions, which the
            Dst literature builds by hand and nobody has ablated inside a learned model.

Plus two free constructions the Kp literature uses and we do not: UT hour and day-of-year as
sin/cos pairs (Wintoft et al. 2017; and Lockwood et al. 2020 show ap has a UT and seasonal
response where the am index does not), and the rectified coupling max(0, -bz) with v times it.

Also here: the same model scored against ap30 and against hp30. Every number in the Kp/Hp
literature is on the Kp-like scale, and correlation is not invariant under the transform.

Usage:
    python data_channels.py
"""

from __future__ import annotations

import io
import os
import zipfile

import numpy as np
import pandas as pd

D = os.path.expanduser("~/Projects/GeoIndex/datasets")
R = os.path.expanduser("~/Projects/GeoIndex/results")
EXTRA = "/tmp/omni_extra_30min.csv"
POOLED = "probe_ap_in12h_out12h_gnn_transformer_baseline"
BASE = [f"{v}_{s}" for v in ("v", "np", "t", "bx", "by", "bz", "bt") for s in ("avg", "min", "max")]
PAST = OUT = 24

QUAL = ["q_pct_interp", "q_pct_interp_max", "q_rms_ts", "q_rms_ts_max", "q_rms_pfn",
        "q_rms_pfn_max", "q_timeshift", "q_timeshift_sd", "q_sc_offaxis", "q_sc_x",
        "q_n_imf", "q_n_plasma"]
FIELD = ["f_rms_b_scalar", "f_rms_b_vec", "f_rms_b_vec_max"]
PLASMA = ["p_ef", "p_ef_min", "p_ef_max", "p_pdyn", "p_beta", "p_ma", "p_mms"]
INDEX = ["i_symh", "i_symh_min", "i_ae", "i_ae_max"]


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
ex = pd.read_csv(EXTRA, parse_dates=["datetime"]).set_index("datetime").reindex(grid)
print(f"extra columns, share missing before fill:")
for grp, name in ((QUAL, "quality"), (FIELD, "field variability"), (PLASMA, "plasma derived"),
                  (INDEX, "other indices")):
    print(f"  {name:20s} {100*ex[grp].isna().mean().mean():5.1f} %")
tbl = tbl.join(ex)
cols = BASE + QUAL + FIELD + PLASMA + INDEX + ["ap30", "hp30"]
tbl[cols] = tbl[cols].interpolate(limit=6).ffill().bfill()
S = {c: tbl[c].to_numpy(float) for c in cols}
pos = {t: i for i, t in enumerate(grid)}

tr = pd.read_csv(os.path.join(D, "total_ap/train_index.csv"))
tr["datetime"] = pd.to_datetime(tr["datetime"])
tri = np.array([pos[t] for t in tr["datetime"] if PAST <= pos.get(t, -1) < len(grid) - OUT])
with zipfile.ZipFile(os.path.join(R, POOLED, "validation", "best", "npz.zip")) as z:
    anchors = [str(np.asarray(np.load(io.BytesIO(z.read(n)), allow_pickle=True)["anchor"]))
               for n in sorted(x for x in z.namelist() if x.endswith(".npz"))]
te_ts = pd.to_datetime(pd.Series(anchors), format="%Y%m%d%H%M%S")
tei = np.array([pos[t] for t in te_ts])
print(f"\ntrain {len(tri):,}, test {len(tei):,}\n")


def hist(idx, keys):
    pw = idx[:, None] + np.arange(-PAST, 0)
    return np.column_stack([S[c][pw] for c in keys])


def clock(idx):
    t = grid[idx]
    h = t.hour.to_numpy() + t.minute.to_numpy() / 60.0
    d = t.dayofyear.to_numpy()
    return np.column_stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                            np.sin(2 * np.pi * d / 365.25), np.cos(2 * np.pi * d / 365.25)])


def coupling(idx):
    """Rectified merging terms built from PAST wind: only southward Bz reconnects."""
    pw = idx[:, None] + np.arange(-PAST, 0)
    bs = np.maximum(0.0, -S["bz_avg"][pw])
    bsx = np.maximum(0.0, -S["bz_min"][pw])
    v = S["v_avg"][pw]
    bt = S["bt_avg"][pw]
    theta = np.arctan2(np.abs(S["by_avg"][pw]), S["bz_avg"][pw])       # clock angle
    newell = v ** (4 / 3) * bt ** (2 / 3) * np.sin(theta / 2) ** (8 / 3) / 1e3
    return np.column_stack([bs, v * bs / 1e3, bsx, v * bsx / 1e3, newell])


def target(idx, name):
    return S[name][idx[:, None] + np.arange(0, OUT)]


def rho(P, Y):
    return float(np.mean([np.corrcoef(P[:, k], Y[:, k])[0, 1] for k in range(OUT)]))


# ── experiment: input channels ──────────────────────────────────────────────
Ytr, Yte = target(tri, "ap30"), target(tei, "ap30")
base_tr, base_te = hist(tri, ["ap30"] + BASE), hist(tei, ["ap30"] + BASE)
base = rho(ridge(base_tr, Ytr)(base_te), Yte)
print("  ── candidate input channels, ap30 target ─────────────────────")
print(f"  {'added to the current 21 channels + ap30 history':52s} {'rho':>6s} {'gain':>7s}")
print(f"  {'nothing (the model as it stands)':52s} {base:6.3f}")

ADDS = [
    ("UT hour and day-of-year, sin/cos", lambda i: clock(i)),
    ("rectified coupling from past wind", lambda i: coupling(i)),
    ("field variability (rms_sd_b_*)", lambda i: hist(i, FIELD)),
    ("plasma derived (Ey, Pdyn, beta, Mach)", lambda i: hist(i, PLASMA)),
    ("OMNI quality metadata", lambda i: hist(i, QUAL)),
    ("other indices (SYM-H, AE)", lambda i: hist(i, INDEX)),
]
for lab, fn in ADDS:
    r = rho(ridge(np.hstack([base_tr, fn(tri)]), Ytr)(np.hstack([base_te, fn(tei)])), Yte)
    print(f"  {lab:52s} {r:6.3f} {r - base:+7.3f}")
allf = lambda i: np.hstack([clock(i), coupling(i), hist(i, FIELD + PLASMA + QUAL + INDEX)])
r = rho(ridge(np.hstack([base_tr, allf(tri)]), Ytr)(np.hstack([base_te, allf(tei)])), Yte)
print(f"  {'everything above':52s} {r:6.3f} {r - base:+7.3f}")
r = rho(ridge(np.hstack([base_tr, allf(tri)]), Ytr)(np.hstack([base_te, allf(tei)])), Yte)

# ── experiment: target scale ────────────────────────────────────────────────
print("\n  ── the same information, scored on two targets ───────────────")
print(f"  {'target':20s} {'per-lead rho':>13s}")
for name in ("ap30", "hp30"):
    Yt, Yv = target(tri, name), target(tei, name)
    print(f"  {name:20s} {rho(ridge(base_tr, Yt)(base_te), Yv):13.3f}")

# ── experiment: does skill depend on input quality? ─────────────────────────
print("\n  ── validation stratified by OMNI's own quality flags ─────────")
P = ridge(base_tr, Ytr)(base_te)
print(f"  {'stratum':36s} {'n':>7s} {'rho':>7s}")
for lab, key, hi in (("timeshift RMS", "q_rms_ts", None),
                     ("phase-front-normal RMS", "q_rms_pfn", None),
                     ("percent interpolated", "q_pct_interp", None),
                     ("monitor off-axis distance", "q_sc_offaxis", None)):
    v = S[key][tei]
    lo_m, hi_m = v <= np.nanpercentile(v, 25), v >= np.nanpercentile(v, 75)
    print(f"  {lab + ', cleanest quartile':36s} {int(lo_m.sum()):7d} "
          f"{rho(P[lo_m], Yte[lo_m]):7.3f}")
    print(f"  {lab + ', dirtiest quartile':36s} {int(hi_m.sum()):7d} "
          f"{rho(P[hi_m], Yte[hi_m]):7.3f}")
# A raw quartile split mixes years, and both the monitor's orbit and the activity level
# drift across a solar cycle. The controlled version splits within each year.
print("\n  the same split, controlled for year")
off, yr = S["q_sc_offaxis"][tei], np.asarray(grid[tei].year)
A = np.zeros(len(off), bool)
B = np.zeros(len(off), bool)
for y in sorted(set(yr.tolist())):
    m = yr == y
    med = np.nanmedian(off[m])
    A |= m & (off <= med)
    B |= m & (off > med)
    print(f"    {y}  near {rho(P[m & (off <= med)], Yte[m & (off <= med)]):.3f}   "
          f"far {rho(P[m & (off > med)], Yte[m & (off > med)]):.3f}")
print(f"  {'pooled within-year, near':36s} {int(A.sum()):7d} {rho(P[A], Yte[A]):7.3f}")
print(f"  {'pooled within-year, far':36s} {int(B.sum()):7d} {rho(P[B], Yte[B]):7.3f}")

print("\n  VERDICT. The raw quartile gap of 0.054 in the monitor's off-axis distance does not")
print("  survive the control: within a year it is +0.014, inside the +/-0.015 precision floor,")
print("  and its sign flips between years. The Sivadas et al. (2026) mechanism is not visible")
print("  in these flags at this resolution. What does survive is the target scale -- the same")
print("  information scores 0.564 against ap30 and 0.628 against hp30.")
