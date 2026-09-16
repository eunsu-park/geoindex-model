"""Score forecast npz files (from `infer_checkpoint.py`) against each other and the references.

Every model is scored on the identical anchors; persistence, 27-day recurrence and the
training-period climatology are built from the same table so the skill scores are
like-for-like. Written for the co-author's focal/oversampling model vs our sweep models,
but model-agnostic.

    python analysis/score_forecast_npz.py --out-dir ~/Projects/GeoIndex/results/coauthor_focal_2026-09/scores \\
        --npz focal_dup248=.../val_focal_dup248.npz ours_gnn_transformer=.../val_ours_gnn_transformer.npz \\
        --ref ours_gnn_transformer

Outputs (CSV unless noted): overall.csv · by_lead.csv · by_target_bin.csv · by_peak_bin.csv ·
events_pointwise.csv · events_peak.csv · storm_rise.csv · lag.csv · uncertainty.csv ·
bootstrap.csv · summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.uncertainty import fit_sigma_scale  # noqa: E402

STEP = pd.Timedelta(minutes=30)
TARGET_BINS = [(0, 15), (15, 30), (30, 50), (50, 100), (100, np.inf)]
THRESHOLDS = (30, 50, 100)
_erf = np.vectorize(math.erf, otypes=[np.float64])


# --- loading ------------------------------------------------------------------------------

def load_npz(path: Path) -> dict:
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    d["time"] = pd.to_datetime(d["anchors"], format="%Y%m%d%H%M%S")
    return d


def table_series(table: Path) -> pd.Series:
    """Observed ap30 series: a parquet with datetime/ap30, or GFZ's Hp30_ap30_*.txt."""
    if table.suffix == ".txt":
        raw = pd.read_csv(table, sep=r"\s+", comment="#", header=None,
                          names=["y", "m", "d", "hh", "hm", "days", "days_m", "hp30", "ap30", "D"])
        dt = pd.to_datetime(dict(year=raw.y, month=raw.m, day=raw.d)) + pd.to_timedelta(raw.hh, unit="h")
        s = pd.Series(raw.ap30.to_numpy(dtype=float), index=pd.DatetimeIndex(dt).as_unit("ns"))
        return s.where(s >= 0)
    df = pd.read_parquet(table, columns=["datetime", "ap30"])
    return pd.Series(df.ap30.to_numpy(dtype=float), index=pd.DatetimeIndex(df.datetime))


def lookup(s: pd.Series, times: pd.DatetimeIndex, offsets_steps: np.ndarray) -> np.ndarray:
    """ap30 at `times[i] + offsets_steps[j] * 30 min` → (N, J), NaN where missing."""
    out = np.full((len(times), len(offsets_steps)), np.nan)
    for j, k in enumerate(offsets_steps):
        out[:, j] = s.reindex(times + int(k) * STEP).to_numpy()
    return out


# --- metrics -------------------------------------------------------------------------------

def pooled(y: np.ndarray, p: np.ndarray, pers: np.ndarray | None = None) -> dict:
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    e = p - y
    out = {"n": int(m.sum()), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
           "bias": float(np.mean(e)), "cc": float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 and p.std() > 0 else np.nan,
           "log_mae": float(np.mean(np.abs(np.log1p(np.clip(p, 0, None)) - np.log1p(y))))}
    if pers is not None:
        pe = np.abs(pers[m] - y)
        out["mae_persistence"] = float(pe.mean())
        out["skill_vs_persistence"] = float(1 - out["mae"] / pe.mean())
    return out


def contingency(obs_event: np.ndarray, fc_event: np.ndarray) -> dict:
    a = int((obs_event & fc_event).sum()); b = int((~obs_event & fc_event).sum())
    c = int((obs_event & ~fc_event).sum()); d = int((~obs_event & ~fc_event).sum())
    n = a + b + c + d
    exp = ((a + b) * (a + c) + (c + d) * (b + d)) / n if n else np.nan
    hss = (a + d - exp) / (n - exp) if n and n != exp else np.nan
    return {"hits": a, "false_alarms": b, "misses": c, "correct_neg": d,
            "pod": a / (a + c) if a + c else np.nan, "far": b / (a + b) if a + b else np.nan,
            "csi": a / (a + b + c) if a + b + c else np.nan, "hss": hss,
            "bias_freq": (a + b) / (a + c) if a + c else np.nan}


def crps_gaussian(y, mu, sd):
    sd = np.clip(sd, 1e-6, None)
    z = (y - mu) / sd
    pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1 + _erf(z / np.sqrt(2)))
    return float(np.mean(sd * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))))


def coverage(y, mu, sd, k):
    return float(np.mean(np.abs(y - mu) <= k * sd))


# --- main ----------------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", nargs="+", required=True, help="name=path pairs")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--table", default="~/Projects/GeoIndex/datasets/data.parquet")
    p.add_argument("--train-end", default="2021-12-31 23:30", help="climatology = table mean up to here")
    p.add_argument("--ref", default=None, help="model the bootstrap differences are taken against (default: 2nd)")
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--calib-split", default="2024-01-01", help="σ-scale fitted before this date, applied after")
    p.add_argument("--clim-start", default="1995-01-01")
    p.add_argument("--start", default=None, help="keep anchors at/after this time")
    p.add_argument("--end", default=None, help="keep anchors before this time")
    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    models = {}
    for item in args.npz:
        name, path = item.split("=", 1)
        models[name] = load_npz(Path(path).expanduser())
    names = list(models)
    ref = args.ref or (names[1] if len(names) > 1 else names[0])
    base = models[names[0]]
    # identical anchor sets required
    for n_, d in models.items():
        assert np.array_equal(d["anchors"], base["anchors"]), f"{n_}: anchor set differs from {names[0]}"
    keep = np.ones(len(base["time"]), dtype=bool)
    if args.start:
        keep &= np.asarray(base["time"] >= pd.Timestamp(args.start))
    if args.end:
        keep &= np.asarray(base["time"] < pd.Timestamp(args.end))
    for d in models.values():
        for k_ in list(d):
            if isinstance(d[k_], np.ndarray) and d[k_].ndim >= 1 and len(d[k_]) == len(keep):
                d[k_] = d[k_][keep]
        d["time"] = d["time"][keep]
    times = base["time"]
    Y = base["target"].astype(float)
    N, L = Y.shape
    K = base["input_ap30"].shape[1]
    s = table_series(Path(args.table).expanduser())
    # references
    pers = np.repeat(base["input_ap30"][:, -1:].astype(float), L, axis=1)
    rec = lookup(s, times, np.arange(L) - 27 * 48)
    clim = np.full_like(Y, float(s[pd.Timestamp(args.clim_start): pd.Timestamp(args.train_end)].mean()))
    refs = {"persistence": pers, "recurrence27": rec, "climatology": clim}
    preds = {n_: d["pred"].astype(float) for n_, d in models.items()}
    preds.update({f"{n_}_mcdmean": d["mcd_mean"].astype(float) for n_, d in models.items() if "mcd_mean" in d})
    allp = {**preds, **refs}

    # 1. overall + 2. by lead
    overall = pd.DataFrame({n_: pooled(Y, P, pers) for n_, P in allp.items()}).T
    overall.to_csv(out_dir / "overall.csv")
    rows = []
    for n_, P in allp.items():
        for j in range(L):
            rows.append({"model": n_, "lead_h": (j + 1) * 0.5, **pooled(Y[:, j], P[:, j], pers[:, j])})
    by_lead = pd.DataFrame(rows)
    by_lead.to_csv(out_dir / "by_lead.csv", index=False)

    # 3. by pointwise target bin, 4. by target-window peak bin
    rows, rows_pk = [], []
    tpeak = Y.max(axis=1); ipeak = base["input_ap30"].max(axis=1).astype(float)
    for lo, hi in TARGET_BINS:
        m = (Y >= lo) & (Y < hi)
        mk = (tpeak >= lo) & (tpeak < hi)
        for n_, P in allp.items():
            rows.append({"model": n_, "bin": f"[{lo},{hi})", **pooled(Y[m], P[m], pers[m])})
            rows_pk.append({"model": n_, "peak_bin": f"[{lo},{hi})", "n_anchors": int(mk.sum()),
                            **pooled(Y[mk], P[mk], pers[mk]),
                            "peak_mae": float(np.nanmean(np.abs(P[mk].max(axis=1) - tpeak[mk]))) if mk.any() else np.nan,
                            "peak_bias": float(np.nanmean(P[mk].max(axis=1) - tpeak[mk])) if mk.any() else np.nan})
    pd.DataFrame(rows).to_csv(out_dir / "by_target_bin.csv", index=False)
    pd.DataFrame(rows_pk).to_csv(out_dir / "by_peak_bin.csv", index=False)

    # 5. events: pointwise and window-peak
    rows, rows_pk = [], []
    for thr in THRESHOLDS:
        for n_, P in allp.items():
            m = np.isfinite(P)
            rows.append({"model": n_, "threshold": thr, **contingency(Y[m] >= thr, P[m] >= thr)})
            rows_pk.append({"model": n_, "threshold": thr, **contingency(tpeak >= thr, np.nanmax(P, axis=1) >= thr)})
    pd.DataFrame(rows).to_csv(out_dir / "events_pointwise.csv", index=False)
    pd.DataFrame(rows_pk).to_csv(out_dir / "events_peak.csv", index=False)

    # 6. storm rise (the co-author's target): target_peak - input_peak >= delta
    rows = []
    delta = tpeak - ipeak
    for dmin in (30, 50, 100):
        m = delta >= dmin
        q = delta < 30  # non-rise anchors: the false-alarm side
        for n_, P in allp.items():
            ppeak = np.nanmax(P, axis=1)
            ratio = (ppeak[m] - ipeak[m]) / np.maximum(delta[m], 1e-6)
            rows.append({"model": n_, "rise_min": dmin, "n": int(m.sum()),
                         "peak_mae": float(np.nanmean(np.abs(ppeak[m] - tpeak[m]))),
                         "peak_bias": float(np.nanmean(ppeak[m] - tpeak[m])),
                         "capture_ratio_mean": float(np.nanmean(np.clip(ratio, -1, 2))),
                         "capture_ratio_median": float(np.nanmedian(ratio)),
                         "frac_ratio_ge_0.5": float(np.nanmean(ratio >= 0.5)),
                         "window_mae": float(np.nanmean(np.abs(P[m] - Y[m]))),
                         "window_mae_persistence": float(np.nanmean(np.abs(pers[m] - Y[m]))),
                         "nonrise_n": int(q.sum()),
                         "nonrise_peak_bias": float(np.nanmean(ppeak[q] - tpeak[q])),
                         "nonrise_false_rise_frac_ge30": float(np.nanmean((ppeak[q] - ipeak[q]) >= 30))})
    pd.DataFrame(rows).to_csv(out_dir / "storm_rise.csv", index=False)

    # 7. lag diagnostic: which observation time does the lead-j forecast track best?
    rows = []
    deltas = np.arange(-24, 13)  # steps; negative = the forecast tracks the PAST
    for j in (0, 5, 11):
        obs = lookup(s, times, j - deltas)  # obs at anchor + (j - δ) steps
        for n_ in preds:
            P = preds[n_][:, j]
            cc = []
            for c in range(len(deltas)):
                m = np.isfinite(obs[:, c])
                cc.append(np.corrcoef(P[m], obs[m, c])[0, 1] if m.sum() > 10 and P[m].std() > 0 else np.nan)
            cc = np.array(cc)
            best = int(np.nanargmax(cc))
            rows.append({"model": n_, "lead_h": (j + 1) * 0.5, "cc_at_target": float(cc[deltas == 0][0]),
                         "best_delta_h": float(-deltas[best] * 0.5), "cc_at_best": float(cc[best])})
    pd.DataFrame(rows).to_csv(out_dir / "lag.csv", index=False)

    # 8. uncertainty
    rows = []
    split_t = pd.Timestamp(args.calib_split)
    early = np.asarray(times < split_t)
    for n_, d in models.items():
        if "mcd_std" not in d:
            continue
        mu, sd = d["mcd_mean"].astype(float), d["mcd_std"].astype(float)
        q_lo, q_hi = d["mcd_q025"].astype(float), d["mcd_q975"].astype(float)
        scale_all = fit_sigma_scale(Y.ravel(), mu.ravel(), sd.ravel(), coverage=0.95, k=2.0)
        scale_early = fit_sigma_scale(Y[early].ravel(), mu[early].ravel(), sd[early].ravel(), coverage=0.95, k=2.0)
        act = Y >= 15; storm = Y >= 50
        late_storm = (~early)[:, None] & storm
        rows.append({"model": n_, "n_samples": int(d.get("n_samples", 0)) if "n_samples" in d else None,
                     "picp_1s": coverage(Y, mu, sd, 1), "picp_2s": coverage(Y, mu, sd, 2),
                     "picp_q025_q975": float(np.mean((Y >= q_lo) & (Y <= q_hi))),
                     "mpiw_2s": float(np.mean(4 * sd)), "mean_std": float(sd.mean()),
                     "picp_2s_active_ge15": coverage(Y[act], mu[act], sd[act], 2),
                     "picp_2s_storm_ge50": coverage(Y[storm], mu[storm], sd[storm], 2),
                     "crps_gauss": crps_gaussian(Y, mu, sd),
                     "sigma_scale_fit_all": float(scale_all),
                     "picp_2s_recal_all_insample": coverage(Y, mu, sd * scale_all, 2),
                     "sigma_scale_fit_pre_split": float(scale_early),
                     "picp_2s_recal_post_split": coverage(Y[~early], mu[~early], sd[~early] * scale_early, 2),
                     "picp_2s_recal_post_split_storm": coverage(Y[late_storm], mu[late_storm], sd[late_storm] * scale_early, 2),
                     "mpiw_2s_recal": float(np.mean(4 * sd * scale_early)),
                     "std_vs_abs_err_cc": float(np.corrcoef(sd.ravel(), np.abs(mu - Y).ravel())[0, 1])})
    pd.DataFrame(rows).to_csv(out_dir / "uncertainty.csv", index=False)

    # 9. monthly block bootstrap of differences vs the reference model
    rng = np.random.default_rng(0)
    months = np.asarray(times.to_period("M"))
    uniq = np.unique(months)
    idx_by_month = {m_: np.where(months == m_)[0] for m_ in uniq}
    rows = []
    Pref = preds[ref]
    storm_pt = Y >= 50
    for n_ in preds:
        if n_ == ref:
            continue
        P = preds[n_]
        stats = {"d_mae": [], "d_cc": [], "d_mae_storm": [], "d_peak_mae_rise50": []}
        for _ in range(args.n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            ii = np.concatenate([idx_by_month[m_] for m_ in pick])
            y, a, b = Y[ii], P[ii], Pref[ii]
            stats["d_mae"].append(np.mean(np.abs(a - y)) - np.mean(np.abs(b - y)))
            stats["d_cc"].append(np.corrcoef(y.ravel(), a.ravel())[0, 1] - np.corrcoef(y.ravel(), b.ravel())[0, 1])
            sm = storm_pt[ii]
            stats["d_mae_storm"].append(np.mean(np.abs(a[sm] - y[sm])) - np.mean(np.abs(b[sm] - y[sm])) if sm.any() else np.nan)
            rm = delta[ii] >= 50
            if rm.any():
                tp = y[rm].max(axis=1)
                stats["d_peak_mae_rise50"].append(np.mean(np.abs(a[rm].max(axis=1) - tp)) - np.mean(np.abs(b[rm].max(axis=1) - tp)))
        for k_, v in stats.items():
            v = np.array(v, dtype=float)
            rows.append({"model": n_, "ref": ref, "stat": k_, "point": float(np.nanmean(v)),
                         "ci_lo": float(np.nanpercentile(v, 2.5)), "ci_hi": float(np.nanpercentile(v, 97.5))})
    pd.DataFrame(rows).to_csv(out_dir / "bootstrap.csv", index=False)

    summary = {"n_anchors": int(N), "output_steps": int(L), "input_steps": int(K), "models": names, "ref": ref,
               "period": [str(times.min()), str(times.max())],
               "overall": overall.round(4).to_dict(orient="index")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 220)
    print(overall.round(3).to_string())
    print("\nstorm rise (>=50):")
    sr = pd.read_csv(out_dir / "storm_rise.csv")
    print(sr[sr.rise_min == 50].set_index("model").round(3).to_string())
    print("\nuncertainty:")
    print(pd.read_csv(out_dir / "uncertainty.csv").set_index("model").round(3).T.to_string())
    print("\nbootstrap vs", ref)
    print(pd.read_csv(out_dir / "bootstrap.csv").round(3).to_string(index=False))
    print(f"\n→ {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
