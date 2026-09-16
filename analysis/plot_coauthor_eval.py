"""Figures for the co-author evaluation package (focal/oversampling model vs our sweep models).

Reads the npz forecasts from `infer_checkpoint.py` and the CSVs from `score_forecast_npz.py`.

    python analysis/plot_coauthor_eval.py --root ~/Projects/GeoIndex/results/coauthor_focal_2026-09 \\
        --table ~/Projects/GeoIndex/datasets/data.parquet \\
        --nrt-table ~/Projects/GeoIndex/datasets/realtime_nrt/Hp30_ap30_complete_series.txt

Figures (PNG, `<root>/figures/`):
  f01_by_lead_val          MAE, CC and skill vs persistence by lead (validation 2022–2025)
  f02_peak_bins_val        MAE and peak bias by target-window peak bin
  f03_storm_rise_val       predicted vs observed window peak for storm-rise anchors (Δpeak ≥ 50)
  f04_events_val           POD / FAR / HSS of the window-peak event at 30, 50, 100
  f05_coverage_val         MC-dropout 2σ coverage by lead, raw and after the σ-scale fitted on 2022–2023
  f06_cases_val            the eight largest storms 2022–2025, both models with 2σ bands
  f07_nrt_series           July–September 2026 real-time run: observed ap30 and the 3-h-lead forecasts
  f08_nrt_by_lead          NRT MAE / skill by lead
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.score_forecast_npz import load_npz, table_series  # noqa: E402

# categorical slots (validated default palette): candidate = orange, reference = blue, third = aqua
C = {"focal_dup248": "#eb6834", "ours_gnn_transformer": "#2a78d6", "ours_transformer": "#1baf7a",
     "ours_patchtst": "#4a3aa7", "persistence": "#8a8984", "obs": "#0b0b0b"}
LABEL = {"focal_dup248": "Focal + oversampling (co-author)", "ours_gnn_transformer": "GNN-Transformer (ours, same arch.)",
         "ours_transformer": "Transformer (ours)", "ours_patchtst": "PatchTST (ours)", "persistence": "Persistence",
         "obs": "Observed ap30"}
MODELS = ["focal_dup248", "ours_gnn_transformer", "ours_transformer"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.edgecolor": "#c3c2b7",
                     "legend.frameon": False, "figure.dpi": 130, "savefig.dpi": 160, "savefig.bbox": "tight"})


def line_by_lead(ax, df, metric, models, ylabel):
    for m in models:
        d = df[df.model == m]
        ax.plot(d.lead_h, d[metric], color=C[m], lw=2, label=LABEL[m])
    ax.set_xlabel("lead (h)"); ax.set_ylabel(ylabel); ax.set_xticks([0.5, 1, 2, 3, 4, 5, 6])


def f01(scores, out):
    df = pd.read_csv(scores / "by_lead.csv")
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    line_by_lead(axs[0], df, "mae", MODELS + ["persistence"], "MAE (ap)")
    line_by_lead(axs[1], df, "cc", MODELS + ["persistence"], "correlation")
    line_by_lead(axs[2], df, "skill_vs_persistence", MODELS, "MAE skill vs persistence")
    axs[2].axhline(0, color="#8a8984", lw=1)
    axs[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Validation split 2022–2025 (23,514 anchors, in6h_out6h)", x=0.01, ha="left", fontsize=10)
    fig.savefig(out / "f01_by_lead_val.png"); plt.close(fig)


def f02(scores, out):
    pk = pd.read_csv(scores / "by_peak_bin.csv")
    order = ["[0,15)", "[15,30)", "[30,50)", "[50,100)", "[100,inf)"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, metric, ylabel in ((axs[0], "mae", "window MAE (ap)"), (axs[1], "peak_bias", "peak bias: forecast − observed peak (ap)")):
        x = np.arange(len(order)); w = 0.2
        for i, m in enumerate(MODELS + ["persistence"]):
            d = pk[pk.model == m].set_index("peak_bin").loc[order, metric]
            ax.bar(x + (i - 1.5) * w, d, width=w * 0.92, color=C[m], label=LABEL[m])
        ax.set_xticks(x); ax.set_xticklabels([f"{o}\nn={int(pk[pk.model == 'persistence'].set_index('peak_bin').loc[o, 'n_anchors']):,}" for o in order], fontsize=8)
        ax.set_ylabel(ylabel); ax.set_xlabel("observed peak of the 6-h target window")
        ax.grid(axis="x", visible=False)
    axs[1].axhline(0, color="#8a8984", lw=1)
    axs[0].legend(fontsize=8)
    fig.savefig(out / "f02_peak_bins_val.png"); plt.close(fig)


def f03(npz, out):
    base = npz["focal_dup248"]
    tpeak = base["target"].max(1); ipeak = base["input_ap30"].max(1)
    m = (tpeak - ipeak) >= 50
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True)
    for ax, name in zip(axs, ["focal_dup248", "ours_gnn_transformer"]):
        pp = npz[name]["pred"].max(1)
        ax.plot([0, 420], [0, 420], color="#8a8984", lw=1)
        ax.scatter(tpeak[m], pp[m], s=12, color=C[name], alpha=0.6, edgecolor="white", linewidth=0.3)
        mae = np.mean(np.abs(pp[m] - tpeak[m])); ratio = np.mean((pp[m] - ipeak[m]) / (tpeak[m] - ipeak[m]))
        ax.set_title(f"{LABEL[name]}\npeak MAE {mae:.1f} · mean capture ratio {ratio:.2f} · n={m.sum()}", fontsize=9)
        ax.set_xlabel("observed window peak (ap)")
    axs[0].set_ylabel("forecast window peak (ap)")
    axs[0].set_xlim(0, 420); axs[0].set_ylim(0, 420)
    fig.suptitle("Storm-rise anchors: target peak − input peak ≥ 50 (the co-author's oversampled class)", x=0.01, ha="left", fontsize=10)
    fig.savefig(out / "f03_storm_rise_val.png"); plt.close(fig)


def f04(scores, out):
    ev = pd.read_csv(scores / "events_peak.csv")
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, metric in zip(axs, ["pod", "far", "hss"]):
        x = np.arange(3); w = 0.2
        for i, m in enumerate(MODELS + ["persistence"]):
            d = ev[ev.model == m].set_index("threshold").loc[[30, 50, 100], metric]
            ax.bar(x + (i - 1.5) * w, d, width=w * 0.92, color=C[m], label=LABEL[m])
        ax.set_xticks(x); ax.set_xticklabels(["ap ≥ 30", "ap ≥ 50", "ap ≥ 100"])
        ax.set_title(metric.upper(), fontsize=9); ax.grid(axis="x", visible=False)
    axs[0].legend(fontsize=7.5)
    fig.suptitle("Window-peak event detection (forecast peak ≥ threshold vs observed peak ≥ threshold)", x=0.01, ha="left", fontsize=10)
    fig.savefig(out / "f04_events_val.png"); plt.close(fig)


def f05(npz, scores, out, split="2024-01-01"):
    unc = pd.read_csv(scores / "uncertainty.csv").set_index("model")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    leads = np.arange(1, 13) * 0.5
    for name in MODELS:
        d = npz[name]; y, mu, sd = d["target"], d["mcd_mean"], d["mcd_std"]
        late = np.asarray(d["time"] >= pd.Timestamp(split))
        scale = unc.loc[name, "sigma_scale_fit_pre_split"]
        raw = [np.mean(np.abs(y[:, j] - mu[:, j]) <= 2 * sd[:, j]) for j in range(12)]
        rec = [np.mean(np.abs(y[late, j] - mu[late, j]) <= 2 * scale * sd[late, j]) for j in range(12)]
        axs[0].plot(leads, raw, color=C[name], lw=2, ls="--")
        axs[0].plot(leads, rec, color=C[name], lw=2, label=f"{LABEL[name]} (σ×{scale:.1f})")
        axs[1].plot(leads, 4 * sd.mean(0), color=C[name], lw=2, ls="--")
        axs[1].plot(leads, 4 * scale * sd[late].mean(0), color=C[name], lw=2, label=LABEL[name])
    axs[0].axhline(0.954, color="#8a8984", lw=1); axs[0].set_ylim(0, 1)
    axs[0].set_ylabel("2σ coverage (PICP)"); axs[0].set_xlabel("lead (h)")
    axs[0].set_title("dashed: raw MC-dropout (all anchors) · solid: recalibrated, scored on 2024–2025 only", fontsize=8.5)
    axs[1].set_ylabel("mean 2σ band width (ap)"); axs[1].set_xlabel("lead (h)"); axs[1].set_title("dashed: raw · solid: recalibrated", fontsize=8.5)
    axs[0].legend(fontsize=7.5, loc="center right")
    fig.savefig(out / "f05_coverage_val.png"); plt.close(fig)


def pick_events(times, tpeak, n=8, min_gap="3D"):
    order = np.argsort(-tpeak)
    chosen = []
    for i in order:
        if all(abs(times[i] - times[j]) > pd.Timedelta(min_gap) for j in chosen):
            chosen.append(i)
        if len(chosen) == n:
            break
    return sorted(chosen, key=lambda i: times[i])


def f06(npz, s, out, scores):
    base = npz["focal_dup248"]
    times = base["time"]; tpeak = base["target"].max(1)
    unc = pd.read_csv(scores / "uncertainty.csv").set_index("model")
    ev = pick_events(times, tpeak)
    rows = []
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.2))
    for ax, i in zip(axs.ravel(), ev):
        # anchor 3 h before the observed peak time so the peak sits at lead 3 h
        peak_time = times[i] + pd.Timedelta(minutes=30 * int(base["target"][i].argmax()))
        anchor = peak_time - pd.Timedelta(hours=3)
        k = np.where(times == anchor)[0]
        k = int(k[0]) if len(k) else i
        t0 = times[k]
        ctx = s[t0 - pd.Timedelta(hours=12): t0 + pd.Timedelta(hours=12)]
        ax.plot(ctx.index, ctx.values, color=C["obs"], lw=1.6, label=LABEL["obs"])
        tt = [t0 + pd.Timedelta(minutes=30 * j) for j in range(12)]
        for name in ["focal_dup248", "ours_gnn_transformer"]:
            d = npz[name]; sc = unc.loc[name, "sigma_scale_fit_pre_split"]
            ax.plot(tt, d["pred"][k], color=C[name], lw=2, label=LABEL[name])
            lo = np.clip(d["mcd_mean"][k] - 2 * sc * d["mcd_std"][k], 0, None); hi = d["mcd_mean"][k] + 2 * sc * d["mcd_std"][k]
            ax.fill_between(tt, lo, hi, color=C[name], alpha=0.15, lw=0)
        ax.axvline(t0, color="#8a8984", lw=1, ls=":")
        ax.set_title(f"anchor {t0:%Y-%m-%d %H:%M} UT · obs peak {tpeak[i]:.0f}", fontsize=8.5)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
        rows.append({"anchor": str(t0), "obs_peak_6h": float(base["target"][k].max()),
                     "input_last": float(base["input_ap30"][k, -1]),
                     **{f"{n_}_peak": float(npz[n_]["pred"][k].max()) for n_ in MODELS},
                     **{f"{n_}_mae": float(np.mean(np.abs(npz[n_]["pred"][k] - base["target"][k]))) for n_ in MODELS}})
    axs[0, 0].legend(fontsize=7.5, loc="upper left")
    axs[0, 0].set_ylabel("ap30"); axs[1, 0].set_ylabel("ap30")
    fig.suptitle("Eight largest storms of 2022–2025, forecast issued 3 h before the observed peak (bands: recalibrated 2σ)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "f06_cases_val.png"); plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "f06_cases_val.csv", index=False)


def f07(nrt, s_nrt, out, scores):
    base = nrt["focal_dup248"]; times = base["time"]
    unc = pd.read_csv(scores / "uncertainty.csv").set_index("model")
    fig, axs = plt.subplots(2, 1, figsize=(14, 6.4), gridspec_kw={"height_ratios": [1.2, 1]})
    j = 5  # lead 3 h
    valid = times + pd.Timedelta(hours=3)
    obs = s_nrt[pd.Timestamp("2026-06-19"): pd.Timestamp("2026-09-16")]
    for ax, (lo, hi) in zip(axs, [(pd.Timestamp("2026-06-19"), pd.Timestamp("2026-09-16")),
                                  (pd.Timestamp("2026-07-03 12:00"), pd.Timestamp("2026-07-05 12:00"))]):
        o = obs[lo:hi]
        ax.plot(o.index, o.values, color=C["obs"], lw=1.4, label=LABEL["obs"])
        m = (valid >= lo) & (valid <= hi)
        # break the line across snapshot gaps instead of drawing straight segments through them
        vt = valid[m]
        gap = np.r_[False, np.diff(vt.values) > np.timedelta64(60, "m")]
        for name in ["focal_dup248", "ours_gnn_transformer"]:
            d = nrt[name]
            yv = d["pred"][m, j].astype(float).copy()
            yv[gap] = np.nan
            ax.plot(vt, yv, color=C[name], lw=1.4, label=f"{LABEL[name]}, lead 3 h")
            if hi - lo < pd.Timedelta(days=10):
                sc = unc.loc[name, "sigma_scale_fit_pre_split"]
                ax.fill_between(valid[m], np.clip(d["mcd_mean"][m, j] - 2 * sc * d["mcd_std"][m, j], 0, None),
                                d["mcd_mean"][m, j] + 2 * sc * d["mcd_std"][m, j], color=C[name], alpha=0.15, lw=0)
        ax.set_xlim(lo, hi); ax.set_ylabel("ap30")
    axs[0].axvline(pd.Timestamp("2026-07-02"), color="#8a8984", lw=1, ls=":")
    axs[0].text(pd.Timestamp("2026-07-02 06:00"), axs[0].get_ylim()[1] * 0.9, "RTSW (SOLAR1) feed →", fontsize=8, color="#52514e")
    axs[0].text(pd.Timestamp("2026-06-20"), axs[0].get_ylim()[1] * 0.9, "← retired 7-day feed", fontsize=8, color="#52514e")
    axs[0].legend(fontsize=8, loc="upper right", ncol=3)
    axs[0].set_title("Real-time run on the archived NOAA snapshots (gaps = no snapshot); forecasts plotted at valid time", fontsize=9)
    axs[1].set_title("2026-07-04 storm (observed ap30 peak 207); bands: recalibrated 2σ (σ-scale from July 2–Aug 9)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "f07_nrt_series.png"); plt.close(fig)


def f08(scores, out):
    df = pd.read_csv(scores / "by_lead.csv")
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    line_by_lead(axs[0], df, "mae", MODELS + ["persistence"], "MAE (ap)")
    line_by_lead(axs[1], df, "cc", MODELS + ["persistence"], "correlation")
    line_by_lead(axs[2], df, "skill_vs_persistence", MODELS, "MAE skill vs persistence")
    axs[2].axhline(0, color="#8a8984", lw=1)
    axs[0].legend(loc="upper left", fontsize=8)
    n = int(df[df.model == "persistence"].n.iloc[0])
    fig.suptitle(f"Real-time feed, 2026-07-02 → 09-15 ({n:,} anchors with complete inputs)", x=0.01, ha="left", fontsize=10)
    fig.savefig(out / "f08_nrt_by_lead.png"); plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--table", default="~/Projects/GeoIndex/datasets/data.parquet")
    p.add_argument("--nrt-table", default="~/Projects/GeoIndex/datasets/realtime_nrt/Hp30_ap30_complete_series.txt")
    args = p.parse_args()
    root = Path(args.root).expanduser()
    out = root / "figures"; out.mkdir(exist_ok=True)
    val = {m: load_npz(root / f"val_{m}.npz") for m in MODELS}
    nrt = {m: load_npz(root / f"nrt_{m}.npz") for m in MODELS}
    s = table_series(Path(args.table).expanduser())
    s_nrt = table_series(Path(args.nrt_table).expanduser())
    sv, sn = root / "scores_val", root / "scores_nrt_rtsw"
    f01(sv, out); f02(sv, out); f03(val, out); f04(sv, out); f05(val, sv, out); f06(val, s, out, sv)
    f07(nrt, s_nrt, out, sn); f08(sn, out)
    print(f"figures → {out}")
    for f in sorted(out.glob("*.png")):
        print(" ", f.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
