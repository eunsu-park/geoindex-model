"""What does the solar wind actually buy, and how much of the rest is Bz?

The question this answers: an ap forecaster fed solar wind must be inferring the future wind and
using it -- otherwise why feed it wind at all? That is right, and this decomposes it.

Past wind buys two different things and they are worth separating:

  (a) the initial condition -- ap is an integrated response with a multi-hour memory, so past Bz
      already determines part of the next 12 h through the ring current's decay, not through any
      prediction of future Bz;
  (b) an estimate of the future wind -- E[wind(t+d) | past], which is what the correlation table
      measures and which collapses for Bz.

And the future wind splits into a part that sets the AMPLITUDE of the coupling (v, np, |B|) and a
part that sets whether the coupling is switched on at all (the Bz sign). The Newell coupling
v^(4/3) B_T^(2/3) sin^(8/3)(theta_c/2) is rectified: only southward Bz reconnects. So the model
can know an envelope without knowing a phase.

Six ridges, same anchors, same target:

  1  ap history only                    -- no wind at all
  2  ap + past wind                     -- today's model's information
  3  2 + true future wind EXCEPT bz     -- perfect envelope, no phase
  4  2 + true future bz ONLY            -- perfect phase, no extra envelope
  5  2 + all true future wind           -- the ceiling
  6  persistence                        -- the operational baseline

Usage:
    python wind_ablation.py [--table data.parquet]
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


def ridge(X, Y, alpha=100.0):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    yb = Y.mean(0)
    W = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (Y - yb))
    return lambda Xn: ((Xn - mu) / sd) @ W + yb


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap_.add_argument("--table", default="data.parquet")
    args = ap_.parse_args()

    tbl = pd.read_parquet(os.path.join(D, args.table)).set_index("datetime").sort_index()
    grid = pd.date_range(tbl.index[0], tbl.index[-1], freq="30min")
    tbl = tbl.reindex(grid)
    cols = WIND + ["ap30"]
    tbl[cols] = tbl[cols].interpolate(limit=6).ffill().bfill()
    ap = tbl["ap30"].to_numpy(float)
    wind = {w: tbl[w].to_numpy(float) for w in WIND}
    pos = {t: i for i, t in enumerate(grid)}

    def blocks(stamps):
        idx = np.array([pos[t] for t in stamps])
        past_w = np.column_stack([wind[w][idx[:, None] + np.arange(-PAST, 0)] for w in WIND])
        past_a = ap[idx[:, None] + np.arange(-PAST, 0)]
        fut_nobz = np.column_stack([wind[w][idx[:, None] + np.arange(0, OUT)]
                                    for w in WIND if w != "bz_avg"])
        fut_bz = wind["bz_avg"][idx[:, None] + np.arange(0, OUT)]
        Y = ap[idx[:, None] + np.arange(0, OUT)]
        return past_a, past_w, fut_nobz, fut_bz, Y, ap[idx]

    tr = pd.read_csv(os.path.join(D, "total_ap/train_index.csv"))
    tr["datetime"] = pd.to_datetime(tr["datetime"])
    tr_stamps = [t for t in tr["datetime"] if PAST <= pos.get(t, -1) < len(grid) - OUT]

    path = os.path.join(R, POOLED, "validation", "best", "npz.zip")
    anchors, arch_true = [], []
    with zipfile.ZipFile(path) as z:
        for n in sorted(x for x in z.namelist() if x.endswith(".npz")):
            d = np.load(io.BytesIO(z.read(n)), allow_pickle=True)
            anchors.append(str(np.asarray(d["anchor"])))
            arch_true.append(np.asarray(d["targets"])[:, 0])
    te_stamps = list(pd.to_datetime(pd.Series(anchors), format="%Y%m%d%H%M%S"))

    A_tr = blocks(tr_stamps)
    A_te = blocks(te_stamps)
    assert np.allclose(A_te[4], np.asarray(arch_true), atol=1e-3), "targets differ from the archive"
    print(f"train {len(tr_stamps):,} anchors, test {len(te_stamps):,} "
          f"(table {args.table}, same anchors as the deep runs)\n")

    VARIANTS = [
        ("1  ap history only",                 lambda b: b[0]),
        ("2  ap + past wind  <- the model",    lambda b: np.hstack([b[0], b[1]])),
        ("3  2 + true future wind, no bz",     lambda b: np.hstack([b[0], b[1], b[2]])),
        ("4  2 + true future bz only",         lambda b: np.hstack([b[0], b[1], b[3]])),
        ("5  2 + all true future wind",        lambda b: np.hstack([b[0], b[1], b[2], b[3]])),
    ]

    Ytr, Yte = A_tr[4], A_te[4]
    peak_obs = Yte.max(1)
    storm = peak_obs >= 47.5

    print(f"  {'':34s} {'per-lead':>9s} {'peak':>7s} {'peak rho':>9s} {'storm':>7s}")
    print(f"  {'':34s} {'rho':>9s} {'rho':>7s} {'>=48':>9s} {'MAE':>7s}")

    def report(label, P):
        per_lead = float(np.mean([np.corrcoef(P[:, k], Yte[:, k])[0, 1] for k in range(OUT)]))
        pk = P.max(1)
        rho = float(np.corrcoef(pk, peak_obs)[0, 1])
        rho_s = float(np.corrcoef(pk[storm], peak_obs[storm])[0, 1])
        mae = float(np.abs(pk[storm] - peak_obs[storm]).mean())
        print(f"  {label:34s} {per_lead:9.3f} {rho:7.3f} {rho_s:9.3f} {mae:7.2f}")
        return per_lead, rho

    base = None
    for label, sel in VARIANTS:
        f = ridge(sel(A_tr), Ytr)
        r = report(label, f(sel(A_te)))
        if label.startswith("2 "):
            base = r
    report("6  persistence (anchor value held)", np.repeat(A_te[5][:, None], OUT, axis=1))

    print(f"\n  headroom over variant 2 (per-lead rho / peak rho), storm rows n={int(storm.sum())}")
    print("  read: 3 is the envelope, 4 is the phase, 5 is both")


if __name__ == "__main__":
    main()
