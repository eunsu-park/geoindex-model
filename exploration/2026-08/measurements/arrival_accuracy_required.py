"""The imagery decision, inverted: how accurate must an arrival forecast be?

`arrival_oracle.py` bounds the coronagraph channel from above and then degrades
the oracle at a few fixed accuracies. That answers "what is it worth at today's
accuracy". It does not answer the question a foundation-model proposal actually
raises, which is "how much better than today would it have to be to be worth
building" -- and that one can be answered without an image, by sweeping the
arrival-time error finely and reading the curve backwards.

Same substitution as the original: the model is handed the true arrival step and
true amplitude of every disturbance front inside its forecast window, then that
handover is corrupted by a per-anchor timing error drawn uniformly from
[-J, +J] and the ridge is retrained on the corrupted version, which is what a
forecaster actually gets. Fronts sliding out of the window are lost rather than
wrapped, because a forecast that puts the arrival outside the window carries no
information about it.

Nothing built from imagery can beat the row at its own timing accuracy, so the
inversion at the end is a requirement, not an estimate: to be worth G, the
pipeline must forecast arrival to better than the quoted number.

Usage:
    python arrival_accuracy_required.py
"""

from __future__ import annotations

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
JITTERS_H = [0.0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 18, 24]
N_DRAWS = 12
TARGETS = [0.02, 0.05, 0.10]


def ridge(X, Y, alpha=100.0):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    yb = Y.mean(0)
    W = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (Y - yb))
    return lambda Xn: ((Xn - mu) / sd) @ W + yb


def shift_rows(block, steps):
    """Shift each row of (n, OUT) by its own signed step count, zero-filling."""
    n, w = block.shape
    out = np.zeros_like(block)
    src = np.arange(w)[None, :] - steps[:, None]
    ok = (src >= 0) & (src < w)
    rows = np.repeat(np.arange(n)[:, None], w, axis=1)
    out[ok] = block[rows[ok], src[ok]]
    return out


def main():
    tbl = (pd.read_parquet(os.path.join(D, "data.parquet"))
           .set_index("datetime").sort_index())
    grid = pd.date_range(tbl.index[0], tbl.index[-1], freq="30min")
    tbl = tbl.reindex(grid)
    cols = WIND + ["ap30"]
    tbl[cols] = tbl[cols].interpolate(limit=6).ffill().bfill()
    ap = tbl["ap30"].to_numpy(float)
    wind = {w: tbl[w].to_numpy(float) for w in WIND}
    pos = {t: i for i, t in enumerate(grid)}

    v, bt = wind["v_avg"], wind["bt_avg"]
    dv, dbt = np.zeros_like(v), np.zeros_like(bt)
    dv[2:] = v[2:] - v[:-2]
    dbt[2:] = bt[2:] - bt[:-2]
    thr_v = float(np.percentile(dv, 99.5))
    front = ((dv >= thr_v) & (dbt > 0)).astype(float)
    amp = np.where(front > 0, dv, 0.0)

    def blocks(stamps):
        idx = np.array([pos[t] for t in stamps])
        past_a = ap[idx[:, None] + np.arange(-PAST, 0)]
        past_w = np.column_stack(
            [wind[w][idx[:, None] + np.arange(-PAST, 0)] for w in WIND])
        fw = idx[:, None] + np.arange(0, OUT)
        return dict(a=past_a, w=past_w, f=front[fw], m=amp[fw],
                    Y=ap[fw], hasfront=front[fw].max(1))

    tr = pd.read_csv(os.path.join(D, "total_ap/train_index.csv"))
    tr["datetime"] = pd.to_datetime(tr["datetime"])
    tr_stamps = [t for t in tr["datetime"]
                 if PAST <= pos.get(t, -1) < len(grid) - OUT]
    with zipfile.ZipFile(os.path.join(R, POOLED, "validation", "best",
                                      "npz.zip")) as z:
        anchors = [str(np.asarray(np.load(io.BytesIO(z.read(n)),
                                          allow_pickle=True)["anchor"]))
                   for n in sorted(x for x in z.namelist()
                                   if x.endswith(".npz"))]
    te_stamps = list(pd.to_datetime(pd.Series(anchors), format="%Y%m%d%H%M%S"))

    TR, TE = blocks(tr_stamps), blocks(te_stamps)
    Yte = TE["Y"]
    storm = Yte.max(1) >= 47.5
    hasf = TE["hasfront"] > 0
    onset = storm & hasf
    print(f"train {len(TR['Y']):,}  test {len(TE['Y']):,}  "
          f"front windows {int(hasf.sum()):,}  onset windows {int(onset.sum()):,}")

    def per_lead(P, mask=None):
        y, p = (Yte, P) if mask is None else (Yte[mask], P[mask])
        return float(np.mean([np.corrcoef(p[:, k], y[:, k])[0, 1]
                              for k in range(OUT)]))

    f0 = ridge(np.hstack([TR["a"], TR["w"]]), TR["Y"])
    P0 = f0(np.hstack([TE["a"], TE["w"]]))
    base_all, base_on = per_lead(P0), per_lead(P0, onset)
    print(f"baseline (no arrival information)  all {base_all:.4f}   "
          f"onset {base_on:.4f}\n")

    print(f"{'timing error':>14}{'all anchors':>26}{'onset windows':>26}")
    print(f"{'(uniform +-h)':>14}{'per-lead rho':>14}{'gain':>12}"
          f"{'per-lead rho':>14}{'gain':>12}")
    curve = []
    for J in JITTERS_H:
        ga, go = [], []
        for draw in range(1 if J == 0 else N_DRAWS):
            rng = np.random.default_rng(1000 + draw)
            hw = int(round(J * 2))
            def degrade(B, n):
                if hw == 0:
                    return B["f"], B["m"]
                s = rng.integers(-hw, hw + 1, size=n)
                return shift_rows(B["f"], s), shift_rows(B["m"], s)
            ftr, mtr = degrade(TR, len(TR["Y"]))
            fte, mte = degrade(TE, len(TE["Y"]))
            fit = ridge(np.hstack([TR["a"], TR["w"], ftr, mtr]), TR["Y"])
            P = fit(np.hstack([TE["a"], TE["w"], fte, mte]))
            ga.append(per_lead(P) - base_all)
            go.append(per_lead(P, onset) - base_on)
        gain_a, gain_o = float(np.mean(ga)), float(np.mean(go))
        curve.append((J, gain_a, gain_o))
        print(f"{J:>14.1f}{base_all + gain_a:>14.4f}{gain_a:>+12.4f}"
              f"{base_on + gain_o:>14.4f}{gain_o:>+12.4f}")

    J = np.array([c[0] for c in curve])
    GA = np.array([c[1] for c in curve])
    GO = np.array([c[2] for c in curve])

    def required(gains, target):
        """Largest timing error still delivering `target`; None if never."""
        if gains[0] < target:
            return None
        below = np.flatnonzero(gains < target)
        if not len(below):
            return float(J[-1])          # still above target at the widest sweep
        i = below[0]
        x0, x1, y0, y1 = J[i - 1], J[i], gains[i - 1], gains[i]
        return float(x0 + (y0 - target) * (x1 - x0) / (y0 - y1))

    print("\nInverted: the arrival accuracy an imagery pipeline must reach")
    print(f"{'to be worth':>14}{'all anchors':>18}{'onset windows':>18}")
    for t in TARGETS:
        ra, ro = required(GA, t), required(GO, t)
        sa = f"+-{ra:.1f} h" if ra is not None else "unreachable"
        so = f"+-{ro:.1f} h" if ro is not None else "unreachable"
        print(f"{('+' + format(t, '.2f')):>14}{sa:>18}{so:>18}")
    print(f"\nCeiling with perfect timing: all {GA[0]:+.4f}, "
          f"onset {GO[0]:+.4f} -- nothing built from images exceeds this.")
    print("Published CME arrival forecasts sit at +-6 to 10 h; read the table there.")


if __name__ == "__main__":
    main()
