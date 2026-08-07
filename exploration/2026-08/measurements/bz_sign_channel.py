"""The one channel the imagery bounds never priced: the SIGN of the arriving field.

Every substitution in README-imagery-bounds.md hands the model an arrival time
and a speed amplitude, neither of which says which way the field points. If a
learned encoder could read flux-rope orientation from an eruption's magnetic
context, it would supply information none of those bounds covered. This prices
it, still without an image.

Three questions, in order:

  1  What is the sign worth at all? Give the true future sign of Bz and nothing
     else about Bz, at two granularities -- every 30-minute step (the upper
     bound on any sign information) and one bit per window (what an orientation
     forecast actually produces, since a flux rope has one orientation).
  2  What does it cost to be wrong? Flip the window bit with probability
     1 - p and sweep p from no skill to perfect.
  3  Inverted: what per-event orientation accuracy would a pipeline need?

The window bit is deliberately the weakest useful form: one binary per forecast,
"does this window contain a geoeffective southward excursion". That is what a
chirality prediction delivers, and it is far less than the per-step sign.
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
SOUTH_NT = -10.0          # threshold defining a geoeffective southward excursion
ACCURACIES = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
N_DRAWS = 12
TARGETS = [0.02, 0.05, 0.10]


def ridge(X, Y, alpha=100.0):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    yb = Y.mean(0)
    W = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (Y - yb))
    return lambda Xn: ((Xn - mu) / sd) @ W + yb


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
    front = ((dv >= float(np.percentile(dv, 99.5))) & (dbt > 0)).astype(float)

    def blocks(stamps):
        idx = np.array([pos[t] for t in stamps])
        fw = idx[:, None] + np.arange(0, OUT)
        bz = wind["bz_avg"][fw]
        return dict(
            a=ap[idx[:, None] + np.arange(-PAST, 0)],
            w=np.column_stack([wind[x][idx[:, None] + np.arange(-PAST, 0)]
                               for x in WIND]),
            sign=np.sign(bz),                       # per-step sign, no magnitude
            absbz=np.abs(bz),                       # per-step magnitude, no sign
            bz=bz,
            fut=np.column_stack([wind[x][fw] for x in WIND]),
            bit=(bz.min(1) <= SOUTH_NT).astype(float)[:, None],
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
    onset = (Yte.max(1) >= 47.5) & (TE["hasfront"] > 0)
    print(f"train {len(TR['Y']):,}  test {len(TE['Y']):,}  "
          f"onset {int(onset.sum()):,}")
    print(f"southward bit (min Bz <= {SOUTH_NT:g} nT): base rate "
          f"{TE['bit'].mean():.3f} on all windows, "
          f"{TE['bit'][onset].mean():.3f} on onset windows\n")

    def per_lead(P, mask=None):
        y, p = (Yte, P) if mask is None else (Yte[mask], P[mask])
        return float(np.mean([np.corrcoef(p[:, k], y[:, k])[0, 1]
                              for k in range(OUT)]))

    def run(keys, tr_over=None, te_over=None):
        Xtr = np.hstack([tr_over.get(k, TR[k]) if tr_over else TR[k]
                         for k in keys])
        Xte = np.hstack([te_over.get(k, TE[k]) if te_over else TE[k]
                         for k in keys])
        P = ridge(Xtr, TR["Y"])(Xte)
        return per_lead(P), per_lead(P, onset)

    base_all, base_on = run(["a", "w"])
    print(f"{'true future information given':>44}{'all':>10}{'gain':>10}"
          f"{'onset':>10}{'gain':>10}")
    print(f"{'none (the model)':>44}{base_all:>10.4f}{'':>10}"
          f"{base_on:>10.4f}{'':>10}")
    for label, keys in [
        ("one bit per window: is it southward?", ["a", "w", "bit"]),
        ("per-step SIGN of Bz, no magnitude", ["a", "w", "sign"]),
        ("per-step |Bz|, no sign", ["a", "w", "absbz"]),
        ("|Bz| + the one-bit sign", ["a", "w", "absbz", "bit"]),
        ("|Bz| + the per-step sign", ["a", "w", "absbz", "sign"]),
        ("Bz itself (reference)", ["a", "w", "bz"]),
        ("all future wind (reference)", ["a", "w", "fut"]),
    ]:
        a, o = run(keys)
        print(f"{label:>44}{a:>10.4f}{a - base_all:>+10.4f}"
              f"{o:>10.4f}{o - base_on:>+10.4f}")

    # ---- 2. degrade the window bit by per-event orientation accuracy -------
    print(f"\ndegrading the one-bit forecast: flip it with probability 1 - p")
    print(f"{'accuracy p':>12}{'all':>12}{'gain':>10}{'onset':>12}{'gain':>10}")
    curve = []
    for p in ACCURACIES:
        ga, go = [], []
        for draw in range(1 if p == 1.0 else N_DRAWS):
            rng = np.random.default_rng(2000 + draw)
            def flip(B, n):
                keep = rng.random(n) < p
                b = B["bit"].copy()
                b[~keep] = 1.0 - b[~keep]
                return b
            a, o = run(["a", "w", "bit"],
                       {"bit": flip(TR, len(TR["Y"]))},
                       {"bit": flip(TE, len(TE["Y"]))})
            ga.append(a - base_all)
            go.append(o - base_on)
        curve.append((p, float(np.mean(ga)), float(np.mean(go))))
        print(f"{p:>12.2f}{base_all + curve[-1][1]:>12.4f}{curve[-1][1]:>+10.4f}"
              f"{base_on + curve[-1][2]:>12.4f}{curve[-1][2]:>+10.4f}")

    P = np.array([c[0] for c in curve])
    GA = np.array([c[1] for c in curve])
    GO = np.array([c[2] for c in curve])

    def required(gains, target):
        if gains[-1] < target:
            return None
        above = np.flatnonzero(gains >= target)
        i = above[0]
        if i == 0:
            return float(P[0])
        x0, x1, y0, y1 = P[i - 1], P[i], gains[i - 1], gains[i]
        return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))

    print("\nInverted: the per-event orientation accuracy a pipeline would need")
    print(f"{'to be worth':>14}{'all anchors':>18}{'onset windows':>18}")
    for t in TARGETS:
        ra, ro = required(GA, t), required(GO, t)
        sa = f"{ra*100:.0f} % correct" if ra is not None else "unreachable"
        so = f"{ro*100:.0f} % correct" if ro is not None else "unreachable"
        print(f"{('+' + format(t, '.2f')):>14}{sa:>18}{so:>18}")
    print(f"\nPerfect orientation knowledge, one bit per window: "
          f"all {GA[-1]:+.4f}, onset {GO[-1]:+.4f}.")


if __name__ == "__main__":
    main()
