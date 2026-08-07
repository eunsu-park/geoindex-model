"""How much of the oracle does the RIDGE fail to extract?

Every imagery bound in this investigation is a ridge given perfect information.
The record already flags that as a floor on the bound rather than the bound --
a ridge extracts less from perfect information than a deep model would, and
adding hand-built products moved two of the rows by +0.009 and +0.042. That
matters for a specific question: whether a model TRAINED FROM SCRATCH with an
image encoder faces the same ceiling as one that has the bound bolted on.

The information argument does not care about the fitting procedure, so the
ceiling transfers. What does not transfer is the ridge's extraction loss. This
measures it, by running the same feature blocks through a ridge and through an
MLP of the size the real architecture uses:

    baseline        past ap + past wind
    + arrival       + true arrival step and amplitude of every front (the
                    coronagraph ceiling)
    + future wind   + the true future wind, all of it (the ceiling on ANY
                    exogenous driver, imagery included)

If the deep model lifts the oracle rows much more than the baseline row, the
bounds understate what a jointly trained model could reach and every imagery
number here is too low. If it lifts both about equally, the bounds stand as
measured and the gap is a property of the problem rather than the estimator.
"""
import io
import os
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

D = os.path.expanduser("~/Projects/GeoIndex/datasets")
R = os.path.expanduser("~/Projects/GeoIndex/results")
POOLED = "probe_ap_in12h_out12h_gnn_transformer_baseline"
WIND = ["v_avg", "np_avg", "t_avg", "bx_avg", "by_avg", "bz_avg", "bt_avg"]
PAST = OUT = 24
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
SEEDS = 3


def ridge(X, Y, alpha=100.0):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    yb = Y.mean(0)
    W = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (Y - yb))
    return lambda Xn: ((Xn - mu) / sd) @ W + yb


def mlp(Xtr, Ytr, Xte, seed, epochs=25, bs=2048, lr=1e-3):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-9] = 1.0
    ym, ys = Ytr.mean(), Ytr.std()
    xtr = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=DEV)
    ytr = torch.tensor((Ytr - ym) / ys, dtype=torch.float32, device=DEV)
    xte = torch.tensor((Xte - mu) / sd, dtype=torch.float32, device=DEV)
    net = nn.Sequential(
        nn.Linear(xtr.shape[1], 512), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(512, 512), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(512, OUT)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, lr, total_steps=epochs * (len(xtr) // bs + 1))
    for _ in range(epochs):
        perm = torch.randperm(len(xtr), device=DEV)
        net.train()
        for s in range(0, len(xtr), bs):
            i = perm[s:s + bs]
            loss = ((net(xtr[i]) - ytr[i]) ** 2).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sch.step()
    net.eval()
    with torch.no_grad():
        P = torch.cat([net(xte[s:s + 8192]) for s in range(0, len(xte), 8192)])
    return P.cpu().numpy() * ys + ym


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
    amp = np.where(front > 0, dv, 0.0)

    def blocks(stamps):
        idx = np.array([pos[t] for t in stamps])
        past_a = ap[idx[:, None] + np.arange(-PAST, 0)]
        past_w = np.column_stack(
            [wind[w][idx[:, None] + np.arange(-PAST, 0)] for w in WIND])
        fw = idx[:, None] + np.arange(0, OUT)
        return dict(a=past_a, w=past_w,
                    orc=np.hstack([front[fw], amp[fw]]),
                    fut=np.column_stack([wind[w][fw] for w in WIND]),
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
    print(f"device={DEV}  train {len(TR['Y']):,}  test {len(TE['Y']):,}  "
          f"onset {int(onset.sum()):,}\n")

    def per_lead(P, mask=None):
        y, p = (Yte, P) if mask is None else (Yte[mask], P[mask])
        return float(np.mean([np.corrcoef(p[:, k], y[:, k])[0, 1]
                              for k in range(OUT)]))

    VARIANTS = [
        ("baseline (past only)", ["a", "w"]),
        ("+ arrival oracle", ["a", "w", "orc"]),
        ("+ true future wind", ["a", "w", "fut"]),
    ]

    print(f"{'features':>22}{'ridge':>18}{'MLP (3 seeds)':>22}"
          f"{'deep lift':>12}")
    print(f"{'':>22}{'all':>9}{'onset':>9}{'all':>11}{'onset':>11}"
          f"{'all':>12}")
    rows = {}
    for label, keys in VARIANTS:
        Xtr = np.hstack([TR[k] for k in keys])
        Xte = np.hstack([TE[k] for k in keys])
        Pr = ridge(Xtr, TR["Y"])(Xte)
        r_all, r_on = per_lead(Pr), per_lead(Pr, onset)
        m_all, m_on = [], []
        for s in range(SEEDS):
            Pm = mlp(Xtr, TR["Y"], Xte, seed=s)
            m_all.append(per_lead(Pm))
            m_on.append(per_lead(Pm, onset))
        ma, mo = float(np.mean(m_all)), float(np.mean(m_on))
        rows[label] = (r_all, r_on, ma, mo)
        print(f"{label:>22}{r_all:>9.4f}{r_on:>9.4f}{ma:>11.4f}{mo:>11.4f}"
              f"{ma - r_all:>+12.4f}")

    b = rows["baseline (past only)"]
    print("\ngain over the baseline, by estimator")
    print(f"{'':>22}{'ridge all':>12}{'MLP all':>10}{'ridge onset':>14}"
          f"{'MLP onset':>12}")
    for label, _ in VARIANTS[1:]:
        r = rows[label]
        print(f"{label:>22}{r[0] - b[0]:>+12.4f}{r[2] - b[2]:>+10.4f}"
              f"{r[1] - b[1]:>+14.4f}{r[3] - b[3]:>+12.4f}")


if __name__ == "__main__":
    main()
