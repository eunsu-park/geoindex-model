"""The salvageable half of proposal 1.

"Forecast every input parameter, not just ap" and "roll a one-step model out"
are two separate ideas that arrived together. The rollout was measured
separately (recursive_nonlinear.py). This measures the other half on its own:
predict all 22 channels DIRECTLY at all 24 leads as an auxiliary task, and see
whether the shared trunk makes the ap head better than an ap-only head of the
same size.

  ap-only        528 -> 24 ap leads
  multitask      528 -> 24 x 22, scored on the ap slice only
  multitask-w    same, with the ap channel weighted 10x in the loss
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DATA = '/Users/eunsupark/Projects/GeoIndex/datasets/data.parquet'
WIND = ['v_avg', 'v_min', 'v_max', 'np_avg', 'np_min', 'np_max',
        't_avg', 't_min', 't_max', 'bx_avg', 'bx_min', 'bx_max',
        'by_avg', 'by_min', 'by_max', 'bz_avg', 'bz_min', 'bz_max',
        'bt_avg', 'bt_min', 'bt_max']
COLS = WIND + ['ap30']
C, L, H = len(COLS), 24, 24
AP = C - 1
TRAIN_END, TEST_START = '2021-12-01', '2022-01-01'
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
import sys
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
torch.manual_seed(SEED)


def rolling_all_finite(mask, w):
    c = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
    n = len(mask) - w + 1
    return (c[w:w + n] - c[:n]) == w


class Net(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(L * C, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, n_out))

    def forward(self, x):
        return self.f(x.flatten(1))


def main():
    df = pd.read_parquet(DATA)
    dt = pd.to_datetime(df['datetime'])
    X = df[COLS].to_numpy(np.float64)
    ok = np.isfinite(X).all(1)
    anchors = np.flatnonzero(rolling_all_finite(ok, L + H)) + L
    t = dt.to_numpy()
    tr = anchors[t[anchors] < np.datetime64(TRAIN_END)]
    te = anchors[t[anchors] >= np.datetime64(TEST_START)]
    rows = np.unique((tr[:, None] + np.arange(-L, H)[None, :]).ravel())
    mu, sd = X[rows].mean(0), X[rows].std(0)
    Z = torch.tensor((X - mu) / sd, dtype=torch.float32, device=DEV)
    tr_t, te_t = torch.tensor(tr, device=DEV), torch.tensor(te, device=DEV)
    off_in = torch.arange(-L, 0, device=DEV)
    off_out = torch.arange(H, device=DEV)
    Ote = X[te[:, None] + np.arange(H)[None, :]][:, :, AP]
    print(f'seed={SEED} device={DEV}  train={len(tr):,}  test={len(te):,}')

    w = torch.ones(C, device=DEV)
    w10 = w.clone(); w10[AP] = 10.0

    def train(n_out, loss_fn, epochs=15, bs=2048, lr=1e-3):
        net = Net(n_out).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt, lr, total_steps=epochs * (len(tr) // bs + 1))
        for ep in range(epochs):
            perm = torch.randperm(len(tr), device=DEV)
            net.train()
            for s in range(0, len(tr), bs):
                a = tr_t[perm[s:s + bs]]
                loss = loss_fn(net, a)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step(); sch.step()
        net.eval()
        return net

    def win(a):
        return Z[a[:, None] + off_in[None, :]]

    def tgt(a):
        return Z[a[:, None] + off_out[None, :]]

    def report(name, fn):
        with torch.no_grad():
            P = np.concatenate([fn(te_t[s:s + 4096]).cpu().numpy()
                                for s in range(0, len(te), 4096)])
        P = P * sd[AP] + mu[AP]
        rr = [np.corrcoef(P[:, k], Ote[:, k])[0, 1] for k in range(H)]
        print(f'{name:>22}{np.mean(rr):>10.3f}{rr[1]:>9.3f}{rr[11]:>9.3f}'
              f'{rr[23]:>9.3f}{np.sqrt(np.mean((P - Ote) ** 2)):>9.3f}'
              f'{np.mean(np.abs(P - Ote)):>8.3f}'
              f'{np.corrcoef(P.max(1), Ote.max(1))[0, 1]:>10.3f}')

    print(f'\n{"scheme":>22}{"mean rho":>10}{"rho@1h":>9}{"rho@6h":>9}'
          f'{"rho@12h":>9}{"RMSE":>9}{"MAE":>8}{"peak rho":>10}')

    a1 = train(H, lambda n, a: ((n(win(a)) - tgt(a)[:, :, AP]) ** 2).mean())
    report('ap-only direct', lambda a: a1(win(a)))

    def mt_loss(weights):
        def f(n, a):
            p = n(win(a)).view(-1, H, C)
            return (((p - tgt(a)) ** 2) * weights).mean()
        return f

    a2 = train(H * C, mt_loss(w))
    report('multitask (equal)', lambda a: a2(win(a)).view(-1, H, C)[:, :, AP])



if __name__ == '__main__':
    main()
