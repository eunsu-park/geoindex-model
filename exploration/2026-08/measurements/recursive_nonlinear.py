"""Proposal 1, the part ridge cannot answer.

A linear one-step model rolled out 24 times is still a linear function of the
input window, so `recursive` and `direct` are the same hypothesis class and the
ridge comparison is close to a tautology. The question only becomes real for a
nonlinear model, where rollout error compounds off the training manifold.

Four nets, same trunk, same budget, same anchors:

  direct        528 -> 24 ap leads, trained on the 12 h window at once
  onestep       528 -> 22 channels at +30 min, trained on one step, rolled out 24x
  onestep-uw    same net, trained THROUGH the 24-step rollout on the ap loss
                (the standard fix for exposure bias; what the proposal would
                have to do to be competitive)
  onestep-oracle  onestep, but the true wind is injected at every rollout step
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
torch.manual_seed(0)


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


def rollout(net, win, Zt, base, oracle=False):
    """win: (n, L, C) z-scored buffer. Returns (n, H) ap in z units."""
    out = []
    for k in range(H):
        nxt = net(win)
        out.append(nxt[:, AP])
        if oracle:
            nxt = torch.cat([Zt[base + k][:, :AP], nxt[:, AP:]], 1)
        win = torch.cat([win[:, 1:], nxt[:, None, :]], 1)
    return torch.stack(out, 1)


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
    Z = torch.tensor(((X - mu) / sd), dtype=torch.float32, device=DEV)
    tr_t = torch.tensor(tr, device=DEV)
    te_t = torch.tensor(te, device=DEV)
    off_in = torch.arange(-L, 0, device=DEV)
    off_out = torch.arange(H, device=DEV)
    print(f'device={DEV}  train={len(tr):,}  test={len(te):,}')

    def win_of(a):
        return Z[a[:, None] + off_in[None, :]]

    def ap_of(a):
        return Z[a[:, None] + off_out[None, :]][:, :, AP]

    def train(net, step_fn, epochs, bs=2048, lr=1e-3):
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, lr, total_steps=epochs * (len(tr) // bs + 1))
        for ep in range(epochs):
            perm = torch.randperm(len(tr), device=DEV)
            net.train()
            tot = nb = 0
            for s in range(0, len(tr), bs):
                a = tr_t[perm[s:s + bs]]
                loss = step_fn(net, a)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step(); sched.step()
                tot += loss.item(); nb += 1
            print(f'    epoch {ep + 1:>2}/{epochs}  loss {tot / nb:.4f}')
        net.eval()
        return net

    def evaluate(pred_z):
        p = pred_z * sd[AP] + mu[AP]
        o = X[te[:, None] + np.arange(H)[None, :]][:, :, AP]
        rr = [np.corrcoef(p[:, k], o[:, k])[0, 1] for k in range(H)]
        return dict(mean_rho=float(np.mean(rr)), rho12=rr[-1], rho1=rr[1],
                    rmse=float(np.sqrt(np.mean((p - o) ** 2))),
                    mae=float(np.mean(np.abs(p - o))),
                    peak_rho=float(np.corrcoef(p.max(1), o.max(1))[0, 1]),
                    sd_ratio=float(np.mean([p[:, k].std() / o[:, k].std()
                                            for k in range(H)])), per_lead=rr)

    @torch.no_grad()
    def predict(fn, bs=4096):
        out = []
        for s in range(0, len(te), bs):
            out.append(fn(te_t[s:s + bs]).cpu().numpy())
        return np.concatenate(out)

    res = {}

    print('\n[direct] 528 -> 24 ap leads')
    d = train(Net(H).to(DEV),
              lambda n, a: ((n(win_of(a)) - ap_of(a)) ** 2).mean(), 15)
    res['direct'] = evaluate(predict(lambda a: d(win_of(a))))

    print('\n[onestep] 528 -> 22 channels, teacher forcing, rolled out 24x')
    o1 = train(Net(C).to(DEV),
               lambda n, a: ((n(win_of(a)) - Z[a]) ** 2).mean(), 15)
    res['onestep (teacher forced)'] = evaluate(
        predict(lambda a: rollout(o1, win_of(a), Z, a)))
    res['onestep + oracle wind'] = evaluate(
        predict(lambda a: rollout(o1, win_of(a), Z, a, oracle=True)))

    print('\n[onestep-unrolled] same net, backprop through the 24-step rollout')
    def unrolled_loss(n, a):
        return ((rollout(n, win_of(a), Z, a) - ap_of(a)) ** 2).mean()
    o2 = train(Net(C).to(DEV), unrolled_loss, 8, bs=1024)
    res['onestep (unrolled loss)'] = evaluate(
        predict(lambda a: rollout(o2, win_of(a), Z, a)))

    print('\n' + '=' * 96)
    hdr = (f'{"scheme":>28}{"mean rho":>10}{"rho@1h":>9}{"rho@12h":>9}'
           f'{"RMSE":>9}{"MAE":>8}{"peak rho":>10}{"sd ratio":>10}')
    print(hdr)
    for k, v in res.items():
        print(f'{k:>28}{v["mean_rho"]:>10.3f}{v["rho1"]:>9.3f}{v["rho12"]:>9.3f}'
              f'{v["rmse"]:>9.3f}{v["mae"]:>8.3f}{v["peak_rho"]:>10.3f}'
              f'{v["sd_ratio"]:>10.3f}')

    print('\nper-lead rho')
    print('lead(h) ' + ''.join(f'{k[:16]:>18}' for k in res))
    for k in range(H):
        print(f'{(k + 1) * 0.5:>7.1f}'
              + ''.join(f'{v["per_lead"][k]:>18.3f}' for v in res.values()))

    import json
    with open('/private/tmp/claude-501/-Users-eunsupark-GitHub-njit-geoindex-'
              'geoindex/2fa134a0-debc-458e-b54a-ce50829dbdb1/scratchpad/'
              'recursive_nonlinear.json', 'w') as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
