"""Both proposals are really bets on the same thing: getting the future solar
wind. This prices that bet.

Setting: identical anchors, identical 12 h ap targets. The model is given the
past 12 h of everything, PLUS the true future wind out to `delta` hours ahead
(and nothing beyond). Sweeping delta says how much lead an exogenous wind
forecast has to deliver before it pays, and the per-channel variants say which
channel has to be right.

The full-horizon column is the ceiling: what ap forecasting looks like if the
wind problem were solved outright.
"""
import numpy as np
import pandas as pd

DATA = '/Users/eunsupark/Projects/GeoIndex/datasets/data.parquet'
WIND = ['v_avg', 'v_min', 'v_max', 'np_avg', 'np_min', 'np_max',
        't_avg', 't_min', 't_max', 'bx_avg', 'bx_min', 'bx_max',
        'by_avg', 'by_min', 'by_max', 'bz_avg', 'bz_min', 'bz_max',
        'bt_avg', 'bt_min', 'bt_max']
COLS = WIND + ['ap30']
C, L, H = len(COLS), 24, 24
AP = C - 1
BZ = [COLS.index(c) for c in ('bz_avg', 'bz_min', 'bz_max')]
V = [COLS.index(c) for c in ('v_avg', 'v_min', 'v_max')]
BT = [COLS.index(c) for c in ('bt_avg', 'bt_min', 'bt_max')]
TRAIN_END, TEST_START = '2021-12-01', '2022-01-01'


def rolling_all_finite(mask, w):
    c = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
    n = len(mask) - w + 1
    return (c[w:w + n] - c[:n]) == w


def ridge(F, Y, alpha=1.0):
    p = F.shape[1]
    R = np.eye(p) * alpha * len(F) / 1e4
    R[-1, -1] = 0.0
    return np.linalg.solve(F.T @ F + R, F.T @ Y)


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
    Z = (X - mu) / sd
    Otr = X[tr[:, None] + np.arange(H)[None, :]][:, :, AP]
    Ote = X[te[:, None] + np.arange(H)[None, :]][:, :, AP]
    ym, ys = Otr.mean(), Otr.std()
    print(f'anchors  train={len(tr):,}  test={len(te):,}\n')

    def build(a, steps, chans):
        past = Z[a[:, None] + np.arange(-L, 0)[None, :]].reshape(len(a), -1)
        parts = [past]
        if steps:
            fut = Z[a[:, None] + np.arange(steps)[None, :]][:, :, chans]
            parts.append(fut.reshape(len(a), -1))
        parts.append(np.ones((len(a), 1)))
        return np.concatenate(parts, 1)

    def run(name, steps, chans):
        W = ridge(build(tr, steps, chans), (Otr - ym) / ys)
        P = build(te, steps, chans) @ W * ys + ym
        rr = [np.corrcoef(P[:, k], Ote[:, k])[0, 1] for k in range(H)]
        pk = np.corrcoef(P.max(1), Ote.max(1))[0, 1]
        sdr = np.mean([P[:, k].std() / Ote[:, k].std() for k in range(H)])
        # storm detection at the operational threshold
        obs_st, prd_st = Ote.max(1) >= 100, P.max(1)
        cut = np.quantile(prd_st, 1 - obs_st.mean())      # equal-frequency issue
        hit = (prd_st >= cut) & obs_st
        pod = hit.sum() / obs_st.sum()
        print(f'{name:>32}{np.mean(rr):>10.3f}{rr[11]:>9.3f}{rr[23]:>9.3f}'
              f'{np.sqrt(np.mean((P - Ote) ** 2)):>9.3f}{pk:>10.3f}'
              f'{sdr:>10.3f}{pod:>9.3f}')
        return np.mean(rr)

    hdr = (f'{"known future wind":>32}{"mean rho":>10}{"rho@6h":>9}{"rho@12h":>9}'
           f'{"RMSE":>9}{"peak rho":>10}{"sd ratio":>10}{"POD100":>9}')
    print(hdr)
    base = run('none (current design)', 0, None)
    for hrs in (0.5, 1, 2, 3, 6, 12):
        s = int(hrs * 2)
        r = run(f'all 21 channels, +{hrs:g} h', s, slice(0, AP))
        print(f'{"":>32}{"":>10}gain vs current: {r - base:+.3f}')
    print()
    print(hdr)
    for nm, ch in [('Bz only (3 cols)', BZ), ('V only (3 cols)', V),
                   ('|B| only (3 cols)', BT), ('Bz + V (6 cols)', BZ + V)]:
        r = run(f'{nm}, full 12 h', H, ch)
        print(f'{"":>32}{"":>10}gain vs current: {r - base:+.3f}')


if __name__ == '__main__':
    main()
