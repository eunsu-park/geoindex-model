"""Proposal 2a asks for a 1 h cadence model with ~7 days of input.

The earlier scan on this data was at 30 min cadence and found the ridge peak at
12 h of input, declining monotonically past it. This re-runs the scan at the
proposed 1 h cadence and out to the proposed 7 days, and separately tests the
long tail as SUMMARIES (mean ap over the last 1/3/7 days) instead of as raw
sequence -- the distinction matters, because a 7 day raw window is 168 steps of
mostly-uninformative sequence while the same span as three scalars is nearly
free.

Targets are the 12 h ap window at 30 min cadence in every case, so the numbers
are comparable to everything else in this investigation.
"""
import numpy as np
import pandas as pd

DATA = '/Users/eunsupark/Projects/GeoIndex/datasets/data.parquet'
WIND = ['v_avg', 'v_min', 'v_max', 'np_avg', 'np_min', 'np_max',
        't_avg', 't_min', 't_max', 'bx_avg', 'bx_min', 'bx_max',
        'by_avg', 'by_min', 'by_max', 'bz_avg', 'bz_min', 'bz_max',
        'bt_avg', 'bt_min', 'bt_max']
COLS = WIND + ['ap30']
C, H = len(COLS), 24
AP = C - 1
MAXBACK = 7 * 48                       # 7 days at 30 min
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


def score(P, O):
    rr = [np.corrcoef(P[:, k], O[:, k])[0, 1] for k in range(O.shape[1])]
    return (np.mean(rr), np.corrcoef(P.max(1), O.max(1))[0, 1],
            np.sqrt(np.mean((P - O) ** 2)), np.mean(np.abs(P - O)))


def main():
    df = pd.read_parquet(DATA)
    dt = pd.to_datetime(df['datetime'])
    X = df[COLS].to_numpy(np.float64)
    ok = np.isfinite(X).all(1)
    anchors = np.flatnonzero(rolling_all_finite(ok, MAXBACK + H)) + MAXBACK
    t = dt.to_numpy()
    tr = anchors[t[anchors] < np.datetime64(TRAIN_END)]
    te = anchors[t[anchors] >= np.datetime64(TEST_START)]
    print(f'anchors  train={len(tr):,}  test={len(te):,}  '
          f'(all require {MAXBACK / 48:.0f} clean days of history)')

    rows = np.unique((tr[:, None] + np.arange(-MAXBACK, H)[None, :]).ravel())
    mu, sd = X[rows].mean(0), X[rows].std(0)
    Z = (X - mu) / sd
    Otr = X[tr[:, None] + np.arange(H)[None, :]][:, :, AP]
    Ote = X[te[:, None] + np.arange(H)[None, :]][:, :, AP]
    ym, ys = Otr.mean(), Otr.std()

    def run(name, build):
        Ftr, Fte = build(tr), build(te)
        W = ridge(Ftr, (Otr - ym) / ys)
        P = Fte @ W * ys + ym
        r, pk, rmse, mae = score(P, Ote)
        print(f'{name:>34}{Ftr.shape[1]:>8}{r:>11.4f}{pk:>11.4f}'
              f'{rmse:>10.3f}{mae:>9.3f}')

    def seq(hours, stride):
        """last `hours` of history at `stride`*30 min cadence, all 22 channels."""
        off = np.arange(-int(hours * 2), 0, stride)
        def build(a):
            F = Z[a[:, None] + off[None, :]].reshape(len(a), -1)
            return np.concatenate([F, np.ones((len(a), 1))], 1)
        return build

    print(f'\n{"input":>34}{"cols":>8}{"mean rho":>11}{"peak rho":>11}'
          f'{"RMSE":>10}{"MAE":>9}')
    print('--- raw sequence, 30 min cadence (the current design) ---')
    for h in (1, 3, 6, 12, 24, 48):
        run(f'{h:>3} h @ 30 min', seq(h, 1))
    print('--- raw sequence, 1 h cadence (the proposal) ---')
    for h in (6, 12, 24, 48, 72, 168):
        run(f'{h:>3} h @ 1 h  ({h / 24:.0f} d)', seq(h, 2))
    print('--- 12 h fine + long tail as SUMMARIES ---')

    def summary_build(days):
        off = np.arange(-24, 0)
        def build(a):
            F = [Z[a[:, None] + off[None, :]].reshape(len(a), -1)]
            for d in days:
                w = int(d * 48)
                idx = a[:, None] + np.arange(-w, 0)[None, :]
                F.append(Z[idx][:, :, AP].mean(1, keepdims=True))
                F.append(Z[idx][:, :, AP].max(1, keepdims=True))
                F.append(Z[idx][:, :, COLS.index('v_avg')].mean(1, keepdims=True))
                F.append(Z[idx][:, :, COLS.index('bz_min')].min(1, keepdims=True))
            F.append(np.ones((len(a), 1)))
            return np.concatenate(F, 1)
        return build

    run('12 h fine only (reference)', seq(12, 1))
    run('12 h fine + 1 d summary', summary_build([1]))
    run('12 h fine + 1,3 d summaries', summary_build([1, 3]))
    run('12 h fine + 1,3,7 d summaries', summary_build([1, 3, 7]))


if __name__ == '__main__':
    main()
