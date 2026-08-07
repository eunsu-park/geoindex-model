"""Co-author proposal 2b: forecast ap at 1 h cadence, then downscale to 30 min
using the earlier ap and the earlier Bz.

Three things get measured, each one a ceiling the downscaler cannot exceed:

  Q1  how much of ap30 is already fixed by the hour it sits in
      (correlation of ap30 with the ap60 of its own hour, and the residual sd)
  Q2  with the TRUE ap60 handed over for free, how much of the remaining
      within-hour split is recoverable from data available at forecast time,
      as a function of how stale that data is
  Q3  end to end on identical anchors and identical ap30 targets:
        direct30   -- ridge straight to 24 ap30 leads
        hourly+fill-- ridge to 12 ap60 leads, each hour copied into both halves
        hourly+split-- same, plus the Q2 downscaler
"""
import numpy as np
import pandas as pd

DATA = '/Users/eunsupark/Projects/GeoIndex/datasets/data.parquet'
AP60 = ('/private/tmp/claude-501/-Users-eunsupark-GitHub-njit-geoindex-geoindex/'
        '2fa134a0-debc-458e-b54a-ce50829dbdb1/scratchpad/ap60.csv')
WIND = ['v_avg', 'v_min', 'v_max', 'np_avg', 'np_min', 'np_max',
        't_avg', 't_min', 't_max', 'bx_avg', 'bx_min', 'bx_max',
        'by_avg', 'by_min', 'by_max', 'bz_avg', 'bz_min', 'bz_max',
        'bt_avg', 'bt_min', 'bt_max']
COLS = WIND + ['ap30']
L, H = 24, 24
TRAIN_END, TEST_START = '2021-12-01', '2022-01-01'


def rolling_all_finite(mask, w):
    c = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
    n = len(mask) - w + 1
    return (c[w:w + n] - c[:n]) == w


def ridge(F, Y, alpha):
    p = F.shape[1]
    R = np.eye(p) * alpha * len(F) / 1e4
    R[-1, -1] = 0.0
    return np.linalg.solve(F.T @ F + R, F.T @ Y)


def main():
    df = pd.read_parquet(DATA)
    dt = pd.to_datetime(df['datetime'])
    X = df[COLS].to_numpy(np.float64)
    ap30 = X[:, -1]

    h60 = pd.read_csv(AP60, parse_dates=['datetime']).set_index('datetime')['ap60']
    # ap60 of the hour each 30-min slot belongs to
    hour_key = dt.dt.floor('h')
    ap60_of_slot = hour_key.map(h60).to_numpy(np.float64)

    # ------------------------------------------------------------------ Q1
    m = np.isfinite(ap30) & np.isfinite(ap60_of_slot)
    a30, a60 = ap30[m], ap60_of_slot[m]
    resid = a30 - a60
    print('Q1  ap30 against the ap60 of its own hour')
    print(f'    n = {m.sum():,}')
    print(f'    corr(ap30, ap60_of_hour)      = {np.corrcoef(a30, a60)[0, 1]:.4f}')
    print(f'    variance of ap30 explained    = {np.corrcoef(a30, a60)[0, 1] ** 2:.4f}')
    print(f'    sd(ap30) = {a30.std():.3f}   sd(ap30 - ap60) = {resid.std():.3f}'
          f'   -> {1 - resid.var() / a30.var():.4f} of variance removed')
    first = (dt.dt.minute == 0).to_numpy()[m]
    split = a30[first] - a30[~first][:first.sum()]        # first half minus second
    print(f'    sd of the within-hour split   = {split.std():.3f} ap'
          f'   (sd of ap30 itself = {a30.std():.3f})')
    for thr in (30, 50, 100):
        sel = a60 >= thr
        if sel.sum() > 500:
            print(f'    ap60 >= {thr:>3}: n={sel.sum():>7,}  sd(ap30-ap60) ='
                  f' {resid[sel].std():>7.3f}  mean|ap30-ap60| = '
                  f'{np.abs(resid[sel]).mean():>6.3f}')

    # ------------------------------------------------------------ anchors
    ok = np.isfinite(X).all(1) & np.isfinite(ap60_of_slot)
    good = rolling_all_finite(ok, L + H)
    anchors = np.flatnonzero(good) + L
    anchors = anchors[dt.dt.minute.to_numpy()[anchors] == 0]   # anchor on the hour
    t = dt.to_numpy()
    tr = anchors[t[anchors] < np.datetime64(TRAIN_END)]
    te = anchors[t[anchors] >= np.datetime64(TEST_START)]
    print(f'\nanchors  train={len(tr):,}  test={len(te):,}  (on the hour)')

    rows_tr = np.unique((tr[:, None] + np.arange(-L, H)[None, :]).ravel())
    mu, sd = X[rows_tr].mean(0), X[rows_tr].std(0)
    Z = (X - mu) / sd

    def feats(a):
        idx = a[:, None] + np.arange(-L, 0)[None, :]
        return np.concatenate([Z[idx].reshape(len(a), -1), np.ones((len(a), 1))], 1)

    Ftr, Fte = feats(tr), feats(te)
    lead = np.arange(H)
    y30_tr = ap30[tr[:, None] + lead[None, :]]
    y30_te = ap30[te[:, None] + lead[None, :]]
    y60_tr = ap60_of_slot[tr[:, None] + lead[None, :]][:, ::2]     # 12 hourly leads
    y60_te = ap60_of_slot[te[:, None] + lead[None, :]][:, ::2]

    # ------------------------------------------------------------- Q3 (a)
    W30 = ridge(Ftr, (y30_tr - mu[-1]) / sd[-1], 1.0)
    p30 = Fte @ W30 * sd[-1] + mu[-1]

    # ------------------------------------------------------------- Q3 (b)
    m60, s60 = y60_tr.mean(), y60_tr.std()
    W60 = ridge(Ftr, (y60_tr - m60) / s60, 1.0)
    p60 = Fte @ W60 * s60 + m60
    p_fill = np.repeat(p60, 2, axis=1)

    # ------------------------------------------------------------------ Q2
    # target: the within-hour deviation of ap30 from its own hour's ap60,
    # given the TRUE ap60. Predictors available at anchor time only.
    dev_tr = y30_tr - np.repeat(y60_tr, 2, axis=1)
    dev_te = y30_te - np.repeat(y60_te, 2, axis=1)
    print('\nQ2  within-hour deviation, with the true ap60 given for free')
    print(f'{"lead(h)":>8}{"sd(dev) ap":>12}{"rho from anchor data":>24}'
          f'{"var explained":>15}')
    Wd = ridge(Ftr, dev_tr, 1.0)
    pd_te = Fte @ Wd
    for k in range(0, H, 2):
        d, q = dev_te[:, k], pd_te[:, k]
        r = np.corrcoef(d, q)[0, 1]
        print(f'{(k + 1) * 0.5:>8.1f}{d.std():>12.3f}{r:>24.3f}{r ** 2:>15.4f}')

    # ------------------------------------------------------------- Q3 (c)
    p_split = p_fill + pd_te

    print('\nQ3  scored against the same ap30 targets')
    hdr = (f'{"scheme":>26}{"mean rho":>10}{"RMSE":>9}{"MAE":>9}'
           f'{"peak rho":>10}{"sd ratio":>10}')
    print(hdr)
    for name, p in [('direct 30-min', p30),
                    ('hourly + constant fill', p_fill),
                    ('hourly + learned split', p_split)]:
        rr = [np.corrcoef(p[:, k], y30_te[:, k])[0, 1] for k in range(H)]
        pk = np.corrcoef(p.max(1), y30_te.max(1))[0, 1]
        print(f'{name:>26}{np.mean(rr):>10.3f}'
              f'{np.sqrt(np.mean((p - y30_te) ** 2)):>9.3f}'
              f'{np.mean(np.abs(p - y30_te)):>9.3f}{pk:>10.3f}'
              f'{np.mean([p[:, k].std() / y30_te[:, k].std() for k in range(H)]):>10.3f}')

    # what if the hourly model were perfect?
    perfect = np.repeat(y60_te, 2, axis=1)
    rr = [np.corrcoef(perfect[:, k], y30_te[:, k])[0, 1] for k in range(H)]
    print(f'{"PERFECT ap60 + fill":>26}{np.mean(rr):>10.3f}'
          f'{np.sqrt(np.mean((perfect - y30_te) ** 2)):>9.3f}'
          f'{np.mean(np.abs(perfect - y30_te)):>9.3f}'
          f'{np.corrcoef(perfect.max(1), y30_te.max(1))[0, 1]:>10.3f}'
          f'{np.mean([perfect[:, k].std() / y30_te[:, k].std() for k in range(H)]):>10.3f}')
    perfect_split = perfect + dev_te
    print(f'{"PERFECT ap60 + true split":>26}  (identity, sanity check) '
          f'RMSE={np.sqrt(np.mean((perfect_split - y30_te) ** 2)):.6f}')


if __name__ == '__main__':
    main()
