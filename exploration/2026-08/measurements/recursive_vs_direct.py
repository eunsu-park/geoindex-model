"""Co-author proposal 1: one-step (30 min) all-channel forecaster rolled out to 12 h.

Compares, on identical anchors and identical targets:
  direct     -- one ridge per lead k, features = 12 h of all 22 channels
  recursive  -- one ridge for lead 1 over all 22 channels, fed back 24 times
  rec+oracle -- recursive in ap only; wind channels are replaced by the TRUE
                future observations at every step (upper bound on what a
                perfect wind forecaster would buy the recursive scheme)
  persistence-- ap_hat(k) = ap at the last input step

Ridge is used as a capacity-controlled stand-in for the deep model: prior work
in this repo measured closed-form ridge peak rho 0.695 vs the best of 14 deep
architectures at 0.699, so a gap that shows up here is a property of the
forecasting scheme, not of the model class.
"""
import numpy as np
import pandas as pd

DATA = '/Users/eunsupark/Projects/GeoIndex/datasets/data.parquet'
WIND = ['v_avg', 'v_min', 'v_max', 'np_avg', 'np_min', 'np_max',
        't_avg', 't_min', 't_max', 'bx_avg', 'bx_min', 'bx_max',
        'by_avg', 'by_min', 'by_max', 'bz_avg', 'bz_min', 'bz_max',
        'bt_avg', 'bt_min', 'bt_max']
COLS = WIND + ['ap30']
AP = len(COLS) - 1          # index of the ap channel
L = 24                      # 12 h of input (measured optimum for this data)
H = 24                      # 12 h of output at 30 min cadence
TRAIN_END = '2021-12-01'
TEST_START = '2022-01-01'


def rolling_all_finite(mask, w):
    """True at i when mask[i:i+w] is all True."""
    c = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
    n = len(mask) - w + 1
    return (c[w:w + n] - c[:n]) == w


def build_anchors(df):
    X = df[COLS].to_numpy(np.float64)
    ok = np.isfinite(X).all(1)
    span = L + H
    good = rolling_all_finite(ok, span)          # good[j] -> rows j..j+span-1 clean
    starts = np.flatnonzero(good)
    anchors = starts + L                          # input rows a-L..a-1, target a..a+H-1
    return X, anchors


def gram(X, anchors, chunk=20000):
    """Accumulate normal equations for features = flattened input window."""
    p = L * len(COLS) + 1
    XtX = np.zeros((p, p))
    XtY = np.zeros((p, len(COLS) * 1 + H))       # [22 next-step channels | H ap leads]
    n = 0
    for s in range(0, len(anchors), chunk):
        a = anchors[s:s + chunk]
        idx = a[:, None] + np.arange(-L, 0)[None, :]
        F = X[idx].reshape(len(a), -1)
        F = np.concatenate([F, np.ones((len(a), 1))], 1)
        tgt_idx = a[:, None] + np.arange(H)[None, :]
        Y = np.concatenate([X[a], X[tgt_idx][:, :, AP]], 1)
        XtX += F.T @ F
        XtY += F.T @ Y
        n += len(a)
    return XtX, XtY, n


def solve(XtX, XtY, alpha):
    p = XtX.shape[0]
    R = np.eye(p) * alpha
    R[-1, -1] = 0.0                               # do not penalise the bias
    return np.linalg.solve(XtX + R, XtY)


def features(X, a):
    idx = a[:, None] + np.arange(-L, 0)[None, :]
    F = X[idx].reshape(len(a), -1)
    return np.concatenate([F, np.ones((len(a), 1))], 1)


def metrics(pred, obs):
    """pred/obs are (n, H) in physical ap units."""
    out = []
    for k in range(pred.shape[1]):
        p, o = pred[:, k], obs[:, k]
        rho = np.corrcoef(p, o)[0, 1]
        rmse = np.sqrt(np.mean((p - o) ** 2))
        out.append((rho, rmse, np.mean(np.abs(p - o)), p.std() / o.std()))
    return np.array(out)


def main():
    df = pd.read_parquet(DATA)
    dt = pd.to_datetime(df['datetime'])
    X, anchors = build_anchors(df)

    tr = anchors[dt.to_numpy()[anchors] < np.datetime64(TRAIN_END)]
    te = anchors[dt.to_numpy()[anchors] >= np.datetime64(TEST_START)]
    print(f'anchors  train={len(tr):,}  test={len(te):,}')

    # z-score on train rows only
    rows_tr = np.unique((tr[:, None] + np.arange(-L, H)[None, :]).ravel())
    mu, sd = X[rows_tr].mean(0), X[rows_tr].std(0)
    Z = (X - mu) / sd

    XtX, XtY, n = gram(Z, tr)
    # small internal split for alpha: reuse train gram, score on test-free tail of train
    hold = tr[int(len(tr) * 0.9):]
    fit = tr[:int(len(tr) * 0.9)]
    XtXf, XtYf, _ = gram(Z, fit)
    best, best_rho = None, -9
    for alpha in [1e-2, 1e-1, 1, 10, 100, 1000]:
        W = solve(XtXf, XtYf, alpha * len(fit) / 1e4)
        Fh = features(Z, hold)
        yh = Fh @ W[:, len(COLS):]
        oh = Z[hold[:, None] + np.arange(H)[None, :]][:, :, AP]
        r = np.mean([np.corrcoef(yh[:, k], oh[:, k])[0, 1] for k in range(H)])
        print(f'  alpha={alpha:>7}  mean per-lead rho (holdout) = {r:.4f}')
        if r > best_rho:
            best, best_rho = alpha, r
    print(f'chosen alpha = {best}')
    W = solve(XtX, XtY, best * n / 1e4)
    W_step, W_direct = W[:, :len(COLS)], W[:, len(COLS):]

    obs = X[te[:, None] + np.arange(H)[None, :]][:, :, AP]

    # ---- direct -----------------------------------------------------------
    pred_direct = (features(Z, te) @ W_direct) * sd[AP] + mu[AP]

    # ---- recursive --------------------------------------------------------
    def rollout(oracle_wind):
        out = np.zeros((len(te), H))
        buf = Z[te[:, None] + np.arange(-L, 0)[None, :]].copy()   # (n, L, C)
        for k in range(H):
            F = np.concatenate([buf.reshape(len(te), -1),
                                np.ones((len(te), 1))], 1)
            nxt = F @ W_step                                       # (n, C)
            out[:, k] = nxt[:, AP]
            if oracle_wind:
                nxt[:, :AP] = Z[te + k][:, :AP]
            buf = np.concatenate([buf[:, 1:], nxt[:, None, :]], 1)
        return out * sd[AP] + mu[AP]

    pred_rec = rollout(False)
    pred_orc = rollout(True)
    pred_per = np.repeat(X[te - 1, AP][:, None], H, 1)

    names = ['direct', 'recursive', 'rec+oracle wind', 'persistence']
    res = {n_: metrics(p, obs) for n_, p in
           zip(names, [pred_direct, pred_rec, pred_orc, pred_per])}

    print('\nper-lead correlation rho')
    print('lead(h) ' + ''.join(f'{n_:>18}' for n_ in names))
    for k in range(H):
        row = ''.join(f'{res[n_][k, 0]:>18.3f}' for n_ in names)
        print(f'{(k + 1) * 0.5:>7.1f}' + row)

    print('\nsummary over the 12 h window')
    hdr = f'{"scheme":>18}{"mean rho":>10}{"RMSE":>9}{"MAE":>9}{"peak rho":>10}{"sigma_p/sigma_o":>18}'
    print(hdr)
    for n_, p in zip(names, [pred_direct, pred_rec, pred_orc, pred_per]):
        m = res[n_]
        pk_p, pk_o = p.max(1), obs.max(1)
        print(f'{n_:>18}{m[:, 0].mean():>10.3f}'
              f'{np.sqrt(np.mean((p - obs) ** 2)):>9.3f}'
              f'{np.mean(np.abs(p - obs)):>9.3f}'
              f'{np.corrcoef(pk_p, pk_o)[0, 1]:>10.3f}'
              f'{m[:, 3].mean():>18.3f}')

    # how fast does the recursive wind state die?
    print('\nrecursive wind-channel decay (fraction of observed sd retained)')
    buf = Z[te[:, None] + np.arange(-L, 0)[None, :]].copy()
    keep = []
    for k in range(H):
        F = np.concatenate([buf.reshape(len(te), -1), np.ones((len(te), 1))], 1)
        nxt = F @ W_step
        keep.append(nxt[:, COLS.index('bz_min')].std() /
                    Z[te + k][:, COLS.index('bz_min')].std())
        buf = np.concatenate([buf[:, 1:], nxt[:, None, :]], 1)
    for k in range(0, H, 2):
        print(f'  lead {(k + 1) * 0.5:>4.1f} h   bz_min sd ratio = {keep[k]:.3f}')

    np.savez('/private/tmp/claude-501/-Users-eunsupark-GitHub-njit-geoindex-geoindex/'
             '2fa134a0-debc-458e-b54a-ce50829dbdb1/scratchpad/recursive_vs_direct.npz',
             obs=obs, direct=pred_direct, recursive=pred_rec,
             oracle=pred_orc, persistence=pred_per)


if __name__ == '__main__':
    main()
