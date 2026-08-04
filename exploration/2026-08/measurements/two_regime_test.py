"""Would splitting into a quiet model and a storm model recover the damping?

The single curve is damped because one number has to hedge across every outcome consistent with
the input: sigma_pred/sigma_obs = rho. Conditioning removes that hedge, and the decomposition is
exact:

    E[y|x] = P(storm|x) E[y|x, storm] + (1 - P(storm|x)) E[y|x, quiet]

So the question is quantitative, and the law of total variance answers it:

    Var(y|x) = E[Var(y|x, branch)]  +  Var(E[y|x, branch])
               ^ within a branch        ^ between branches -- this is what splitting recovers

If most of the conditional variance is BETWEEN branches, two models recover most of the damping.
If it is WITHIN, they recover little and the scenarios are two damped curves instead of one.

Also measured: what a storm-only model actually predicts on storms, whether the two-branch
mixture beats the pooled forecast on a proper score, and how many anchors a storm model would
have to train on.

Ridges, trained 1998-2021, scored 2022-2025, on the observation-time table.
"""

import os

import numpy as np
import pandas as pd

D = os.path.expanduser("~/Projects/GeoIndex/datasets")
WIND = ["v_avg", "np_avg", "t_avg", "bx_avg", "by_avg", "bz_avg", "bt_avg"]
PAST, OUT = 24, 24
TRAIN_FROM, TEST_FROM = "1998-01-01", "2022-01-01"


def ridge(X, y, alpha=100.0):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    yb = y.mean()
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - yb))
    return lambda Xn: ((Xn - mu) / sd) @ w + yb


df = pd.read_parquet(os.path.join(D, "data_obs.parquet")).set_index("datetime").sort_index()
grid = pd.date_range(df.index[0], df.index[-1], freq="30min")
df = df.reindex(grid)
cols = WIND + ["ap30"]
df[cols] = df[cols].interpolate(limit=6).ffill().bfill()
ap = df["ap30"].to_numpy(float)
n = len(grid)
n_a = n - OUT
idx = np.arange(n_a)
times = grid[:n_a]
peak = np.lib.stride_tricks.sliding_window_view(ap, OUT)[:n_a].max(axis=1)


def lags(a, k=PAST):
    out = []
    for b in range(k):
        lo = idx - k + b
        c = np.full(n_a, np.nan)
        ok = lo >= 0
        c[ok] = a[lo[ok]]
        out.append(c)
    return out


X = np.column_stack([c for w in WIND for c in lags(df[w].to_numpy(float))] + lags(ap))
ok = ~np.isnan(X).any(1)
tr = ok & (times >= pd.Timestamp(TRAIN_FROM)) & (times < pd.Timestamp(TEST_FROM))
te = ok & (times >= pd.Timestamp(TEST_FROM))
print(f"train {tr.sum()}, test {te.sum()}")

for THR in (39.0, 48.0, 56.0):
    print(f"\n{'='*72}\nSTORM DEFINED AS peak >= {THR:.0f} ap\n{'='*72}")
    storm = peak >= THR
    n_tr_storm = int((tr & storm).sum())
    print(f"training anchors: quiet {int((tr & ~storm).sum()):,}  "
          f"storm {n_tr_storm:,} ({100*n_tr_storm/tr.sum():.1f} %)")

    # ── law of total variance on the held-out set ────────────────────────────
    y = peak[te]
    s = storm[te]
    within = s.mean() * y[s].var() + (1 - s.mean()) * y[~s].var()
    between = s.mean() * (y[s].mean() - y.mean()) ** 2 + \
        (1 - s.mean()) * (y[~s].mean() - y.mean()) ** 2
    print(f"\nvariance of the observed peak: total {y.var():.0f}")
    print(f"  within branches  {within:9.0f}  ({100*within/y.var():4.1f} %)  "
          f"-- stays damped even after splitting")
    print(f"  between branches {between:9.0f}  ({100*between/y.var():4.1f} %)  "
          f"-- this is what two models recover")

    # ── the three forecasts ─────────────────────────────────────────────────
    f_pool = ridge(X[tr], peak[tr])
    f_quiet = ridge(X[tr & ~storm], peak[tr & ~storm])
    f_storm = ridge(X[tr & storm], peak[tr & storm])
    p_pool, p_quiet, p_storm = f_pool(X[te]), f_quiet(X[te]), f_storm(X[te])

    def line(lab, p, mask=None):
        m = np.ones(len(y), bool) if mask is None else mask
        rho = float(np.corrcoef(p[m], y[m])[0, 1])
        b = float(np.polyfit(y[m], p[m], 1)[0])
        hi = y[m] >= 100
        rep = float(p[m][hi].mean() / y[m][hi].mean()) if hi.sum() > 30 else float("nan")
        print(f"  {lab:34s} {rho:7.3f} {b:7.3f} {p[m].std()/y[m].std():7.3f} "
              f"{rep:8.3f} {np.abs(p[m]-y[m]).mean():8.2f}")

    print(f"\nscored on ALL held-out anchors      {'rho':>7s} {'slope':>7s} {'disp':>7s} "
          f"{'repro':>8s} {'MAE':>8s}")
    line("pooled model (today)", p_pool)
    print(f"\nscored on STORM anchors only (n={int(s.sum())})")
    line("pooled model", p_pool, s)
    line("storm-only model", p_storm, s)
    print(f"\nscored on QUIET anchors only (n={int((~s).sum())})")
    line("pooled model", p_pool, ~s)
    line("quiet-only model", p_quiet, ~s)

    # ── the mixture: does presenting both, weighted, beat the pooled curve? ──
    # P(storm|x) from a logistic-ish ridge on the same features, fitted on train only.
    f_p = ridge(X[tr], storm[tr].astype(float))
    ps = np.clip(f_p(X[te]), 0.001, 0.999)
    mix = ps * p_storm + (1 - ps) * p_quiet
    print(f"\nmixture  P(storm|x)*storm + (1-P)*quiet")
    print(f"  {'':34s} {'rho':>7s} {'slope':>7s} {'disp':>7s} {'repro':>8s} {'MAE':>8s}")
    line("mixture", mix)
    line("pooled model (same rows)", p_pool)
    print(f"  P(storm|x): mean {ps.mean():.3f}, observed rate {s.mean():.3f}, "
          f"AUC-ish rho with outcome {np.corrcoef(ps, s)[0,1]:.3f}")

    # what the storm branch would tell a user, and how often it is right
    print(f"\n  the scenario a user would read:")
    print(f"    'if a storm comes'   -> {p_storm[s].mean():6.1f} ap   "
          f"(observed on storms {y[s].mean():6.1f})")
    print(f"    'if it stays quiet'  -> {p_quiet[~s].mean():6.1f} ap   "
          f"(observed on quiet   {y[~s].mean():6.1f})")
