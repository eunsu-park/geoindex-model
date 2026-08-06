# How `per-lead rho` is computed, and how precise it is

Every table in this folder collapses a forecast into one number with this:

```python
def rho(P):                                  # P, Y are (anchors, 24 leads)
    return float(np.mean([np.corrcoef(P[:, k], Y[:, k])[0, 1] for k in range(OUT)]))
```

Two steps, and the first is the one that gets misread.

1. **Fix a lead, correlate across anchors.** For lead k — half an hour, one hour, … twelve hours —
   take the 23,514 predicted values at that lead and the 23,514 observed values, and compute the
   Pearson correlation between those two vectors. This is anomaly correlation at a fixed lead
   time, the standard forecast-verification quantity. It is **not** the correlation between a
   predicted curve and an observed curve within one event.
2. **Average the 24 numbers, unweighted.** Every lead counts the same, so the result sits between
   the easy short leads and the hard long ones.

For the deep baseline the 24 numbers run from 0.851 at +0.5 h to 0.384 at +12 h.

## The collapse is a choice, and it is worth 0.08

The same 24 correlations, combined differently:

| | value |
|---|---|
| **arithmetic mean — what every table here uses** | **0.5598** |
| Fisher-z mean, back-transformed | 0.5791 |
| pooled: all anchors × leads flattened into one correlation | 0.5674 |
| weighted by the observed variance at each lead | 0.5604 |
| arithmetic mean, computed on `log1p(ap)` instead of raw ap | 0.6432 |

The raw-versus-`log1p` row is the largest. Correlation is not invariant under a monotone
non-linear transform, and ap30 is heavy-tailed, so quoting a correlation without saying which
scale it is on is ambiguous by up to 0.08. Everything in this folder is on the **raw denormalized
ap30 scale**, matching what the stored validation archives hold.

## Precision: about ±0.015, not ±0.001

Anchors are thirty minutes apart and their twelve-hour windows overlap, so they are nowhere near
independent and the textbook standard error is far too small. Block bootstrap over 45 monthly
blocks, 300 draws:

| quantity | estimate | sd | 95 % interval |
|---|---|---|---|
| ridge with past wind only | 0.564 | 0.015 | [0.536, 0.594] |
| paired gain from true future v and bt | +0.107 | 0.013 | [0.083, 0.133] |

The naive iid standard error would be 0.0045 — **four times too small**.

Note the paired difference is *not* much more precise than the level it is a difference of. Pairing
removes the anchor-to-anchor variation but not the month-to-month variation, and the gain from
knowing the future wind is itself much larger in storm-rich months than in quiet ones.

**So a difference below about 0.03 is not resolvable by these measurements**, three printed
decimals notwithstanding. Read the tables accordingly:

- solid: +0.107 for the coronal-hole channel, +0.216 for the full headroom, the +0.000 rows for
  recurrence and long input windows, the correction from +0.037 to +0.107;
- **within noise**: ap30's +0.216 against hp30's +0.195, the envelope at +0.104 on average
  columns against +0.118 on the full set, and any two rows of the degradation sweeps that differ
  by a hundredth.

The degradation sweeps are a partial exception. There the same fitted model is scored on the same
anchors under successive corruptions, so the *ordering* down a column is reliable even where the
individual gaps are not.

## An inconsistency this check turned up

The hub documents quote the deep baseline's per-lead `rho` as **0.567**. The arithmetic mean of
its per-lead correlations is **0.5598**; 0.5674 is the *pooled* value, which matches 0.567 to three
decimals, so the earlier scripts were almost certainly reporting the pooled estimator.

That matters for one claim made repeatedly in this folder: that the ridge tracks the deep model,
"0.569 against 0.567". Those are two different estimators. Compared like with like the ridge is
0.569 against the deep model's 0.560 — a gap of 0.009, well inside the ±0.015 above. **The
conclusion survives and the quoted precision never existed.**
