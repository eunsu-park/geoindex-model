# Sampling the wind instead of forecasting it

`wind_ensemble.py`, run 2026-08-06 on the 23,514 validation anchors.

The tower property closes deterministic joint forecasting: for any deterministic `g`,
`E[Y | X, g(X)] = E[Y | X]`. It does not close the generative version. If `g` is a *sample*, the
ensemble spread is produced where the physics puts it. This is the one formulation left open
after every other axis closed.

An earlier attempt dressed the **ap output** in a residual process, which fixed the shape
(sharpness 2.73 against an observed 2.59) and left the level wrong (storm PIT 0.815). This
resamples the **wind**, before the mapping:

```
1  W_wind  (past ap, past wind) -> the next 12 h of wind
2  residual trajectories kept whole, so cross-variable and temporal structure survives
3  W_map   (past ap, past wind, future wind) -> the next 12 h of ap
4  W_map has its own irreducible error, so each member also carries a resampled W_map residual
5  50 members; both residual pools binned by predicted level
```

## The mapping, and a physical feature that pays

| given the TRUE future wind | per-lead `rho` |
|---|---|
| no future wind — the plain forecast | 0.569 |
| + future wind, linear | 0.785 |
| **+ future wind, rectified coupling** | **0.822** |

Only southward Bz reconnects, so a map linear in bz treats +5 nT and −5 nT alike. Adding
`Bs = max(0, −bz)` and `v·Bs` — two columns per step, the shape of the Newell coupling — is worth
**+0.037**. It pays only when the future wind is in hand, so it is not a forecast improvement, but
it is the difference between an ensemble that is physically meaningful and one that is not.

## What the ensemble buys

**The mean buys nothing, exactly as the theorem says.** Per-lead `rho` 0.557 against the plain
forecast's 0.569. A mean that had *improved* would have meant a leak, so this is the check that
the plumbing is right rather than a disappointment.

**The members are shaped like observations.** This is the first thing all August to fix the shape
without a degenerate trick.

| | sharpness (peak / window mean) |
|---|---|
| observed | 2.46 (storms 2.72) |
| direct forecast | 1.29 |
| ensemble mean | 1.34 |
| **a single member** | **2.51** |

**Overall calibration is right, and the exceedance probabilities beat a calibrated point
forecast.** Brier, against climatology and against the point forecast passed through an empirical
exceedance map:

| | base rate | climatology | point forecast | **ensemble** | ensemble skill |
|---|---|---|---|---|---|
| ap ≥ 48 | 0.205 | 0.1644 | 0.1076 | **0.1057** | 0.357 (point: 0.345) |
| ap ≥ 100 | 0.031 | 0.0303 | 0.0263 | **0.0246** | **0.188** (point: 0.132) |

The ap ≥ 100 gain matters most, because that is the threshold the investigation concluded cannot
be issued as a warning at all and has to be a probability. Skill rises from 0.132 to 0.188, a
42 % relative improvement, with no new information and no retrained network.

Mean PIT over all anchors is 0.502 against a calibrated 0.500, and the observed peak lands above
every one of the 50 members on 2.7 % of anchors against an expected 2.0 %.

## Where it still fails, and why

**Storms stay under-dispersed.** Mean PIT 0.757 on the 4,829 storm anchors, and the observed peak
is above every member 11.2 % of the time. Better than the ap-side attempt's 0.815, not fixed.

It is not a resolution problem — 10, 20 and 40 residual bins give 0.757, 0.752, 0.750, saturated.
The reason is structural and worth stating plainly: **`W_wind` is itself a conditional mean, so it
is damped by the same law that damps the index.** Members are correctly *shaped* but centered on a
wind forecast that is too quiet, and adding zero-mean residuals cannot move a center. The damping
moved down one level rather than disappearing.

That is the third time this month the level has been the hard part — the regime split fixed branch
level and not shape, the peak head fixed magnitude and not ranking, and the ensemble now fixes
shape and not level. Whatever is attacked, what survives is the conditional mean of the driver.

## What this is good for

Not `rho`, which the theorem forbids and the measurement confirms. It is a better **probability
product**: correctly shaped members, calibrated overall, and exceedance probabilities that beat
the calibrated point forecast at both thresholds, most where the point forecast is weakest. That
is the direction section 10 of the co-author report already recommended on other grounds.

The obvious next move, if this is pursued, is to attack the center rather than the spread — draw
wind trajectories from a conditional generator that is not a conditional mean, rather than from a
damped forecast plus zero-mean residuals.
