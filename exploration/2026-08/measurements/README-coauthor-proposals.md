# Two proposed redesigns, priced

Run 2026-08-07 in response to a co-author proposal. Everything below uses the same
`data.parquet`, the same train/test split (train < 2021-12, test 2022-01 onward), and the
same 12 h ap30 target window as the rest of this investigation, so the numbers sit next to
the existing ones without translation.

```
recursive_vs_direct.py    proposal 1, linear   -- and why linear cannot answer it
recursive_nonlinear.py    proposal 1, nonlinear -- rollout vs direct vs unrolled loss
multitask_direct.py       the salvageable half of proposal 1
cadence_and_downscale.py  proposal 2b -- the 1 h -> 30 min interpolator
input_length_1h.py        proposal 2a -- 1 h cadence, 7 days of input
wind_oracle_curve.py      what all three proposals are actually bets on
```

## Proposal 1 — a 30 min one-step model over all channels, rolled out to 12 h

Ridge says recursive 0.592 and direct 0.594, but that is a near-tautology: a linear one-step
model applied 24 times is a matrix power, still a linear function of the input window, so the
two are the same hypothesis class. The question only exists for a nonlinear model.

Same trunk, same budget, same anchors, an MLP:

| scheme | mean rho | rho@1h | rho@6h | rho@12h | RMSE | MAE | sd ratio |
|---|---|---|---|---|---|---|---|
| direct, 528 -> 24 ap leads | 0.539 | 0.841 | 0.522 | **0.337** | 13.933 | 7.139 | 0.697 |
| one-step over 22 channels, rolled out 24x | 0.538 | **0.851** | 0.520 | 0.307 | 14.377 | **6.835** | 0.643 |
| the same net trained through the rollout | 0.506 | 0.815 | 0.487 | 0.299 | 14.319 | 7.142 | 0.672 |

A wash on the mean, with the profiles crossing near 4–5 h: **the rollout wins at short lead
and loses at long lead.** Backpropagating through the 24 steps — the textbook fix for
exposure bias — is worse everywhere. (It got 8 epochs against 15, at 24x the cost per epoch,
with a flattening loss curve; the gap is too large to be that.)

The mechanism is visible directly. Fraction of the observed standard deviation the rolled-out
`bz_min` still carries:

| lead | 0.5 h | 2.5 h | 6.5 h | 11.5 h |
|---|---|---|---|---|
| sd retained | 0.805 | 0.509 | 0.319 | 0.203 |

The rollout does not forecast the wind, it relaxes it to climatology. Past ~3 h the model is
coasting on an input that has already gone flat.

## The half of proposal 1 that works

"Forecast every parameter" and "roll a one-step model out" arrived together and are separable.
Predicting all 22 channels **directly** at all 24 leads, scored on the ap slice only, 4 seeds:

| | ap-only | multitask | delta |
|---|---|---|---|
| mean per-lead rho | 0.5413 ±0.0023 | **0.5710 ±0.0029** | **+0.0297** |
| rho at 12 h | 0.3340 ±0.0019 | 0.3758 ±0.0061 | +0.0418 |
| RMSE | 13.900 ±0.031 | 13.465 ±0.038 | −0.435 |
| MAE | 7.119 ±0.014 | 6.938 ±0.014 | −0.181 |
| peak rho | 0.6645 ±0.0021 | 0.6788 ±0.0025 | +0.0143 |

Seed ranges do not overlap on any row; the gap is ~10x the seed spread. This is the largest
model-side gain in the investigation — the previous best was the peak head at +0.017.

Weighting ap 10x inside the auxiliary loss **halves** it (0.575 → 0.555 at seed 0), so the
gain is the wind task shaping the trunk, not extra ap effort. Shipped as
`training.aux_forecast` (off by default); the sweep is `analysis/aux_forecast_probe.sh`.

## Proposal 2a — 1 h cadence, 7 days of input

Anchors here require 7 clean days of history, so the set is smaller and slightly quieter than
elsewhere; read the rows against each other, not against other tables.

| input | cols | mean rho | peak rho | MAE |
|---|---|---|---|---|
| 1–3 h @ 30 min | 45–133 | **0.5833** | 0.663 | 7.43 |
| 12 h @ 30 min | 529 | 0.5764 | 0.660 | 7.49 |
| 12 h @ 1 h | 265 | 0.5545 | 0.636 | 7.65 |
| **7 d @ 1 h (the proposal)** | 3697 | **0.5287** | 0.622 | 8.31 |
| 12 h fine + 1/3/7 d summaries | 541 | 0.5781 | 0.661 | 7.49 |

The worst configuration tested. Downsampling 30 min → 1 h costs −0.022 on its own; the length
takes the rest. The 7 day span is not worthless — as four summary scalars it is slightly
positive — but as 168 raw timesteps it is a liability.

## Proposal 2b — forecast ap hourly, then interpolate to 30 min from earlier ap and Bz

`ap30` against the `ap60` of its own hour: rho **0.960**, 92.0 % of the variance already
fixed, residual sd 4.51 ap.

The remaining within-hour split, with the true `ap60` handed over for free:

| lead | 0.5 h | 1.5 h | 2.5 h | 3.5 h+ |
|---|---|---|---|---|
| rho from anchor data | 0.180 | 0.036 | 0.009 | ≤0.03 |
| variance explained | 0.032 | 0.001 | 0.000 | ≤0.001 |

End to end, against the same ap30 targets:

| scheme | mean rho | RMSE | MAE | peak rho |
|---|---|---|---|---|
| direct 30 min | 0.590 | 13.201 | 7.080 | 0.675 |
| hourly + constant fill | 0.590 | 13.212 | 7.102 | 0.680 |
| hourly + learned split | 0.590 | 13.201 | 7.080 | 0.675 |
| *(perfect ap60 + constant fill)* | *0.954* | *4.999* | *2.495* | *0.982* |

The learned split is indistinguishable from copying the hourly value into both halves. Its
whole budget is the 4.51 ap of within-hour structure, and the hourly forecast currently errs
by 13.2 — the correction is buried. It would only surface once the hourly model is ~2.5x
better, and the row above says it would not be recoverable from ap and Bz history even then.

## What all three proposals are really bets on

Same anchors, same targets, past 12 h of everything plus the **true** future wind out to
`delta`:

| known future wind | mean rho | rho@12h | peak rho | POD(ap≥100) | gain |
|---|---|---|---|---|---|
| none (current design) | 0.594 | 0.405 | 0.681 | 0.411 | — |
| all channels, +1 h | 0.622 | 0.428 | 0.709 | 0.443 | +0.029 |
| all channels, +3 h | 0.676 | 0.474 | 0.756 | 0.503 | +0.083 |
| all channels, +6 h | 0.743 | 0.543 | 0.807 | 0.562 | +0.149 |
| **all channels, +12 h** | **0.813** | **0.792** | **0.862** | **0.651** | **+0.220** |

On the nonlinear model the same oracle runs 0.539 → **0.843**, rho at 12 h 0.337 → **0.833**.

Split by channel, each given the full 12 h:

| channel | gain |
|---|---|
| **Bz (3 cols)** | **+0.166** |
| \|B\| (3 cols) | +0.109 |
| V (3 cols) | +0.048 |
| Bz + V | +0.175 |

The ap forecasting problem is the Bz forecasting problem. Loss, architecture, cadence and
input length rearrange information the model already has, and the best any of them returned in
this investigation is +0.017. The **smallest** row in the table above is worth twice that.

The oracle is not attainable — that is the point of calling it one. It prices the bet, and it
says any proposal that does not add an exogenous driver for the wind is competing for the
scraps.

## On the imagery half of proposal 2

Solar imagery is the only exogenous driver on the table, so it aims at the right prize. Two
things bound what it can collect.

The prize is specifically Bz, and imagery has no established skill for ICME internal field
orientation at 1 AU; its established skill is arrival time and speed. In the table above that
is the V (+0.048) and |B| (+0.109) share, not the Bz (+0.166) share.

And SDO starts in 2010-05. Training anchors surviving that cut, by storm size:

| 12 h peak ap | 1995+ | 2010.5+ | kept |
|---|---|---|---|
| ≥ 0 | 471,888 | 203,136 | 43.0 % |
| ≥ 50 | 45,833 | 14,134 | 30.8 % |
| ≥ 100 | 10,742 | 2,848 | 26.5 % |
| ≥ 200 | 2,142 | 272 | **12.7 %** |
| ≥ 400 | 161 | 0 | **0 %** |

Cycle 23 maximum and the 2003 Halloween storm (12 h peak ap 617) leave entirely. Imagery must
therefore be an optional auxiliary channel behind a missing-data mask, never a required input,
with the backbone trained on 1995+ and the imagery adapter fitted on 2010+ only.
