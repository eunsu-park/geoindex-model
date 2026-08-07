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

## The half of proposal 1 that looked promising — and did not survive

**Verdict up front: it failed its pass mark on the real architecture.** The MLP result below is
kept because it is what motivated the feature and because the size of its error is the point;
read it with the two subsections that follow it.

"Forecast every parameter" and "roll a one-step model out" arrived together and are separable.
Predicting all 22 channels **directly** at all 24 leads, scored on the ap slice only, 4 seeds:

| | ap-only | multitask | delta |
|---|---|---|---|
| mean per-lead rho | 0.5413 ±0.0023 | **0.5710 ±0.0029** | **+0.0297** |
| rho at 12 h | 0.3340 ±0.0019 | 0.3758 ±0.0061 | +0.0418 |
| RMSE | 13.900 ±0.031 | 13.465 ±0.038 | −0.435 |
| MAE | 7.119 ±0.014 | 6.938 ±0.014 | −0.181 |
| peak rho | 0.6645 ±0.0021 | 0.6788 ±0.0025 | +0.0143 |

Seed ranges do not overlap on any row; the gap is ~10x the seed spread — on an MLP. Reported at
the time as the largest model-side gain in the investigation, beating the peak head's +0.017.
**That reading is retracted:** it did not transfer, and the transfer risks were named before the
real run rather than after it.

Weighting ap 10x inside the auxiliary loss **halves** it (0.575 → 0.555 at seed 0), so the
gain is the wind task shaping the trunk, not extra ap effort — on an MLP. Shipped as
`training.aux_forecast` (off by default, and it stays off); the sweep is
`analysis/aux_forecast_probe.sh`.

### On the real architecture it does not hold — the pass mark failed

Run 2026-08-07, `gnn_transformer`, `in12h_out12h`, 6 variants x 2 seeds, scored on the 23,514
validation anchors with `analysis/score_per_lead.py`.

| variant | mean rho | rho@12h | peak rho | MAE |
|---|---|---|---|---|
| baseline | 0.5605 ±0.0008 | 0.3858 ±0.0017 | 0.6946 ±0.0029 | **7.165** |
| aux1 (w=1, MSE) | **0.5656** ±0.0015 | **0.4043** ±0.0001 | 0.6907 ±0.0004 | 7.258 |
| aux_mae | 0.5657 ±0.0059 | 0.3977 ±0.0075 | 0.6933 ±0.0032 | 7.450 |
| aux05 | 0.5652 ±0.0063 | 0.3969 ±0.0089 | 0.6926 ±0.0042 | 7.428 |
| aux2 | 0.5645 ±0.0011 | 0.4010 ±0.0025 | 0.6884 ±0.0014 | 7.335 |
| aux01 (w=0.1) | 0.5615 ±0.0022 | 0.3874 ±0.0032 | 0.6952 ±0.0023 | 7.312 |

The pass mark, fixed before the run, was per-lead rho beating baseline by more than half the
predicted +0.030 with peak rho not falling. **Both clauses fail**: per-lead rho moves +0.0051,
a sixth of the prediction, and peak rho drops 0.0039. MAE gets worse too.

The effect is nonetheless real rather than noise. Seed ranges do not overlap on mean rho
(baseline {0.5598, 0.5613} against aux1 {0.5641, 0.5671}); **rho at 12 h moves +0.0185**, some
11x the baseline seed sd, with the two aux1 seeds landing at 0.4043 and 0.4042; and the dose
response is clean — w=0.1 sits on the baseline and w=0.5/1/2 saturate together. What the
architecture does not reproduce is the *size*: the MLP prediction was wrong by a factor of six.

So the honest statement of the finding is narrow: **the auxiliary task improves long-lead
discrimination by about +0.019 rho and nothing else.** It does not improve the curve as a
whole, the peak, or the error.

#### The confound was tested and is not the explanation

Best epoch, out of 100:

| run | patience 10 | patience 30 |
|---|---|---|
| baseline_s0 | 5 (15 epochs run) | **5** (35 run) |
| baseline_s1 | 4 (14 run) | **4** (34 run) |
| aux1_s0 | 3 (13 run) | **3** (33 run) |
| aux1_s1 | 3 (13 run) | **3** (33 run) |

Every run early-stops between epoch 3 and 6 of 100, and the auxiliary runs stop earlier than the
baseline (3 against 4–5), which looked like the stopping rule cutting the representation off
before it could pay — the MLP probe had no early stopping at all, a fixed 15 epochs on OneCycle,
and that was one of the three transfer risks named in advance.

Tripling the patience refutes it. The patience-30 runs really did train 33–35 epochs, and the
best epoch and best validation regression loss are **unchanged to five decimals**; thirty further
epochs found nothing better. Training is deterministic given the seed, so the selected checkpoint
is the same file and the validation scores are identical digit for digit. The validation loss
genuinely bottoms early and does not recover.

That closes the auxiliary question — `training.aux_forecast` stays off by default, and the
finding stands as the narrow one above.

It also clears a worry this raised about the rest of the investigation. If patience 10 had been
binding, every null measured under it would have been suspect, including the 24 loss-probe
variants. It was not binding, so those nulls do not need re-testing on these grounds. Directly
measured on the baseline and aux1 configurations only, but all twelve runs bottom in the same
3–6 epoch band, so the mechanism looks common rather than particular.
