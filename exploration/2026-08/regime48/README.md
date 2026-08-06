# Two-regime training at ap30 ≥ 48 (Kp 5o)

## Verdict — negative. Do not train a separate storm model.

Measured on 23,514 validation anchors: the split plus a peak head raises the predicted storm
peak from 43 % of observed to 82 % and cuts peak MAE from 45.70 to 22.78. **Almost all of that
is bias removal, and bias removal is one free scalar.** Predicting a single constant — the
observed median — for every storm anchor gives MAE 25.59. The trained pair of models earns
2.8 ap beyond that. Correlation did not move at all (0.482 → 0.470), so no information was
added; the curve is as flat as it started (sharpness 1.31 against the observed 2.72).

The actionable finding is about the *pooled* model, not about splitting: **it is biased low on
storms, and a scalar correction on its output recovers most of the gap without retraining
anything.** Everything below is kept because the measurements are sound and the traps are worth
not falling into twice — but read it with this verdict in hand.

---

**Nothing in `src/`, `configs/` or `scripts/` is touched.** A regime is a property of the target,
known at training time, and table mode already reads its training anchors from a CSV — so the
whole split is two index files plus a `train_index` override. The manuscript tree stays exactly
as it was restored.

```
build_regime_indices.py    writes the two index files
score_regimes.py           scores the two runs against the pooled model, conditionally
```

## What this is for, and what it is not for

The single curve is damped because one number has to hedge across every outcome consistent with
the input. Conditioning removes that hedge, and the decomposition is exact:

```
E[y|x] = P(storm|x)·E[y|x, storm] + (1 − P(storm|x))·E[y|x, quiet]
```

Measured with ridges on the same features, **47.6 % of the variance of the observed peak is
between the two branches** at this threshold — that is what splitting recovers, and it is the
part no loss function could reach.

| scored on storm anchors | pooled | storm-only |
|---|---|---|
| reproduction ≥100 ap | 0.580 | **0.649** |
| MAE | 31.32 | **26.65** |
| rho | 0.594 | 0.599 |
| "if a storm comes" | — | **82.4 ap** (observed 80.7) |

| scored on quiet anchors | pooled | quiet-only |
|---|---|---|
| MAE | 8.64 | **5.97** |

**It is not an accuracy improvement.** Recombining the branches with `P(storm|x)` gives rho
0.721 / MAE 12.80 against the pooled model's 0.724 / 12.88 — slightly worse, as it must be,
because a single number is the conditional mean and the pooled model estimates that directly.
What the split buys is a **conditional statement** the single curve cannot make.

**The bottleneck moves rather than disappears.** `P(storm|x)` is the discrimination problem and
is bounded at the same ceiling. At this threshold it is exactly calibrated — mean 0.187 against
an observed rate of 0.187 — which is the main reason 48 was chosen over 39 or 56.

## Why 48

ap30 is a 36 discrete levels, not a continuous variable. Between 48 and 56 there is nothing, so a
threshold written as `50` silently means 56, and `100` means 111. `build_regime_indices.py`
refuses any threshold not on the discrete level set for that reason.

| | ≥ 39 | **≥ 48** | ≥ 56 |
|---|---|---|---|
| storm training anchors | 79,246 | **56,571** | 40,806 |
| between-branch variance | 43.0 % | **47.6 %** | 50.3 % |
| `P(storm\|x)` vs observed | 0.251 / 0.254 | **0.187 / 0.187** | 0.142 / 0.136 |
| rho within the storm branch | 0.619 | **0.599** | 0.581 |

(pooled model on the same storm rows: rho 0.594. At ≥ 56 the storm branch no longer beats it;
at ≥ 111 it falls to 0.444 and the branch predicts one number for every storm.)

Call it **Kp 5o**, not "the G1 threshold" — G1 is Kp 5, which spans ap 39–56. This project has
mislabelled the G scale once already.

## Run

```bash
cd ~/GitHub/njit-geoindex/geoindex-model && conda activate geoindex

python exploration/2026-08/regime48/build_regime_indices.py

for R in quiet storm; do
  N=regime48_ap_in12h_out12h_gnn_transformer_$R

  # TRAIN on the branch validation index -- early stopping and checkpoint selection use
  # whatever validation set training is given, and a branch model judged on the full set is
  # selected on a distribution it was never trained for.
  python scripts/train.py --config-name=server_ap +io=in12h_out12h +model=gnn_transformer \
      data.timeseries.train_index=regime48_ap/train_index_$R.csv \
      data.timeseries.validation_index=regime48_ap/validation_index_$R.csv \
      experiment.name=$N

  # SCORE on the full index, so both branches are comparable with the pooled run.
  python scripts/validate.py --config-name=server_ap +io=in12h_out12h +model=gnn_transformer \
      data.timeseries.train_index=regime48_ap/train_index_$R.csv \
      experiment.name=$N validation.epoch=best validation.mcd_samples=0
done

python exploration/2026-08/regime48/score_regimes.py
```

The pooled comparison run already exists as
`probe_ap_in12h_out12h_gnn_transformer_baseline`, so no third training run is needed.

Both regimes deliberately keep `stat_file: table_stats_ap.pkl`, the pooled statistics, so the
two models see identically scaled inputs and their outputs are directly comparable — and the
storm model does not get a normalization fitted to 17 % of the record.

**The two validation indices are not interchangeable.** Training uses its validation set for
early stopping and for choosing which checkpoint to keep, so it must see the branch it is being
trained for. Scoring uses the full index so both branches land on the same 23,514 anchors as the
pooled run. The first attempt at these runs used the full index for both and the result was
invalid: the storm model's training loss was still falling at epoch 14 while its validation loss
— computed on an 85 %-quiet set — had bottomed at epoch 4, so early stopping kept a
barely-trained checkpoint that predicted 43 ap on storms whose observed mean was 88.

## Results — the split works on the level, and the maximum throws it away

Measured on the 23,514 validation anchors, storm = observed 12 h peak ≥ 48 (n = 4,829):

| scored on storm anchors | pooled | regime split |
|---|---|---|
| **curve level** vs observed | 93 % | **109 %** |
| **peak** vs observed | 43 % | 55.5 % |
| MAE | 45.70 | **36.00** |
| rho | 0.482 | 0.471 |
| sharpness (peak / window mean) | 1.20 | 1.23 — observed is **2.58** |

The storm branch learns the level completely — it slightly overshoots the observed curve mean.
What it does not learn is the shape, and the maximum of a flat curve is low whatever its level.
That is the double shrinkage of section 2.7 again: the maximum of 24 separately shrunk
conditional means is not the shrunk maximum.

The quiet branch is the same effect in reverse: rho improves (0.546 → 0.584) but MAE gets worse
(9.79 → 10.45), because it learns a lower level and then misses the small peaks inside quiet
windows.

## What the ridge proxy says next — and how far to trust it

`ridge_proxy.py` fits **both** structures per regime and, unlike the first pre-test, validates
itself before predicting anything. It reproduces the trained storm branch (103 % / 54 % against
the measured 109 % / 52 %) but **not** the pooled model (79 % / 40 % against 93 % / 43 %), and it
says so in its own output.

| storm rows, peak vs observed | |
|---|---|
| pooled, max of the curve | 40 % |
| regime split, max of the curve | 54 % |
| pooled, separate peak output | 72 % |
| **regime split + separate peak output** | **102 %** |

The two fixes address different defects — the split fixes the branch *level*, the peak output
avoids the double shrinkage — and they stack close to additively rather than multiplicatively.
The same pattern holds on quiet rows (64 % → 96 %, MAE 8.30 → 6.28).

**Treat that 102 % as a ridge result, not a prediction.** Of the three ridge-led predictions in
this investigation, one transferred (the clock change, predicted +0.0104 of AUC and delivered
+0.0109) and two did not (the block maximum promised +0.11 of correlation and returned +0.014;
the first regime pre-test promised reproduction 0.649 and returned 0.360, because it fitted a
different target from the one the deep model fits).

## Pass mark for the next run — regime split + peak head

Set from the **measured deep numbers**, not from the ridge:

| criterion | value to beat | where it comes from |
|---|---|---|
| storm peak recovery | **> 55.5 %** | the regime split measured here |
| storm MAE | **< 36.00** | the regime split measured here |
| quiet MAE | **< 9.79** | the pooled model on quiet rows |
| rho in the storm branch | **≥ 0.482** | the pooled model on storm rows |

`score_regimes.py` applies these only to a run that emits `peak_prediction`; against the
curve-only runs it prints the reference values instead, since scoring them against themselves
would be circular.

## Two traps this experiment fell into

Both were the discrete ap30 levels in different disguises, and both are now guarded in code.

1. **A threshold off the discrete level set.** `50` silently means `56`; `100` means `111`.
   `build_regime_indices.py` refuses any threshold that is not one of the discrete ap30 levels.
2. **The denormalization round trip.** The stored archives hold `47.99999` where the discrete level set
   value is `48`, so a bare `peak >= 48` drops every window whose maximum is exactly 48 and
   silently scores the `>= 56` subset — which the first version of `score_regimes.py` did, on
   3,634 rows instead of 4,829. It now applies a half-unit tolerance and prints what a bare
   comparison would have given.

## Caveats to carry into the write-up

- The storm model trains on 13,095 anchors against the pooled model's 76,585. Early stopping
  patience and learning rate were tuned for the larger set; if the storm run early-stops in a
  handful of epochs, that is the first thing to look at.
- Applied to a quiet input the storm model still returns a large number — it has never seen a
  quiet window. **Any interface showing both branches must make the probability visually
  dominant**, or a 4 %-likely 168 ap reads as a forecast of 168 ap.
- The conditional numbers above are computed by selecting rows on the **observed** label. That
  is the correct verification for a conditional statement and it is **not** a forecast
  improvement. Say so explicitly wherever they are quoted.
- This is the two-atom discretisation of a conditional distribution. It fixes the branch *level*,
  which the trajectory test could not (storm PIT 0.815); it does not fix the curve *shape*, which
  the trajectory test could (sharpness 2.73 against observed 2.59). Sampling trajectories
  **within** each branch is the natural next step and combines the two.

## What the full visual pass shows — the gain is a constant in disguise

All 23,514 validation anchors were rendered with `../analysis/plot_all_anchors.py` (pooled plus
both branches, each branch's peak head drawn as a level spanning the window) to
`~/Projects/GeoIndex/results/plots_all_regime48ph/`. Reading the index that pass produced turns
the headline numbers over.

**The head barely beats a constant.** On the 4,829 storm anchors:

| predictor | peak MAE |
|---|---|
| pooled model, maximum of the curve | 45.70 |
| **a single constant, the observed median** | **25.59** |
| regime split + peak head | 22.78 |

45.70 → 25.59 is free — the pooled model was biased low and one scalar removes it. The trained
models earn 2.8 ap on top of that.

**The band recoveries are the same illusion.** An earlier version of this file reported 98 %
recovery on the 48–99 band as if it were calibration.

| observed peak | n | observed | storm head | recovery | head MAE | constant MAE | ρ in band |
|---|---|---|---|---|---|---|---|
| 48–99 | 4,093 | 63.5 | 62.2 | 98 % | 11.80 | 12.89 | 0.339 |
| ≥ 100 | 736 | 160.1 | 76.9 | 48 % | — | — | 0.346 |

A predictor pinned near a band's mean recovers ~100 % of that band by construction. The head
beats the band constant by 1.1 ap. The 82 % aggregate is that near-constant weighted by a
48–99 band that outnumbers the ≥ 100 band 5.6 to 1.

**There is almost no dispersion to work with.**

| | sd |
|---|---|
| observed peak, storm anchors | 41.2 |
| storm head, storm anchors | 14.9 |
| storm head, **quiet** anchors | **3.8** |

A conditional mean satisfies `sigma_pred/sigma_obs = rho`. The measured ratio is 0.362 against
`rho` 0.470, so the head does not even use the width its own correlation allows. On a quiet day
"if a storm comes" answers about 57 ap almost regardless of the input — it carries no
information there and is only honest behind a calibrated `P(storm|x)`.

**It is not an under-training artefact.** All three runs early-stop at a comparable point —
pooled best at epoch 5 of 15, quiet at 9 of 19, storm at 7 of 17 — so the storm branch is not
short-changed relative to the pooled model it is being compared with.

**And the tail has no data.** The 13,095 storm training anchors are 12 h windows that overlap;
counted as episodes more than 24 h apart they are **239 independent storms**, and the subset
reaching 100 ap is far smaller. The 48 % recovery above 100 ap is a sample-size limit, not a
loss-function or architecture one.

Navigation for the contact sheets (24 panels each, ordered by observed peak):

| sheets | band |
|---|---|
| 1–31 | ≥ 100 ap |
| 31–202 | 48–99 ap |
| 202–448 | 27–39 ap |
| 448–878 | 9–22 ap |
| 878–980 | < 9 ap |

Sheet 1 is the April 2023 storm (observed 294): the storm head calls 60, and every branch curve
is flat where the observation has a two-step structure.

## The runs behind these numbers

The peak-head code is not on `main` — it is branch `exploration/regime48-peakhead`, reproduced
byte-for-byte by `../patches/01-model-and-config.patch` and `../patches/02-tests.patch` applied
to the manuscript commit `e7697e8`. The peak head is independent of `regression_loss_type`, so
the curve is scored exactly as the paper model scores it and the head is added alongside:

```bash
git checkout exploration/regime48-peakhead        # or: git apply ../patches/*.patch
python exploration/2026-08/regime48/build_regime_indices.py

for R in quiet storm; do
  N=regime48ph_ap_in12h_out12h_gnn_transformer_$R
  python scripts/train.py --config-name=server_ap +io=in12h_out12h +model=gnn_transformer \
      data.timeseries.train_index=regime48_ap/train_index_$R.csv \
      data.timeseries.validation_index=regime48_ap/validation_index_$R.csv \
      training.peak_head.enabled=true experiment.name=$N
  python scripts/validate.py --config-name=server_ap +io=in12h_out12h +model=gnn_transformer \
      data.timeseries.train_index=regime48_ap/train_index_$R.csv \
      experiment.name=$N validation.epoch=best validation.mcd_samples=0
done

python exploration/2026-08/regime48/score_regimes.py --stem regime48ph_ap_in12h_out12h_gnn_transformer
```

`training.peak_head.base_loss` was left at its config value; the training log records the
per-lead loss (`solar_wind_weighted`) but not the head's, so treat the head loss as unpinned
if these runs are ever reproduced.
