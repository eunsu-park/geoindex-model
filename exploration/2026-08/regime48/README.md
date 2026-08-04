# Two-regime training at ap30 ≥ 48 (Kp 5o)

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

ap30 is a 36-level ladder, not a continuous variable. Between 48 and 56 there is nothing, so a
threshold written as `50` silently means 56, and `100` means 111. `build_regime_indices.py`
refuses any threshold not on the ladder for that reason.

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

## Pass mark, fixed before the runs

Scored on the storm anchors of the validation set (observed 12 h peak ≥ 48):

| criterion | value to beat | why |
|---|---|---|
| reproduction ≥100 ap | **> 0.580** | the pooled model on the same rows; this is the whole point |
| rho within the branch | **≥ 0.594** | if it falls, the storm sample is too thin — this is what happens at ≥ 111 |
| `P(storm\|x)` mean | within **±10 %** of the observed rate | a scenario without a calibrated probability misleads |
| quiet-branch MAE | **< 8.64** | the pooled model on quiet rows |

And one that is **not** a criterion: the mixture will not beat the pooled model on rho or MAE.
The ridge says 0.721 vs 0.724. Do not read that as failure — it is what the decomposition
predicts.

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
