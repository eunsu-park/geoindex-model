# Exploration — August 2026 ceiling investigation

**Not part of the paper pipeline.** `src/`, `configs/` and `tests/` were reverted to `e7697e8`
(2026-07-31), the state the DeepAP manuscript is based on. Everything the investigation produced
is kept here instead, so the main tree stays reproducible and nothing is lost.

The full findings, with numbers and corrections, are in the `geoindex` repo:

- [`docs/investigation-2026-08.md`](../../../geoindex/docs/investigation-2026-08.md) — the summary and the verdict
- [`docs/geoindex-model/damping-and-warning-calibration-2026-08.md`](../../../geoindex/docs/geoindex-model/damping-and-warning-calibration-2026-08.md) — the detailed record
- [`docs/geoindex-realtime/clock-audit-2026-08.md`](../../../geoindex/docs/geoindex-realtime/clock-audit-2026-08.md) — the service audit

---

## What was established

The ap30 forecast is bounded at per-lead `rho` ≈ 0.60 and peak `rho` ≈ 0.72 by the information
in L1 solar wind, not by the model. Three axes were closed independently, and the physics
explains why:

| axis | evidence | result |
|---|---|---|
| loss and target | 17 variants + two output heads | `rho` invariant at 0.55–0.60 |
| architecture | 14 networks vs a closed-form ridge | the ridge beats 12 of them |
| input window | 1 hour to 54 days | 12 h optimal; 1 h gets 99 % of it |

The one thing that moved discrimination was the **clock**: OMNI HRO is time-shifted to the bow
shock nose, so the ~50 minutes a monitor sees the wind coming was never in the training data.
Re-indexing to observation time gives AUC +0.0119 and per-lead `rho` 0.567 → 0.600, replicated
in 5 CV folds of 5.

And the mechanism behind the ceiling, measured: **the predictable part of the wind is not
geoeffective and the geoeffective part is not predictable.**

| lead | v | bt | **bz** |
|---|---|---|---|
| 0.5 h | 0.993 | 0.979 | **0.832** |
| 2 h | 0.977 | 0.910 | **0.533** |
| 12 h | 0.866 | 0.565 | **0.098** |

---

## `analysis/` — reusable tooling

Written during the investigation; none of it is needed to reproduce the paper, but all of it
is reusable against any stored validation archive.

| script | what it does |
|---|---|
| `compare_loss_variants.py` | continuous + categorical comparison across probe runs, with a ridge reference row |
| `attenuation_diagnostics.py` | `rho` / `rho^2` scoring, dispersion, discrimination, heteroscedasticity, the change control |
| `calibrate_warning.py` | refit a warning threshold to a false-alarm budget; isotonic probability; deploy JSON |
| `liemohn_metrics.py` | the Liemohn et al. (2018) verification set — slope B, threshold sweep, ROC, activity-binned error |
| `check_cv_leakage.py` | fold contiguity and input-window overlap |
| `loss_probe.sh` | trains and validates one io × model config under 21 loss/target variants |
| `make_observation_time_table.py` | re-index the wind from bow-shock time to L1 observation time (`--verify` refits the ridge on both) |
| `plot_all_anchors.py` | one forecast panel per validation anchor, plus index and contact sheets |

`plot_all_anchors.py` is the one most likely to be wanted again — it rebuilds per-sample plots
from the stored npz with no checkpoint and no GPU, and can overlay several runs.

## `measurements/` — one-off scripts, kept for their numbers

Each answered one question and is not general tooling. Listed with what it found.

| script | question | answer |
|---|---|---|
| `input_length_scan.py` | how far back is worth looking? | 12 h optimal, 1 h gets 99 %, everything longer is worse; recurrence and cycle add nothing, preconditioning +0.010 |
| `phase0_l1time.py` | is the clock worth anything? | bow-shock 0.706 → observation time 0.724, AUC +0.0104 |
| `score_peak.py`, `peak_control.py` | did the peak loss work? | it hit its numbers by making output step 21 a maximum register — 99.7 % of anchors peak at 11.0 h |
| `score_peakhead.py` | does a separate peak head fix that? | yes for shape, but stacking it on the clock change fails the concentration criterion |
| `score_obs.py`, `score_cv5.py` | does the clock change hold? | 5 CV folds of 5 on AUC, per-lead `rho` and per-lead MAE |
| `eval_obs.py` | full-sample verification | all 23,514 anchors; the figure data behind the verification page |
| `trajectory_test.py` | can sampled trajectories fix the curve's shape? | shape yes (2.73 vs observed 2.59), conditional calibration no (storm PIT 0.815) |
| `joint_wind_test.py` | what would forecasting the wind jointly buy? | **exactly zero for a deterministic forecast** — `E[Y\|X, g(X)] = E[Y\|X]`; the Bz predictability table above is from here |
| `availability.py` | does the feed fail during storms? | no — 30.0 of 30 usable minutes at ap ≥ 100 |
| `validate_palette.py` | — | Python port of the dataviz palette validator (no node on this machine) |

Most read `~/Projects/GeoIndex/results/probe_ap_in12h_out12h_gnn_transformer_*`; some also read
the `space_weather` database or `~/Projects/GeoIndex/datasets/data.parquet`.

## `patches/` — the reverted code

Applies onto `e7697e8`.

| patch | contains |
|---|---|
| `01-model-and-config.patch` | `PinballLoss`, `LeadNormalizedLoss`, `ExceedanceBCELoss`, `PeakAugmentedLoss`, `PeakHeadLoss`; the optional `peak_head` on `GNNOnlyModel` with the opt-in `return_peak`; trainer and validator wiring; `configs/server_ap_obs.yaml`; the `training.peak`, `training.peak_head` and `training.exceedance` config blocks |
| `02-tests.patch` | the tests for all of the above |

```bash
git apply exploration/2026-08/patches/01-model-and-config.patch
git apply exploration/2026-08/patches/02-tests.patch
cp exploration/2026-08/server_ap_obs.yaml configs/
cp exploration/2026-08/analysis/*.py exploration/2026-08/analysis/*.sh analysis/
cp exploration/2026-08/tests/*.py tests/
```

`server_ap_obs.yaml` and the two test files that only exercise the exploration analysis scripts
sit here rather than in the patches, because they are whole new files rather than edits.

Two things in there are worth restoring if the work is picked up again:

- **`server_ap_obs.yaml` + `make_observation_time_table.py`** — the only change that moved
  discrimination, replicated 5/5. It also removes a train/serve clock mismatch as a side effect.
- **`PeakHeadLoss` + the `peak_head`** — a separate scalar peak output. Natively calibrated
  (dispersion ratio 1.02 against the baseline's 0.75) and leaves the curve better than it found
  it, but do not stack it on the clock change.

`PeakAugmentedLoss` is kept for the record and should not be used: it derives the peak from the
same 24 curve outputs, and the model satisfies it by nominating one output step as a maximum
register rather than by learning a taller storm.

## Data artefacts, outside the repo

Under `~/Projects/GeoIndex/` (cloud-synced), not committed:

```
datasets/data_obs.parquet              wind re-indexed to L1 observation time
datasets/omni_timeshift_30min.csv      the applied OMNI shift, cached from the database
datasets/table_stats_ap_obs.pkl        its normalization statistics
datasets/cv5_ap_obs/fold{1..5}/        CV indices for the observation-time runs
results/probe_ap_in12h_out12h_gnn_transformer_*    21 probe runs
results/cv5obs_ap_in12h_out12h_gnn_transformer_*   the 5 observation-time folds
results/plots_all_obs/                 23,514 per-anchor panels, index.csv, 980 contact sheets
results/analysis_202608/               warning calibrations at ap >= 30, 50, 100
```
