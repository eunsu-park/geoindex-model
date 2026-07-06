# Solar Wind Prediction - Geomagnetic Index Regression

Deep learning system for predicting a geomagnetic index from solar wind time series data. The target index is a Hydra config choice (index-agnostic); ap30 is the default example, and hp30 is also defined. Supports 14 model architectures including GNN, PatchTST, and TimesNet.

---

## Features

- **Modular data pipeline**: CSV time series (active), HDF5 multi-modal (SDO+OMNI, future)
- **14 model architectures**: Linear, LSTM, BiLSTM, Transformer, TCN, PatchTST, TimesNet, GNN×7
- **SolarWindWeightedLoss**: NOAA G-Scale based weighted loss for geomagnetic storms
- **Hydra configuration**: Easy experiment management with config inheritance
- **GNN with dynamic node groups**: Config-based variable grouping with validation

---

## Quick Start

```bash
# Using ap CLI (recommended)
../geoindex/ap train --profile standard           # GNN+Transformer, 2-day input, 12h output
../geoindex/ap train --profile quick              # Linear baseline, fast (~2 min)
../geoindex/ap validate --profile standard --epoch best
../geoindex/ap analyze attention --profile standard

# Using Hydra config groups directly
python scripts/train.py --config-name=local +io=in2d_out12h +model=gnn_transformer
python scripts/train.py --config-name=local +io=in1d_out6h +model=linear

# Train all 336 experiments
./train.sh

# Train specific subset
./train.sh --filter out12h          # 12h output only
./train.sh --model transformer      # Transformer only
./train.sh --filter "in(6|12)h"     # Regex supported
./train.sh --max-jobs 4             # Limit parallel jobs

# Validate all
./validation.sh --epoch best
```

---

## Project Structure

```
geoindex-model/
├── configs/                # Hydra configuration files
│   ├── base.yaml           # Shared defaults
│   ├── local.yaml          # Environment settings
│   ├── io/                 # I/O window configs (24)
│   │   ├── in6h_out6h.yaml ... in3d_out24h.yaml
│   ├── model/              # Model configs (14)
│   │   ├── linear.yaml, transformer.yaml, gnn_transformer.yaml ...
│   └── experiments/        # Experiment overrides
├── src/                    # Core modules
│   ├── networks/           # Model architectures (package)
│   │   ├── _registry.py    # @register_model decorator
│   │   ├── transformer.py, tcn.py, gnn.py, patchtst.py ...
│   ├── pipeline/           # Data loading (package)
│   │   ├── normalizer.py, readers.py, datasets_csv.py, factory.py ...
│   ├── plotting.py         # Shared visualization
│   ├── trainers.py         # Training loop
│   ├── validators.py       # Validation loop
│   ├── testers.py          # Inference
│   ├── losses.py           # Loss functions
│   └── utils.py            # Utilities
├── scripts/                # Entry points
├── analysis/               # Interpretability (attention, MCD)
├── tests/                  # Unit tests (156 tests)
├── train.sh                # Parallel training runner
└── validation.sh           # Parallel validation runner
```

Documentation lives in the sibling hub repo under
`../geoindex/docs/geoindex-model/`, not inside this repo.

---

## Data

Current dataset: CSV-based 30-min solar wind time series from `geoindex-data`.

See: [dataset-guide.md](../geoindex/docs/geoindex-model/dataset-guide.md)

- **22 input variables**: solar wind (v, np, t, bx, by, bz, bt) × (avg/min/max) + ap30
- **Target**: configurable geomagnetic index (Hydra config choice) — ap30 by default (30-min equivalent amplitude geomagnetic index); hp30 also defined
- **Output windows**: 6h (12 timesteps), 12h (24 timesteps), 24h (48 timesteps)

---

## Models (14)

See: [model-guide.md](../geoindex/docs/geoindex-model/model-guide.md)

| # | `+model=` | Type | Description |
|---|-----------|------|-------------|
| 1 | `linear` | `linear` | MLP baseline |
| 2 | `lstm` | `lstm` | LSTM recurrent encoder |
| 3 | `bilstm` | `bilstm` | Bidirectional LSTM |
| 4 | `transformer` | `transformer` | Transformer encoder |
| 5 | `tcn` | `tcn` | Temporal Convolutional Network |
| 6 | `patchtst` | `patchtst` | Patch-based Transformer (ICLR 2023) |
| 7 | `timesnet` | `timesnet` | FFT + 2D Inception Conv (ICLR 2023) |
| 8 | `gnn_linear` | `gnn` | GNN + Linear |
| 9 | `gnn_lstm` | `gnn` | GNN + LSTM |
| 10 | `gnn_bilstm` | `gnn` | GNN + BiLSTM |
| 11 | `gnn_transformer` | `gnn` | GNN + Transformer |
| 12 | `gnn_tcn` | `gnn` | GNN + TCN |
| 13 | `gnn_patchtst` | `gnn` | GNN + PatchTST |
| 14 | `gnn_timesnet` | `gnn` | GNN + TimesNet |

GNN variants build a graph over config-driven node groups (variable node count; 8 groups is the default grouping).

---

## Experiment Matrix

`train.sh` cross-products every `configs/io/*.yaml` with every `configs/model/*.yaml`:
336 experiments = 24 I/O windows × 14 models.

I/O windows = 6 inputs (6h/12h/18h/1d/2d/3d) × 4 outputs (6h/12h/18h/24h).

See: [experiments.md](../geoindex/docs/geoindex-model/experiments.md)

---

## Config System

Hydra config groups compose I/O windows and models independently.

```bash
# Syntax:
python scripts/train.py --config-name=local +io={window} +model={model}

# Examples:
python scripts/train.py --config-name=local +io=in2d_out12h +model=transformer
python scripts/train.py --config-name=local +io=in1d_out6h +model=linear
python scripts/train.py --config-name=local +io=in3d_out24h +model=gnn_patchtst
```

Sample I/O windows (3 of 24; full list in `configs/io/`):

| I/O Window | Input | Output | `+io=` |
|------------|-------|--------|--------|
| 1 day → 6h | 48 steps | 12 steps | `in1d_out6h` |
| 2 days → 12h | 96 steps | 24 steps | `in2d_out12h` |
| 3 days → 24h | 144 steps | 48 steps | `in3d_out24h` |

All 14 models (full list in `configs/model/`):

| Model | `+model=` |
|-------|-----------|
| Linear | `linear` |
| LSTM | `lstm` |
| BiLSTM | `bilstm` |
| Transformer | `transformer` |
| TCN | `tcn` |
| PatchTST | `patchtst` |
| TimesNet | `timesnet` |
| GNN+Linear | `gnn_linear` |
| GNN+LSTM | `gnn_lstm` |
| GNN+BiLSTM | `gnn_bilstm` |
| GNN+Transformer | `gnn_transformer` |
| GNN+TCN | `gnn_tcn` |
| GNN+PatchTST | `gnn_patchtst` |
| GNN+TimesNet | `gnn_timesnet` |

---

## Analysis

Post-training analysis tools for model interpretability and evaluation.

See: [analysis.md](../geoindex/docs/geoindex-model/analysis.md)

### Primary Analysis (require checkpoint + data)

| Script | Shell | Description |
|--------|-------|-------------|
| `analysis/run_attention.py` | `./attention.sh` | Transformer attention weight extraction |
| `analysis/run_mcd.py` | `./mcd.sh` | Monte Carlo Dropout uncertainty |
| `analysis/run_saliency.py` | `./saliency.sh` | Gradient-based saliency maps (SDO) |

### Post-hoc Evaluation (process existing results)

| Script | Description |
|--------|-------------|
| `analysis/evaluate_storm_performance.py` | Storm-tier filtered MAE/RMSE |
| `analysis/compare_predictions.py` | Multi-model prediction overlay plots |
| `analysis/evaluate_mcd.py` | MCD coverage and calibration |
| `analysis/visualize_gnn_graph.py` | GNN learned adjacency heatmap |

---

## Testing

```bash
conda activate ap
pytest tests/ -v
```

---

## License

See: [LICENSE](LICENSE)
