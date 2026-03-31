# Solar Wind Prediction - ap30 Regression

Deep learning system for predicting the ap30 geomagnetic index using solar wind time series data.
Supports multiple model architectures and extensible multi-modal data pipelines.

## Features

- **Modular data pipeline**: CSV time series (active), HDF5 multi-modal (SDO+OMNI, future)
- **Flexible model selection**: Transformer, TCN, Linear (time series); ConvLSTM, Fusion, Baseline (multi-modal)
- **SolarWindWeightedLoss**: NOAA G-Scale based weighted loss for geomagnetic storms
- **Hydra configuration**: Easy experiment management with config inheritance
- **Extensible**: Add new data modalities via `config.data.modalities` flags

## Quick Start

```bash
# Train (transformer, default)
python scripts/train.py --config-name=local

# Train with different model
python scripts/train.py --config-name=local model.model_type=tcn

# Validate
python scripts/validate.py --config-name=local experiment.name=local validation.epoch=best

# Using shell scripts
./train.sh v1                  # transformer_v1
./train.sh v1 tcn              # tcn_v1
```

## Project Structure

```
regression-sw/
├── configs/           # Hydra configuration files
│   ├── base.yaml      # Shared settings (modalities, data, model, training)
│   ├── local.yaml     # Local development (macOS/MPS)
│   └── experiments/   # Experiment-specific overrides
├── src/               # Core modules
│   ├── pipeline.py    # Data loading (CSV + HDF5), normalization, datasets
│   ├── networks.py    # Model architectures
│   ├── trainers.py    # Training loop
│   ├── validators.py  # Validation loop
│   ├── losses.py      # Loss functions
│   └── utils.py       # Utilities
├── scripts/           # Entry points (train.py, validate.py, test.py)
├── analysis/          # Interpretability (attention, saliency, MCD)
├── tests/             # Unit tests
├── legacy/            # Archived HPC scripts and previous experiment results
├── DATASET_GUIDE.md   # Data format documentation
└── EXPERIMENTS.md     # Experiment log
```

## Data

Current dataset: CSV-based 30-min solar wind time series from `setup-sw-db`.
See [DATASET_GUIDE.md](DATASET_GUIDE.md) for details.

- **23 variables**: solar wind (v, np, t, bx, by, bz, bt) x (avg/min/max) + ap30 + hp30
- **Event window**: 5 days input (240 timesteps) + 3 days target (144 timesteps)
- **Target**: ap30 (30-min equivalent amplitude geomagnetic index)

### Modality Configuration

```yaml
# In configs/base.yaml
data:
  modalities:
    timeseries: true    # CSV solar wind (active)
    sdo: false          # SDO images (future)
    omni_hdf5: false    # Legacy OMNI HDF5
```

## Model Types

| Model | Type | Input | Description |
|-------|------|-------|-------------|
| `transformer` | Time series | CSV/OMNI | Transformer encoder (default) |
| `tcn` | Time series | CSV/OMNI | Temporal Convolutional Network |
| `linear` | Time series | CSV/OMNI | Linear encoder |
| `convlstm` | Image | SDO | ConvLSTM (requires SDO) |
| `fusion` | Multi-modal | SDO + OMNI | Cross-modal attention |
| `baseline` | Multi-modal | SDO + OMNI | Conv3D + Linear (Son et al. 2023) |

## Testing

```bash
conda activate ap
pytest tests/ -v
```

## License

See [LICENSE](LICENSE).
