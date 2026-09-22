"""Five-fold cross-validation of the co-author's focal-loss + oversampling recipe.

Request 2 of the 2026-09-22 co-author delivery: train gnn_transformer, tcn and lstm
with THEIR training code on OUR chronological folds (`datasets/cv5_ap/fold{1..5}`).

The loss, the oversampling scheme, the optimizer, the gradient clipping, the metric
function and the checkpoint-selection rule are copied verbatim from
`~/PostBox/GeoIndex/20260922/DeepAP/code/geoindex-model/Zhenduo_baseline/train_{tcn,lstm}.py`
and from the 2026-09-16 `Zhenduo_focal/train_focal_oversampling.py` (gnn_transformer).
What this file adds is only parametrisation:

* `--model` / `--fold` select the architecture dict (their exact dicts) and the fold
  indices + per-fold `table_stats.pkl` (the recipe reads its log1p statistics from the
  training dataset, so the fold statistics flow through automatically);
* epochs / patience are unified to the gnn_transformer script's 30 / 10 for all three
  models (the delivered baseline scripts say 10 / 5, the delivered checkpoints' metadata
  say 30 / 10);
* the selection rule is the delivered scripts' "best validation overall MAE" rule. The
  target-MAE-window rule that produced the delivered TCN/LSTM checkpoints (see
  `tmp/build_baseline_notebooks.py` in the package) is not used: it has no meaning
  across folds;
* Apple-silicon `mps` is accepted as a device.

    python analysis/coauthor_focal_cv.py --model tcn --fold 1 --device mps \
        --out-root ~/Projects/GeoIndex/results/coauthor_focal_cv_2026-09

Outputs per run: `{model}_fold{k}/best.pt` (same payload layout as their checkpoints),
`history.csv` (per-epoch train loss + validation metrics) and `metrics.json`.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.networks import create_model  # noqa: E402
from src.pipeline.datasets_table import TableTrainDataset, TableValidationDataset  # noqa: E402

# --------------------------------------------------------------------------------------
# Their hyper-parameters (Zhenduo_baseline/train_*.py, Zhenduo_focal/train_focal_oversampling.py)
# --------------------------------------------------------------------------------------
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 1e-4
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.0
SEED = 250104
NUM_WORKERS = 0

SCHEME_NAME = "dup_2_4_8"
SCHEME = {"30_50": 2, "50_100": 4, "100_plus": 8}
DIFF_THRESHOLD = 50.0
FOCAL_THRESHOLD = 30.0
FOCAL_BASE_LOSS = "huber"
FOCAL_GAMMA = 2.0
OUTLIER_WEIGHT = 4.0
QUIET_WEIGHT = 1.0
NORMALIZE_WEIGHT = True

# Their exact model dicts (copied from the EXPERIMENT_CONFIG['model'] of each script).
MODEL_SECTIONS = {
    "tcn": {"model_type": "tcn", "d_model": 128, "output_seq_len": None,
            "tcn_channels": [64, 128, 256], "tcn_kernel_size": 3, "tcn_dropout": 0.1},
    "lstm": {"model_type": "lstm", "d_model": 128, "output_seq_len": None,
             "bilstm_hidden_size": 128, "bilstm_num_layers": 2, "gnn_dropout": 0.1},
    "gnn_transformer": {"model_type": "gnn", "d_model": 128, "transformer_nhead": 4,
                        "transformer_num_layers": 2, "transformer_dim_feedforward": 256,
                        "transformer_dropout": 0.1, "gnn_temporal_type": "transformer",
                        "gnn_node_feature_dim": 32, "gnn_gcn_hidden_dim": 64,
                        "gnn_num_gcn_layers": 2, "gnn_dropout": 0.1, "gnn_node_embed_dim": 16,
                        "patch_len": 4, "patch_stride": 2, "output_seq_len": None},
}

INPUT_VARIABLES = ["v_avg", "v_min", "v_max", "np_avg", "np_min", "np_max", "t_avg", "t_min",
                   "t_max", "bx_avg", "bx_min", "bx_max", "by_avg", "by_min", "by_max",
                   "bz_avg", "bz_min", "bz_max", "bt_avg", "bt_min", "bt_max", "ap30"]
GNN_GROUPS = {"v": ["v_avg", "v_min", "v_max"], "np": ["np_avg", "np_min", "np_max"],
              "t": ["t_avg", "t_min", "t_max"], "bx": ["bx_avg", "bx_min", "bx_max"],
              "by": ["by_avg", "by_min", "by_max"], "bz": ["bz_avg", "bz_min", "bz_max"],
              "bt": ["bt_avg", "bt_min", "bt_max"], "ap30": ["ap30"]}
NORMALIZATION = {"default": "zscore", "methods": {
    **{f"{v}_{s}": "log_zscore" for v in ("v", "np", "t", "bt") for s in ("avg", "min", "max")},
    **{f"{v}_{s}": "zscore" for v in ("bx", "by", "bz") for s in ("avg", "min", "max")},
    "ap30": "log1p_zscore", "hp30": "log1p_zscore"}}


class Settings(dict):
    """Dictionary with attribute access for the existing src APIs."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def settings(value):
    if isinstance(value, dict):
        return Settings({k: settings(v) for k, v in value.items()})
    if isinstance(value, list):
        return [settings(v) for v in value]
    return value


def build_config(model: str, fold: int, data_root: Path, out_dir: Path, device: torch.device) -> Settings:
    """Their EXPERIMENT_CONFIG with the fold's indices and statistics substituted."""
    cv = f"cv5_ap/fold{fold}"
    cfg = {
        "sampling": {"enable_undersampling": False, "undersampling_mode": "static"},
        "data": {"modalities": {"timeseries": True, "sdo": False, "omni_hdf5": False},
                 "timeseries": {"dataset_mode": "table", "table_file": "data.parquet",
                                "train_index": f"{cv}/train_index.csv",
                                "validation_index": f"{cv}/validation_index.csv",
                                "test_index": f"{cv}/validation_index.csv",
                                "stat_file": f"{cv}/table_stats.pkl",
                                "interval_minutes": 30, "points_per_day": 48,
                                "input_start": -12, "input_end": 0, "target_start": 0, "target_end": 12,
                                "input_variables": INPUT_VARIABLES, "target_variables": ["ap30"],
                                "gnn_variable_groups": GNN_GROUPS, "normalization": NORMALIZATION,
                                "augmentation": {"gaussian_noise_std": 0.0},
                                "train_filter": {"variable": "ap30", "peak_min": None, "peak_max": None}}},
        "model": copy.deepcopy(MODEL_SECTIONS[model]),
        "environment": {"data_root": str(data_root), "save_root": str(out_dir),
                        "device": device.type, "num_workers": NUM_WORKERS},
        "experiment": {"name": f"coauthor_focal_{model}_fold{fold}", "seed": SEED, "batch_size": BATCH_SIZE},
    }
    return settings(cfg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------------------
# Verbatim pieces of their recipe (only `denorm_ap30` takes the statistics as arguments)
# --------------------------------------------------------------------------------------
class FocalRegressionLoss(nn.Module):
    def __init__(self, denorm, threshold=30.0, base="huber", gamma=2.0, outlier_weight=4.0,
                 quiet_weight=1.0, normalize_weight=True):
        super().__init__()
        self.denorm = denorm
        self.threshold = threshold
        self.base = base
        self.gamma = gamma
        self.outlier_weight = outlier_weight
        self.quiet_weight = quiet_weight
        self.normalize_weight = normalize_weight

    def forward(self, pred, target):
        raw_target = self.denorm(target)
        outlier = raw_target >= self.threshold
        if self.base == "mse":
            base_loss = (pred - target).pow(2)
        elif self.base == "huber":
            base_loss = F.smooth_l1_loss(pred, target, reduction="none")
        else:
            raise ValueError(self.base)
        focal_factor = torch.pow(1.0 - torch.exp(-base_loss), self.gamma)
        class_weight = torch.where(
            outlier,
            torch.tensor(self.outlier_weight, device=target.device, dtype=target.dtype),
            torch.tensor(self.quiet_weight, device=target.device, dtype=target.dtype))
        weight = class_weight * focal_factor
        if self.normalize_weight:
            weight = weight / weight.mean().detach().clamp_min(1e-6)
        return (weight * base_loss).mean()


def dataset_peak_info(dataset, idx):
    name, _ = dataset.file_list[idx]
    row = dataset.dt_to_row[dataset._name_to_dt[name]]
    ap_idx = dataset.all_variables.index("ap30")
    input_peak = float(np.nanmax(dataset.array[row + dataset.input_start:row + dataset.input_end, ap_idx]))
    output_peak = float(np.nanmax(dataset.array[row + dataset.target_start:row + dataset.target_end, ap_idx]))
    return output_peak - input_peak, input_peak, output_peak


def multiplier_for_diff(diff, scheme):
    if 30 <= diff < 50:
        return scheme["30_50"]
    if 50 <= diff < 100:
        return scheme["50_100"]
    if diff >= 100:
        return scheme["100_plus"]
    return 1


def build_oversampled_indices(base_indices, train_diff, scheme):
    indices = []
    for idx, diff in zip(base_indices, train_diff):
        indices.extend([int(idx)] * int(multiplier_for_diff(float(diff), scheme)))
    return indices


def train_one_epoch(model, criterion, optimizer, loader, device):
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        preds = model(inputs, None, return_features=False)
        loss = criterion(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * inputs.shape[0]
        total_count += inputs.shape[0]
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_mae(model, loader, device, denorm):
    model.eval()
    preds_all, targets_all, inputs_all = [], [], []
    for batch in loader:
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        preds = model(inputs, None, return_features=False)
        preds_all.append(denorm(preds).cpu().numpy()[..., 0])
        targets_all.append(denorm(targets).cpu().numpy()[..., 0])
        inputs_all.append(denorm(inputs[:, :, -1]).cpu().numpy())
    pred = np.concatenate(preds_all)
    target = np.concatenate(targets_all)
    inp = np.concatenate(inputs_all)
    err = pred - target
    input_peak = inp.max(axis=1)
    target_peak = target.max(axis=1)
    pred_peak = pred.max(axis=1)
    peak_delta = target_peak - input_peak
    diff50 = peak_delta >= DIFF_THRESHOLD
    persistence = np.mean(np.abs(inp[:, -1:] - target))
    return {
        "overall_mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "persistence_mae": float(persistence),
        "mae_skill": float(1 - np.mean(np.abs(err)) / persistence) if persistence > 0 else float("nan"),
        "overall_rmse": float(np.sqrt(np.mean(err ** 2))),
        "peak_mae": float(np.mean(np.abs(pred_peak - target_peak))),
        "diff50_n": int(diff50.sum()),
        "diff50_peak_mae": float(np.mean(np.abs(pred_peak[diff50] - target_peak[diff50]))) if diff50.any() else float("nan"),
        "ap30_ge_50_mae": float(np.mean(np.abs(err[target >= 50]))) if (target >= 50).any() else float("nan"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(MODEL_SECTIONS))
    p.add_argument("--fold", required=True, type=int, choices=[1, 2, 3, 4, 5])
    p.add_argument("--data-root", default="~/Projects/GeoIndex/datasets")
    p.add_argument("--out-root", default="~/Projects/GeoIndex/results/coauthor_focal_cv_2026-09")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--limit", type=int, default=None, help="first N train / val anchors only (smoke test)")
    args = p.parse_args()

    device = torch.device(args.device)
    data_root = Path(args.data_root).expanduser()
    out_dir = Path(args.out_root).expanduser() / f"{args.model}_fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_config(args.model, args.fold, data_root, out_dir, device)
    print(f"model={args.model} fold={args.fold} device={device} out={out_dir}")

    set_seed(SEED)
    train_dataset_full = TableTrainDataset(cfg)
    val_dataset_full = TableValidationDataset(cfg)
    ap30_stats = train_dataset_full.stat_dict["ap30"]
    log1p_mean = float(ap30_stats["log1p_mean"])
    log1p_std = float(ap30_stats["log1p_std"])
    print(f"fold statistics: log1p_mean={log1p_mean:.6f} log1p_std={log1p_std:.6f}")

    def denorm_ap30(z):
        return torch.clamp(torch.expm1(z * log1p_std + log1p_mean), min=0.0)

    n_train = len(train_dataset_full) if args.limit is None else min(args.limit, len(train_dataset_full))
    n_val = len(val_dataset_full) if args.limit is None else min(args.limit, len(val_dataset_full))
    base_train_indices = np.arange(n_train)
    train_diff = np.array([dataset_peak_info(train_dataset_full, int(i))[0] for i in base_train_indices])
    indices = build_oversampled_indices(base_train_indices, train_diff, SCHEME)
    pin = device.type == "cuda"
    train_loader = DataLoader(Subset(train_dataset_full, indices), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader = DataLoader(Subset(val_dataset_full, list(range(n_val))), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)
    print(f"train anchors {n_train} -> oversampled {len(indices)} "
          f"(+{100 * (len(indices) / max(n_train, 1) - 1):.1f}%); validation anchors {n_val}")

    set_seed(SEED + len(SCHEME_NAME))
    model = create_model(copy.deepcopy(cfg)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = FocalRegressionLoss(denorm_ap30, FOCAL_THRESHOLD, FOCAL_BASE_LOSS, FOCAL_GAMMA,
                                    OUTLIER_WEIGHT, QUIET_WEIGHT, NORMALIZE_WEIGHT).to(device)

    best_path = out_dir / "best.pt"
    history_csv = out_dir / "history.csv"
    history = []
    best_mae = float("inf")
    best_epoch = None
    patience = 0
    for epoch in range(1, args.max_epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)
        metrics = evaluate_mae(model, val_loader, device, denorm_ap30)
        mae = metrics["overall_mae"]
        seconds = time.time() - start
        history.append({"epoch": epoch, "train_loss": train_loss, "seconds": seconds, **metrics})
        pd.DataFrame(history).to_csv(history_csv, index=False)
        print(f"{args.model} fold{args.fold} epoch {epoch}/{args.max_epochs}: train_loss={train_loss:.6f} "
              f"val_MAE={mae:.4f} seconds={seconds:.1f}", flush=True)
        if not np.isfinite(mae):
            raise ValueError(f"Validation MAE is not finite at epoch {epoch}")
        if mae < best_mae - EARLY_STOP_MIN_DELTA:
            best_mae = mae
            best_epoch = epoch
            patience = 0
            torch.save({
                "model": args.model, "fold": args.fold, "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": metrics, "experiment_config": dict(cfg),
                "normalization": {"log1p_mean": log1p_mean, "log1p_std": log1p_std},
                "seed": SEED + len(SCHEME_NAME),
                "training": {"max_epochs": args.max_epochs, "patience": EARLY_STOPPING_PATIENCE,
                             "min_delta": EARLY_STOP_MIN_DELTA, "batch_size": BATCH_SIZE,
                             "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                             "loss": "FocalRegressionLoss", "oversampling": SCHEME,
                             "selection_metric": "validation overall MAE"},
            }, best_path)
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping after {patience} epochs without improvement")
                break

    (out_dir / "metrics.json").write_text(json.dumps({
        "model": args.model, "fold": args.fold, "best_epoch": best_epoch, "best_val_mae": best_mae,
        "epochs_run": len(history), "n_train": int(n_train), "n_train_oversampled": len(indices),
        "n_val": int(n_val), "device": device.type, "checkpoint": str(best_path),
        **{f"best_{k}": v for k, v in history[best_epoch - 1].items() if k not in ("epoch",)},
    }, indent=2))
    print(f"{args.model} fold{args.fold} best validation MAE {best_mae:.4f} at epoch {best_epoch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
