"""Run one checkpoint over an anchor index: deterministic + MC-dropout passes → one npz.

Hydra-free harness for scoring checkpoints that did not come out of `train.sh` (e.g. a
co-author's retrained model) on exactly the same anchors, table, statistics file and
normalizer as the sweep runs. Also used to apply a checkpoint to a real-time 30-min table
(`--table` / `--anchor-index` point at any parquet with the training schema).

    # co-author focal/oversampling model, validation split, MPS
    python analysis/infer_checkpoint.py \
        --checkpoint /path/to/gnn_transformer_oversample_dup_2_4_8_best.pt \
        --io in6h_out6h --model gnn_transformer --mc-samples 50 --device mps \
        --out ~/Projects/GeoIndex/results/coauthor_focal_2026-09/focal_val.npz

    # our sweep checkpoint through the identical path (sanity: MAE must match its validation_results.txt)
    python analysis/infer_checkpoint.py --checkpoint .../ap_in6h_out6h_gnn_transformer/checkpoint/model_best.pth ...

Output npz keys (N anchors, L output steps, K input steps):
    anchors (U14, anchor UTC as YYYYMMDDHHMMSS) · input_ap30 (N,K) · target (N,L) · pred (N,L)
    mcd_mean / mcd_std / mcd_median / mcd_q025 / mcd_q975 (N,L) — all in raw ap units.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.networks import create_model  # noqa: E402
from src.pipeline.datasets_table import TableValidationDataset  # noqa: E402
from src.uncertainty import mcd_sample_stats  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
INPUT_VARIABLES = ["v_avg", "v_min", "v_max", "np_avg", "np_min", "np_max", "t_avg", "t_min", "t_max",
                   "bx_avg", "bx_min", "bx_max", "by_avg", "by_min", "by_max", "bz_avg", "bz_min", "bz_max",
                   "bt_avg", "bt_min", "bt_max", "ap30"]
GNN_GROUPS = {"v": ["v_avg", "v_min", "v_max"], "np": ["np_avg", "np_min", "np_max"],
              "t": ["t_avg", "t_min", "t_max"], "bx": ["bx_avg", "bx_min", "bx_max"],
              "by": ["by_avg", "by_min", "by_max"], "bz": ["bz_avg", "bz_min", "bz_max"],
              "bt": ["bt_avg", "bt_min", "bt_max"], "ap30": ["ap30"]}
NORMALIZATION = {"default": "zscore", "methods": {
    **{f"{v}_{s}": "log_zscore" for v in ("v", "np", "t", "bt") for s in ("avg", "min", "max")},
    **{f"{v}_{s}": "zscore" for v in ("bx", "by", "bz") for s in ("avg", "min", "max")},
    "ap30": "log1p_zscore", "hp30": "log1p_zscore"}}


class Settings(dict):
    """Dict with attribute access — what the src APIs expect from the Hydra config."""

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


def parse_io(io: str) -> tuple[int, int, int, int]:
    """'in6h_out6h' / 'in1d_out12h' → (input_start, input_end, target_start, target_end) in 30-min steps."""
    def steps(tok: str) -> int:
        num, unit = float(tok[:-1]), tok[-1]
        return int(round(num * (48 if unit == "d" else 2)))
    a, b = io.split("_")
    return -steps(a[2:]), 0, 0, steps(b[3:])


def model_section(name: str, io: str) -> dict:
    """Model hyperparameters: configs/base.yaml `model` block overlaid with configs/model/<name>.yaml
    and the io group's model overrides (configs/io/<io>.yaml sets e.g. patch_len for short windows)."""
    base = yaml.safe_load((REPO / "configs" / "base.yaml").read_text()).get("model", {})
    out = {k: v for k, v in base.items() if not isinstance(v, dict)}
    for path in (REPO / "configs" / "model" / f"{name}.yaml", REPO / "configs" / "io" / f"{io}.yaml"):
        if path.exists():
            over = yaml.safe_load(path.read_text()) or {}
            over = over.get("model", over if path.parent.name == "model" else {})
            out.update({k: v for k, v in over.items() if not isinstance(v, dict)})
    return out


def build_config(args) -> Settings:
    i0, i1, t0, t1 = parse_io(args.io)
    cfg = {
        "sampling": {"enable_undersampling": False, "undersampling_mode": "static"},
        "data": {"modalities": {"timeseries": True, "sdo": False, "omni_hdf5": False},
                 "timeseries": {"dataset_mode": "table", "table_file": args.table,
                                "train_index": args.train_index, "validation_index": args.anchor_index,
                                "test_index": args.anchor_index, "stat_file": args.stat_file,
                                "interval_minutes": 30, "points_per_day": 48,
                                "input_start": i0, "input_end": i1, "target_start": t0, "target_end": t1,
                                "input_variables": INPUT_VARIABLES, "target_variables": ["ap30"],
                                "gnn_variable_groups": GNN_GROUPS, "normalization": NORMALIZATION,
                                "augmentation": {"gaussian_noise_std": 0.0},
                                "train_filter": {"variable": "ap30", "peak_min": None, "peak_max": None}}},
        "model": model_section(args.model, args.io),
        "environment": {"data_root": str(Path(args.data_root).expanduser()), "save_root": str(Path(args.out).parent),
                        "device": args.device, "num_workers": 0},
        "experiment": {"name": Path(args.out).stem, "seed": 0, "batch_size": args.batch_size},
    }
    return settings(cfg)


def load_state_dict(path: Path, device) -> dict:
    ck = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ck, dict) and "model_state_dict" in ck:
        return ck["model_state_dict"]
    if isinstance(ck, dict) and "state_dict" in ck:
        return ck["state_dict"]
    return ck


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, help="output npz path")
    p.add_argument("--io", default="in6h_out6h")
    p.add_argument("--model", default="gnn_transformer", help="name of configs/model/<name>.yaml")
    p.add_argument("--data-root", default="~/Projects/GeoIndex/datasets")
    p.add_argument("--table", default="data.parquet")
    p.add_argument("--stat-file", default="table_stats_ap.pkl")
    p.add_argument("--train-index", default="total_ap/train_index.csv")
    p.add_argument("--anchor-index", default="total_ap/validation_index.csv",
                   help="index csv (datetime[,label]) of the anchors to run")
    p.add_argument("--mc-samples", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None, help="first N anchors only (smoke test)")
    args = p.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(args.device)
    cfg = build_config(args)
    ds = TableValidationDataset(cfg)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    loader = DataLoader(torch.utils.data.Subset(ds, list(range(n))), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    model = create_model(copy.deepcopy(cfg)).to(device)
    missing, unexpected = model.load_state_dict(load_state_dict(Path(args.checkpoint).expanduser(), device), strict=True)
    model.eval()
    norm = ds.normalizer
    ap_col = ds.input_variables.index("ap30")

    def denorm(z: np.ndarray) -> np.ndarray:
        return np.clip(norm.denormalize_omni(z, "ap30"), 0.0, None)

    keys = ["input_ap30", "target", "pred", "mcd_mean", "mcd_std", "mcd_median", "mcd_q025", "mcd_q975"]
    acc = {k: [] for k in keys}
    anchors = []
    t0 = time.time()
    for b, batch in enumerate(loader):
        x = batch["inputs"].to(device)
        with torch.no_grad():
            out = model(x, None, return_features=False)
        acc["pred"].append(denorm(out[..., 0].cpu().numpy()))
        acc["target"].append(denorm(batch["targets"][..., 0].numpy()))
        acc["input_ap30"].append(denorm(batch["inputs"][:, :, ap_col].numpy()))
        if args.mc_samples > 0:
            st = mcd_sample_stats(model, x, None, norm, ["ap30"], num_samples=args.mc_samples)
            for k in ("mcd_mean", "mcd_std", "mcd_median", "mcd_q025", "mcd_q975"):
                acc[k].append(st[k][..., 0])
        anchors.extend(list(batch["file_names"]))
        if b % 5 == 0:
            done = min((b + 1) * args.batch_size, n)
            print(f"  {done}/{n} anchors, {time.time() - t0:.0f} s", flush=True)

    out = {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in acc.items() if v}
    out["anchors"] = np.array(anchors, dtype="U14")
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    mae = float(np.mean(np.abs(out["pred"] - out["target"])))
    meta = {"checkpoint": str(Path(args.checkpoint).expanduser()), "io": args.io, "model": args.model,
            "table": args.table, "stat_file": args.stat_file, "anchor_index": args.anchor_index,
            "n_anchors": int(n), "mc_samples": args.mc_samples, "device": args.device,
            "deterministic_mae_raw": mae, "seconds": round(time.time() - t0, 1)}
    if "mcd_mean" in out:
        meta["mcd_mean_mae_raw"] = float(np.mean(np.abs(out["mcd_mean"] - out["target"])))
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
