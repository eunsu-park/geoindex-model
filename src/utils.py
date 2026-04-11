"""Utility functions for solar wind prediction model.

Functions:
    setup_seed: Set random seeds for reproducibility.
    setup_device: Initialize compute device (CPU/CUDA/MPS).
    setup_experiment: Combined setup for seed and device.
    load_model: Load model checkpoint.
    save_model: Save model checkpoint.
    resolve_paths: Resolve checkpoint_path and output_dir from epoch or explicit paths.

Example:
    >>> from src.utils import setup_experiment, load_model, resolve_paths
    >>> device = setup_experiment(config)
    >>> checkpoint_path, output_dir = resolve_paths(config, 'validation')
    >>> model = load_model(model, checkpoint_path, device)
"""

import os
import random
import logging
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch


def setup_seed(seed: int = 250104) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value. Defaults to 250104.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed: {seed}")


def setup_device(requested_device: str) -> torch.device:
    """Initialize compute device.

    Args:
        requested_device: Requested device ('cuda', 'mps', 'cpu').

    Returns:
        torch.device object for the selected device.
    """
    # CUDA
    if requested_device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Using CUDA: {gpu_name}")
        else:
            device = torch.device('cpu')
            print("CUDA not available, using CPU")

    # MPS (Apple Silicon)
    elif requested_device == 'mps':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS (Apple Silicon)")
        else:
            device = torch.device('cpu')
            print("MPS not available, using CPU")

    # CPU
    elif requested_device == 'cpu':
        device = torch.device('cpu')
        print("Using CPU")

    # Unknown
    else:
        device = torch.device('cpu')
        print(f"Unknown device '{requested_device}', using CPU")

    return device


def setup_experiment(config) -> torch.device:
    """Combined setup for seed and device.

    Args:
        config: Configuration object with experiment.seed and environment.device.

    Returns:
        torch.device object for the selected device.
    """
    setup_seed(config.experiment.seed)
    device = setup_device(config.environment.device)
    return device


def load_model(model: torch.nn.Module, checkpoint_path: str,
               device: torch.device) -> torch.nn.Module:
    """Load model checkpoint.

    Args:
        model: PyTorch model instance.
        checkpoint_path: Path to checkpoint file.
        device: Device for computation.

    Returns:
        Model with loaded weights.

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Load into model
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Model loaded: {checkpoint_path}")
    return model


def save_model(model: torch.nn.Module, save_path: str,
               epoch: Optional[int] = None,
               optimizer: Optional[torch.optim.Optimizer] = None):
    """Save model checkpoint.

    Args:
        model: PyTorch model to save.
        save_path: Path to save checkpoint.
        epoch: Optional epoch number to include.
        optimizer: Optional optimizer state to include.
    """
    checkpoint = {
        'model_state_dict': model.state_dict()
    }

    if epoch is not None:
        checkpoint['epoch'] = epoch

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Model saved: {save_path}")


def resolve_paths(config, phase: str) -> tuple:
    """Resolve checkpoint_path and output_dir from config.

    Supports two modes:
    1. Explicit paths: Use checkpoint_path and output_dir directly if specified
    2. Epoch-based: Auto-generate paths from epoch number or special keywords

    Epoch can be:
    - Integer (e.g., 10, 100): Uses model_epoch_{XXXX}.pth
    - "best": Uses model_best.pth (saved during training when val_loss improves)
    - "final": Uses model_final.pth (saved at end of training)

    Args:
        config: Hydra configuration object.
        phase: Phase name ('validation', 'test', 'mcd', 'saliency', 'attention').

    Returns:
        Tuple of (checkpoint_path, output_dir).

    Raises:
        ValueError: If neither explicit paths nor epoch are specified.

    Example:
        >>> checkpoint_path, output_dir = resolve_paths(config, 'validation')
        >>> # With epoch: validation.epoch=10
        >>> # checkpoint_path = "{save_root}/{name}/checkpoint/model_epoch_0010.pth"
        >>> # output_dir = "{save_root}/{name}/validation/epoch_0010"
        >>> # With epoch: validation.epoch="best"
        >>> # checkpoint_path = "{save_root}/{name}/checkpoint/model_best.pth"
        >>> # output_dir = "{save_root}/{name}/validation/best"
    """
    phase_config = getattr(config, phase)

    # For saliency/attention, they may use validation's checkpoint
    # but have their own output directory
    checkpoint_phase = phase
    if phase in ('saliency', 'attention'):
        # These analysis tools use validation checkpoint by default
        # but can specify their own epoch
        checkpoint_phase = 'validation'

    # Resolve checkpoint path
    checkpoint_path = getattr(phase_config, 'checkpoint_path', '')
    if checkpoint_path:
        # Explicit path specified - use it directly
        pass
    elif hasattr(phase_config, 'epoch') and phase_config.epoch is not None:
        # Auto-generate from epoch (supports int, "best", "final")
        epoch = phase_config.epoch
        base_path = f"{config.environment.save_root}/{config.experiment.name}/checkpoint"

        if isinstance(epoch, str):
            epoch_lower = epoch.lower()
            if epoch_lower == "best":
                checkpoint_path = f"{base_path}/model_best.pth"
            elif epoch_lower == "final":
                checkpoint_path = f"{base_path}/model_final.pth"
            else:
                raise ValueError(
                    f"Invalid epoch string: '{epoch}'. "
                    f"Use integer, 'best', or 'final'."
                )
        else:
            checkpoint_path = f"{base_path}/model_epoch_{epoch:04d}.pth"
    else:
        raise ValueError(
            f"Either {phase}.checkpoint_path or {phase}.epoch must be specified. "
            f"Use CLI: {phase}.epoch=10 or {phase}.epoch=best or "
            f"{phase}.checkpoint_path=/path/to/model.pth"
        )

    # Resolve output directory
    output_dir = getattr(phase_config, 'output_dir', '')
    if output_dir:
        # Explicit path specified - use it directly
        pass
    elif hasattr(phase_config, 'epoch') and phase_config.epoch is not None:
        # Auto-generate from epoch
        epoch = phase_config.epoch
        base_path = f"{config.environment.save_root}/{config.experiment.name}/{phase}"

        if isinstance(epoch, str):
            output_dir = f"{base_path}/{epoch.lower()}"
        else:
            output_dir = f"{base_path}/epoch_{epoch:04d}"
    else:
        raise ValueError(
            f"Either {phase}.output_dir or {phase}.epoch must be specified. "
            f"Use CLI: {phase}.epoch=10 or {phase}.output_dir=/path/to/output"
        )

    return checkpoint_path, output_dir


def create_local_output_dir(
    nas_output_dir: str, experiment_name: str, phase: str
) -> Path:
    """Create local temp directory for analysis output.

    Writes go to local disk first (fast I/O), then compress_and_move()
    archives the result to NAS. This avoids creating many small files
    directly on network storage.

    Args:
        nas_output_dir: Original NAS output path (used to derive subdir name).
        experiment_name: Experiment name for directory structure.
        phase: Analysis phase name (validation, test, mcd, attention, saliency).

    Returns:
        Path to local temp directory.
    """
    nas_path = Path(nas_output_dir)
    local_dir = Path.home() / "tmp" / experiment_name / phase / nas_path.name
    local_dir.mkdir(parents=True, exist_ok=True)
    return local_dir


def compress_and_move(
    local_dir: Path, nas_output_dir: str, cleanup: bool = True
):
    """Compress subdirectories individually and move to NAS.

    Keeps the output folder structure on NAS. Top-level files (TXT, CSV)
    are copied as-is, while each subdirectory is compressed into its own
    ZIP archive within the NAS folder.

    Result structure example:
        nas_output_dir/
        ├── validation_results.txt   (copied)
        ├── validation_results.csv   (copied)
        ├── plots.zip                (from plots/)
        └── npz.zip                  (from npz/)

    Args:
        local_dir: Local temp directory containing results.
        nas_output_dir: Target NAS directory path.
        cleanup: If True, delete local temp directory after move.
    """
    nas_path = Path(nas_output_dir)
    nas_path.mkdir(parents=True, exist_ok=True)

    # Copy top-level files directly
    for item in sorted(local_dir.iterdir()):
        if item.is_file():
            dest = nas_path / item.name
            shutil.copy2(str(item), str(dest))
            print(f"Copied: {item.name} → {dest}")

    # Compress each subdirectory into an individual ZIP
    for item in sorted(local_dir.iterdir()):
        if not item.is_dir():
            continue
        # Skip empty directories
        if not any(item.iterdir()):
            print(f"Skipped empty directory: {item.name}")
            continue

        archive_local = local_dir / f"{item.name}.zip"
        print(f"Compressing {item.name}/ → {archive_local.name}")
        shutil.make_archive(
            str(archive_local.with_suffix('')),
            'zip',
            root_dir=str(local_dir),
            base_dir=item.name
        )

        file_size_mb = archive_local.stat().st_size / (1024 ** 2)
        archive_dest = nas_path / archive_local.name
        shutil.move(str(archive_local), str(archive_dest))
        print(f"  → {archive_dest} ({file_size_mb:.1f} MB)")

    if cleanup:
        shutil.rmtree(local_dir)
        print(f"Cleaned up: {local_dir}")


def _log_message(logger: Optional[logging.Logger], message: str,
                level: int = logging.INFO) -> None:
    """Helper function for logging.

    Args:
        logger: Optional logger instance.
        message: Message to log.
        level: Logging level.
    """
    if logger:
        logger.log(level, message)
    else:
        print(message)
