"""Utility functions for solar wind prediction model.

Functions:
    setup_seed: Set random seeds for reproducibility.
    setup_device: Initialize compute device (CPU/CUDA/MPS).
    setup_experiment: Combined setup for seed and device.
    load_model: Load model checkpoint.
    save_model: Save model checkpoint.
    resolve_paths: Resolve checkpoint_path and output_dir from epoch or explicit paths.
    save_plot: Save comparison plot and data.
    denormalize_predictions: Denormalize predictions using statistics.
    create_comparison_plot: Create and save comparison plot.
    save_data_h5: Save data to HDF5 file.

Example:
    >>> from src.utils import setup_experiment, load_model, resolve_paths
    >>> device = setup_experiment(config)
    >>> checkpoint_path, output_dir = resolve_paths(config, 'validation')
    >>> model = load_model(model, checkpoint_path, device)
"""

import os
import random
import logging
from typing import List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py


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


def save_plot(targets: np.ndarray, outputs: np.ndarray,
              target_variables: List[str], stat_dict: dict,
              plot_path: str, plot_title: str,
              logger: Optional[logging.Logger] = None) -> None:
    """Save comparison plot and data with improved error handling.

    Args:
        targets: Ground truth values of shape (seq_len, n_vars).
        outputs: Model predictions of shape (seq_len, n_vars).
        target_variables: List of target variable names.
        stat_dict: Dictionary containing statistics for denormalization.
        plot_path: Path to save the plot (without extension).
        plot_title: Title of the plot.
        logger: Optional logger for output.

    Raises:
        ValueError: If input shapes don't match or are invalid.
        OSError: If file saving fails.
    """
    # Validate inputs
    if targets.shape != outputs.shape:
        raise ValueError(f"Shape mismatch: targets {targets.shape} != outputs {outputs.shape}")

    if targets.shape[1] != len(target_variables):
        raise ValueError(f"Variable count mismatch: got {targets.shape[1]}, expected {len(target_variables)}")

    try:
        # Denormalize data
        targets_denorm, outputs_denorm = denormalize_predictions(
            targets, outputs, target_variables, stat_dict
        )

        # Create and save plot
        create_comparison_plot(
            targets_denorm, outputs_denorm, target_variables,
            plot_title, f"{plot_path}.png"
        )

        # Save data as HDF5
        save_data_h5(targets_denorm, outputs_denorm, f"{plot_path}.h5")

        message = f"Plot and data saved: {plot_path}"
        _log_message(logger, message, logging.DEBUG)

    except Exception as e:
        error_msg = f"Failed to save plot {plot_path}: {e}"
        _log_message(logger, error_msg, logging.ERROR)
        raise OSError(error_msg)


def denormalize_predictions(targets: np.ndarray, outputs: np.ndarray,
                           target_variables: List[str], stat_dict: dict) -> tuple:
    """Denormalize predictions using statistics.

    Args:
        targets: Normalized target values.
        outputs: Normalized prediction values.
        target_variables: List of variable names.
        stat_dict: Statistics dictionary with mean/std for each variable.

    Returns:
        Tuple of (denormalized_targets, denormalized_outputs).
    """
    zero_clip_variables = {"ap_index", "ap_index_nt"}  # Variables that should be clipped to >= 0

    targets_denorm_list = []
    outputs_denorm_list = []

    for idx, variable in enumerate(target_variables):
        # Process targets
        target_var = targets[:, idx:idx+1]
        if variable in stat_dict:
            mean = stat_dict[variable]['mean']
            std = stat_dict[variable]['std']
            target_denorm = (target_var * std) + mean
        else:
            target_denorm = target_var

        # Clip if necessary
        if variable in zero_clip_variables:
            target_denorm = np.clip(target_denorm, 0, None)
        targets_denorm_list.append(target_denorm)

        # Process outputs
        output_var = outputs[:, idx:idx+1]
        if variable in stat_dict:
            output_denorm = (output_var * std) + mean
        else:
            output_denorm = output_var

        # Clip if necessary
        if variable in zero_clip_variables:
            output_denorm = np.clip(output_denorm, 0, None)
        outputs_denorm_list.append(output_denorm)

    return (np.concatenate(targets_denorm_list, axis=1),
            np.concatenate(outputs_denorm_list, axis=1))


def create_comparison_plot(targets: np.ndarray, outputs: np.ndarray,
                          target_variables: List[str], title: str,
                          save_path: str) -> None:
    """Create and save comparison plot.

    Args:
        targets: Denormalized target values.
        outputs: Denormalized output values.
        target_variables: List of variable names.
        title: Plot title.
        save_path: Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(title, fontsize=14, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(target_variables)))

    for idx, (variable, color) in enumerate(zip(target_variables, colors)):
        ax.plot(targets[:, idx], label=f'True {variable}',
               color=color, linewidth=2, alpha=0.8)
        ax.plot(outputs[:, idx], label=f'Predicted {variable}',
               color=color, linewidth=2, linestyle='--', alpha=0.8)

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def save_data_h5(targets: np.ndarray, outputs: np.ndarray, save_path: str) -> None:
    """Save denormalized data to HDF5 file.

    Args:
        targets: Denormalized target values.
        outputs: Denormalized output values.
        save_path: Path to save the HDF5 file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    with h5py.File(save_path, 'w') as f:
        f.create_dataset("targets", data=targets, compression='gzip')
        f.create_dataset("outputs", data=outputs, compression='gzip')


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
