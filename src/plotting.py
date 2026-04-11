"""Shared plotting utilities for solar wind prediction model.

Provides unified plotting functions used by validators, testers, and trainers
to eliminate code duplication across modules.

Functions:
    plot_prediction_timeseries: Plot input, target, and prediction time series.
    extract_file_names: Extract file names from data dictionary.
    denormalize_arrays: Denormalize input/target/prediction arrays using Normalizer.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_prediction_timeseries(
    inputs: np.ndarray,
    predictions: np.ndarray,
    input_variables: List[str],
    target_variables: List[str],
    save_path: Path,
    targets: Optional[np.ndarray] = None,
    title: str = "Prediction",
    logger=None,
    normalizer=None
):
    """Plot input, target, and prediction time series.

    Supports both validation (with targets) and operation (without targets)
    modes. Optionally denormalizes data using the provided normalizer.

    Args:
        inputs: Input data of shape (seq_len, num_input_vars).
        predictions: Prediction data of shape (pred_len, num_target_vars).
        input_variables: List of input variable names.
        target_variables: List of target variable names.
        save_path: Path to save the plot.
        targets: Target data of shape (target_len, num_target_vars), or None
            for inference without ground truth.
        title: Plot title.
        logger: Optional logger for warning messages.
        normalizer: Optional Normalizer instance for denormalization.
            When provided, all arrays are denormalized to original scale.
    """
    try:
        # Denormalize data if normalizer is available
        if normalizer is not None:
            inputs, targets, predictions = denormalize_arrays(
                inputs, predictions, input_variables, target_variables,
                normalizer, targets=targets
            )

        input_len = inputs.shape[0]
        pred_len = predictions.shape[0]
        num_target_vars = len(target_variables)

        # Find which target variables are also in input
        target_in_input = {}
        for target_var in target_variables:
            if target_var in input_variables:
                target_in_input[target_var] = input_variables.index(target_var)

        # Create figure
        fig, axes = plt.subplots(num_target_vars, 1, figsize=(14, 4 * num_target_vars))
        if num_target_vars == 1:
            axes = [axes]

        for var_idx, target_var in enumerate(target_variables):
            ax = axes[var_idx]

            # Time axis
            input_time = np.arange(-input_len, 0)
            pred_time = np.arange(0, pred_len)

            # Plot input if target variable is in input
            if target_var in target_in_input:
                input_var_idx = target_in_input[target_var]
                input_values = inputs[:, input_var_idx]
                ax.plot(input_time, input_values, 'b-', linewidth=1.5,
                        label=f'Input ({target_var})', alpha=0.7)

            # Plot target if available
            if targets is not None:
                target_values = targets[:, var_idx]
                ax.plot(pred_time, target_values, 'g-', linewidth=2,
                        label='Target (Ground Truth)', marker='o', markersize=3)

            # Plot prediction
            pred_values = predictions[:, var_idx]
            ax.plot(pred_time, pred_values, 'r--', linewidth=2,
                    label='Prediction', marker='x', markersize=4)

            # Formatting
            ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='Reference Time')
            ax.set_xlabel('Time Step (relative)', fontsize=10)
            ax.set_ylabel(f'{target_var}', fontsize=10)
            ax.set_title(f'{title} - {target_var}', fontsize=12, fontweight='bold')
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)

            # Add metrics if targets available
            if targets is not None:
                target_values = targets[:, var_idx]
                mae = np.abs(target_values - pred_values).mean()
                rmse = np.sqrt(((target_values - pred_values) ** 2).mean())
                ax.text(0.98, 0.95, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}',
                        transform=ax.transAxes, fontsize=9, verticalalignment='top',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    except Exception as e:
        if logger:
            logger.warning(f"Failed to create prediction plot: {e}")
        else:
            print(f"Warning: Failed to create prediction plot: {e}")


def denormalize_arrays(
    inputs: np.ndarray,
    predictions: np.ndarray,
    input_variables: List[str],
    target_variables: List[str],
    normalizer,
    targets: Optional[np.ndarray] = None
) -> tuple:
    """Denormalize input, prediction, and optionally target arrays.

    Uses Normalizer.denormalize_omni() which correctly handles all
    normalization methods (zscore, log_zscore, log1p_zscore, minmax).

    Args:
        inputs: Normalized input array of shape (seq_len, num_input_vars).
        predictions: Normalized predictions of shape (pred_len, num_target_vars).
        input_variables: List of input variable names.
        target_variables: List of target variable names.
        normalizer: Normalizer instance with denormalize_omni method.
        targets: Optional normalized targets of shape (target_len, num_target_vars).

    Returns:
        Tuple of (denormalized_inputs, denormalized_targets, denormalized_predictions).
        If targets is None, denormalized_targets will be None.
    """
    inputs = inputs.copy()
    for var_idx, var_name in enumerate(input_variables):
        inputs[:, var_idx] = normalizer.denormalize_omni(
            inputs[:, var_idx], var_name
        )

    predictions = predictions.copy()
    for var_idx, var_name in enumerate(target_variables):
        predictions[:, var_idx] = normalizer.denormalize_omni(
            predictions[:, var_idx], var_name
        )

    if targets is not None:
        targets = targets.copy()
        for var_idx, var_name in enumerate(target_variables):
            targets[:, var_idx] = normalizer.denormalize_omni(
                targets[:, var_idx], var_name
            )

    return inputs, targets, predictions


def extract_file_names(data_dict: Dict[str, Any], batch_idx: int) -> List[str]:
    """Extract file names from a data dictionary.

    Handles multiple input formats: tensor, list, or single value.
    Falls back to generated names if 'file_names' key is missing.

    Args:
        data_dict: Data dictionary, optionally containing 'file_names'.
        batch_idx: Batch index used for fallback naming.

    Returns:
        List of file name strings.
    """
    if 'file_names' not in data_dict:
        batch_size = data_dict['inputs'].size(0)
        return [f"batch_{batch_idx}_sample_{i}" for i in range(batch_size)]

    file_names_raw = data_dict['file_names']

    if isinstance(file_names_raw, torch.Tensor):
        return [str(name) for name in file_names_raw.tolist()]
    elif isinstance(file_names_raw, list):
        return [str(name) for name in file_names_raw]
    else:
        return [str(file_names_raw)]
