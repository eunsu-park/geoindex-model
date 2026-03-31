"""
Monte Carlo Dropout for Uncertainty Estimation

This script performs MC Dropout inference on the validation dataset
to estimate prediction uncertainty.

MC Dropout:
1. Enable dropout during inference (training mode for Dropout layers)
2. Run multiple forward passes with different dropout masks
3. Compute mean and std of predictions as uncertainty estimate

Usage:
    cd /opt/projects/10_Harim/01_AP/02_Regression

    # Method 1: Epoch-based (recommended - auto-generates paths)
    python analysis/monte_carlo_dropout.py --config-name=local mcd.epoch=10

    # Method 2: Explicit paths (for custom locations)
    python analysis/monte_carlo_dropout.py --config-name=local \\
        mcd.checkpoint_path=/path/to/checkpoint.pth \\
        mcd.output_dir=/path/to/output
"""

# Python standard library
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party library
import hydra
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Custom library
from src.pipeline import create_dataloader
from src.networks import create_model
from src.utils import setup_experiment, resolve_paths


def create_mcd_plot(
    mean: np.ndarray,
    std: np.ndarray,
    target: np.ndarray,
    target_variable: str,
    save_path: str,
    title: str = "MC Dropout Prediction",
    n_std: float = 2.0,
    input_data: np.ndarray = None,
    input_variables: list = None
):
    """Create MC Dropout prediction plot with uncertainty band.

    Args:
        mean: Prediction mean (seq_len,)
        std: Prediction std (seq_len,)
        target: Ground truth (seq_len,)
        target_variable: Name of target variable
        save_path: Path to save the plot
        title: Plot title
        n_std: Number of standard deviations for uncertainty band
        input_data: Input data (input_len, num_vars), optional
        input_variables: List of input variable names, optional
    """
    pred_len = len(mean)

    # Check if target variable is in input variables
    has_input = False
    input_values = None
    if input_data is not None and input_variables is not None:
        if target_variable in input_variables:
            var_idx = input_variables.index(target_variable)
            input_values = input_data[:, var_idx]
            has_input = True

    # Time axis: input is negative, prediction is positive
    if has_input:
        input_len = len(input_values)
        input_time = np.arange(-input_len, 0)
        pred_time = np.arange(0, pred_len)
    else:
        pred_time = np.arange(pred_len)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Plot input if available
    if has_input:
        ax.plot(
            input_time, input_values, 'b-', linewidth=1.5,
            label=f'Input ({target_variable})', alpha=0.7
        )

    # Uncertainty band (mean ± n_std * std)
    lower = mean - n_std * std
    upper = mean + n_std * std
    ax.fill_between(
        pred_time, lower, upper,
        alpha=0.3, color='red',
        label=f'Uncertainty (±{n_std}σ)'
    )

    # Target (Ground Truth)
    ax.plot(
        pred_time, target, 'g-', linewidth=2,
        label='Target (Ground Truth)', marker='o', markersize=4
    )

    # Prediction (Mean)
    ax.plot(
        pred_time, mean, 'r--', linewidth=2,
        label='Prediction (Mean)', marker='x', markersize=5
    )

    # Calculate coverage (% of targets within uncertainty band)
    in_band = (target >= lower) & (target <= upper)
    coverage = in_band.mean() * 100

    # Calculate metrics
    mae = np.abs(target - mean).mean()
    rmse = np.sqrt(((target - mean) ** 2).mean())
    mean_std = std.mean()

    # Formatting
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='Reference Time')
    ax.set_xlabel('Time Step (relative to reference)', fontsize=11)
    ax.set_ylabel(f'{target_variable}', fontsize=11)
    ax.set_title(f'{title}', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add metrics text box
    metrics_text = (
        f'MAE: {mae:.2f}\n'
        f'RMSE: {rmse:.2f}\n'
        f'Mean σ: {mean_std:.2f}\n'
        f'Coverage: {coverage:.1f}%'
    )
    ax.text(
        0.98, 0.95, metrics_text,
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    # Mark points outside uncertainty band
    outside_band = ~in_band
    if outside_band.any():
        ax.scatter(
            pred_time[outside_band], target[outside_band],
            color='darkgreen', s=80, marker='s', zorder=5,
            label=f'Outside band ({(~in_band).sum()} pts)'
        )
        ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


@hydra.main(config_path="../configs", version_base=None)
def main(config):
    """Run MC Dropout inference on validation dataset."""

    device = setup_experiment(config)

    # Resolve paths (epoch-based or explicit)
    checkpoint_path, output_dir = resolve_paths(config, 'mcd')

    # Update config with resolved paths
    OmegaConf.update(config, "mcd.checkpoint_path", checkpoint_path)
    OmegaConf.update(config, "mcd.output_dir", output_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get settings from config
    n_samples = getattr(config.mcd, 'num_mc_samples', 100)
    create_plots = getattr(config.mcd, 'create_plots', True)
    n_std = getattr(config.mcd, 'n_std', 2.0)  # Number of std for uncertainty band

    # Create subdirectories
    npz_dir = output_dir / "npz"
    plot_dir = output_dir / "plots"
    npz_dir.mkdir(exist_ok=True, parents=True)
    if create_plots:
        plot_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("MONTE CARLO DROPOUT - UNCERTAINTY ESTIMATION")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    print(f"MC samples: {n_samples}")
    print(f"Create plots: {create_plots}")
    if create_plots:
        print(f"Uncertainty band: ±{n_std}σ")

    # Create validation dataloader
    validation_dataloader = create_dataloader(config, 'validation')
    print(
        f"Validation dataloader: {len(validation_dataloader.dataset)} samples, "
        f"{len(validation_dataloader)} batches"
    )

    model = create_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")

    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=config.environment.device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Enable dropout during inference
    dropout_count = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
            dropout_count += 1
    print(f"Enabled {dropout_count} Dropout layers for MC sampling")

    # Get variable names
    target_variable = config.data.target_variables[0] if hasattr(config.data, 'target_variables') else 'ap_index_nt'
    input_variables = list(config.data.input_variables) if hasattr(config.data, 'input_variables') else []

    # Process all batches
    total_processed = 0
    total_skipped = 0
    all_coverages = []

    print("\n" + "=" * 70)
    print("PROCESSING")
    print("=" * 70)

    for batch_idx, data_dict in enumerate(validation_dataloader):
        sdo = data_dict["sdo"].to(device)
        inputs = data_dict["inputs"].to(device)
        targets = data_dict["targets"].to(device)
        file_names = data_dict["file_names"]

        batch_size = len(file_names)

        for n in range(batch_size):
            _sdo = sdo[n:n+1]
            _input = inputs[n:n+1]
            _target = targets[n:n+1]
            file_name = file_names[n]

            npz_path = npz_dir / f"{file_name}.npz"
            plot_path = plot_dir / f"{file_name}.png"

            # Skip if already exists
            if npz_path.exists():
                total_skipped += 1
                continue

            # MC Dropout sampling
            predictions = []
            with torch.no_grad():
                for _ in range(n_samples):
                    output, _, _ = model(_input, _sdo, return_features=True)
                    output = output.cpu().numpy()
                    output = output[:, :, 0]  # (1, seq_len)
                    predictions.append(output)

            predictions = np.concatenate(predictions, 0)  # (n_samples, seq_len)

            # Denormalize predictions
            predictions = validation_dataloader.dataset.normalizer.denormalize_omni(
                predictions, target_variable
            )

            # Denormalize target
            target_np = _target[:, :, 0].cpu().numpy()  # (1, seq_len)
            target_denorm = validation_dataloader.dataset.normalizer.denormalize_omni(
                target_np, target_variable
            )[0]  # (seq_len,)

            # Denormalize input (for plotting)
            input_np = _input[0].cpu().numpy()  # (input_len, num_vars)
            input_denorm = np.zeros_like(input_np)
            for var_idx, var_name in enumerate(input_variables):
                input_denorm[:, var_idx] = validation_dataloader.dataset.normalizer.denormalize_omni(
                    input_np[:, var_idx], var_name
                )

            # Compute statistics
            mean = predictions.mean(0)  # (seq_len,)
            std = predictions.std(0)    # (seq_len,)

            # Calculate coverage
            lower = mean - n_std * std
            upper = mean + n_std * std
            in_band = (target_denorm >= lower) & (target_denorm <= upper)
            coverage = in_band.mean() * 100
            all_coverages.append(coverage)

            # Save NPZ (include target for later analysis)
            np.savez(
                npz_path,
                mean=mean,
                std=std,
                target=target_denorm,
                n_samples=n_samples,
                coverage=coverage
            )

            # Create plot
            if create_plots:
                create_mcd_plot(
                    mean=mean,
                    std=std,
                    target=target_denorm,
                    target_variable=target_variable,
                    save_path=str(plot_path),
                    title=f"MC Dropout - {file_name}",
                    n_std=n_std,
                    input_data=input_denorm,
                    input_variables=input_variables
                )

            total_processed += 1
            if total_processed % 50 == 0:
                print(f"  Processed {total_processed} samples (coverage: {coverage:.1f}%)")

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(validation_dataloader)} complete")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Processed: {total_processed} samples")
    print(f"Skipped (existing): {total_skipped} samples")
    print(f"Output directory: {output_dir}")
    print(f"  NPZ files: {npz_dir}")
    if create_plots:
        print(f"  Plots: {plot_dir}")

    if all_coverages:
        avg_coverage = np.mean(all_coverages)
        min_coverage = np.min(all_coverages)
        max_coverage = np.max(all_coverages)
        print(f"\nCoverage Statistics (±{n_std}σ band):")
        print(f"  Average: {avg_coverage:.1f}%")
        print(f"  Min: {min_coverage:.1f}%")
        print(f"  Max: {max_coverage:.1f}%")

        # Expected coverage for Gaussian: ±2σ ≈ 95.4%, ±1σ ≈ 68.3%
        if n_std == 2.0:
            expected = 95.4
        elif n_std == 1.0:
            expected = 68.3
        else:
            from scipy.stats import norm
            expected = (norm.cdf(n_std) - norm.cdf(-n_std)) * 100

        print(f"  Expected (Gaussian): {expected:.1f}%")

        if avg_coverage < expected - 10:
            print("\n  WARNING: Coverage is lower than expected!")
            print("  This may indicate underestimated uncertainty.")
        elif avg_coverage > expected + 10:
            print("\n  NOTE: Coverage is higher than expected.")
            print("  This may indicate overestimated uncertainty (conservative).")


if __name__ == "__main__":
    main()
