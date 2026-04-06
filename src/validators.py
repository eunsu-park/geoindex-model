"""Validation components for solar wind prediction model.

Contains Validator, MetricsAggregator, and ResultsWriter classes.

Classes:
    MetricsAggregator: Aggregate validation metrics across batches.
    ResultsWriter: Write validation results to files.
    Validator: Main validation loop for multimodal solar wind prediction.

Example:
    >>> from src.validators import Validator, MetricsAggregator
    >>> validator = Validator(config, model, criterion, device)
    >>> results = validator.validate(dataloader)
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Import save_plot from utils
try:
    from .utils import save_plot
    SAVE_PLOT_AVAILABLE = True
except ImportError:
    SAVE_PLOT_AVAILABLE = False


def plot_prediction_timeseries(
    inputs: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    input_variables: List[str],
    target_variables: List[str],
    save_path: Path,
    title: str = "Prediction",
    logger=None,
    normalizer=None
):
    """Plot input, target, and prediction time series.

    Args:
        inputs: Input data (seq_len, num_input_vars)
        targets: Target data (target_len, num_target_vars)
        predictions: Prediction data (target_len, num_target_vars)
        input_variables: List of input variable names
        target_variables: List of target variable names
        save_path: Path to save the plot
        title: Plot title
        logger: Optional logger
        normalizer: Optional normalizer for denormalization
    """
    try:
        # Denormalize data if normalizer is available
        if normalizer is not None:
            # Denormalize inputs (variable by variable)
            inputs = inputs.copy()
            for var_idx, var_name in enumerate(input_variables):
                inputs[:, var_idx] = normalizer.denormalize_omni(
                    inputs[:, var_idx], var_name
                )

            # Denormalize targets and predictions
            targets = targets.copy()
            predictions = predictions.copy()
            for var_idx, var_name in enumerate(target_variables):
                targets[:, var_idx] = normalizer.denormalize_omni(
                    targets[:, var_idx], var_name
                )
                predictions[:, var_idx] = normalizer.denormalize_omni(
                    predictions[:, var_idx], var_name
                )
        input_len = inputs.shape[0]
        target_len = targets.shape[0]
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
            target_time = np.arange(0, target_len)

            # Plot input if target variable is in input
            if target_var in target_in_input:
                input_var_idx = target_in_input[target_var]
                input_values = inputs[:, input_var_idx]
                ax.plot(input_time, input_values, 'b-', linewidth=1.5,
                        label=f'Input ({target_var})', alpha=0.7)

            # Plot target and prediction
            target_values = targets[:, var_idx]
            pred_values = predictions[:, var_idx]

            ax.plot(target_time, target_values, 'g-', linewidth=2,
                    label='Target (Ground Truth)', marker='o', markersize=3)
            ax.plot(target_time, pred_values, 'r--', linewidth=2,
                    label='Prediction', marker='x', markersize=4)

            # Formatting
            ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='Reference Time')
            ax.set_xlabel('Time Step (relative)', fontsize=10)
            ax.set_ylabel(f'{target_var}', fontsize=10)
            ax.set_title(f'{title} - {target_var}', fontsize=12, fontweight='bold')
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)

            # Add metrics
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


class MetricsAggregator:
    """Aggregate validation metrics across batches.

    Attributes:
        target_variables: List of target variable names.
        losses: List of batch losses.
        file_results: List of per-file validation results.
    """

    def __init__(self, target_variables: List[str]):
        """Initialize metrics aggregator.

        Args:
            target_variables: List of target variable names.
        """
        self.target_variables = target_variables
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.losses = []
        self.maes = []
        self.rmses = []
        self.r2_scores = []
        self.cosine_sims = []
        self.all_targets = []
        self.all_predictions = []
        self.file_results = []

    def update(self, batch_result: Dict[str, Any], file_names: Optional[List[str]] = None):
        """Update metrics with batch results.

        Args:
            batch_result: Dictionary containing batch validation results.
            file_names: Optional list of file names for this batch.
        """
        self.losses.append(batch_result['loss'])
        self.maes.append(batch_result['mae'])
        self.rmses.append(batch_result['rmse'])
        self.r2_scores.append(batch_result['r2_score'])

        if 'cosine_sim' in batch_result:
            self.cosine_sims.append(batch_result['cosine_sim'])

        self.all_targets.append(batch_result['targets'])
        self.all_predictions.append(batch_result['predictions'])

        # Store file-level results
        batch_size = batch_result['targets'].shape[0]
        for i in range(batch_size):
            file_name = file_names[i] if file_names else f"sample_{len(self.file_results)}"
            self.file_results.append({
                'file_name': file_name,
                'targets': batch_result['targets'][i],
                'predictions': batch_result['predictions'][i]
            })

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics.

        Returns:
            Dictionary containing overall metrics and per-variable metrics.
        """
        if not self.losses:
            raise ValueError("No data to summarize")

        # Overall metrics
        overall = {
            'average_loss': float(np.mean(self.losses)),
            'std_loss': float(np.std(self.losses)),
            'average_mae': float(np.mean(self.maes)),
            'std_mae': float(np.std(self.maes)),
            'average_rmse': float(np.mean(self.rmses)),
            'std_rmse': float(np.std(self.rmses)),
            'average_r2': float(np.mean(self.r2_scores)),
            'std_r2': float(np.std(self.r2_scores)),
        }

        if self.cosine_sims:
            overall['average_cosine_sim'] = float(np.mean(self.cosine_sims))
            overall['std_cosine_sim'] = float(np.std(self.cosine_sims))
        else:
            overall['average_cosine_sim'] = None
            overall['std_cosine_sim'] = None

        # Concatenate all targets and predictions
        all_targets = np.concatenate(self.all_targets, axis=0)
        all_predictions = np.concatenate(self.all_predictions, axis=0)

        # Per-variable metrics
        per_variable = self._calculate_per_variable_metrics(all_targets, all_predictions)

        return {
            'overall': overall,
            'per_variable': per_variable,
            'file_results': self.file_results,
            'total_samples': len(self.file_results),
            'success_rate': 100.0  # Assuming all processed batches succeeded
        }

    def _calculate_per_variable_metrics(
        self,
        all_targets: np.ndarray,
        all_predictions: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Calculate regression metrics for each target variable.

        Args:
            all_targets: Array of shape (n_samples, n_groups, n_variables).
            all_predictions: Array of shape (n_samples, n_groups, n_variables).

        Returns:
            Dictionary containing metrics for each variable.
        """
        n_samples, n_groups, n_variables = all_targets.shape
        metrics_dict = {}

        for var_idx, var_name in enumerate(self.target_variables):
            # Flatten across samples and groups for this variable
            var_targets = all_targets[:, :, var_idx].flatten()
            var_predictions = all_predictions[:, :, var_idx].flatten()

            # Calculate regression metrics
            mae = float(np.mean(np.abs(var_targets - var_predictions)))
            mse = float(np.mean((var_targets - var_predictions) ** 2))
            rmse = float(np.sqrt(mse))

            # R2 Score
            ss_res = np.sum((var_targets - var_predictions) ** 2)
            ss_tot = np.sum((var_targets - var_targets.mean()) ** 2)
            r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

            # Additional metrics
            max_error = float(np.max(np.abs(var_targets - var_predictions)))
            median_ae = float(np.median(np.abs(var_targets - var_predictions)))

            # MAPE (with epsilon to avoid division by zero)
            mape = float(np.mean(np.abs((var_targets - var_predictions) / (np.abs(var_targets) + 1e-8))) * 100)

            # Bias
            bias = float(np.mean(var_predictions - var_targets))

            metrics_dict[var_name] = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2_score': r2,
                'max_error': max_error,
                'median_absolute_error': median_ae,
                'mape': mape,
                'bias': bias,
                'mean_target': float(var_targets.mean()),
                'std_target': float(var_targets.std()),
                'mean_prediction': float(var_predictions.mean()),
                'std_prediction': float(var_predictions.std()),
                'min_target': float(var_targets.min()),
                'max_target': float(var_targets.max()),
                'min_prediction': float(var_predictions.min()),
                'max_prediction': float(var_predictions.max())
            }

        return metrics_dict


class ResultsWriter:
    """Write validation results to files.

    Attributes:
        output_dir: Path to output directory.
    """

    def __init__(self, output_dir: str, logger=None):
        """Initialize results writer.

        Args:
            output_dir: Directory to save results.
            logger: Optional logger for output.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def write_summary(self, results: Dict[str, Any]):
        """Write summary text file.

        Args:
            results: Dictionary containing validation results.
        """
        summary_path = self.output_dir / "validation_results.txt"

        try:
            with open(summary_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("VALIDATION RESULTS SUMMARY\n")
                f.write("=" * 80 + "\n\n")

                # Overall metrics
                overall = results['overall']
                f.write("Overall Metrics:\n")
                f.write(f"  Average Loss: {overall['average_loss']:.6f} (+/-{overall['std_loss']:.6f})\n")
                f.write(f"  Average MAE:  {overall['average_mae']:.4f} (+/-{overall['std_mae']:.4f})\n")
                f.write(f"  Average RMSE: {overall['average_rmse']:.4f} (+/-{overall['std_rmse']:.4f})\n")
                f.write(f"  Average R2:   {overall['average_r2']:.4f} (+/-{overall['std_r2']:.4f})\n")

                if overall['average_cosine_sim'] is not None:
                    f.write(f"  Average Cosine Sim: {overall['average_cosine_sim']:.4f} "
                           f"(+/-{overall['std_cosine_sim']:.6f})\n")

                f.write(f"\n  Total Samples:  {results['total_samples']}\n")
                f.write(f"  Success Rate:   {results['success_rate']:.1f}%\n")

                # Per-variable metrics
                f.write("\n" + "=" * 80 + "\n")
                f.write("METRICS BY VARIABLE\n")
                f.write("=" * 80 + "\n\n")

                for var_name, metrics in results['per_variable'].items():
                    f.write(f"{var_name}:\n")
                    f.write(f"  MAE:   {metrics['mae']:.4f}\n")
                    f.write(f"  RMSE:  {metrics['rmse']:.4f}\n")
                    f.write(f"  R2:    {metrics['r2_score']:.4f}\n")
                    f.write(f"  Max Error: {metrics['max_error']:.4f}\n")
                    f.write(f"  Median AE: {metrics['median_absolute_error']:.4f}\n")
                    f.write(f"  MAPE:  {metrics['mape']:.2f}%\n")
                    f.write(f"  Bias:  {metrics['bias']:.4f}\n")
                    f.write("\n")

            message = f"Summary saved: {summary_path}"
            if self.logger:
                self.logger.info(message)
            else:
                print(message)

        except Exception as e:
            error_msg = f"Failed to save summary: {e}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(f"Error: {error_msg}")

    def write_csv(self, file_results: List[Dict[str, Any]], target_variables: List[str]):
        """Write detailed CSV file.

        Args:
            file_results: List of per-file results.
            target_variables: List of target variable names.
        """
        csv_path = self.output_dir / "validation_results.csv"

        try:
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = ['file_name', 'target', 'prediction', 'error',
                            'absolute_error', 'squared_error']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for result in file_results:
                    file_name = result['file_name']
                    targets = result['targets']  # Shape: (n_groups, n_variables)
                    predictions = result['predictions']

                    n_groups, n_variables = targets.shape

                    for var_idx, var_name in enumerate(target_variables):
                        for group_idx in range(n_groups):
                            target_val = float(targets[group_idx, var_idx])
                            pred_val = float(predictions[group_idx, var_idx])
                            error = pred_val - target_val
                            abs_error = abs(error)
                            sq_error = error ** 2

                            full_identifier = f"{file_name}_group{group_idx}_{var_name}"

                            writer.writerow({
                                'file_name': full_identifier,
                                'target': target_val,
                                'prediction': pred_val,
                                'error': error,
                                'absolute_error': abs_error,
                                'squared_error': sq_error
                            })

            message = f"CSV saved: {csv_path}"
            if self.logger:
                self.logger.info(message)
            else:
                print(message)

        except Exception as e:
            error_msg = f"Failed to save CSV: {e}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(f"Error: {error_msg}")


class Validator:
    """Validator for multimodal solar wind prediction model.

    Supports three model types:
    - "fusion": Full multimodal model with cross-attention
    - "transformer": OMNI time series only
    - "convlstm": SDO images only

    Attributes:
        config: Configuration object.
        model: PyTorch model.
        criterion: Loss function.
        device: Device for computation.
        metrics_aggregator: MetricsAggregator instance.
        results_writer: ResultsWriter instance.
    """

    def __init__(
        self,
        config,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
        logger=None
    ):
        """Initialize validator.

        Args:
            config: Configuration object.
            model: PyTorch model.
            criterion: Loss function.
            device: Device for computation.
            logger: Optional logger for output.
        """
        self.config = config
        self.model = model
        self.criterion = criterion
        self.device = device
        self.logger = logger

        # Settings from config
        self.save_plots = getattr(config.validation, 'save_plots', True)
        self.save_npz = getattr(config.validation, 'save_npz', True)

        # Model type determines validation behavior
        self.model_type = getattr(config.model, 'model_type', 'fusion')

        # Get variable lists for plotting
        self._setup_variable_info()

        # Components
        self.metrics_aggregator = MetricsAggregator(self.target_variables)
        self.results_writer = ResultsWriter(
            output_dir=config.validation.output_dir,
            logger=logger
        )

        # Compute alignment flag (for fusion and baseline models)
        self.compute_alignment = (
            getattr(config.validation, 'compute_alignment', True)
            and self.model_type in ['fusion', 'baseline']
        )

        # Normalizer for denormalization (set in validate())
        self.normalizer = None

    def _setup_variable_info(self):
        """Setup variable information for plotting."""
        use_csv = getattr(self.config.data.modalities, 'timeseries', False)

        if use_csv:
            self.input_variables = list(self.config.data.timeseries.input_variables)
            self.target_variables = list(self.config.data.timeseries.target_variables)
        elif hasattr(self.config.data, 'omni') and hasattr(self.config.data.omni, 'input'):
            self.input_variables = list(self.config.data.omni.input.variables)
            self.target_variables = list(self.config.data.omni.target.variables)
        else:
            self.input_variables = list(self.config.data.input_variables)
            self.target_variables = list(self.config.data.target_variables)

    def validate_batch(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Validate single batch.

        Args:
            data_dict: Dictionary containing input data.

        Returns:
            Dictionary containing loss, metrics, and predictions.
        """
        self.model.eval()

        # Move data to device (sdo is optional — absent in CSV timeseries mode)
        sdo = data_dict["sdo"].to(self.device) if "sdo" in data_dict else None
        inputs = data_dict["inputs"].to(self.device)
        targets = data_dict["targets"].to(self.device)

        with torch.no_grad():
            # Forward pass depends on model type
            if self.compute_alignment and self.model_type in ["fusion", "baseline"]:
                outputs, feature_1, feature_2 = self.model(
                    inputs, sdo, return_features=True
                )
                cosine_sim = F.cosine_similarity(
                    feature_1, feature_2, dim=1
                ).mean().item()
            else:
                outputs = self.model(inputs, sdo, return_features=False)
                cosine_sim = None

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Calculate metrics
            mae = F.l1_loss(outputs, targets).item()
            mse = F.mse_loss(outputs, targets).item()
            rmse = np.sqrt(mse)

            # R2 score
            ss_res = torch.sum((targets - outputs) ** 2).item()
            ss_tot = torch.sum((targets - targets.mean()) ** 2).item()
            r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        result = {
            'loss': loss.item(),
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2_score,
            'inputs': inputs.cpu().numpy(),
            'targets': targets.cpu().numpy(),
            'predictions': outputs.cpu().numpy()
        }

        if cosine_sim is not None:
            result['cosine_sim'] = cosine_sim

        return result

    def validate(self, dataloader) -> Dict[str, Any]:
        """Run validation on entire dataset.

        Args:
            dataloader: Validation data loader.

        Returns:
            Dictionary containing validation results.
        """
        # Store normalizer for denormalization in plots
        if hasattr(dataloader.dataset, 'normalizer'):
            self.normalizer = dataloader.dataset.normalizer

        if self.logger:
            self.logger.info(f"Running validation (model_type: {self.model_type})...")
        else:
            print(f"Running validation (model_type: {self.model_type})...")

        self.metrics_aggregator.reset()
        failed_batches = 0

        # Create plots subdirectory if saving plots
        # Create output directories
        plots_dir = None
        npz_dir = None

        if self.save_plots:
            plots_dir = Path(self.results_writer.output_dir) / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

        if self.save_npz:
            npz_dir = Path(self.results_writer.output_dir) / "npz"
            npz_dir.mkdir(parents=True, exist_ok=True)

        report_freq = getattr(self.config.validation, 'report_freq', 50)

        for batch_idx, data_dict in enumerate(dataloader):
            try:
                # Validate batch
                batch_result = self.validate_batch(data_dict)

                # Extract file names
                file_names = self._extract_file_names(data_dict, batch_idx)

                # Save individual plots if enabled
                if self.save_plots:
                    self._save_prediction_plots(batch_result, file_names, plots_dir)

                # Save NPZ files if enabled
                if self.save_npz:
                    self._save_npz_files(batch_result, file_names, npz_dir)

                # Update aggregator
                self.metrics_aggregator.update(batch_result, file_names)

                # Log progress periodically
                if (batch_idx + 1) % report_freq == 0:
                    self.log_progress(batch_idx, len(dataloader))

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Validation failed for batch {batch_idx}: {e}")
                else:
                    print(f"Warning: Batch {batch_idx} failed: {e}")
                failed_batches += 1
                continue

        # Get summary
        try:
            results = self.metrics_aggregator.get_summary()
            results['failed_batches'] = failed_batches
            results['success_rate'] = ((len(dataloader) - failed_batches) / len(dataloader)) * 100
            results['output_directory'] = str(self.results_writer.output_dir)

            # Log summary
            self.log_summary(results)

            # Write results
            self.results_writer.write_summary(results)
            self.results_writer.write_csv(results['file_results'], self.target_variables)

            return results

        except Exception as e:
            error_msg = f"Validation failed: {e}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(f"Error: {error_msg}")
            raise RuntimeError(error_msg)

    def _extract_file_names(self, data_dict: Dict[str, Any], batch_idx: int) -> List[str]:
        """Extract file names from data dict.

        Args:
            data_dict: Data dictionary.
            batch_idx: Batch index for fallback naming.

        Returns:
            List of file names.
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

    def _save_prediction_plots(
        self,
        batch_result: Dict[str, Any],
        file_names: List[str],
        plots_dir: Path
    ):
        """Save prediction time series plots for each sample.

        Args:
            batch_result: Batch validation results.
            file_names: List of file names.
            plots_dir: Directory to save plots.
        """
        inputs = batch_result['inputs']  # (batch_size, seq_len, num_input_vars)
        targets = batch_result['targets']  # (batch_size, target_len, num_target_vars)
        predictions = batch_result['predictions']

        for i, file_name in enumerate(file_names):
            file_name_base = os.path.splitext(file_name)[0]
            save_path = plots_dir / f"{file_name_base}.png"

            plot_prediction_timeseries(
                inputs=inputs[i],
                targets=targets[i],
                predictions=predictions[i],
                input_variables=self.input_variables,
                target_variables=self.target_variables,
                save_path=save_path,
                title=f"Validation - {file_name_base}",
                logger=self.logger,
                normalizer=self.normalizer
            )

    def _save_npz_files(
        self,
        batch_result: Dict[str, Any],
        file_names: List[str],
        npz_dir: Path
    ):
        """Save NPZ files for each sample (denormalized to original scale).

        Args:
            batch_result: Batch validation results.
            file_names: List of file names.
            npz_dir: Directory to save NPZ files.
        """
        inputs = batch_result['inputs']
        targets = batch_result['targets']
        predictions = batch_result['predictions']

        for i, file_name in enumerate(file_names):
            file_name_base = os.path.splitext(file_name)[0]
            npz_path = npz_dir / f"{file_name_base}.npz"

            inp = inputs[i].copy()
            tgt = targets[i].copy()
            pred = predictions[i].copy()

            # Denormalize to original scale
            if self.normalizer is not None:
                for var_idx, var_name in enumerate(self.input_variables):
                    inp[:, var_idx] = self.normalizer.denormalize_omni(
                        inp[:, var_idx], var_name
                    )
                for var_idx, var_name in enumerate(self.target_variables):
                    tgt[:, var_idx] = self.normalizer.denormalize_omni(
                        tgt[:, var_idx], var_name
                    )
                    pred[:, var_idx] = self.normalizer.denormalize_omni(
                        pred[:, var_idx], var_name
                    )

            np.savez_compressed(
                npz_path,
                inputs=inp,
                targets=tgt,
                predictions=pred,
                input_variables=self.input_variables,
                target_variables=self.target_variables
            )

    def _save_overall_plot(self, results: Dict[str, Any], dataloader):
        """Save overall validation plot with averaged results.

        Args:
            results: Validation results dictionary.
            dataloader: Dataloader (for accessing stat_dict).
        """
        try:
            # Check if stat_dict is available
            stat_dict = None
            if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'stat_dict'):
                stat_dict = dataloader.dataset.stat_dict

            if stat_dict is None:
                if self.logger:
                    self.logger.warning("Skipping overall plot: stat_dict not available in dataset")
                return

            # Extract all targets and predictions
            all_targets = []
            all_predictions = []

            for file_result in results['file_results']:
                all_targets.append(file_result['targets'])
                all_predictions.append(file_result['predictions'])

            # Convert to arrays and compute mean
            all_targets = np.array(all_targets)  # (n_samples, n_groups, n_variables)
            all_predictions = np.array(all_predictions)

            mean_targets = np.mean(all_targets, axis=0)  # (n_groups, n_variables)
            mean_predictions = np.mean(all_predictions, axis=0)

            # Save overall plot
            plots_dir = Path(self.results_writer.output_dir) / "plots"
            overall_plot_path = str(plots_dir / "overall_validation_results")
            overall_plot_title = "Overall Validation Results (Mean)"

            save_plot(
                targets=mean_targets,
                outputs=mean_predictions,
                target_variables=self.target_variables,
                stat_dict=stat_dict,
                plot_path=overall_plot_path,
                plot_title=overall_plot_title,
                logger=self.logger
            )

            if self.logger:
                self.logger.info(f"Overall validation plot saved: {overall_plot_path}.png")

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to create overall validation plot: {e}")
            else:
                print(f"Warning: Failed to create overall validation plot: {e}")

    def log_progress(self, batch_idx: int, total_batches: int):
        """Log validation progress.

        Args:
            batch_idx: Current batch index.
            total_batches: Total number of batches.
        """
        summary = self.metrics_aggregator.get_summary()
        overall = summary['overall']

        log_msg = (
            f"Processed {batch_idx + 1}/{total_batches} batches | "
            f"Avg Loss: {overall['average_loss']:.6f} | "
            f"Avg MAE: {overall['average_mae']:.4f} | "
            f"Avg RMSE: {overall['average_rmse']:.4f} | "
            f"Avg R2: {overall['average_r2']:.4f}"
        )

        if overall['average_cosine_sim'] is not None:
            log_msg += f" | Avg Cosine Sim: {overall['average_cosine_sim']:.4f}"

        if self.logger:
            self.logger.info(log_msg)
        else:
            print(log_msg)

    def log_summary(self, results: Dict[str, Any]):
        """Log validation summary.

        Args:
            results: Dictionary containing validation results.
        """
        overall = results['overall']

        if self.logger:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("VALIDATION RESULTS SUMMARY")
            self.logger.info("=" * 80)
            self.logger.info(f"Average Loss: {overall['average_loss']:.6f} (+/-{overall['std_loss']:.6f})")
            self.logger.info(f"Average MAE:  {overall['average_mae']:.4f} (+/-{overall['std_mae']:.4f})")
            self.logger.info(f"Average RMSE: {overall['average_rmse']:.4f} (+/-{overall['std_rmse']:.4f})")
            self.logger.info(f"Average R2:   {overall['average_r2']:.4f} (+/-{overall['std_r2']:.4f})")

            if overall['average_cosine_sim'] is not None:
                self.logger.info(f"Average Cosine Sim: {overall['average_cosine_sim']:.4f} "
                               f"(+/-{overall['std_cosine_sim']:.6f})")

            self.logger.info(f"Total Samples:  {results['total_samples']}")
            self.logger.info(f"Failed Batches: {results['failed_batches']}")
            self.logger.info(f"Success Rate:   {results['success_rate']:.1f}%")

            self.logger.info("\n" + "=" * 80)
            self.logger.info("METRICS BY VARIABLE")
            self.logger.info("=" * 80)

            for var_name, metrics in results['per_variable'].items():
                self.logger.info(f"\n{var_name}:")
                self.logger.info(f"  MAE:   {metrics['mae']:.4f}")
                self.logger.info(f"  RMSE:  {metrics['rmse']:.4f}")
                self.logger.info(f"  R2:    {metrics['r2_score']:.4f}")
                self.logger.info(f"  Max Error: {metrics['max_error']:.4f}")
                self.logger.info(f"  Median AE: {metrics['median_absolute_error']:.4f}")
                self.logger.info(f"  MAPE:  {metrics['mape']:.2f}%")
                self.logger.info(f"  Bias:  {metrics['bias']:.4f}")

            self.logger.info("\n" + "=" * 80 + "\n")
        else:
            print("=" * 80)
            print("VALIDATION RESULTS SUMMARY")
            print("=" * 80)
            print(f"Average Loss: {overall['average_loss']:.6f}")
            print(f"Average MAE:  {overall['average_mae']:.4f}")
            print(f"Average RMSE: {overall['average_rmse']:.4f}")
            print(f"Average R2:   {overall['average_r2']:.4f}")

            if overall['average_cosine_sim'] is not None:
                print(f"Average Cosine Sim: {overall['average_cosine_sim']:.4f}")

            print(f"\nTotal Samples:  {results['total_samples']}")
            print(f"Success Rate:   {results['success_rate']:.1f}%")
            print("\nKey Metrics by Variable:")

            for var_name, metrics in results['per_variable'].items():
                print(f"  {var_name}:")
                print(f"    MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, "
                      f"R2: {metrics['r2_score']:.4f}")

            print(f"\nResults saved to: {results['output_directory']}")
            print("=" * 80)
