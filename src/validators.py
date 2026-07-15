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

# Import shared plotting and utility functions
from .plotting import plot_prediction_timeseries, extract_file_names, denormalize_arrays
from .uncertainty import mcd_sample_stats, uncertainty_metrics



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

                # MC-dropout calibration (raw; pre-recalibration) for the primary target.
                cal = results.get('calibration')
                if cal is not None:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("MC-DROPOUT CALIBRATION (raw)\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"  PICP 1sigma:      {cal['picp_1sigma']:.3f}  (ideal 0.683)\n")
                    f.write(f"  PICP 2sigma:      {cal['picp_2sigma']:.3f}  (ideal 0.954)\n")
                    f.write(f"  Sharpness (2s):   {cal['sharpness_2sigma']:.4f}\n")
                    f.write(f"  NLL (Gaussian):   {cal['nll_gaussian']:.4f}\n")
                    f.write(f"  CRPS (Gaussian):  {cal['crps_gaussian']:.4f}\n")
                    f.write(f"  MAE (MCD mean):   {cal['mae_mcd_mean']:.4f}\n")

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
        # MC-dropout is folded into the validation pass (always on, over every event).
        # mcd_samples stochastic forwards per event build the predictive interval.
        self.mcd_samples = int(getattr(config.validation, 'mcd_samples', 100))

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

        # Fold MC-dropout uncertainty into the same pass. The deterministic `predictions`
        # above stay the headline (metrics are computed from them); the MCD stats are an
        # additional, per-event predictive interval. Needs the normalizer (denorm happens
        # per sample), which is set in validate() before the batch loop.
        if self.normalizer is not None and self.mcd_samples > 0:
            result['mcd'] = mcd_sample_stats(
                self.model, inputs, sdo, self.normalizer,
                self.target_variables, num_samples=self.mcd_samples
            )
            self.model.eval()  # mcd_sample_stats leaves eval; keep the model deterministic

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

        # Calibration accumulators for the primary target (denormalized true / MCD mean+std),
        # pooled across all events to score the folded-in MC-dropout intervals.
        self._cal_true, self._cal_mean, self._cal_std = [], [], []

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

                # Accumulate calibration inputs for the primary target (index 0).
                if 'mcd' in batch_result and self.normalizer is not None:
                    tv = self.target_variables[0]
                    true0 = self.normalizer.denormalize_omni(
                        batch_result['targets'][..., 0], tv)
                    self._cal_true.append(np.asarray(true0).ravel())
                    self._cal_mean.append(batch_result['mcd']['mcd_mean'][..., 0].ravel())
                    self._cal_std.append(batch_result['mcd']['mcd_std'][..., 0].ravel())

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

            # Raw MC-dropout calibration for the primary target (headline stays deterministic).
            # Post-hoc recalibration (per-run sigma scale) is a separate step:
            # analysis/recalibrate_mcd.py reads validation/<epoch>/npz.zip.
            if self._cal_true:
                results['calibration'] = uncertainty_metrics(
                    np.concatenate(self._cal_true),
                    np.concatenate(self._cal_mean),
                    np.concatenate(self._cal_std),
                )

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

        Delegates to shared extract_file_names() utility.

        Args:
            data_dict: Data dictionary.
            batch_idx: Batch index for fallback naming.

        Returns:
            List of file names.
        """
        return extract_file_names(data_dict, batch_idx)

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
                predictions=predictions[i],
                input_variables=self.input_variables,
                target_variables=self.target_variables,
                save_path=save_path,
                targets=targets[i],
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

        mcd = batch_result.get('mcd')

        for i, file_name in enumerate(file_names):
            file_name_base = os.path.splitext(file_name)[0]
            npz_path = npz_dir / f"{file_name_base}.npz"

            inp = inputs[i]
            tgt = targets[i]
            pred = predictions[i]

            # Denormalize to original scale
            if self.normalizer is not None:
                inp, tgt, pred = denormalize_arrays(
                    inp, pred, self.input_variables, self.target_variables,
                    self.normalizer, targets=tgt
                )

            arrays = {
                'anchor': file_name_base,
                'inputs': inp,
                'targets': tgt,
                'predictions': pred,
                'input_variables': self.input_variables,
                'target_variables': self.target_variables,
            }
            # MC-dropout predictive interval (already denormalized), per-event slice.
            # mcd_mean/std/min/max/median + empirical 90% (q05/q95) and 95% (q025/q975)
            # bands so downstream can use a Gaussian or a distribution-free interval.
            if mcd is not None:
                for key, val in mcd.items():
                    arrays[key] = val if key == 'n_samples' else val[i]

            np.savez_compressed(npz_path, **arrays)

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
