"""Testing components for solar wind prediction model.

Contains Tester and TestResultsWriter classes for inference.
Supports both TestDataset (with targets) and OperationDataset (without targets).

Classes:
    TestResultsWriter: Write test results (predictions) to files.
    Tester: Inference loop for multimodal solar wind prediction.

Example:
    >>> from src.testers import Tester
    >>> tester = Tester(config, model, device)
    >>> results = tester.test(dataloader)
"""

import os
import csv
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def plot_prediction_timeseries(
    inputs: np.ndarray,
    targets: Optional[np.ndarray],
    predictions: np.ndarray,
    input_variables: List[str],
    target_variables: List[str],
    save_path: Path,
    title: str = "Prediction",
    logger=None
):
    """Plot input, target, and prediction time series.

    Args:
        inputs: Input data (seq_len, num_input_vars)
        targets: Target data (target_len, num_target_vars) or None
        predictions: Prediction data (target_len, num_target_vars)
        input_variables: List of input variable names
        target_variables: List of target variable names
        save_path: Path to save the plot
        title: Plot title
        logger: Optional logger
    """
    try:
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
            ax.set_ylabel(f'{target_var} (normalized)', fontsize=10)
            ax.set_title(f'{title} - {target_var}', fontsize=12, fontweight='bold')
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)

            # Add metrics if targets available
            if targets is not None:
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


class TestResultsWriter:
    """Write test results to files.

    Attributes:
        output_dir: Path to output directory.
    """

    def __init__(self, output_dir: str):
        """Initialize test results writer.

        Args:
            output_dir: Directory to save results.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_predictions(
        self,
        file_results: List[Dict[str, Any]],
        target_variables: List[str]
    ):
        """Write predictions to CSV file.

        Args:
            file_results: List of per-file prediction results.
            target_variables: List of target variable names.
        """
        csv_path = self.output_dir / "test_predictions.csv"

        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = ["file_name", "timestep", "variable", "prediction"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in file_results:
                file_name = result["file_name"]
                predictions = result["predictions"]

                n_timesteps, n_variables = predictions.shape

                for var_idx, var_name in enumerate(target_variables):
                    for t in range(n_timesteps):
                        pred_val = float(predictions[t, var_idx])

                        writer.writerow({
                            "file_name": file_name,
                            "timestep": t,
                            "variable": var_name,
                            "prediction": pred_val
                        })

        logging.info(f"Predictions saved: {csv_path}")

    def write_npz(self, file_results: List[Dict[str, Any]]):
        """Write predictions to compressed NPZ files.

        Args:
            file_results: List of per-file prediction results.
        """
        npz_dir = self.output_dir / "predictions"
        npz_dir.mkdir(parents=True, exist_ok=True)

        for result in file_results:
            file_name = result["file_name"]
            predictions = result["predictions"]

            file_name_base = os.path.splitext(file_name)[0]
            npz_path = npz_dir / f"{file_name_base}.npz"

            np.savez_compressed(npz_path, predictions=predictions)

        logging.info(f"NPZ files saved to: {npz_dir}")

    def write_summary(self, total_samples: int, output_dir: str):
        """Write summary text file.

        Args:
            total_samples: Total number of samples processed.
            output_dir: Output directory path.
        """
        summary_path = self.output_dir / "test_summary.txt"

        with open(summary_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("TEST RESULTS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total Samples Processed: {total_samples}\n")
            f.write(f"Output Directory: {output_dir}\n")
            f.write("\n" + "=" * 80 + "\n")

        logging.info(f"Summary saved: {summary_path}")


class Tester:
    """Tester for multimodal solar wind prediction model (inference only).

    Supports three model types:
    - "fusion": Full multimodal model with cross-attention
    - "transformer": OMNI time series only
    - "convlstm": SDO images only

    Attributes:
        config: Configuration object.
        model: PyTorch model.
        device: Device for computation.
        results_writer: TestResultsWriter instance.
    """

    def __init__(
        self,
        config,
        model: nn.Module,
        device: torch.device
    ):
        """Initialize tester.

        Args:
            config: Configuration object.
            model: PyTorch model.
            device: Device for computation.
        """
        self.config = config
        self.model = model
        self.device = device

        # Model type
        self.model_type = getattr(config.model, 'model_type', 'fusion')

        # Components
        self.results_writer = TestResultsWriter(output_dir=config.test.output_dir)

        # Test settings from config
        self.report_freq = getattr(config.test, 'report_freq', 50)
        self.save_npz = getattr(config.test, 'save_npz', True)
        self.save_plots = getattr(config.test, 'save_plots', True)

        # Get variable lists
        self._setup_variable_info()

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

    def predict_batch(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Run inference on single batch.

        Args:
            data_dict: Dictionary containing input data (sdo, inputs, optionally targets).

        Returns:
            Dictionary with inputs, predictions, and optionally targets.
        """
        self.model.eval()

        sdo = data_dict["sdo"].to(self.device) if "sdo" in data_dict else None
        inputs = data_dict["inputs"].to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs, sdo, return_features=False)

        result = {
            'inputs': data_dict["inputs"].cpu().numpy(),
            'predictions': outputs.cpu().numpy()
        }

        # Include targets if available (TestDataset has targets)
        if "targets" in data_dict:
            result['targets'] = data_dict["targets"].cpu().numpy()

        return result

    def test(self, dataloader) -> Dict[str, Any]:
        """Run inference on entire test dataset.

        Args:
            dataloader: Test data loader (OperationDataset or TestDataset).

        Returns:
            Dictionary containing file results and summary information.
        """
        logging.info(f"Running inference (model_type: {self.model_type})...")

        # Create output directories
        plots_dir = None
        npz_dir = None

        if self.save_plots:
            plots_dir = Path(self.results_writer.output_dir) / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

        if self.save_npz:
            npz_dir = Path(self.results_writer.output_dir) / "npz"
            npz_dir.mkdir(parents=True, exist_ok=True)

        file_results = []
        total_samples = 0
        has_targets = False

        for batch_idx, data_dict in enumerate(dataloader):
            batch_result = self.predict_batch(data_dict)
            file_names = self._extract_file_names(data_dict, batch_idx)
            has_targets = 'targets' in batch_result

            batch_size = batch_result['predictions'].shape[0]

            for idx in range(batch_size):
                file_name = file_names[idx]
                file_name_base = os.path.splitext(file_name)[0]

                result = {
                    "file_name": file_name,
                    "inputs": batch_result['inputs'][idx],
                    "predictions": batch_result['predictions'][idx]
                }
                if has_targets:
                    result["targets"] = batch_result['targets'][idx]

                file_results.append(result)
                total_samples += 1

                # Save plot
                if self.save_plots:
                    save_path = plots_dir / f"{file_name_base}.png"
                    plot_prediction_timeseries(
                        inputs=result['inputs'],
                        targets=result.get('targets'),
                        predictions=result['predictions'],
                        input_variables=self.input_variables,
                        target_variables=self.target_variables,
                        save_path=save_path,
                        title=f"Test - {file_name_base}"
                    )

                # Save NPZ
                if self.save_npz:
                    npz_path = npz_dir / f"{file_name_base}.npz"
                    npz_data = {
                        'inputs': result['inputs'],
                        'predictions': result['predictions'],
                        'input_variables': self.input_variables,
                        'target_variables': self.target_variables
                    }
                    if has_targets:
                        npz_data['targets'] = result['targets']
                    np.savez_compressed(npz_path, **npz_data)

            if (batch_idx + 1) % self.report_freq == 0:
                logging.info(
                    f"Processed {batch_idx + 1}/{len(dataloader)} batches | "
                    f"Samples: {total_samples}"
                )

        # Write CSV results
        self.results_writer.write_predictions(file_results, self.target_variables)

        self.results_writer.write_summary(
            total_samples=total_samples,
            output_dir=str(self.results_writer.output_dir)
        )

        # Log summary
        self.log_summary(total_samples, has_targets)

        return {
            "file_results": file_results,
            "total_samples": total_samples,
            "has_targets": has_targets,
            "output_directory": str(self.results_writer.output_dir)
        }

    def _extract_file_names(self, data_dict: Dict[str, Any], batch_idx: int) -> List[str]:
        """Extract file names from data dict.

        Args:
            data_dict: Data dictionary.
            batch_idx: Batch index for fallback naming.

        Returns:
            List of file names.
        """
        if "file_names" not in data_dict:
            batch_size = data_dict["inputs"].size(0)
            return [f"batch_{batch_idx}_sample_{idx}" for idx in range(batch_size)]

        file_names_raw = data_dict["file_names"]

        if isinstance(file_names_raw, torch.Tensor):
            return [str(name) for name in file_names_raw.tolist()]
        elif isinstance(file_names_raw, list):
            return [str(name) for name in file_names_raw]
        else:
            return [str(file_names_raw)]

    def log_summary(self, total_samples: int, has_targets: bool = False):
        """Log test summary.

        Args:
            total_samples: Total number of samples processed.
            has_targets: Whether targets were available.
        """
        logging.info("=" * 80)
        logging.info("TEST COMPLETED")
        logging.info("=" * 80)
        logging.info(f"Model Type: {self.model_type}")
        logging.info(f"Total Samples: {total_samples}")
        logging.info(f"Has Targets: {has_targets}")
        logging.info(f"Plots Saved: {self.save_plots}")
        logging.info(f"NPZ Saved: {self.save_npz}")
        logging.info(f"Output Directory: {self.results_writer.output_dir}")
        logging.info("=" * 80)
