#!/usr/bin/env python
"""Collect and analyze experiment results.

This module provides functionality to:
- Collect results from all completed experiments
- Extract performance metrics (loss, MAE, RMSE, R²)
- Generate summary reports
- Create comparison visualizations

Usage:
    # Collect results from completed experiments
    python experiments/result_aggregator.py collect

    # Generate summary report
    python experiments/result_aggregator.py report

    # Compare models by type
    python experiments/result_aggregator.py compare --by=model_type

    # Generate visualizations
    python experiments/result_aggregator.py plot --type=heatmap
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from experiment_tracker import ExperimentTracker


class ResultAggregator:
    """Collect and analyze experiment results."""

    def __init__(self, tracking_file: Path, outputs_dir: Path):
        """Initialize aggregator.

        Args:
            tracking_file: Path to experiments.csv tracking file
            outputs_dir: Path to outputs directory containing experiment results
        """
        self.tracking_file = Path(tracking_file)
        self.outputs_dir = Path(outputs_dir)
        self.tracker = ExperimentTracker(tracking_file)
        self.results = []

    def collect_results(self, force: bool = False) -> int:
        """Collect results from all completed experiments.

        Args:
            force: If True, recollect all results (otherwise skip already collected)

        Returns:
            Number of results collected
        """
        completed = self.tracker.get_experiments_by_status("completed")

        if not completed:
            print("No completed experiments found.")
            return 0

        print(f"Collecting results from {len(completed)} completed experiments...")

        collected = 0
        for exp in completed:
            exp_id = exp["exp_id"]
            exp_name = exp["exp_name"]

            # Skip if already has best_loss and not forcing
            if exp.get("best_loss") and not force:
                collected += 1
                continue

            # Find experiment output directory
            exp_output_dir = self._find_experiment_output(exp_name)

            if not exp_output_dir:
                print(f"  Warning: Output not found for {exp_id} ({exp_name})")
                continue

            # Extract metrics from training logs
            metrics = self._extract_metrics(exp_output_dir)

            if metrics:
                # Update tracker with best loss
                self.tracker.update_status(
                    exp_id=exp_id,
                    status="completed",
                    best_loss=str(metrics.get("best_val_loss", ""))
                )

                self.results.append({
                    "exp_id": exp_id,
                    "exp_name": exp_name,
                    **exp,
                    **metrics
                })
                collected += 1

                if collected % 50 == 0:
                    print(f"  Collected {collected}/{len(completed)}...")

        print(f"\nCollected {collected} results.")
        return collected

    def _find_experiment_output(self, exp_name: str) -> Optional[Path]:
        """Find experiment output directory.

        Args:
            exp_name: Experiment name

        Returns:
            Path to output directory or None if not found
        """
        # Check for direct match
        direct_path = self.outputs_dir / exp_name
        if direct_path.exists():
            return direct_path

        # Check for timestamped directories
        for subdir in self.outputs_dir.iterdir():
            if subdir.is_dir() and exp_name in subdir.name:
                return subdir

        return None

    def _extract_metrics(self, exp_output_dir: Path) -> Dict:
        """Extract metrics from experiment output directory.

        Args:
            exp_output_dir: Path to experiment output directory

        Returns:
            Dict containing extracted metrics
        """
        metrics = {}

        # Try to read metrics.json if exists
        metrics_file = exp_output_dir / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                return metrics
            except (json.JSONDecodeError, IOError):
                pass

        # Try to parse training log
        log_file = exp_output_dir / "training.log"
        if log_file.exists():
            metrics = self._parse_training_log(log_file)

        # Try to find best checkpoint info
        checkpoints_dir = exp_output_dir / "checkpoints"
        if checkpoints_dir.exists():
            best_info = self._find_best_checkpoint_info(checkpoints_dir)
            metrics.update(best_info)

        return metrics

    def _parse_training_log(self, log_file: Path) -> Dict:
        """Parse training log for metrics.

        Args:
            log_file: Path to training.log

        Returns:
            Dict containing parsed metrics
        """
        metrics = {
            "best_val_loss": None,
            "best_epoch": None,
            "final_train_loss": None,
            "final_val_loss": None,
        }

        try:
            with open(log_file, 'r') as f:
                content = f.read()

            # Look for validation loss patterns
            val_loss_pattern = r"val_loss[:\s]+([0-9.]+)"
            val_losses = re.findall(val_loss_pattern, content, re.IGNORECASE)
            if val_losses:
                val_losses = [float(v) for v in val_losses]
                metrics["best_val_loss"] = min(val_losses)
                metrics["best_epoch"] = val_losses.index(min(val_losses)) + 1
                metrics["final_val_loss"] = val_losses[-1]

            # Look for training loss patterns
            train_loss_pattern = r"train_loss[:\s]+([0-9.]+)"
            train_losses = re.findall(train_loss_pattern, content, re.IGNORECASE)
            if train_losses:
                metrics["final_train_loss"] = float(train_losses[-1])

            # Look for MAE, RMSE, R² patterns
            mae_pattern = r"MAE[:\s]+([0-9.]+)"
            rmse_pattern = r"RMSE[:\s]+([0-9.]+)"
            r2_pattern = r"R2|R²[:\s]+([0-9.-]+)"

            mae_matches = re.findall(mae_pattern, content, re.IGNORECASE)
            if mae_matches:
                metrics["final_mae"] = float(mae_matches[-1])

            rmse_matches = re.findall(rmse_pattern, content, re.IGNORECASE)
            if rmse_matches:
                metrics["final_rmse"] = float(rmse_matches[-1])

            r2_matches = re.findall(r2_pattern, content, re.IGNORECASE)
            if r2_matches:
                metrics["final_r2"] = float(r2_matches[-1])

        except (IOError, ValueError):
            pass

        return metrics

    def _find_best_checkpoint_info(self, checkpoints_dir: Path) -> Dict:
        """Find best checkpoint information.

        Args:
            checkpoints_dir: Path to checkpoints directory

        Returns:
            Dict with checkpoint info
        """
        info = {}

        # Look for best checkpoint files
        best_checkpoints = list(checkpoints_dir.glob("best*.pt")) + \
                          list(checkpoints_dir.glob("best*.pth"))

        if best_checkpoints:
            # Try to extract loss from filename
            for ckpt in best_checkpoints:
                match = re.search(r"loss[_=]?([0-9.]+)", ckpt.name)
                if match:
                    info["best_val_loss"] = float(match.group(1))
                    break

        return info

    def generate_summary(self, output_file: Path = None) -> Dict:
        """Generate summary statistics.

        Args:
            output_file: Optional path to save summary CSV

        Returns:
            Dict containing summary statistics
        """
        if not self.results:
            self.collect_results()

        if not self.results:
            print("No results to summarize.")
            return {}

        summary = {
            "total_experiments": len(self.results),
            "by_model_type": defaultdict(list),
            "by_input_days": defaultdict(list),
            "by_target_days": defaultdict(list),
        }

        for result in self.results:
            val_loss = result.get("best_val_loss")
            if val_loss is not None:
                model_type = result.get("model_type", "unknown")
                input_days_count = result.get("input_days_count", 0)
                target_days_count = result.get("target_days_count", 0)

                summary["by_model_type"][model_type].append(val_loss)
                summary["by_input_days"][input_days_count].append(val_loss)
                summary["by_target_days"][target_days_count].append(val_loss)

        # Calculate statistics
        for category in ["by_model_type", "by_input_days", "by_target_days"]:
            stats = {}
            for key, values in summary[category].items():
                if values:
                    stats[key] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "count": len(values)
                    }
            summary[f"{category}_stats"] = stats

        # Save to file if requested
        if output_file:
            self._save_summary_csv(output_file)

        return summary

    def _save_summary_csv(self, output_file: Path):
        """Save results summary to CSV.

        Args:
            output_file: Path to output CSV file
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "exp_id", "exp_name", "model_type",
            "input_days_count", "target_days_count", "subsample_index",
            "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
            "final_mae", "final_rmse", "final_r2"
        ]

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.results)

        print(f"Saved summary to {output_file}")

    def print_report(self):
        """Print formatted report to console."""
        summary = self.generate_summary()

        if not summary:
            return

        print("\n" + "=" * 70)
        print("EXPERIMENT RESULTS SUMMARY")
        print("=" * 70)

        print(f"\nTotal Experiments Analyzed: {summary['total_experiments']}")

        # By model type
        print("\n" + "-" * 50)
        print("By Model Type:")
        print("-" * 50)
        print(f"  {'Model':<15} {'Mean Loss':>12} {'Std':>10} {'Min':>10} {'Count':>8}")
        print("  " + "-" * 55)

        for model_type, stats in sorted(summary.get("by_model_type_stats", {}).items()):
            print(f"  {model_type:<15} {stats['mean']:>12.6f} {stats['std']:>10.6f} "
                  f"{stats['min']:>10.6f} {stats['count']:>8d}")

        # By input days
        print("\n" + "-" * 50)
        print("By Input Days Count:")
        print("-" * 50)
        print(f"  {'Input Days':>10} {'Mean Loss':>12} {'Std':>10} {'Min':>10} {'Count':>8}")
        print("  " + "-" * 50)

        for input_days, stats in sorted(summary.get("by_input_days_stats", {}).items()):
            print(f"  {input_days:>10d} {stats['mean']:>12.6f} {stats['std']:>10.6f} "
                  f"{stats['min']:>10.6f} {stats['count']:>8d}")

        # By target days
        print("\n" + "-" * 50)
        print("By Target Days Count:")
        print("-" * 50)
        print(f"  {'Target Days':>11} {'Mean Loss':>12} {'Std':>10} {'Min':>10} {'Count':>8}")
        print("  " + "-" * 51)

        for target_days, stats in sorted(summary.get("by_target_days_stats", {}).items()):
            print(f"  {target_days:>11d} {stats['mean']:>12.6f} {stats['std']:>10.6f} "
                  f"{stats['min']:>10.6f} {stats['count']:>8d}")

        # Best experiments
        print("\n" + "-" * 50)
        print("Top 10 Best Experiments:")
        print("-" * 50)

        sorted_results = sorted(
            [r for r in self.results if r.get("best_val_loss") is not None],
            key=lambda x: x["best_val_loss"]
        )[:10]

        for i, result in enumerate(sorted_results, 1):
            print(f"  {i:2d}. {result['exp_name']:<35} loss: {result['best_val_loss']:.6f}")

        print("=" * 70 + "\n")

    def plot_comparison(
        self,
        plot_type: str = "heatmap",
        output_file: Path = None,
        show: bool = False
    ):
        """Generate comparison plots.

        Args:
            plot_type: Type of plot ("heatmap", "bar", "box")
            output_file: Path to save plot
            show: Whether to display plot
        """
        if not HAS_MATPLOTLIB:
            print("Error: matplotlib not installed. Cannot generate plots.")
            return

        if not self.results:
            self.collect_results()

        if not self.results:
            print("No results to plot.")
            return

        if plot_type == "heatmap":
            self._plot_heatmap(output_file, show)
        elif plot_type == "bar":
            self._plot_bar(output_file, show)
        elif plot_type == "box":
            self._plot_box(output_file, show)
        else:
            print(f"Unknown plot type: {plot_type}")

    def _plot_heatmap(self, output_file: Path = None, show: bool = False):
        """Plot heatmap of model performance vs input/target days.

        Args:
            output_file: Path to save plot
            show: Whether to display plot
        """
        # Organize data for heatmap
        model_types = ["baseline", "convlstm", "transformer", "fusion"]
        input_days_counts = sorted(set(int(r.get("input_days_count", 0)) for r in self.results))
        target_days_counts = sorted(set(int(r.get("target_days_count", 0)) for r in self.results))

        fig, axes = plt.subplots(1, len(model_types), figsize=(16, 5))

        for idx, model_type in enumerate(model_types):
            ax = axes[idx] if len(model_types) > 1 else axes

            # Create matrix for heatmap
            matrix = np.full((len(input_days_counts), len(target_days_counts)), np.nan)

            for result in self.results:
                if result.get("model_type") == model_type and result.get("best_val_loss"):
                    i = input_days_counts.index(int(result["input_days_count"]))
                    j = target_days_counts.index(int(result["target_days_count"]))
                    # Average over subsamples
                    current = matrix[i, j]
                    if np.isnan(current):
                        matrix[i, j] = result["best_val_loss"]
                    else:
                        matrix[i, j] = (current + result["best_val_loss"]) / 2

            im = ax.imshow(matrix, aspect='auto', cmap='viridis_r')
            ax.set_title(f"{model_type.capitalize()}")
            ax.set_xlabel("Target Days")
            ax.set_ylabel("Input Days")
            ax.set_xticks(range(len(target_days_counts)))
            ax.set_xticklabels(target_days_counts)
            ax.set_yticks(range(len(input_days_counts)))
            ax.set_yticklabels(input_days_counts)

            plt.colorbar(im, ax=ax, label="Val Loss")

        plt.suptitle("Model Performance by Input/Target Days Configuration")
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {output_file}")

        if show:
            plt.show()

        plt.close()

    def _plot_bar(self, output_file: Path = None, show: bool = False):
        """Plot bar chart comparing model types.

        Args:
            output_file: Path to save plot
            show: Whether to display plot
        """
        summary = self.generate_summary()
        stats = summary.get("by_model_type_stats", {})

        if not stats:
            print("No data to plot.")
            return

        model_types = list(stats.keys())
        means = [stats[m]["mean"] for m in model_types]
        stds = [stats[m]["std"] for m in model_types]

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(model_types))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

        ax.set_xlabel("Model Type")
        ax.set_ylabel("Validation Loss (Mean ± Std)")
        ax.set_title("Model Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in model_types])

        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f'{mean:.4f}', ha='center', va='bottom')

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {output_file}")

        if show:
            plt.show()

        plt.close()

    def _plot_box(self, output_file: Path = None, show: bool = False):
        """Plot box plot of model performance distributions.

        Args:
            output_file: Path to save plot
            show: Whether to display plot
        """
        # Organize data by model type
        data_by_model = defaultdict(list)
        for result in self.results:
            if result.get("best_val_loss") is not None:
                data_by_model[result.get("model_type", "unknown")].append(
                    result["best_val_loss"]
                )

        if not data_by_model:
            print("No data to plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        model_types = sorted(data_by_model.keys())
        data = [data_by_model[m] for m in model_types]

        bp = ax.boxplot(data, labels=[m.capitalize() for m in model_types], patch_artist=True)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel("Model Type")
        ax.set_ylabel("Validation Loss")
        ax.set_title("Model Performance Distribution")

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {output_file}")

        if show:
            plt.show()

        plt.close()

    def find_best_config(self, model_type: str = None) -> Optional[Dict]:
        """Find best performing configuration.

        Args:
            model_type: Optional filter by model type

        Returns:
            Best experiment configuration dict
        """
        if not self.results:
            self.collect_results()

        filtered = self.results
        if model_type:
            filtered = [r for r in filtered if r.get("model_type") == model_type]

        if not filtered:
            return None

        best = min(
            [r for r in filtered if r.get("best_val_loss") is not None],
            key=lambda x: x["best_val_loss"],
            default=None
        )

        return best


def main():
    parser = argparse.ArgumentParser(description="Collect and analyze experiment results")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Collect command
    collect_parser = subparsers.add_parser("collect", help="Collect results from experiments")
    collect_parser.add_argument("--force", action="store_true", help="Recollect all results")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate summary report")
    report_parser.add_argument("--output", type=str, help="Output CSV file path")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare results")
    compare_parser.add_argument("--by", choices=["model_type", "input_days", "target_days"],
                                default="model_type", help="Grouping variable")

    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Generate plots")
    plot_parser.add_argument("--type", choices=["heatmap", "bar", "box"],
                            default="heatmap", help="Plot type")
    plot_parser.add_argument("--output", type=str, help="Output file path")
    plot_parser.add_argument("--show", action="store_true", help="Display plot")

    # Best command
    best_parser = subparsers.add_parser("best", help="Find best configuration")
    best_parser.add_argument("--model-type", type=str, help="Filter by model type")

    args = parser.parse_args()

    # Determine paths
    experiments_root = Path(__file__).parent
    tracking_file = experiments_root / "tracking" / "experiments.csv"
    outputs_dir = experiments_root.parent / "outputs"

    aggregator = ResultAggregator(tracking_file, outputs_dir)

    if args.command == "collect":
        aggregator.collect_results(force=args.force)

    elif args.command == "report":
        output_file = Path(args.output) if args.output else \
                     experiments_root / "tracking" / "results_summary.csv"
        aggregator.generate_summary(output_file)
        aggregator.print_report()

    elif args.command == "compare":
        aggregator.print_report()

    elif args.command == "plot":
        output_file = Path(args.output) if args.output else \
                     experiments_root / "tracking" / f"comparison_{args.type}.png"
        aggregator.plot_comparison(
            plot_type=args.type,
            output_file=output_file,
            show=args.show
        )

    elif args.command == "best":
        best = aggregator.find_best_config(model_type=args.model_type)
        if best:
            print(f"\nBest Configuration:")
            print("=" * 50)
            print(f"  Experiment: {best['exp_name']}")
            print(f"  Model Type: {best.get('model_type', 'N/A')}")
            print(f"  Input Days: {best.get('input_days', 'N/A')}")
            print(f"  Target Days: {best.get('target_days', 'N/A')}")
            print(f"  Subsample: {best.get('subsample_index', 'N/A')}")
            print(f"  Best Loss: {best.get('best_val_loss', 'N/A')}")
            print("=" * 50)
        else:
            print("No results found.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
