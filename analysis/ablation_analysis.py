"""
Ablation Analysis for Baseline Model

Measures the contribution of each modality (SDO vs OMNI) by zeroing out
encoder outputs and comparing performance.

Methods:
1. Full model: Both SDO + OMNI (baseline)
2. OMNI only: SDO features zeroed out
3. SDO only: OMNI features zeroed out

This reveals which modality contributes more to predictions.

Usage:
    python analysis/ablation_analysis.py --config-name=local \
        experiment.name=baseline_v2 validation.epoch=best
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src.networks import create_model
from src.pipeline import create_dataloader


class AblationAnalyzer:
    """Ablation analysis for multi-modal models."""

    def __init__(self, model: nn.Module, device: str = 'mps'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # Verify model type
        if not hasattr(model, 'image_encoder') or not hasattr(model, 'ts_encoder'):
            raise ValueError(
                "Model must have image_encoder and ts_encoder. "
                "This analyzer is designed for BaselineModel."
            )

    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute regression metrics."""
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()

        # Flatten for metric computation
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()

        # MAE
        mae = np.abs(pred_flat - target_flat).mean()

        # RMSE
        rmse = np.sqrt(((pred_flat - target_flat) ** 2).mean())

        # R²
        ss_res = ((target_flat - pred_flat) ** 2).sum()
        ss_tot = ((target_flat - target_flat.mean()) ** 2).sum()
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        return {'mae': mae, 'rmse': rmse, 'r2': r2}

    def run_ablation(
        self,
        dataloader,
        ablation_mode: str = 'full'
    ) -> Dict[str, float]:
        """Run model with specified ablation mode.

        Args:
            dataloader: Validation dataloader
            ablation_mode: One of:
                - 'full': Both modalities (no ablation)
                - 'omni_only': Zero out SDO features
                - 'sdo_only': Zero out OMNI features

        Returns:
            Dictionary of metrics
        """
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Ablation ({ablation_mode})"):
                solar_wind = batch['inputs'].to(self.device)
                images = batch['sdo'].to(self.device)
                targets = batch['targets']

                # Get encoder outputs
                ts_features = self.model.ts_encoder(solar_wind)
                img_features = self.model.image_encoder(images)

                # Apply ablation
                if ablation_mode == 'full':
                    pass  # Use both
                elif ablation_mode == 'omni_only':
                    img_features = torch.zeros_like(img_features)
                elif ablation_mode == 'sdo_only':
                    ts_features = torch.zeros_like(ts_features)
                else:
                    raise ValueError(f"Unknown ablation_mode: {ablation_mode}")

                # Forward through fusion head
                combined = torch.cat([ts_features, img_features], dim=1)
                predictions = self.model.fusion_head(combined)

                # Reshape
                output = predictions.reshape(
                    predictions.size(0),
                    self.model.target_sequence_length,
                    self.model.num_target_variables
                )

                all_predictions.append(output.cpu())
                all_targets.append(targets)

        # Aggregate
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute metrics
        metrics = self._compute_metrics(all_predictions, all_targets)
        return metrics

    def run_full_ablation(self, dataloader) -> Dict[str, Dict[str, float]]:
        """Run all ablation modes and return comparison."""
        results = {}

        for mode in ['full', 'omni_only', 'sdo_only']:
            results[mode] = self.run_ablation(dataloader, ablation_mode=mode)

        return results

    def print_summary(self, results: Dict[str, Dict[str, float]]):
        """Print ablation analysis summary."""
        print("\n" + "="*70)
        print("ABLATION ANALYSIS SUMMARY")
        print("="*70)

        print("\n[Performance by Ablation Mode]")
        print("-" * 70)
        print(f"{'Mode':<15} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
        print("-" * 70)

        for mode, metrics in results.items():
            mode_name = {
                'full': 'Full (SDO+OMNI)',
                'omni_only': 'OMNI only',
                'sdo_only': 'SDO only'
            }.get(mode, mode)
            print(f"{mode_name:<15} {metrics['mae']:>10.4f} {metrics['rmse']:>10.4f} {metrics['r2']:>+10.4f}")

        print("-" * 70)

        # Compute contribution
        full_mae = results['full']['mae']
        omni_mae = results['omni_only']['mae']
        sdo_mae = results['sdo_only']['mae']

        print("\n[Modality Contribution Analysis]")

        # Performance drop when removing each modality
        omni_drop = (sdo_mae - full_mae) / full_mae * 100  # Drop when removing OMNI
        sdo_drop = (omni_mae - full_mae) / full_mae * 100  # Drop when removing SDO

        print(f"  Removing OMNI (SDO only):  MAE increases by {omni_drop:+.1f}%")
        print(f"  Removing SDO (OMNI only):  MAE increases by {sdo_drop:+.1f}%")

        print("\n[Interpretation]")
        if omni_drop > sdo_drop * 2:
            print("  ⚠️  OMNI 기여도가 SDO보다 훨씬 높음")
            print(f"     OMNI 제거 시 MAE 증가: {omni_drop:+.1f}%")
            print(f"     SDO 제거 시 MAE 증가: {sdo_drop:+.1f}%")
        elif sdo_drop > omni_drop * 2:
            print("  ✓ SDO 기여도가 OMNI보다 높음")
            print(f"     SDO 제거 시 MAE 증가: {sdo_drop:+.1f}%")
            print(f"     OMNI 제거 시 MAE 증가: {omni_drop:+.1f}%")
        else:
            print("  ✓ SDO와 OMNI 기여도가 유사함")
            print(f"     OMNI 제거 시: {omni_drop:+.1f}%")
            print(f"     SDO 제거 시: {sdo_drop:+.1f}%")

        # Single modality comparison
        print("\n[Single Modality Comparison]")
        if sdo_mae < omni_mae:
            print(f"  SDO only 성능이 더 좋음 (MAE: {sdo_mae:.4f} vs {omni_mae:.4f})")
        else:
            print(f"  OMNI only 성능이 더 좋음 (MAE: {omni_mae:.4f} vs {sdo_mae:.4f})")

        print("="*70)

        return {
            'omni_contribution': omni_drop,
            'sdo_contribution': sdo_drop
        }


def load_model_and_data(cfg: DictConfig) -> Tuple[nn.Module, torch.utils.data.DataLoader]:
    """Load model and validation data."""
    model = create_model(cfg)

    result_dir = Path(cfg.environment.save_root) / cfg.experiment.name
    if cfg.validation.epoch == 'best':
        checkpoint_path = result_dir / 'checkpoint' / 'model_best.pth'
    else:
        checkpoint_path = result_dir / 'checkpoint' / f'model_epoch_{cfg.validation.epoch:04d}.pth'

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=cfg.environment.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded from epoch {checkpoint['epoch']}")

    val_loader = create_dataloader(cfg, phase="validation")
    return model, val_loader


@hydra.main(version_base=None, config_path="../configs", config_name="local")
def main(cfg: DictConfig):
    """Main entry point."""
    print("="*70)
    print("ABLATION ANALYSIS")
    print("="*70)
    print(f"Experiment: {cfg.experiment.name}")
    print(f"Epoch: {cfg.validation.epoch}")
    print(f"Model type: {cfg.model.model_type}")

    if cfg.model.model_type != 'baseline':
        print("\n⚠️  Warning: This script is designed for baseline model.")
        print("   For fusion model, use cross_modal_analysis.py instead.")

    device = cfg.environment.device
    result_dir = Path(cfg.environment.save_root) / cfg.experiment.name / 'analysis'
    result_dir.mkdir(parents=True, exist_ok=True)

    model, val_loader = load_model_and_data(cfg)
    analyzer = AblationAnalyzer(model, device=device)

    print("\nRunning ablation analysis...")
    results = analyzer.run_full_ablation(val_loader)

    contribution = analyzer.print_summary(results)

    # Save results
    np.savez(
        result_dir / 'ablation_results.npz',
        full_mae=results['full']['mae'],
        full_rmse=results['full']['rmse'],
        full_r2=results['full']['r2'],
        omni_only_mae=results['omni_only']['mae'],
        omni_only_rmse=results['omni_only']['rmse'],
        omni_only_r2=results['omni_only']['r2'],
        sdo_only_mae=results['sdo_only']['mae'],
        sdo_only_rmse=results['sdo_only']['rmse'],
        sdo_only_r2=results['sdo_only']['r2'],
        omni_contribution=contribution['omni_contribution'],
        sdo_contribution=contribution['sdo_contribution']
    )

    # Save text report
    report_path = result_dir / 'ablation_report.txt'
    with open(report_path, 'w') as f:
        f.write("ABLATION ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Experiment: {cfg.experiment.name}\n")
        f.write(f"Epoch: {cfg.validation.epoch}\n\n")

        f.write("Performance by Mode:\n")
        f.write("-" * 50 + "\n")
        for mode, metrics in results.items():
            f.write(f"{mode}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:+.4f}\n")

        f.write("\nContribution:\n")
        f.write("-" * 50 + "\n")
        f.write(f"OMNI contribution: {contribution['omni_contribution']:+.1f}% (MAE increase when removed)\n")
        f.write(f"SDO contribution: {contribution['sdo_contribution']:+.1f}% (MAE increase when removed)\n")

    print(f"\nResults saved to: {result_dir}")


if __name__ == '__main__':
    main()
