"""
Cross-Modal Fusion Analysis

Analyzes how the fusion model balances OMNI (Transformer) and SDO (ConvLSTM) features.

Key metrics:
1. Gate weights: sigmoid output determining OMNI vs SDO balance
   - gate_weights > 0.5: OMNI dominant
   - gate_weights < 0.5: SDO dominant
2. Feature norms: Magnitude of features from each modality
3. Gradient flow: Which modality receives more gradient signal

Usage:
    python analysis/cross_modal_analysis.py --config-name=local \
        experiment.name=fusion_v2 validation.epoch=best
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src.networks import create_model
from src.pipeline import create_dataloader


class CrossModalAnalyzer:
    """Analyze cross-modal fusion behavior."""

    def __init__(self, model: nn.Module, device: str = 'mps'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # Verify model type
        if not hasattr(model, 'cross_modal_fusion'):
            raise ValueError("Model must be fusion type with cross_modal_fusion module")

        # Storage for analysis
        self.gate_weights_history = []
        self.transformer_norms = []
        self.convlstm_norms = []
        self.attention_weights_t2c = []  # transformer to convlstm
        self.attention_weights_c2t = []  # convlstm to transformer

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture intermediate values."""

        def gate_hook(module, input, output):
            """Capture gate weights from feature_gate."""
            # output is the sigmoid output (gate_weights)
            self.gate_weights_history.append(output.detach().cpu())

        def attention_hook_t2c(module, input, output):
            """Capture attention from transformer_to_convlstm."""
            # We need to modify this to get attention weights
            pass

        # Register hook on feature_gate
        self.model.cross_modal_fusion.feature_gate.register_forward_hook(gate_hook)

    def analyze_batch(
        self,
        solar_wind_input: torch.Tensor,
        image_input: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Analyze a single batch."""

        solar_wind_input = solar_wind_input.to(self.device)
        image_input = image_input.to(self.device)

        with torch.no_grad():
            # Get features from each modality
            transformer_features = self.model.transformer_model(solar_wind_input)
            convlstm_features = self.model.convlstm_model(image_input)

            # Store feature norms
            self.transformer_norms.append(transformer_features.norm(dim=-1).cpu())
            self.convlstm_norms.append(convlstm_features.norm(dim=-1).cpu())

            # Forward through fusion (gate hook will capture weights)
            fused_features = self.model.cross_modal_fusion(
                transformer_features, convlstm_features
            )

            # Get predictions
            predictions = self.model.regression_head(fused_features)

        return {
            'transformer_features': transformer_features.cpu(),
            'convlstm_features': convlstm_features.cpu(),
            'fused_features': fused_features.cpu(),
            'predictions': predictions.cpu()
        }

    def run_analysis(self, dataloader, max_batches: int = None) -> Dict[str, np.ndarray]:
        """Run analysis on entire dataset."""

        self.gate_weights_history = []
        self.transformer_norms = []
        self.convlstm_norms = []

        n_batches = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))

        for batch_idx, batch in enumerate(tqdm(dataloader, total=n_batches, desc="Analyzing")):
            if max_batches and batch_idx >= max_batches:
                break

            solar_wind = batch['inputs']
            images = batch['sdo']

            self.analyze_batch(solar_wind, images)

        # Aggregate results
        gate_weights = torch.cat(self.gate_weights_history, dim=0).numpy()
        transformer_norms = torch.cat(self.transformer_norms, dim=0).numpy()
        convlstm_norms = torch.cat(self.convlstm_norms, dim=0).numpy()

        return {
            'gate_weights': gate_weights,
            'transformer_norms': transformer_norms,
            'convlstm_norms': convlstm_norms
        }

    def visualize_gate_distribution(
        self,
        gate_weights: np.ndarray,
        save_path: Optional[Path] = None
    ):
        """Visualize distribution of gate weights."""

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram
        ax = axes[0]
        mean_gate = gate_weights.mean(axis=1)  # Average over feature dimension
        ax.hist(mean_gate, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Balance point')
        ax.axvline(x=mean_gate.mean(), color='blue', linestyle='-', linewidth=2,
                   label=f'Mean: {mean_gate.mean():.3f}')
        ax.set_xlabel('Gate Weight (0=SDO, 1=OMNI)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Gate Weight Distribution', fontsize=14, fontweight='bold')
        ax.legend()

        # Per-dimension analysis
        ax = axes[1]
        dim_means = gate_weights.mean(axis=0)  # Average over samples
        ax.bar(range(len(dim_means)), dim_means, alpha=0.7)
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Feature Dimension', fontsize=12)
        ax.set_ylabel('Mean Gate Weight', fontsize=12)
        ax.set_title('Gate Weight by Dimension', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
            plt.close()
        else:
            plt.show()

    def visualize_feature_norms(
        self,
        transformer_norms: np.ndarray,
        convlstm_norms: np.ndarray,
        save_path: Optional[Path] = None
    ):
        """Compare feature magnitudes from each modality."""

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Distribution comparison
        ax = axes[0]
        ax.hist(transformer_norms.flatten(), bins=50, alpha=0.6, label='OMNI (Transformer)')
        ax.hist(convlstm_norms.flatten(), bins=50, alpha=0.6, label='SDO (ConvLSTM)')
        ax.set_xlabel('Feature Norm', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Feature Magnitude Distribution', fontsize=14, fontweight='bold')
        ax.legend()

        # Ratio analysis
        ax = axes[1]
        ratio = transformer_norms.flatten() / (convlstm_norms.flatten() + 1e-8)
        ax.hist(np.clip(ratio, 0, 5), bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Equal')
        ax.axvline(x=ratio.mean(), color='blue', linestyle='-', linewidth=2,
                   label=f'Mean: {ratio.mean():.3f}')
        ax.set_xlabel('OMNI/SDO Feature Norm Ratio', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Feature Ratio Distribution', fontsize=14, fontweight='bold')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
            plt.close()
        else:
            plt.show()

    def print_summary(self, results: Dict[str, np.ndarray]):
        """Print analysis summary."""

        gate_weights = results['gate_weights']
        transformer_norms = results['transformer_norms']
        convlstm_norms = results['convlstm_norms']

        mean_gate = gate_weights.mean()
        omni_dominant_ratio = (gate_weights.mean(axis=1) > 0.5).mean()

        print("\n" + "="*60)
        print("CROSS-MODAL FUSION ANALYSIS SUMMARY")
        print("="*60)

        print("\n[Gate Weights Analysis]")
        print(f"  Mean Gate Weight: {mean_gate:.4f}")
        print(f"    → {mean_gate:.1%} OMNI, {1-mean_gate:.1%} SDO")
        print(f"  OMNI Dominant Samples: {omni_dominant_ratio:.1%}")
        print(f"  SDO Dominant Samples: {1-omni_dominant_ratio:.1%}")

        print("\n[Feature Magnitude Analysis]")
        print(f"  OMNI Feature Norm: {transformer_norms.mean():.4f} (±{transformer_norms.std():.4f})")
        print(f"  SDO Feature Norm:  {convlstm_norms.mean():.4f} (±{convlstm_norms.std():.4f})")
        ratio = transformer_norms.mean() / (convlstm_norms.mean() + 1e-8)
        print(f"  OMNI/SDO Ratio: {ratio:.4f}")

        print("\n[Interpretation]")
        if mean_gate > 0.6:
            print("  ⚠️  Model heavily relies on OMNI data (Transformer)")
            print("  → SDO images may not be contributing effectively")
        elif mean_gate < 0.4:
            print("  ✓ Model relies more on SDO images (ConvLSTM)")
            print("  → SDO data is being utilized")
        else:
            print("  ✓ Balanced fusion between OMNI and SDO")

        print("="*60)


def load_model_and_data(cfg: DictConfig) -> Tuple[nn.Module, torch.utils.data.DataLoader]:
    """Load model and validation data."""

    # Create model
    model = create_model(cfg)

    # Load checkpoint
    result_dir = Path(cfg.environment.save_root) / cfg.experiment.name
    if cfg.validation.epoch == 'best':
        checkpoint_path = result_dir / 'checkpoint' / 'model_best.pth'
    else:
        checkpoint_path = result_dir / 'checkpoint' / f'model_epoch_{cfg.validation.epoch:04d}.pth'

    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=cfg.environment.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded from epoch {checkpoint['epoch']}")

    # Create dataloader
    val_loader = create_dataloader(cfg, phase="validation")

    return model, val_loader


@hydra.main(version_base=None, config_path="../configs", config_name="local")
def main(cfg: DictConfig):
    """Main entry point."""

    print("="*60)
    print("CROSS-MODAL FUSION ANALYSIS")
    print("="*60)
    print(f"Experiment: {cfg.experiment.name}")
    print(f"Epoch: {cfg.validation.epoch}")

    # Setup
    device = cfg.environment.device
    result_dir = Path(cfg.environment.save_root) / cfg.experiment.name / 'analysis'
    result_dir.mkdir(parents=True, exist_ok=True)

    # Load model and data
    model, val_loader = load_model_and_data(cfg)

    # Create analyzer
    analyzer = CrossModalAnalyzer(model, device=device)

    # Run analysis
    print("\nRunning cross-modal analysis...")
    results = analyzer.run_analysis(val_loader, max_batches=None)

    # Print summary
    analyzer.print_summary(results)

    # Visualizations
    print("\nGenerating visualizations...")
    analyzer.visualize_gate_distribution(
        results['gate_weights'],
        save_path=result_dir / 'gate_distribution.png'
    )
    analyzer.visualize_feature_norms(
        results['transformer_norms'],
        results['convlstm_norms'],
        save_path=result_dir / 'feature_norms.png'
    )

    # Save raw results
    np.savez(
        result_dir / 'cross_modal_results.npz',
        gate_weights=results['gate_weights'],
        transformer_norms=results['transformer_norms'],
        convlstm_norms=results['convlstm_norms']
    )
    print(f"\nResults saved to: {result_dir}")


if __name__ == '__main__':
    main()
