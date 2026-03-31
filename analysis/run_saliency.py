"""
Saliency Analysis - All Batches

Extract saliency maps (Grad-CAM, Integrated Gradients) from validation data.
Based on Legacy/example_saliency.py with updated project structure.

Features:
- Load trained model from checkpoint
- Process all validation batches
- Generate Grad-CAM and Integrated Gradients
- Per-channel analysis and visualization
- Save results as NPZ and PNG files

Usage:
    cd /opt/projects/10_Harim/01_AP/02_Regression

    # Method 1: Epoch-based (recommended - auto-generates paths)
    python analysis/run_saliency.py --config-name=local saliency.epoch=10

    # Method 2: Explicit paths (for custom locations)
    python analysis/run_saliency.py --config-name=local \\
        saliency.checkpoint_path=/path/to/model.pth \\
        saliency.output_dir=saliency_outputs
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from src.networks import create_model
from src.pipeline import create_dataloader
from src.utils import resolve_paths
from analysis.saliency_maps import SaliencyExtractor


def load_trained_model(checkpoint_path: str, config: DictConfig, device: str):
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        config: Hydra configuration
        device: Target device (cuda, mps, cpu)

    Returns:
        Loaded model in eval mode
    """
    model = create_model(config)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    return model


def debug_gradients(model, solar_wind_input, image_input, device):
    """Check if gradient computation works correctly.

    Args:
        model: The model to test
        solar_wind_input: OMNI input tensor
        image_input: SDO image tensor
        device: Target device

    Returns:
        True if gradients are computed successfully
    """
    print("\n" + "=" * 60)
    print("DEBUG: Gradient Check")
    print("=" * 60)

    solar_wind_input = solar_wind_input.to(device)
    image_input = image_input.to(device)

    # Enable gradient for images
    image_input.requires_grad = True

    # Forward pass
    model.zero_grad()
    output = model(solar_wind_input, image_input)

    print(f"Output shape: {output.shape}")
    print(f"Output value: {output[0, 0, 0].item():.4f}")

    # Backward pass
    target = output[0, 0, 0]
    target.backward()

    # Check gradients
    if image_input.grad is not None:
        grad_magnitude = image_input.grad.abs().mean().item()
        grad_max = image_input.grad.abs().max().item()
        grad_min = image_input.grad.abs().min().item()

        print(f"\nGradient computed successfully!")
        print(f"  Gradient mean: {grad_magnitude:.6f}")
        print(f"  Gradient max: {grad_max:.6f}")
        print(f"  Gradient min: {grad_min:.6f}")

        if grad_magnitude < 1e-10:
            print("\nWARNING: Gradient is extremely small!")
            print("   Model may not be using image input effectively")
            return False

        return True
    else:
        print("\nERROR: No gradient computed!")
        print("   Check if model is in eval mode or if requires_grad=True")
        return False


def process_batch(
    extractor: SaliencyExtractor,
    solar_wind_input: torch.Tensor,
    image_input: torch.Tensor,
    output_dir: Path,
    channel_names: list,
    target_index: int = 0,
    target_variable: int = 0,
    ig_steps: int = 50,
    create_plots: bool = True
):
    """Process a single batch for saliency analysis.

    Args:
        extractor: SaliencyExtractor instance
        solar_wind_input: OMNI input tensor (batch, seq_len, num_vars)
        image_input: SDO image tensor (batch, channels, seq_len, H, W)
        output_dir: Output directory for this batch
        channel_names: Names of SDO channels
        target_index: Target timestep index
        target_variable: Target variable index
        ig_steps: Number of integration steps for IG
        create_plots: Whether to generate visualizations
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    # Multi-channel analysis
    print("\n" + "=" * 60)
    print("MULTI-CHANNEL ANALYSIS")
    print("=" * 60)

    try:
        extractor.visualize_all_channels_analysis(
            solar_wind_input,
            image_input,
            target_index=target_index,
            target_variable=target_variable,
            channel_names=channel_names,
            output_dir=output_dir
        )
    except Exception as e:
        print(f"Error in multi-channel analysis: {e}")
        import traceback
        traceback.print_exc()

    # Integrated Gradients for channel importance
    print("\n" + "-" * 60)
    print("Computing Integrated Gradients for channel importance...")

    try:
        ig_maps = extractor.integrated_gradients(
            solar_wind_input,
            image_input,
            target_index=target_index,
            target_variable=target_variable,
            n_steps=ig_steps
        )

        # Compute channel importance
        channel_importance = np.abs(ig_maps).sum(axis=(1, 2, 3))
        channel_importance = channel_importance / (channel_importance.max() + 1e-10)

        print("  Channel importance:")
        for ch_idx, name in enumerate(channel_names):
            print(f"    {name}: {channel_importance[ch_idx]:.3f}")

        # Save channel importance
        np.savez(
            output_dir / "channel_importance.npz",
            channel_importance=channel_importance,
            channel_names=channel_names
        )
        print("  Saved: channel_importance.npz")

    except Exception as e:
        print(f"  Error: {e}")


@hydra.main(config_path="../configs", config_name="local", version_base=None)
def main(config: DictConfig):
    """Main execution function."""

    # ========================================
    # Setup
    # ========================================
    device = config.environment.device

    # Resolve paths (epoch-based or explicit)
    checkpoint_path, output_dir = resolve_paths(config, 'saliency')

    # Update config with resolved paths
    OmegaConf.update(config, "saliency.checkpoint_path", checkpoint_path)
    OmegaConf.update(config, "saliency.output_dir", output_dir)

    # MPS warning
    if device == "mps":
        print("WARNING: Using MPS (Apple Silicon)")
        print("   Some gradient operations may not work correctly.")
        print("   Consider using 'cpu' if results are abnormal.\n")

    output_root = Path(output_dir)
    output_root.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("SALIENCY ANALYSIS - ALL BATCHES MODE")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Output directory: {output_root}")
    print(f"Checkpoint: {checkpoint_path}")

    # ========================================
    # Load model
    # ========================================
    model = load_trained_model(checkpoint_path, config, device)

    # ========================================
    # Create dataloader
    # ========================================
    dataloader = create_dataloader(config, phase="validation")
    print(f"DataLoader loaded (total batches: {len(dataloader)})\n")

    # ========================================
    # Debug with first batch
    # ========================================
    batch = next(iter(dataloader))
    solar_wind_input = batch["inputs"][:1]
    image_input = batch["sdo"][:1]

    print(f"Data shapes:")
    print(f"  Solar wind: {solar_wind_input.shape}")
    print(f"  SDO images: {image_input.shape}")

    # Gradient test
    gradient_ok = debug_gradients(model, solar_wind_input, image_input, device)

    if not gradient_ok:
        print("\n" + "=" * 60)
        print("RECOMMENDATION")
        print("=" * 60)
        print("Gradient computation failed or too small.")
        print("\nPossible solutions:")
        print("1. Try device='cpu' instead of 'mps'")
        print("2. Check if model architecture has gradient flow issues")
        print("3. Verify model is actually using SDO images in forward pass")
        print("4. Try a different sample (this one might have low activity)")
        print("\nContinuing anyway, but results may not be meaningful...")

    # ========================================
    # Initialize SaliencyExtractor
    # ========================================
    extractor = SaliencyExtractor(model, device=device)

    # Check model type compatibility
    if not extractor.supports_sdo_saliency():
        print("\n" + "=" * 70)
        print("ERROR: MODEL TYPE NOT SUPPORTED")
        print("=" * 70)
        print(f"Model type '{extractor.model_type}' does not support SDO saliency analysis.")
        print("Saliency analysis requires ConvLSTM layers (model_type: 'convlstm' or 'fusion').")
        print("\nTo use saliency analysis:")
        print("  1. Use a fusion model: model.model_type=fusion")
        print("  2. Use a convlstm model: model.model_type=convlstm")
        print("\nFor transformer-only models, use attention analysis instead:")
        print("  python analysis/run_attention.py --config-name=local attention.epoch=10")
        print("=" * 70)
        return

    print("\nSaliencyExtractor initialized")

    # ========================================
    # Configuration
    # ========================================
    max_batches = getattr(config, 'max_batches', None) or len(dataloader)

    # Get wavelength names (new structure first, legacy fallback)
    if hasattr(config.data, 'sdo') and hasattr(config.data.sdo, 'wavelengths'):
        channel_names = list(config.data.sdo.wavelengths)
    else:
        channel_names = list(config.data.wavelengths)

    # Get saliency settings
    ig_steps = getattr(config.saliency, 'ig_steps', 50) if hasattr(config, 'saliency') else 50
    create_plots = getattr(config.saliency, 'create_plots', True) if hasattr(config, 'saliency') else True

    # ========================================
    # Process batches
    # ========================================
    sample_idx = 0  # Global sample counter

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            print(f"\nProcessed {max_batches} batches. Stopping.")
            break

        batch_size = batch["inputs"].shape[0]
        file_names = batch.get("file_names", [f"sample_{sample_idx + i:04d}" for i in range(batch_size)])

        print("\n" + "=" * 70)
        print(f"Processing batch {batch_idx + 1}/{max_batches} ({batch_size} samples)")
        print("=" * 70)

        # Process each sample in the batch
        for i in range(batch_size):
            # Prepare data (single sample)
            solar_wind_input = batch["inputs"][i:i+1]
            image_input = batch["sdo"][i:i+1]

            # Get file name for this sample
            file_name = file_names[i] if i < len(file_names) else f"sample_{sample_idx:04d}"

            # Output directory for this sample (use file_name like validation)
            sample_output_dir = output_root / file_name

            # Skip if already exists
            if sample_output_dir.exists():
                print(f"  [{i+1}/{batch_size}] Skipping {file_name} (already exists)")
                sample_idx += 1
                continue

            print(f"  [{i+1}/{batch_size}] Processing {file_name}...")

            try:
                # Process sample
                process_batch(
                    extractor=extractor,
                    solar_wind_input=solar_wind_input,
                    image_input=image_input,
                    output_dir=sample_output_dir,
                    channel_names=channel_names,
                    target_index=0,
                    target_variable=0,
                    ig_steps=ig_steps,
                    create_plots=create_plots
                )
            except Exception as e:
                print(f"  Error in sample {file_name}: {e}")
                import traceback
                traceback.print_exc()

            sample_idx += 1

        print(f"\nBatch {batch_idx} complete")

    # ========================================
    # Final summary
    # ========================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nProcessed {sample_idx} samples from {min(max_batches, len(dataloader))} batches")
    print(f"Results saved to: {output_root}")

    print("\nGenerated structure per sample:")
    print("  {file_name}/")
    for ch_idx, ch_name in enumerate(channel_names):
        ch_dir_name = f"channel_{ch_idx}_{ch_name.replace('Å', 'A')}"
        print(f"    ├─ {ch_dir_name}/")
        print(f"    │   ├─ grad_cam_{ch_name}.png")
        print(f"    │   ├─ full_sequence_{ch_name}.png")
        print(f"    │   └─ comprehensive_{ch_name}.png")
    print("    ├─ channel_comparison.png")
    print("    ├─ temporal_importance_all_channels.png")
    print("    └─ channel_importance.npz")


if __name__ == '__main__':
    main()
