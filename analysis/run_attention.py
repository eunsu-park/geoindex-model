"""
Attention Analysis - All Batches

Extract and analyze attention weights from all validation batches.
Based on Legacy/example_attention_all_targets.py with updated project structure.

Features:
- Load trained model from checkpoint
- Extract attention weights from all Transformer layers
- Compute temporal importance per layer
- Save results as NPZ files
- Optional visualization

NPZ structure per batch:
- solar_wind_data: (seq_len, num_vars) - Original OMNI data
- sdo_data: (channels, seq_len, H, W) - Original SDO images
- attention_weights: (num_layers, num_heads, seq_len, seq_len) - All layer attention
- temporal_importance: (num_layers, seq_len) - Per-layer temporal importance
- predictions: (n_targets, num_vars) - Model predictions
- targets: (n_targets, num_vars) - Ground truth
- metadata: dict - Configuration info

Note: This is MUCH faster than IG (~50-100x) - forward pass only, no gradients.

Usage:
    cd /opt/projects/10_Harim/01_AP/02_Regression

    # Method 1: Epoch-based (recommended - auto-generates paths)
    python analysis/run_attention.py --config-name=local attention.epoch=10

    # Method 2: Explicit paths (for custom locations)
    python analysis/run_attention.py --config-name=local \\
        attention.checkpoint_path=/path/to/model.pth \\
        attention.output_dir=attention_outputs
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional

import torch
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from src.networks import create_model
from src.pipeline import create_dataloader
from src.utils import resolve_paths
from analysis.attention_analysis import AttentionExtractor


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


def generate_attention_analysis(
    extractor: AttentionExtractor,
    solar_wind_input: torch.Tensor,
    image_input: torch.Tensor,
    targets: torch.Tensor,
    output_path: Path,
    config: DictConfig,
    create_plots: bool = False,
    plot_dir: Optional[Path] = None
):
    """Generate attention analysis for a single batch.

    Args:
        extractor: AttentionExtractor instance
        solar_wind_input: (batch, seq_len, num_vars)
        image_input: (batch, channels, seq_len, H, W)
        targets: (batch, target_seq_len, num_vars)
        output_path: Path for output NPZ file
        config: Hydra config
        create_plots: Whether to generate visualizations
        plot_dir: Directory for saving plots (required if create_plots=True)

    Returns:
        dict with analysis results
    """
    print("\n" + "=" * 70)
    print("ATTENTION ANALYSIS - SINGLE BATCH")
    print("=" * 70)

    # Shape info
    batch, seq_len, num_vars = solar_wind_input.shape
    _, channels, sdo_seq_len, H, W = image_input.shape
    _, n_targets, n_target_vars = targets.shape

    print(f"\nInput shapes:")
    print(f"  Solar wind: {solar_wind_input.shape}")
    print(f"  SDO images: {image_input.shape}")
    print(f"  Targets: {targets.shape}")

    # ================================================================
    # 1. Extract attention weights
    # ================================================================
    print(f"\n[1/3] Extracting attention weights...")

    attention_weights, predictions = extractor.extract_attention_manual_forward(
        solar_wind_input, image_input
    )

    num_layers = len(attention_weights)
    num_heads = attention_weights[0].shape[1]

    print(f"Extracted attention from {num_layers} layers")
    print(f"  Each layer: (batch={batch}, heads={num_heads}, seq_len={seq_len}, seq_len={seq_len})")

    # ================================================================
    # 2. Compute temporal importance
    # ================================================================
    print(f"\n[2/3] Computing temporal importance...")

    temporal_importance_all = []

    for layer_idx, attn in enumerate(attention_weights):
        temporal_imp = extractor.compute_temporal_importance(
            attn[0],  # First batch
            method='incoming'
        )
        temporal_importance_all.append(temporal_imp)

        print(f"  Layer {layer_idx}: Most important timestep = {temporal_imp.argmax()} "
              f"(score={temporal_imp.max():.4f})")

    temporal_importance_all = np.array(temporal_importance_all)  # (num_layers, seq_len)

    # ================================================================
    # 3. Save NPZ
    # ================================================================
    print(f"\n[3/3] Saving to NPZ...")

    # Convert attention weights to numpy
    attention_weights_np = [attn[0].cpu().numpy() for attn in attention_weights]
    attention_weights_np = np.array(attention_weights_np)  # (num_layers, num_heads, seq_len, seq_len)

    # Metadata
    metadata = {
        'num_layers': num_layers,
        'num_heads': num_heads,
        'seq_len': seq_len,
        'num_vars': num_vars,
        'n_targets': n_targets,
        'sdo_channels': channels,
        'sdo_seq_len': sdo_seq_len,
        'image_size': H,
        'd_model': config.model.d_model,
        'checkpoint': config.validation.checkpoint_path
    }

    # Save
    np.savez_compressed(
        output_path,
        solar_wind_data=solar_wind_input[0].cpu().numpy(),         # (seq_len, num_vars)
        sdo_data=image_input[0].cpu().numpy(),                     # (channels, sdo_seq_len, H, W)
        attention_weights=attention_weights_np,                     # (num_layers, num_heads, seq_len, seq_len)
        temporal_importance=temporal_importance_all,                # (num_layers, seq_len)
        predictions=predictions[0].cpu().numpy(),                   # (n_targets, n_target_vars)
        targets=targets[0].cpu().numpy(),                           # (n_targets, n_target_vars)
        metadata=metadata
    )

    # File size
    file_size_mb = output_path.stat().st_size / (1024**2)

    print(f"Saved to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")

    # ================================================================
    # 4. Visualization (Optional)
    # ================================================================
    if create_plots and plot_dir is not None:
        print(f"\n[4/4] Creating visualizations...")

        batch_name = output_path.stem

        # Attention heatmap (last layer)
        extractor.visualize_attention_heatmap(
            attention_weights[-1][0],  # Last layer, first batch
            save_path=plot_dir / f"{batch_name}_heatmap.png",
            title=f"Attention Heatmap - {batch_name}"
        )

        # Attention matrix (last layer, all heads)
        extractor.visualize_attention_matrix(
            attention_weights[-1][0],  # Last layer, first batch
            save_path=plot_dir / f"{batch_name}_matrix.png",
            title=f"Attention Matrix - {batch_name}"
        )

        # Layer comparison
        all_layers_batch0 = [attn[0] for attn in attention_weights]
        extractor.visualize_layer_comparison(
            all_layers_batch0,
            save_path=plot_dir / f"{batch_name}_layers.png"
        )

        print(f"Plots saved to: {plot_dir}")

    # ================================================================
    # 5. Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nAttention analysis complete!")
    print(f"Data saved to: {output_path.absolute()}")

    print(f"\nNPZ Contents:")
    print(f"  'solar_wind_data':       ({seq_len}, {num_vars})")
    print(f"  'sdo_data':              ({channels}, {sdo_seq_len}, {H}, {W})")
    print(f"  'attention_weights':     ({num_layers}, {num_heads}, {seq_len}, {seq_len})")
    print(f"  'temporal_importance':   ({num_layers}, {seq_len})")
    print(f"  'predictions':           ({n_targets}, {n_target_vars})")
    print(f"  'targets':               ({n_targets}, {n_target_vars})")
    print(f"  'metadata':              dict")

    # Temporal importance summary
    print(f"\nTemporal Importance Summary:")
    print(f"  {'Layer':<8} {'Peak Frame':<12} {'Peak Score':<12} {'Mean Score':<12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    for layer_idx in range(num_layers):
        peak_frame = temporal_importance_all[layer_idx].argmax()
        peak_score = temporal_importance_all[layer_idx].max()
        mean_score = temporal_importance_all[layer_idx].mean()
        print(f"  {layer_idx:<8} {peak_frame:<12} {peak_score:<12.4f} {mean_score:<12.4f}")

    return {
        'output_path': output_path,
        'file_size_mb': file_size_mb,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'temporal_importance': temporal_importance_all,
        'predictions': predictions[0].cpu().numpy(),
        'targets': targets[0].cpu().numpy()
    }


@hydra.main(config_path="../configs", config_name="local", version_base=None)
def main(config: DictConfig):
    """Main execution function."""

    # ========================================
    # Check model type compatibility
    # ========================================
    model_type = config.model.model_type
    if model_type in ["convlstm", "baseline"]:
        print("\n" + "=" * 70)
        print("ATTENTION ANALYSIS - SKIPPED")
        print("=" * 70)
        print(f"\nModel type '{model_type}' does not have Transformer layers.")
        print("Attention analysis requires Transformer-based models.")
        print("\nSupported model types for attention analysis:")
        print("  - 'transformer': OMNI-only Transformer model")
        print("  - 'fusion': Multi-modal fusion model with Transformer")
        print("\nAlternative analyses for this model type:")
        print("  - Saliency analysis (Grad-CAM, Integrated Gradients):")
        print("      python analysis/run_saliency.py --config-name=local saliency.epoch=10")
        print("  - Monte Carlo Dropout (uncertainty estimation):")
        print("      python analysis/monte_carlo_dropout.py --config-name=local mcd.epoch=10")
        print("=" * 70 + "\n")
        return

    # ========================================
    # Setup
    # ========================================
    device = config.environment.device

    # Resolve paths (epoch-based or explicit)
    checkpoint_path, output_dir_str = resolve_paths(config, 'attention')

    # Update config with resolved paths
    OmegaConf.update(config, "attention.checkpoint_path", checkpoint_path)
    OmegaConf.update(config, "attention.output_dir", output_dir_str)

    output_dir = Path(output_dir_str)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get create_plots setting (need this before creating directories)
    create_plots = getattr(config.attention, 'create_plots', False) if hasattr(config, 'attention') else False

    # Create subdirectories for npz and plots
    npz_dir = output_dir / "npz"
    plot_dir = output_dir / "plots"
    npz_dir.mkdir(exist_ok=True, parents=True)
    if create_plots:
        plot_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("ATTENTION ANALYSIS - ALL BATCHES MODE")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print()
    print("NOTE: This is MUCH faster than IG (~50-100x)")
    print("   Forward pass only, no gradients needed!")
    print()

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
    # Initialize AttentionExtractor
    # ========================================
    try:
        extractor = AttentionExtractor(model, device=device)
    except ValueError as e:
        print("\n" + "=" * 70)
        print("ERROR: MODEL TYPE NOT SUPPORTED")
        print("=" * 70)
        print(str(e))
        print("\nTo use attention analysis:")
        print("  1. Use a fusion model: model.model_type=fusion")
        print("  2. Use a transformer model: model.model_type=transformer")
        print("\nFor convlstm-only models, use saliency analysis instead:")
        print("  python analysis/run_saliency.py --config-name=local saliency.epoch=10")
        print("=" * 70)
        return

    print("AttentionExtractor initialized\n")

    # ========================================
    # Configuration
    # ========================================
    max_batches = getattr(config, 'max_batches', None) or len(dataloader)

    # ========================================
    # Process batches
    # ========================================
    results = []
    sample_idx = 0  # Global sample counter

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            print(f"\nProcessed {max_batches} batches. Stopping.")
            break

        batch_size = batch["inputs"].shape[0]
        file_names = batch.get("file_names", [f"sample_{sample_idx + i:04d}" for i in range(batch_size)])

        print("\n" + "=" * 70)
        print(f"PROCESSING BATCH {batch_idx + 1}/{max_batches} ({batch_size} samples)")
        print("=" * 70)

        # Process each sample in the batch
        for i in range(batch_size):
            # Prepare data (single sample)
            solar_wind_input = batch["inputs"][i:i+1]
            image_input = batch["sdo"][i:i+1]
            targets = batch["targets"][i:i+1]

            # Get file name for this sample
            file_name = file_names[i] if i < len(file_names) else f"sample_{sample_idx:04d}"

            # Output file path (use file_name like validation)
            output_path = npz_dir / f"{file_name}.npz"

            # Skip if already exists
            if output_path.exists():
                print(f"  [{i+1}/{batch_size}] Skipping {file_name} (already exists)")
                sample_idx += 1
                continue

            print(f"  [{i+1}/{batch_size}] Processing {file_name}...")

            try:
                result = generate_attention_analysis(
                    extractor=extractor,
                    solar_wind_input=solar_wind_input,
                    image_input=image_input,
                    targets=targets,
                    output_path=output_path,
                    config=config,
                    create_plots=create_plots,
                    plot_dir=plot_dir if create_plots else None
                )

                results.append(result)

            except Exception as e:
                print(f"Error in sample {file_name}: {e}")
                import traceback
                traceback.print_exc()

            sample_idx += 1

    # ========================================
    # Final summary
    # ========================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\nProcessed {len(results)} samples successfully")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"  NPZ files: {npz_dir.absolute()}")
    if create_plots:
        print(f"  Plots:     {plot_dir.absolute()}")

    if results:
        total_size = sum(r['file_size_mb'] for r in results)
        num_layers = results[0]['num_layers']
        num_heads = results[0]['num_heads']

        print(f"\nGenerated NPZ files:")
        for i, result in enumerate(results[:5]):  # Show first 5 only
            print(f"  {i+1}. {result['output_path'].name} ({result['file_size_mb']:.2f} MB)")
        if len(results) > 5:
            print(f"  ... and {len(results) - 5} more files")
        print(f"\nTotal size: {total_size:.2f} MB")

        print(f"\nModel architecture:")
        print(f"  Transformer layers: {num_layers}")
        print(f"  Attention heads: {num_heads}")

        # Average temporal importance
        avg_temporal_imp = np.mean([r['temporal_importance'] for r in results], axis=0)
        print(f"\nAverage Temporal Importance (across all batches):")
        for layer_idx in range(num_layers):
            peak_frame = avg_temporal_imp[layer_idx].argmax()
            print(f"  Layer {layer_idx}: Peak at timestep {peak_frame} "
                  f"(score={avg_temporal_imp[layer_idx, peak_frame]:.4f})")

    print("\n" + "=" * 70)
    print("USAGE EXAMPLE")
    print("=" * 70)
    print("""
# Load the data
import numpy as np

data = np.load('npz/{file_name}.npz')

# Access arrays
solar_wind = data['solar_wind_data']            # (seq_len, num_vars)
sdo_images = data['sdo_data']                   # (channels, sdo_seq_len, H, W)
attention = data['attention_weights']           # (num_layers, num_heads, seq_len, seq_len)
temporal_imp = data['temporal_importance']      # (num_layers, seq_len)
predictions = data['predictions']               # (n_targets, n_target_vars)
targets = data['targets']                       # (n_targets, n_target_vars)

# Example 1: Get attention from last layer
last_layer_attention = attention[-1]  # (num_heads, seq_len, seq_len)

# Example 2: Average attention over all heads
avg_attention = attention[-1].mean(axis=0)  # (seq_len, seq_len)

# Example 3: Find most important timestep
most_important = temporal_imp[-1].argmax()
print(f"Most important timestep: {most_important}")

# Example 4: Compare prediction vs target
mse = ((predictions - targets) ** 2).mean()
print(f"MSE: {mse:.4f}")
    """)

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Load and analyze NPZ files:
   data = np.load('attention_batch_0000.npz')
   attention = data['attention_weights']  # (num_layers, num_heads, seq_len, seq_len)

2. Compare with IG results:
   - IG shows gradient-based importance
   - Attention shows what model directly attends to
   - Both should align for well-trained models

3. Analyze attention patterns:
   - Which timesteps get most attention?
   - How does attention change across layers?
   - Is attention focused or uniform?

4. Compare with paper findings (DeepHalo):
   - Positive predictions: Uniform attention (progressive process)
   - Negative predictions: Early-focused attention
    """)


if __name__ == '__main__':
    main()
