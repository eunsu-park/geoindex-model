#!/usr/bin/env python
"""Validation script for solar wind prediction model.

Usage:
    cd /opt/projects/10_Harim/01_AP/02_Regression

    # Method 1: Epoch-based (recommended - auto-generates paths)
    python scripts/validate.py --config-name=local validation.epoch=10

    # Method 2: Explicit paths (for custom locations)
    python scripts/validate.py --config-name=local \\
        validation.checkpoint_path=/path/to/checkpoint.pth \\
        validation.output_dir=/path/to/output

    # Override model type
    python scripts/validate.py --config-name=local validation.epoch=10 model.model_type=transformer
"""

import os
import sys

import torch
import torch.nn as nn
import hydra
from omegaconf import OmegaConf

# Add parent directory to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import setup_experiment, resolve_paths
from src.pipeline import create_dataloader
from src.networks import create_model
from src.validators import Validator


@hydra.main(config_path="../configs", version_base=None)
def main(config):
    """Main validation function.

    Args:
        config: Hydra configuration object.
    """
    # Setup experiment (seed, device)
    device = setup_experiment(config)

    # Resolve paths (epoch-based or explicit)
    checkpoint_path, output_dir = resolve_paths(config, 'validation')

    # Update config with resolved paths (for Validator to use)
    OmegaConf.update(config, "validation.checkpoint_path", checkpoint_path)
    OmegaConf.update(config, "validation.output_dir", output_dir)

    os.makedirs(output_dir, exist_ok=True)

    logger = None

    # Print configuration summary
    model_type = config.model.model_type
    print(f"Model type: {model_type}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")

    # Create validation dataloader
    validation_dataloader = create_dataloader(config, 'validation')
    print(
        f"Validation dataloader: {len(validation_dataloader.dataset)} samples, "
        f"{len(validation_dataloader)} batches"
    )

    # Create model
    model = create_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Checkpoint loaded successfully")

    # Create loss criterion
    criterion = nn.MSELoss()

    # Create validator
    validator = Validator(
        config=config,
        model=model,
        criterion=criterion,
        device=device,
        logger=logger
    )

    # Run validation
    results = validator.validate(validation_dataloader)

    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETED")
    print("=" * 80)
    print(f"Model Type: {model_type}")
    print(f"Average Loss: {results['overall']['average_loss']:.6f}")
    print(f"Average MAE:  {results['overall']['average_mae']:.4f}")
    print(f"Average RMSE: {results['overall']['average_rmse']:.4f}")
    print(f"Average R2:   {results['overall']['average_r2']:.4f}")

    if results['overall']['average_cosine_sim'] is not None:
        print(f"Average Cosine Similarity: {results['overall']['average_cosine_sim']:.4f}")

    print(f"\nSuccess Rate: {results['success_rate']:.1f}%")
    print(f"Results saved to: {results['output_directory']}")
    print("=" * 80 + "\n")

    print("Validation completed successfully")

    return results


if __name__ == '__main__':
    main()
