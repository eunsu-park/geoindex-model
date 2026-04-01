#!/usr/bin/env python
"""Inference script for solar wind prediction model (without ground truth).

Usage:
    cd /opt/projects/10_Harim/01_AP/02_Regression

    # Method 1: Epoch-based (recommended - auto-generates paths)
    python scripts/test.py --config-name=local test.epoch=10

    # Method 2: Explicit paths (for custom locations)
    python scripts/test.py --config-name=local \\
        test.checkpoint_path=/path/to/checkpoint.pth \\
        test.output_dir=/path/to/output

    # Override model type
    python scripts/test.py --config-name=local test.epoch=10 model.model_type=transformer
"""

import os
import sys
import logging

import torch
import hydra
from omegaconf import OmegaConf

# Add parent directory to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import setup_experiment, resolve_paths
from src.pipeline import create_dataloader
from src.networks import create_model
from src.testers import Tester


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@hydra.main(config_path="../configs", version_base=None)
def main(config):
    """Main inference function.

    Args:
        config: Hydra configuration object.
    """
    # Setup experiment (seed, device)
    device = setup_experiment(config)

    # Resolve paths (epoch-based or explicit)
    checkpoint_path, output_dir = resolve_paths(config, 'test')

    # Update config with resolved paths (for Tester to use)
    OmegaConf.update(config, "test.checkpoint_path", checkpoint_path)
    OmegaConf.update(config, "test.output_dir", output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Print configuration summary
    model_type = config.model.model_type
    logging.info(f"Model type: {model_type}")
    logging.info(f"Device: {device}")
    logging.info(f"Checkpoint: {checkpoint_path}")
    logging.info(f"Output directory: {output_dir}")

    # Create test dataloader
    # Note: Use 'test' phase for TestDataset (with targets) or 'operation' for OperationDataset
    test_dataloader = create_dataloader(config, 'test')
    logging.info(
        f"Test dataloader: {len(test_dataloader.dataset)} samples, "
        f"{len(test_dataloader)} batches"
    )

    # Create model
    model = create_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model: {total_params:,} parameters")

    # Load checkpoint
    logging.info(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    logging.info("Checkpoint loaded successfully")

    # Create tester
    tester = Tester(
        config=config,
        model=model,
        device=device
    )

    # Run inference
    results = tester.test(test_dataloader)

    # Print summary
    logging.info("=" * 80)
    logging.info("INFERENCE COMPLETED")
    logging.info("=" * 80)
    logging.info(f"Model Type: {model_type}")
    logging.info(f"Total Samples: {results['total_samples']}")
    logging.info(f"Results saved to: {results['output_directory']}")
    logging.info("=" * 80)

    print("Inference completed successfully")

    return results


if __name__ == '__main__':
    main()
