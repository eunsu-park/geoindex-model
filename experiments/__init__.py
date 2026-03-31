"""Experiment management package for large-scale hyperparameter sweeps.

This package provides tools for managing 1,000+ experiments on HPC clusters:
- config_generator: Generate experiment configurations
- experiment_tracker: Track experiment status
- submit_jobs: Submit SLURM jobs
- result_aggregator: Collect and analyze results

Usage:
    # Generate all experiment configs
    python experiments/config_generator.py --generate-all

    # Submit jobs to SLURM
    python experiments/submit_jobs.py --submit-all

    # Check status
    python experiments/experiment_tracker.py status

    # Collect results
    python experiments/result_aggregator.py collect
"""

from pathlib import Path

# Package paths
EXPERIMENTS_ROOT = Path(__file__).parent
CONFIGS_DIR = EXPERIMENTS_ROOT / "configs"
SLURM_DIR = EXPERIMENTS_ROOT / "slurm"
TRACKING_DIR = EXPERIMENTS_ROOT / "tracking"

# Ensure directories exist
CONFIGS_DIR.mkdir(exist_ok=True)
SLURM_DIR.mkdir(exist_ok=True)
TRACKING_DIR.mkdir(exist_ok=True)
