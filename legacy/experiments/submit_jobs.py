#!/usr/bin/env python
"""Submit SLURM jobs for experiment training.

This module provides functionality to:
- Submit all experiments as a SLURM job array
- Submit specific model types
- Retry failed experiments
- Generate submission commands (dry-run)

Usage:
    # Submit all experiments (dry-run to see command)
    python experiments/submit_jobs.py --submit-all --dry-run

    # Submit all experiments (max 50 concurrent)
    python experiments/submit_jobs.py --submit-all --max-concurrent=50

    # Submit specific model type only
    python experiments/submit_jobs.py --model-type=fusion --submit

    # Retry failed experiments
    python experiments/submit_jobs.py --retry-failed

    # Submit specific experiment IDs
    python experiments/submit_jobs.py --exp-ids 1,5,10,100
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from experiment_tracker import ExperimentTracker


def get_experiment_ids_by_status(tracker: ExperimentTracker, status: str) -> List[int]:
    """Get experiment IDs with given status.

    Args:
        tracker: ExperimentTracker instance
        status: Status to filter by

    Returns:
        List of experiment IDs (as integers)
    """
    experiments = tracker.get_experiments_by_status(status)
    return [int(exp["exp_id"]) for exp in experiments]


def get_experiment_ids_by_model(tracker: ExperimentTracker, model_type: str) -> List[int]:
    """Get experiment IDs for given model type.

    Args:
        tracker: ExperimentTracker instance
        model_type: Model type to filter by

    Returns:
        List of experiment IDs (as integers)
    """
    experiments = tracker.get_experiments_by_model(model_type)
    return [int(exp["exp_id"]) for exp in experiments]


def format_array_spec(exp_ids: List[int], max_concurrent: int = 50) -> str:
    """Format experiment IDs as SLURM array specification.

    Args:
        exp_ids: List of experiment IDs
        max_concurrent: Maximum concurrent jobs

    Returns:
        SLURM array specification string (e.g., "1-100%50" or "1,5,10,15")
    """
    if not exp_ids:
        return ""

    exp_ids = sorted(exp_ids)

    # Check if IDs are consecutive
    is_consecutive = (exp_ids[-1] - exp_ids[0] + 1 == len(exp_ids))

    if is_consecutive and len(exp_ids) > 10:
        # Use range format: start-end%max_concurrent
        return f"{exp_ids[0]}-{exp_ids[-1]}%{max_concurrent}"
    else:
        # Use comma-separated format
        return ",".join(str(i) for i in exp_ids)


def submit_jobs(
    exp_ids: List[int],
    slurm_script: Path,
    max_concurrent: int = 50,
    dry_run: bool = False
) -> bool:
    """Submit SLURM job array for experiments.

    Args:
        exp_ids: List of experiment IDs to submit
        slurm_script: Path to SLURM script
        max_concurrent: Maximum concurrent jobs
        dry_run: If True, only print command without executing

    Returns:
        True if successful (or dry-run), False otherwise
    """
    if not exp_ids:
        print("No experiments to submit.")
        return False

    if not slurm_script.exists():
        print(f"Error: SLURM script not found: {slurm_script}")
        return False

    array_spec = format_array_spec(exp_ids, max_concurrent)

    cmd = ["sbatch", f"--array={array_spec}", str(slurm_script)]

    print(f"\nSubmission Command:")
    print(f"  {' '.join(cmd)}")
    print(f"\nExperiments: {len(exp_ids)}")
    print(f"Array Spec: {array_spec}")
    print(f"Max Concurrent: {max_concurrent}")

    if dry_run:
        print("\n[DRY RUN] Command not executed.")
        return True

    print("\nSubmitting...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"\nSubmitted successfully!")
        print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nSubmission failed!")
        print(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("\nError: 'sbatch' command not found. Are you on a SLURM cluster?")
        return False


def main():
    parser = argparse.ArgumentParser(description="Submit SLURM jobs for experiments")

    # Submission options
    parser.add_argument(
        "--submit-all",
        action="store_true",
        help="Submit all pending experiments"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed experiments"
    )
    parser.add_argument(
        "--model-type",
        choices=["baseline", "convlstm", "transformer", "fusion"],
        help="Submit only experiments for specific model type"
    )
    parser.add_argument(
        "--exp-ids",
        type=str,
        help="Comma-separated experiment IDs to submit (e.g., 1,5,10,100)"
    )

    # Configuration
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=50,
        help="Maximum concurrent jobs (default: 50)"
    )
    parser.add_argument(
        "--slurm-script",
        type=str,
        default=None,
        help="Path to SLURM script (default: experiments/slurm/train_array.slurm)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing"
    )

    args = parser.parse_args()

    # Paths
    experiments_root = Path(__file__).parent
    tracking_file = experiments_root / "tracking" / "experiments.csv"
    slurm_script = Path(args.slurm_script) if args.slurm_script else experiments_root / "slurm" / "train_array.slurm"

    # Load tracker
    tracker = ExperimentTracker(tracking_file)

    if not tracker.experiments:
        print("Error: No experiments found. Run config_generator.py first.")
        sys.exit(1)

    # Determine which experiments to submit
    exp_ids = []

    if args.exp_ids:
        # Specific IDs provided
        exp_ids = [int(i.strip()) for i in args.exp_ids.split(",")]
        print(f"Submitting specific experiments: {exp_ids}")

    elif args.retry_failed:
        # Retry failed experiments
        exp_ids = get_experiment_ids_by_status(tracker, "failed")
        print(f"Retrying {len(exp_ids)} failed experiments")

    elif args.model_type:
        # Specific model type
        model_exp_ids = get_experiment_ids_by_model(tracker, args.model_type)
        # Only submit pending ones
        pending_ids = set(get_experiment_ids_by_status(tracker, "pending"))
        exp_ids = [i for i in model_exp_ids if i in pending_ids]
        print(f"Submitting {len(exp_ids)} pending {args.model_type} experiments")

    elif args.submit_all:
        # All pending experiments
        exp_ids = get_experiment_ids_by_status(tracker, "pending")
        print(f"Submitting all {len(exp_ids)} pending experiments")

    else:
        parser.print_help()
        sys.exit(0)

    if not exp_ids:
        print("No experiments match the criteria.")
        sys.exit(0)

    # Submit
    success = submit_jobs(
        exp_ids=exp_ids,
        slurm_script=slurm_script,
        max_concurrent=args.max_concurrent,
        dry_run=args.dry_run
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
