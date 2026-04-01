#!/usr/bin/env python
"""Track experiment status and manage experiment metadata.

This module provides functionality to:
- Update experiment status (pending, submitted, running, completed, failed)
- Query experiment status
- List experiments by status
- Generate status reports

Usage:
    # Show overall status
    python experiments/experiment_tracker.py status

    # Update experiment status
    python experiments/experiment_tracker.py update 0001 running
    python experiments/experiment_tracker.py update 0001 completed --end-time "2024-01-01T12:00:00"

    # List experiments by status
    python experiments/experiment_tracker.py list-pending
    python experiments/experiment_tracker.py list-failed
    python experiments/experiment_tracker.py list-completed

    # Get specific experiment info
    python experiments/experiment_tracker.py info 0001
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import Counter


# Valid status values
VALID_STATUSES = ["pending", "submitted", "running", "completed", "failed", "cancelled"]


class ExperimentTracker:
    """Track and manage experiment status."""

    def __init__(self, tracking_file: Path):
        """Initialize tracker with tracking file path.

        Args:
            tracking_file: Path to experiments.csv tracking file
        """
        self.tracking_file = Path(tracking_file)
        self.experiments = []
        self.fieldnames = []

        if self.tracking_file.exists():
            self._load()

    def _load(self):
        """Load experiments from CSV file."""
        with open(self.tracking_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            self.fieldnames = reader.fieldnames
            self.experiments = list(reader)

    def _save(self):
        """Save experiments to CSV file."""
        if not self.fieldnames:
            return

        with open(self.tracking_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.experiments)

    def get_experiment(self, exp_id: str) -> Optional[Dict]:
        """Get experiment by ID.

        Args:
            exp_id: Experiment ID (e.g., "0001")

        Returns:
            Experiment dict or None if not found
        """
        # Normalize exp_id to 4 digits
        exp_id = exp_id.zfill(4)

        for exp in self.experiments:
            if exp["exp_id"] == exp_id:
                return exp
        return None

    def update_status(
        self,
        exp_id: str,
        status: str,
        slurm_job_id: str = None,
        start_time: str = None,
        end_time: str = None,
        best_loss: str = None,
        notes: str = None
    ) -> bool:
        """Update experiment status.

        Args:
            exp_id: Experiment ID
            status: New status (pending, submitted, running, completed, failed, cancelled)
            slurm_job_id: SLURM job ID (optional)
            start_time: Start time ISO format (optional)
            end_time: End time ISO format (optional)
            best_loss: Best loss value (optional)
            notes: Additional notes (optional)

        Returns:
            True if updated successfully, False otherwise
        """
        if status not in VALID_STATUSES:
            print(f"Error: Invalid status '{status}'. Valid: {VALID_STATUSES}")
            return False

        exp = self.get_experiment(exp_id)
        if not exp:
            print(f"Error: Experiment {exp_id} not found")
            return False

        # Update fields
        exp["status"] = status

        if slurm_job_id is not None:
            exp["slurm_job_id"] = slurm_job_id
        if start_time is not None:
            exp["start_time"] = start_time
        if end_time is not None:
            exp["end_time"] = end_time
        if best_loss is not None:
            exp["best_loss"] = best_loss
        if notes is not None:
            # Append to existing notes
            existing = exp.get("notes", "")
            if existing:
                exp["notes"] = f"{existing}; {notes}"
            else:
                exp["notes"] = notes

        self._save()
        return True

    def get_status_counts(self) -> Dict[str, int]:
        """Get count of experiments by status.

        Returns:
            Dict mapping status to count
        """
        counter = Counter(exp["status"] for exp in self.experiments)
        return dict(counter)

    def get_experiments_by_status(self, status: str) -> List[Dict]:
        """Get all experiments with given status.

        Args:
            status: Status to filter by

        Returns:
            List of experiment dicts
        """
        return [exp for exp in self.experiments if exp["status"] == status]

    def get_experiments_by_model(self, model_type: str) -> List[Dict]:
        """Get all experiments with given model type.

        Args:
            model_type: Model type to filter by

        Returns:
            List of experiment dicts
        """
        return [exp for exp in self.experiments if exp["model_type"] == model_type]

    def print_status(self):
        """Print overall status summary."""
        if not self.experiments:
            print("No experiments found. Run config_generator.py first.")
            return

        counts = self.get_status_counts()
        total = len(self.experiments)

        print("\n" + "=" * 60)
        print("EXPERIMENT STATUS SUMMARY")
        print("=" * 60)

        print(f"\nTotal Experiments: {total}")
        print("-" * 40)

        for status in VALID_STATUSES:
            count = counts.get(status, 0)
            pct = (count / total * 100) if total > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"  {status:12s}: {count:5d} ({pct:5.1f}%) {bar}")

        # Completed percentage
        completed = counts.get("completed", 0)
        completed_pct = (completed / total * 100) if total > 0 else 0
        print("-" * 40)
        print(f"  Progress: {completed}/{total} ({completed_pct:.1f}%)")

        # Model type breakdown
        print("\n" + "-" * 40)
        print("By Model Type:")
        for model_type in ["baseline", "convlstm", "transformer", "fusion"]:
            model_exps = self.get_experiments_by_model(model_type)
            if model_exps:
                model_completed = sum(1 for e in model_exps if e["status"] == "completed")
                print(f"  {model_type:12s}: {model_completed}/{len(model_exps)} completed")

        print("=" * 60 + "\n")

    def print_list(self, status: str, limit: int = 50):
        """Print list of experiments with given status.

        Args:
            status: Status to filter by
            limit: Maximum number to show
        """
        experiments = self.get_experiments_by_status(status)

        print(f"\n{status.upper()} Experiments ({len(experiments)} total):")
        print("-" * 60)

        if not experiments:
            print("  (none)")
            return

        for exp in experiments[:limit]:
            print(f"  {exp['exp_id']}: {exp['exp_name']}")
            if exp.get("slurm_job_id"):
                print(f"         Job ID: {exp['slurm_job_id']}")
            if exp.get("notes"):
                print(f"         Notes: {exp['notes']}")

        if len(experiments) > limit:
            print(f"\n  ... and {len(experiments) - limit} more")

    def print_info(self, exp_id: str):
        """Print detailed info for specific experiment.

        Args:
            exp_id: Experiment ID
        """
        exp = self.get_experiment(exp_id)

        if not exp:
            print(f"Error: Experiment {exp_id} not found")
            return

        print(f"\nExperiment: {exp['exp_id']}")
        print("=" * 50)
        for key, value in exp.items():
            if value:
                print(f"  {key:20s}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Experiment status tracker")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show overall status")

    # Update command
    update_parser = subparsers.add_parser("update", help="Update experiment status")
    update_parser.add_argument("exp_id", help="Experiment ID (e.g., 0001)")
    update_parser.add_argument("status", choices=VALID_STATUSES, help="New status")
    update_parser.add_argument("--slurm-job-id", help="SLURM job ID")
    update_parser.add_argument("--start-time", help="Start time (ISO format)")
    update_parser.add_argument("--end-time", help="End time (ISO format)")
    update_parser.add_argument("--best-loss", help="Best loss value")
    update_parser.add_argument("--notes", help="Additional notes")

    # List commands
    for status in VALID_STATUSES:
        list_parser = subparsers.add_parser(f"list-{status}", help=f"List {status} experiments")
        list_parser.add_argument("--limit", type=int, default=50, help="Max to show")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show experiment details")
    info_parser.add_argument("exp_id", help="Experiment ID")

    args = parser.parse_args()

    # Determine tracking file path
    experiments_root = Path(__file__).parent
    tracking_file = experiments_root / "tracking" / "experiments.csv"

    tracker = ExperimentTracker(tracking_file)

    if args.command == "status":
        tracker.print_status()

    elif args.command == "update":
        success = tracker.update_status(
            exp_id=args.exp_id,
            status=args.status,
            slurm_job_id=args.slurm_job_id,
            start_time=args.start_time,
            end_time=args.end_time,
            best_loss=args.best_loss,
            notes=args.notes
        )
        if success:
            print(f"Updated experiment {args.exp_id} to '{args.status}'")
        sys.exit(0 if success else 1)

    elif args.command and args.command.startswith("list-"):
        status = args.command.replace("list-", "")
        tracker.print_list(status, limit=args.limit)

    elif args.command == "info":
        tracker.print_info(args.exp_id)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
