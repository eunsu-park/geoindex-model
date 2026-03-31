#!/usr/bin/env python
"""Generate experiment configurations for large-scale hyperparameter sweeps.

This script generates YAML configuration files and a tracking CSV for all
experiment combinations.

Note: num_subsamples varies by target_days:
    - target_days=[1]: 14 subsamples -> 4*7*14 = 392 experiments
    - target_days=[1,2]: 8 subsamples -> 4*7*8 = 224 experiments
    - target_days=[1,2,3]: 6 subsamples -> 4*7*6 = 168 experiments
    Total: 784 experiments

Usage:
    # Generate all 784 experiments
    python experiments/config_generator.py --generate-all

    # Generate test subset (4 experiments)
    python experiments/config_generator.py --test-mode --count=4

    # List all combinations without generating
    python experiments/config_generator.py --list-only

    # Generate specific model type only
    python experiments/config_generator.py --generate-all --model-type=fusion
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

import yaml


# Default experiment parameter space (used when no matrix file provided)
DEFAULT_MODEL_TYPES = ["baseline", "convlstm", "transformer", "fusion"]

DEFAULT_INPUT_DAYS_COMBINATIONS = [
    [-1],
    [-2, -1],
    [-3, -2, -1],
    [-4, -3, -2, -1],
    [-5, -4, -3, -2, -1],
    [-6, -5, -4, -3, -2, -1],
    [-7, -6, -5, -4, -3, -2, -1],
]

DEFAULT_TARGET_DAYS_CONFIG = {
    1: {"days": [1], "num_subsamples": 14},
    2: {"days": [1, 2], "num_subsamples": 8},
    3: {"days": [1, 2, 3], "num_subsamples": 6},
}

# Valid model types for validation
VALID_MODEL_TYPES = ["baseline", "convlstm", "transformer", "fusion"]


def load_experiment_matrix(matrix_path: Path) -> Dict[str, Any]:
    """Load experiment matrix from YAML file.

    Args:
        matrix_path: Path to the experiment matrix YAML file

    Returns:
        Dictionary containing experiment matrix configuration

    Raises:
        FileNotFoundError: If matrix file does not exist
        ValueError: If matrix file has invalid structure
    """
    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

    with open(matrix_path, 'r') as f:
        matrix = yaml.safe_load(f)

    validate_matrix(matrix)
    matrix = apply_filters(matrix)

    return matrix


def validate_matrix(matrix: Dict[str, Any]) -> None:
    """Validate experiment matrix structure.

    Args:
        matrix: Experiment matrix dictionary

    Raises:
        ValueError: If matrix has invalid structure or values
    """
    required_keys = ['model_types', 'input_days_combinations', 'target_days_config']
    for key in required_keys:
        if key not in matrix:
            raise ValueError(f"Missing required key in matrix: {key}")

    # Validate model types
    for model in matrix['model_types']:
        if model not in VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model type: {model}. "
                f"Valid options: {VALID_MODEL_TYPES}"
            )

    # Validate input_days_combinations
    if not isinstance(matrix['input_days_combinations'], list):
        raise ValueError("input_days_combinations must be a list")
    for combo in matrix['input_days_combinations']:
        if not isinstance(combo, list) or not all(isinstance(d, int) for d in combo):
            raise ValueError(f"Invalid input_days combination: {combo}")

    # Validate target_days_config
    if not isinstance(matrix['target_days_config'], dict):
        raise ValueError("target_days_config must be a dictionary")
    for key, cfg in matrix['target_days_config'].items():
        if 'days' not in cfg or 'num_subsamples' not in cfg:
            raise ValueError(
                f"target_days_config[{key}] must have 'days' and 'num_subsamples'"
            )

    # Validate undersampling_modes (optional, defaults to ["static"])
    undersampling_modes = matrix.get('undersampling_modes', ["static"])
    if not isinstance(undersampling_modes, list):
        raise ValueError("undersampling_modes must be a list")
    valid_modes = {True, False, "static", "dynamic"}
    for mode in undersampling_modes:
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid undersampling_mode: {mode}. "
                f"Must be one of: 'static', 'dynamic', true, false"
            )


def apply_filters(matrix: Dict[str, Any]) -> Dict[str, Any]:
    """Apply include_only and exclude filters to matrix.

    Args:
        matrix: Experiment matrix dictionary

    Returns:
        Filtered matrix dictionary
    """
    model_types = matrix['model_types']

    # Apply include_only filter
    include_only = matrix.get('include_only', [])
    if include_only:
        model_types = [m for m in model_types if m in include_only]

    # Apply exclude filter
    exclude = matrix.get('exclude', [])
    if exclude:
        model_types = [m for m in model_types if m not in exclude]

    matrix['model_types'] = model_types
    return matrix


def get_default_matrix() -> Dict[str, Any]:
    """Get default experiment matrix (hardcoded values).

    Returns:
        Default experiment matrix dictionary
    """
    return {
        'model_types': DEFAULT_MODEL_TYPES.copy(),
        'input_days_combinations': [combo.copy() for combo in DEFAULT_INPUT_DAYS_COMBINATIONS],
        'target_days_config': {
            k: v.copy() for k, v in DEFAULT_TARGET_DAYS_CONFIG.items()
        },
        'undersampling_modes': ["static"],
        'include_only': [],
        'exclude': []
    }


def get_num_subsamples(target_days: List[int], target_days_config: Dict) -> int:
    """Get num_subsamples based on target_days count.

    Args:
        target_days: List of target days
        target_days_config: Target days configuration dictionary

    Returns:
        Number of subsamples for this target_days configuration
    """
    return target_days_config[len(target_days)]["num_subsamples"]


def generate_experiment_name(
    model_type: str,
    input_days: List[int],
    target_days: List[int],
    undersampling_mode: Any,
    subsample_index: Optional[int] = None
) -> str:
    """Generate experiment name from parameters.

    Format:
        Static undersampling: {model_type}_in{N}_out{M}_sub{subsample_index:02d}
        Dynamic undersampling: {model_type}_in{N}_out{M}_dyn
        No undersampling: {model_type}_in{N}_out{M}_full

    Examples:
        fusion_in7_out3_sub00     (static undersampling)
        fusion_in7_out3_dyn       (dynamic undersampling)
        baseline_in1_out1_full    (no undersampling)
    """
    base = f"{model_type}_in{len(input_days)}_out{len(target_days)}"
    if undersampling_mode == "dynamic":
        return f"{base}_dyn"
    elif undersampling_mode in (True, "static"):
        return f"{base}_sub{subsample_index:02d}"
    else:
        return f"{base}_full"


def generate_yaml_content(
    exp_id: int,
    exp_name: str,
    model_type: str,
    input_days: List[int],
    target_days: List[int],
    undersampling_mode: Any,
    target_days_config: Dict,
    subsample_index: Optional[int] = None,
    base_config: str = "wulver"
) -> str:
    """Generate YAML configuration content for an experiment.

    Args:
        exp_id: Experiment ID (1-indexed)
        exp_name: Experiment name
        model_type: Model type (baseline, convlstm, transformer, fusion)
        input_days: List of input days (negative integers)
        target_days: List of target days (positive integers)
        undersampling_mode: "static", "dynamic", True (=static), or False (=off)
        target_days_config: Target days configuration dictionary
        subsample_index: Subsample index (required for static mode)
        base_config: Base config to inherit from (default: wulver)

    Returns:
        YAML content as string
    """
    # Format lists for YAML
    input_days_str = "\n".join([f"    - {d}" for d in sorted(input_days)])
    target_days_str = "\n".join([f"    - {d}" for d in sorted(target_days)])

    # Build sampling section based on undersampling mode
    if undersampling_mode == "dynamic":
        sampling_section = f"""sampling:
  enable_undersampling: true
  undersampling_mode: "dynamic"
  input_days:
{input_days_str}
  target_days:
{target_days_str}"""
    elif undersampling_mode in (True, "static"):
        num_subsamples = get_num_subsamples(target_days, target_days_config)
        sampling_section = f"""sampling:
  enable_undersampling: true
  undersampling_mode: "static"
  num_subsamples: {num_subsamples}
  subsample_index: {subsample_index}
  input_days:
{input_days_str}
  target_days:
{target_days_str}"""
    else:
        sampling_section = f"""sampling:
  enable_undersampling: false
  input_days:
{input_days_str}
  target_days:
{target_days_str}"""

    yaml_content = f"""# Experiment {exp_id:04d}: {exp_name}
# Auto-generated by config_generator.py
# Generated at: {datetime.now().isoformat()}

defaults:
  - {base_config}
  - _self_

experiment:
  name: "{exp_name}"

model:
  model_type: "{model_type}"

{sampling_section}
"""
    return yaml_content


def generate_all_combinations(
    matrix: Optional[Dict[str, Any]] = None
) -> List[Tuple[str, List[int], List[int], Any, Optional[int]]]:
    """Generate all parameter combinations from experiment matrix.

    Args:
        matrix: Experiment matrix dictionary. If None, uses default values.

    Returns:
        List of tuples: (model_type, input_days, target_days, undersampling_mode, subsample_index)
        Note: subsample_index is None for dynamic mode or when undersampling is off
    """
    if matrix is None:
        matrix = get_default_matrix()

    model_types = matrix['model_types']
    input_days_combinations = matrix['input_days_combinations']
    target_days_config = matrix['target_days_config']
    target_days_combinations = [cfg["days"] for cfg in target_days_config.values()]
    undersampling_modes = matrix.get('undersampling_modes', ["static"])

    combinations = []
    for model_type in model_types:
        for input_days in input_days_combinations:
            for target_days in target_days_combinations:
                for mode in undersampling_modes:
                    if mode == "dynamic":
                        # Dynamic: single experiment, sampler handles per-epoch
                        combinations.append(
                            (model_type, input_days, target_days, "dynamic", None)
                        )
                    elif mode in (True, "static"):
                        # Static k-fold: one experiment per fold
                        num_subsamples = get_num_subsamples(target_days, target_days_config)
                        for subsample_index in range(num_subsamples):
                            combinations.append(
                                (model_type, input_days, target_days, "static", subsample_index)
                            )
                    else:
                        # No undersampling
                        combinations.append(
                            (model_type, input_days, target_days, False, None)
                        )
    return combinations


def generate_configs(
    output_dir: Path,
    tracking_file: Path,
    matrix: Optional[Dict[str, Any]] = None,
    model_type_filter: str = None,
    test_mode: bool = False,
    test_count: int = 4,
    base_config: str = "wulver"
) -> int:
    """Generate all experiment configuration files.

    Args:
        output_dir: Directory to save config files
        tracking_file: Path to save tracking CSV
        matrix: Experiment matrix dictionary. If None, uses default values.
        model_type_filter: Only generate for specific model type (optional)
        test_mode: If True, only generate test_count experiments
        test_count: Number of experiments in test mode
        base_config: Base config to inherit from

    Returns:
        Number of configs generated
    """
    if matrix is None:
        matrix = get_default_matrix()

    output_dir.mkdir(parents=True, exist_ok=True)
    tracking_file.parent.mkdir(parents=True, exist_ok=True)

    combinations = generate_all_combinations(matrix)

    # Apply filters
    if model_type_filter:
        combinations = [c for c in combinations if c[0] == model_type_filter]

    if test_mode:
        combinations = combinations[:test_count]

    print(f"Generating {len(combinations)} experiment configurations...")

    # Write tracking CSV
    csv_rows = []

    target_days_config = matrix['target_days_config']

    for exp_id, (model_type, input_days, target_days, undersampling_mode, subsample_index) in enumerate(combinations, 1):
        exp_name = generate_experiment_name(
            model_type, input_days, target_days, undersampling_mode, subsample_index
        )

        # Generate YAML file
        yaml_content = generate_yaml_content(
            exp_id=exp_id,
            exp_name=exp_name,
            model_type=model_type,
            input_days=input_days,
            target_days=target_days,
            undersampling_mode=undersampling_mode,
            target_days_config=target_days_config,
            subsample_index=subsample_index,
            base_config=base_config
        )

        config_path = output_dir / f"exp_{exp_id:04d}.yaml"
        config_path.write_text(yaml_content)

        # Add to tracking
        csv_rows.append({
            "exp_id": f"{exp_id:04d}",
            "exp_name": exp_name,
            "model_type": model_type,
            "input_days": str(sorted(input_days)),
            "input_days_count": len(input_days),
            "target_days": str(sorted(target_days)),
            "target_days_count": len(target_days),
            "undersampling_mode": undersampling_mode,
            "subsample_index": subsample_index if subsample_index is not None else "",
            "status": "pending",
            "slurm_job_id": "",
            "start_time": "",
            "end_time": "",
            "best_loss": "",
            "notes": ""
        })

        if exp_id % 100 == 0:
            print(f"  Generated {exp_id}/{len(combinations)} configs...")

    # Write tracking CSV
    fieldnames = [
        "exp_id", "exp_name", "model_type",
        "input_days", "input_days_count",
        "target_days", "target_days_count",
        "undersampling_mode", "subsample_index",
        "status", "slurm_job_id",
        "start_time", "end_time", "best_loss", "notes"
    ]

    with open(tracking_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nGeneration complete!")
    print(f"  Configs: {output_dir}")
    print(f"  Tracking: {tracking_file}")
    print(f"  Total: {len(combinations)} experiments")

    return len(combinations)


def list_combinations(
    matrix: Optional[Dict[str, Any]] = None,
    model_type_filter: str = None
):
    """List all experiment combinations without generating files.

    Args:
        matrix: Experiment matrix dictionary. If None, uses default values.
        model_type_filter: Only list for specific model type (optional)
    """
    if matrix is None:
        matrix = get_default_matrix()

    combinations = generate_all_combinations(matrix)

    if model_type_filter:
        combinations = [c for c in combinations if c[0] == model_type_filter]

    model_types = matrix['model_types']
    input_days_combinations = matrix['input_days_combinations']
    target_days_config = matrix['target_days_config']
    undersampling_modes = matrix.get('undersampling_modes', ["static"])

    print(f"\nTotal combinations: {len(combinations)}")
    print("\nBreakdown:")
    print(f"  Model types: {len(model_types)} ({', '.join(model_types)})")
    print(f"  Input days combinations: {len(input_days_combinations)}")
    print(f"  Target days configurations:")
    for _, cfg in target_days_config.items():
        print(f"    - {cfg['days']} -> num_subsamples={cfg['num_subsamples']}")
    print(f"  Undersampling modes: {undersampling_modes}")

    # Calculate total
    total = 0
    for mode in undersampling_modes:
        if mode == "dynamic":
            count = len(model_types) * len(input_days_combinations) * len(target_days_config)
            total += count
        elif mode in (True, "static"):
            for cfg in target_days_config.values():
                count = len(model_types) * len(input_days_combinations) * cfg['num_subsamples']
                total += count
        else:
            count = len(model_types) * len(input_days_combinations) * len(target_days_config)
            total += count
    print(f"\n  Calculated total: {total}")

    print("\nFirst 10 combinations:")
    for i, (model_type, input_days, target_days, us_mode, subsample_index) in enumerate(combinations[:10], 1):
        exp_name = generate_experiment_name(model_type, input_days, target_days, us_mode, subsample_index)
        print(f"  {i:4d}. {exp_name}")

    if len(combinations) > 10:
        print(f"  ... and {len(combinations) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Generate experiment configurations for hyperparameter sweep"
    )
    parser.add_argument(
        "--generate-all",
        action="store_true",
        help="Generate all experiment configurations"
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List all combinations without generating files"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Generate only a small subset for testing"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=4,
        help="Number of experiments in test mode (default: 4)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=VALID_MODEL_TYPES,
        help="Generate only for specific model type"
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="wulver",
        help="Base config to inherit from (default: wulver)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for configs (default: experiments/configs)"
    )
    parser.add_argument(
        "--matrix",
        type=str,
        default=None,
        help="Path to experiment matrix YAML file (default: experiments/experiment_matrix.yaml)"
    )
    parser.add_argument(
        "--validate-matrix",
        action="store_true",
        help="Validate matrix file without generating configs"
    )

    args = parser.parse_args()

    # Determine paths
    experiments_root = Path(__file__).parent
    output_dir = Path(args.output_dir) if args.output_dir else experiments_root / "configs"
    tracking_file = experiments_root / "tracking" / "experiments.csv"

    # Load experiment matrix
    matrix = None
    if args.matrix:
        matrix_path = Path(args.matrix)
    else:
        # Check for default matrix file
        default_matrix_path = experiments_root / "experiment_matrix.yaml"
        if default_matrix_path.exists():
            matrix_path = default_matrix_path
        else:
            matrix_path = None

    if matrix_path:
        try:
            print(f"Loading experiment matrix from: {matrix_path}")
            matrix = load_experiment_matrix(matrix_path)
            print(f"  Model types: {matrix['model_types']}")
            print(f"  Input combinations: {len(matrix['input_days_combinations'])}")
            print(f"  Target configs: {len(matrix['target_days_config'])}")
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading matrix: {e}")
            return 1

    if args.validate_matrix:
        if matrix:
            print("\nMatrix validation successful!")
            combinations = generate_all_combinations(matrix)
            print(f"Total experiments: {len(combinations)}")
        else:
            print("No matrix file specified. Use --matrix to specify one.")
        return 0

    if args.list_only:
        list_combinations(matrix, args.model_type)
    elif args.generate_all or args.test_mode:
        generate_configs(
            output_dir=output_dir,
            tracking_file=tracking_file,
            matrix=matrix,
            model_type_filter=args.model_type,
            test_mode=args.test_mode,
            test_count=args.count,
            base_config=args.base_config
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
