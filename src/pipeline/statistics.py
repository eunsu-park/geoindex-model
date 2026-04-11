"""Statistics computation and undersampling utilities.

Provides functions for computing and caching normalization statistics
from HDF5, CSV, and in-memory table data sources, plus class-based
undersampling utilities for imbalanced datasets.
"""

import os
import logging
import pickle
from typing import Dict, List, Tuple

import h5py
import numpy as np

from .readers import CSVEventReader
from .normalizer import OnlineStatistics

logger = logging.getLogger(__name__)


def compute_statistics(
    stat_file_path: str,
    data_root: str,
    data_dir_name: str,
    data_file_list: List[str],
    variables: List[str],
    overwrite: bool = False
) -> Dict[str, Dict[str, float]]:
    """Compute and cache normalization statistics for OMNI variables.

    Args:
        stat_file_path: Path to save/load statistics pickle file
        data_root: Root directory for data
        data_dir_name: Subdirectory containing HDF5 files
        data_file_list: List of HDF5 filenames
        variables: List of OMNI variables to compute statistics for
        overwrite: If True, recompute even if cache exists

    Returns:
        Dictionary of statistics for each variable
    """
    # Try to load cached statistics
    if os.path.exists(stat_file_path) and not overwrite:
        try:
            with open(stat_file_path, 'rb') as f:
                loaded_stats = pickle.load(f)

            if all(var in loaded_stats for var in variables):
                logger.info(f"Loaded statistics from {stat_file_path}")
                return {var: loaded_stats[var] for var in variables}
            else:
                logger.info("Incomplete statistics, recomputing...")
        except (pickle.PickleError, KeyError) as e:
            logger.warning(f"Failed to load statistics: {e}, recomputing...")

    # Filter for h5 files
    h5_files = [
        f"{data_root}/{data_dir_name}/{f}"
        for f in data_file_list if f.endswith('.h5')
    ]

    if not h5_files:
        raise ValueError("No valid .h5 files found in data file list")

    # Initialize online statistics
    stats_computers = {var: OnlineStatistics() for var in variables}

    # Process files
    valid_files = 0
    total_files = len(h5_files)

    for i, file_path in enumerate(h5_files):
        if not os.path.exists(file_path):
            continue

        try:
            with h5py.File(file_path, 'r') as f:
                for variable in variables:
                    dataset_name = f"omni/{variable}"
                    if dataset_name in f:
                        data = f[dataset_name][:]
                        stats_computers[variable].update(data)

            valid_files += 1

            # Progress logging
            if valid_files % 100 == 0 or valid_files == total_files:
                logger.info(f"Computing statistics: {valid_files}/{total_files} files processed")

        except (OSError, KeyError) as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            continue

    if valid_files == 0:
        raise ValueError("No valid data files found for statistics computation")

    # Compile final statistics
    stat_dict = {}
    for variable in variables:
        stat_dict[variable] = stats_computers[variable].get_stats()
        logger.info(
            f"{variable}: mean={stat_dict[variable]['mean']:.3f}, "
            f"std={stat_dict[variable]['std']:.3f}"
        )

    # Save statistics
    try:
        os.makedirs(os.path.dirname(stat_file_path), exist_ok=True)
        with open(stat_file_path, 'wb') as f:
            pickle.dump(stat_dict, f)
        logger.info(f"Statistics saved to {stat_file_path}")
    except (OSError, pickle.PickleError) as e:
        logger.warning(f"Failed to save statistics: {e}")

    return stat_dict


def compute_statistics_csv(
    stat_file_path: str,
    csv_file_paths: List[str],
    variables: List[str],
    overwrite: bool = False
) -> Dict[str, Dict[str, float]]:
    """Compute and cache normalization statistics for CSV event files.

    Args:
        stat_file_path: Path to save/load statistics pickle file
        csv_file_paths: List of CSV file paths to compute statistics from
        variables: List of variable names to compute statistics for
        overwrite: If True, recompute even if cache exists

    Returns:
        Dictionary of statistics for each variable
    """
    # Try to load cached statistics
    if os.path.exists(stat_file_path) and not overwrite:
        try:
            with open(stat_file_path, 'rb') as f:
                loaded_stats = pickle.load(f)

            if all(var in loaded_stats for var in variables):
                logger.info(f"Loaded statistics from {stat_file_path}")
                return {var: loaded_stats[var] for var in variables}
            else:
                logger.info("Incomplete statistics, recomputing...")
        except (pickle.PickleError, KeyError) as e:
            logger.warning(f"Failed to load statistics: {e}, recomputing...")

    if not csv_file_paths:
        raise ValueError("No CSV files provided for statistics computation")

    # Initialize online statistics per variable
    stats_computers = {var: OnlineStatistics() for var in variables}

    valid_files = 0
    total_files = len(csv_file_paths)

    for i, file_path in enumerate(csv_file_paths):
        if not os.path.exists(file_path):
            continue

        try:
            data = CSVEventReader.read(file_path, variables=variables, validate=False)
            # data shape: (T, V) — iterate over variables
            for j, variable in enumerate(variables):
                stats_computers[variable].update(data[:, j])

            valid_files += 1

            if valid_files % 100 == 0 or valid_files == total_files:
                logger.info(f"Computing statistics: {valid_files}/{total_files} files processed")

        except (OSError, KeyError) as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            continue

    if valid_files == 0:
        raise ValueError("No valid CSV files found for statistics computation")

    # Compile final statistics
    stat_dict = {}
    for variable in variables:
        stat_dict[variable] = stats_computers[variable].get_stats()
        logger.info(
            f"{variable}: mean={stat_dict[variable]['mean']:.3f}, "
            f"std={stat_dict[variable]['std']:.3f}"
        )

    # Save statistics
    try:
        os.makedirs(os.path.dirname(stat_file_path), exist_ok=True)
        with open(stat_file_path, 'wb') as f:
            pickle.dump(stat_dict, f)
        logger.info(f"Statistics saved to {stat_file_path}")
    except (OSError, pickle.PickleError) as e:
        logger.warning(f"Failed to save statistics: {e}")

    return stat_dict


def compute_statistics_table(
    stat_file_path: str,
    data_array: np.ndarray,
    row_indices: List[int],
    input_start: int,
    input_end: int,
    variables: List[str],
    overwrite: bool = False
) -> Dict[str, Dict[str, float]]:
    """Compute and cache normalization statistics from in-memory table.

    Args:
        stat_file_path: Path to save/load statistics pickle file
        data_array: Full table as numpy array (N_rows, N_vars)
        row_indices: Row indices of reference times in training split
        input_start: Window start offset (negative, relative to ref)
        input_end: Window end offset (positive, relative to ref)
        variables: List of variable names (matching columns of data_array)
        overwrite: If True, recompute even if cache exists

    Returns:
        Dictionary of statistics for each variable
    """
    # Try to load cached statistics
    if os.path.exists(stat_file_path) and not overwrite:
        try:
            with open(stat_file_path, 'rb') as f:
                loaded_stats = pickle.load(f)

            if all(var in loaded_stats for var in variables):
                logger.info(f"Loaded table statistics from {stat_file_path}")
                return {var: loaded_stats[var] for var in variables}
            else:
                logger.info("Incomplete table statistics, recomputing...")
        except (pickle.PickleError, KeyError) as e:
            logger.warning(f"Failed to load table statistics: {e}, recomputing...")

    if not row_indices:
        raise ValueError("No row indices provided for statistics computation")

    # Initialize online statistics per variable
    stats_computers = {var: OnlineStatistics() for var in variables}

    for count, ref_row in enumerate(row_indices):
        window = data_array[ref_row + input_start:ref_row + input_end]
        for j, variable in enumerate(variables):
            stats_computers[variable].update(window[:, j])

        if (count + 1) % 1000 == 0:
            logger.info(f"Computing table statistics: "
                        f"{count + 1}/{len(row_indices)} refs processed")

    # Compile final statistics
    stat_dict = {}
    for variable in variables:
        stat_dict[variable] = stats_computers[variable].get_stats()
        logger.info(
            f"{variable}: mean={stat_dict[variable]['mean']:.3f}, "
            f"std={stat_dict[variable]['std']:.3f}"
        )

    # Save statistics
    try:
        os.makedirs(os.path.dirname(stat_file_path) or '.', exist_ok=True)
        with open(stat_file_path, 'wb') as f:
            pickle.dump(stat_dict, f)
        logger.info(f"Table statistics saved to {stat_file_path}")
    except (OSError, pickle.PickleError) as e:
        logger.warning(f"Failed to save table statistics: {e}")

    return stat_dict


# =============================================================================
# Undersampling Utilities
# =============================================================================

def split_by_class(
    file_list: List[Tuple[str, int]]
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Split file list into positive and negative samples.

    Args:
        file_list: List of (filename, label) tuples

    Returns:
        Tuple of (positive_list, negative_list)
    """
    positive = []
    negative = []
    for file_name, label in file_list:
        if label == 0:
            negative.append((file_name, label))
        else:
            positive.append((file_name, label))
    positive.sort(key=lambda x: x[0])
    negative.sort(key=lambda x: x[0])
    return positive, negative


def undersample(
    file_list: List[Tuple[str, int]],
    num_subsample: int,
    subsample_index: int,
    seed: int = 42
) -> Tuple[List[Tuple[str, int]], int, int]:
    """Undersample negative class to balance dataset.

    Uses deterministic shuffling with seed for reproducibility.
    Purpose: Test model stability across different negative subsamples.

    Args:
        file_list: List of (filename, label) tuples
        num_subsample: Number of folds for negative samples
        subsample_index: Which fold to use (0 to num_subsample-1)
        seed: Random seed for reproducible shuffling

    Returns:
        Tuple of (sampled_list, num_positive, num_negative)
    """
    import random

    positive, negative = split_by_class(file_list)

    # Reproducible shuffle with seed
    rng = random.Random(seed)
    rng.shuffle(negative)

    # Split into folds
    n = len(negative)
    base_size = n // num_subsample
    remainder = n % num_subsample

    start = 0
    for i in range(num_subsample):
        size = base_size + (1 if i < remainder else 0)
        if i == subsample_index:
            selected_negative = negative[start:start + size]
            break
        start += size

    # Combine positive and selected negative
    sampled_list = positive + selected_negative

    return sampled_list, len(positive), len(negative)
