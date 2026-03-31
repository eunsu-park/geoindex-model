"""Data pipeline for multi-modal solar wind prediction.

This module provides data loading, preprocessing, and dataset classes for
training, validation, testing, and operational inference.

Supports multiple data modalities controlled by config.data.modalities:
  - timeseries: CSV-based 30-min solar wind time series (new dataset)
  - sdo: SDO image data via HDF5 (future use)
  - omni_hdf5: Legacy OMNI HDF5 time series

Classes:
    HDF5Reader: Reads HDF5 data files (SDO + OMNI)
    CSVEventReader: Reads CSV event files (30-min solar wind time series)
    Normalizer: Handles data normalization
    OnlineStatistics: Computes statistics using Welford's algorithm
    BaseDataset: Abstract base class for HDF5-based datasets
    TrainDataset: Training dataset with undersampling support (HDF5)
    ValidationDataset: Validation dataset (HDF5)
    TestDataset: Test dataset with targets for evaluation (HDF5)
    OperationDataset: Operational dataset without targets (HDF5)
    CSVBaseDataset: Base class for CSV event-based datasets
    CSVTrainDataset: Training dataset for CSV events
    CSVValidationDataset: Validation dataset for CSV events
    CSVTestDataset: Test dataset for CSV events

Functions:
    hours_to_index: Convert relative hours to array index
    compute_statistics: Compute and cache normalization statistics (HDF5)
    compute_statistics_csv: Compute and cache normalization statistics (CSV)
    create_dataloader: Create DataLoader for specified phase (auto-selects modality)
    verify_pipeline: Verify pipeline with config
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Union
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd


logger = logging.getLogger(__name__)


# =============================================================================
# Time-based Indexing Utilities
# =============================================================================

def hours_to_index(
    hours: int,
    interval: int,
    base_offset: int = -168
) -> int:
    """Convert relative hours to array index.

    Args:
        hours: Relative hours from reference time (e.g., -96 for 4 days before)
        interval: Data interval in hours (e.g., 6 for SDO, 3 for OMNI)
        base_offset: Starting offset of the data in hours (default: -168 = -7 days)

    Returns:
        Array index

    Examples:
        >>> hours_to_index(-96, 6, -168)  # SDO: -96h with 6h interval
        12
        >>> hours_to_index(0, 6, -168)    # SDO: reference time
        28
        >>> hours_to_index(-96, 3, -168)  # OMNI: -96h with 3h interval
        24
        >>> hours_to_index(72, 3, -168)   # OMNI: +72h (3 days)
        80
    """
    return (hours - base_offset) // interval


def days_to_indices_omni(days: List[int], base_offset_hours: int = -168) -> Tuple[int, int]:
    """Convert day list to OMNI array indices.

    OMNI data uses 3-hour intervals (8 points per day).
    Days are numbered: -7 to -1 (input), 1 to 3 (target).
    Day 0 does not exist (reference time is boundary).

    Args:
        days: List of day numbers (negative for input, positive for target)
              Examples: [-7, -6, -5] or [1, 2, 3]
        base_offset_hours: Starting offset in hours (default: -168 = -7 days)

    Returns:
        Tuple of (start_index, end_index) for slicing

    Examples:
        >>> days_to_indices_omni([-7, -6, -5, -4, -3, -2, -1])  # All input days
        (0, 56)
        >>> days_to_indices_omni([1, 2, 3])  # All target days
        (56, 80)
        >>> days_to_indices_omni([-3, -2, -1])  # Last 3 input days
        (32, 56)
        >>> days_to_indices_omni([3])  # Day 3 only
        (72, 80)
    """
    POINTS_PER_DAY = 8  # 24h / 3h = 8

    # Convert days to indices
    # Day -7 -> index 0, Day -1 -> index 48
    # Day +1 -> index 56, Day +3 -> index 72
    indices = []
    for day in sorted(days):
        if day < 0:
            # Input days: day -7 = 0, day -1 = 48
            start_idx = (day + 7) * POINTS_PER_DAY
        elif day > 0:
            # Target days: day +1 = 56, day +3 = 72
            start_idx = 56 + (day - 1) * POINTS_PER_DAY
        else:
            raise ValueError("Day 0 is not valid. Use negative days for input, positive for target.")
        indices.append(start_idx)

    start_index = min(indices)
    # End index is start of last day + points per day
    last_day = sorted(days)[-1]
    if last_day < 0:
        end_index = (last_day + 7 + 1) * POINTS_PER_DAY
    else:
        end_index = 56 + last_day * POINTS_PER_DAY

    return start_index, end_index


def days_to_indices_sdo(days: List[int], base_offset_hours: int = -168) -> Tuple[int, int]:
    """Convert day list to SDO array indices.

    SDO data uses 6-hour intervals (4 points per day).
    Only negative days (input) are valid for SDO.

    Args:
        days: List of negative day numbers (e.g., [-7, -6, -5])
        base_offset_hours: Starting offset in hours (default: -168 = -7 days)

    Returns:
        Tuple of (start_index, end_index) for slicing

    Examples:
        >>> days_to_indices_sdo([-7, -6, -5, -4, -3, -2, -1])  # All 7 days
        (0, 28)
        >>> days_to_indices_sdo([-4, -3, -2, -1])  # Last 4 days
        (12, 28)
        >>> days_to_indices_sdo([-1])  # Last day only
        (24, 28)
    """
    POINTS_PER_DAY = 4  # 24h / 6h = 4

    # Validate: SDO only has input data (negative days)
    if any(d >= 0 for d in days):
        raise ValueError("SDO data only contains input days (negative values)")

    # Convert days to indices
    # Day -7 -> index 0, Day -1 -> index 24
    indices = []
    for day in sorted(days):
        start_idx = (day + 7) * POINTS_PER_DAY
        indices.append(start_idx)

    start_index = min(indices)
    last_day = sorted(days)[-1]
    end_index = (last_day + 7 + 1) * POINTS_PER_DAY

    return start_index, end_index


def get_sdo_indices(config) -> Tuple[int, int]:
    """Get SDO start and end indices from config.

    Supports three configuration methods (in priority order):
    1. Day-based: sampling.input_days list
    2. Time-based: data.sdo.start_hours/end_hours
    3. Legacy index-based: data.sdo_start_index/sdo_end_index

    Args:
        config: Hydra configuration object

    Returns:
        Tuple of (start_index, end_index)
    """
    # Priority 1: Day-based config (sampling.input_days)
    if hasattr(config, 'sampling') and hasattr(config.sampling, 'input_days'):
        input_days = list(config.sampling.input_days)
        return days_to_indices_sdo(input_days)

    # Priority 2: Time-based config (data.sdo.start_hours)
    if hasattr(config.data, 'sdo') and hasattr(config.data.sdo, 'start_hours'):
        interval = config.data.sdo.interval_hours
        base_offset = config.data.sdo.base_offset_hours
        start_idx = hours_to_index(config.data.sdo.start_hours, interval, base_offset)
        end_idx = hours_to_index(config.data.sdo.end_hours, interval, base_offset)
        return start_idx, end_idx

    # Priority 3: Legacy index-based config
    return config.data.sdo_start_index, config.data.sdo_end_index


def get_omni_input_indices(config) -> Tuple[int, int]:
    """Get OMNI input start and end indices from config.

    Supports three configuration methods (in priority order):
    1. Day-based: sampling.input_days list
    2. Time-based: data.omni.input.start_hours/end_hours
    3. Legacy index-based: data.input_start_index/input_end_index

    Args:
        config: Hydra configuration object

    Returns:
        Tuple of (start_index, end_index)
    """
    # Priority 1: Day-based config (sampling.input_days)
    if hasattr(config, 'sampling') and hasattr(config.sampling, 'input_days'):
        input_days = list(config.sampling.input_days)
        return days_to_indices_omni(input_days)

    # Priority 2: Time-based config (data.omni.input.start_hours)
    if hasattr(config.data, 'omni') and hasattr(config.data.omni, 'input'):
        interval = config.data.omni.interval_hours
        base_offset = config.data.omni.base_offset_hours
        start_idx = hours_to_index(config.data.omni.input.start_hours, interval, base_offset)
        end_idx = hours_to_index(config.data.omni.input.end_hours, interval, base_offset)
        return start_idx, end_idx

    # Priority 3: Legacy index-based config
    return config.data.input_start_index, config.data.input_end_index


def get_omni_target_indices(config) -> Tuple[int, int]:
    """Get OMNI target start and end indices from config.

    Supports three configuration methods (in priority order):
    1. Day-based: sampling.target_days list
    2. Time-based: data.omni.target.start_hours/end_hours
    3. Legacy index-based: data.target_start_index/target_end_index

    Args:
        config: Hydra configuration object

    Returns:
        Tuple of (start_index, end_index)
    """
    # Priority 1: Day-based config (sampling.target_days)
    if hasattr(config, 'sampling') and hasattr(config.sampling, 'target_days'):
        target_days = list(config.sampling.target_days)
        return days_to_indices_omni(target_days)

    # Priority 2: Time-based config (data.omni.target.start_hours)
    if hasattr(config.data, 'omni') and hasattr(config.data.omni, 'target'):
        interval = config.data.omni.interval_hours
        base_offset = config.data.omni.base_offset_hours
        start_idx = hours_to_index(config.data.omni.target.start_hours, interval, base_offset)
        end_idx = hours_to_index(config.data.omni.target.end_hours, interval, base_offset)
        return start_idx, end_idx

    # Priority 3: Legacy index-based config
    return config.data.target_start_index, config.data.target_end_index


def get_sdo_wavelengths(config) -> List[str]:
    """Get SDO wavelengths from config."""
    if hasattr(config.data, 'sdo') and hasattr(config.data.sdo, 'wavelengths'):
        return sorted(list(config.data.sdo.wavelengths))
    return sorted(list(config.data.wavelengths))


def get_input_variables(config) -> List[str]:
    """Get OMNI input variables from config."""
    if hasattr(config.data, 'omni') and hasattr(config.data.omni, 'input'):
        return sorted(list(config.data.omni.input.variables))
    return sorted(list(config.data.input_variables))


def get_target_variables(config) -> List[str]:
    """Get OMNI target variables from config."""
    if hasattr(config.data, 'omni') and hasattr(config.data.omni, 'target'):
        return sorted(list(config.data.omni.target.variables))
    return sorted(list(config.data.target_variables))


def get_sdo_image_size(config) -> int:
    """Get SDO image size from config."""
    if hasattr(config.data, 'sdo') and hasattr(config.data.sdo, 'image_size'):
        return config.data.sdo.image_size
    return config.data.sdo_image_size


def is_timeseries_mode(config) -> bool:
    """Check if CSV timeseries modality is active."""
    return getattr(config.data.modalities, 'timeseries', False)


def get_timeseries_config(config):
    """Get timeseries-specific config section."""
    return config.data.timeseries


def get_timeseries_input_variables(config) -> List[str]:
    """Get input variables for CSV timeseries mode."""
    return list(config.data.timeseries.input_variables)


def get_timeseries_target_variables(config) -> List[str]:
    """Get target variables for CSV timeseries mode."""
    return list(config.data.timeseries.target_variables)


def get_timeseries_normalization_config(config) -> Optional[Dict]:
    """Get normalization config for CSV timeseries mode."""
    ts_cfg = config.data.timeseries
    if hasattr(ts_cfg, 'normalization'):
        norm_config = dict(ts_cfg.normalization)
        if 'methods' in norm_config:
            norm_config['methods'] = dict(norm_config['methods'])
        return norm_config
    return None


# =============================================================================
# HDF5 Reader
# =============================================================================

class HDF5Reader:
    """Reader for HDF5 data files containing SDO and OMNI data."""

    @staticmethod
    def read(
        file_path: str,
        sdo_wavelengths: List[str],
        omni_variables: List[str],
        validate: bool = True
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Read SDO and OMNI data from HDF5 file.

        Args:
            file_path: Path to HDF5 file
            sdo_wavelengths: List of SDO wavelength names
            omni_variables: List of OMNI variable names
            validate: If True, validate data for NaN/Inf values

        Returns:
            Tuple of (sdo_data dict, omni_data dict)

        Raises:
            FileNotFoundError: If file does not exist
            KeyError: If required dataset is not found
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        sdo_data = {}
        omni_data = {}

        with h5py.File(file_path, 'r') as f:
            # Read SDO data
            for wavelength in sdo_wavelengths:
                dataset_name = f"sdo/{wavelength}"
                if dataset_name not in f:
                    raise KeyError(f"SDO wavelength {wavelength} not found in {file_path}")
                data = f[dataset_name][:]
                if validate:
                    HDF5Reader._validate_data(data, dataset_name, file_path)
                sdo_data[wavelength] = data

            # Read OMNI data
            for variable in omni_variables:
                dataset_name = f"omni/{variable}"
                if dataset_name not in f:
                    raise KeyError(f"OMNI variable {variable} not found in {file_path}")
                data = f[dataset_name][:]
                if validate:
                    HDF5Reader._validate_data(data, dataset_name, file_path)
                omni_data[variable] = data

        return sdo_data, omni_data

    @staticmethod
    def _validate_data(data: np.ndarray, name: str, file_path: str) -> None:
        """Validate data for NaN and Inf values.

        Args:
            data: Data array to validate
            name: Dataset name for logging
            file_path: File path for logging
        """
        nan_count = np.sum(np.isnan(data))
        inf_count = np.sum(np.isinf(data))

        if nan_count > 0:
            logger.warning(
                f"Found {nan_count} NaN values in {name} from {file_path}"
            )
        if inf_count > 0:
            logger.warning(
                f"Found {inf_count} Inf values in {name} from {file_path}"
            )


# =============================================================================
# CSV Event Reader
# =============================================================================

class CSVEventReader:
    """Reader for CSV event files containing 30-min solar wind time series.

    Each CSV file represents one event window with columns:
    datetime, v_avg, v_min, v_max, ..., ap30, hp30

    See DATASET_GUIDE.md for full variable descriptions.
    """

    @staticmethod
    def read(
        file_path: str,
        variables: Optional[List[str]] = None,
        validate: bool = True
    ) -> np.ndarray:
        """Read solar wind time series from CSV event file.

        Args:
            file_path: Path to CSV file
            variables: List of column names to read. If None, reads all
                      numeric columns (excludes datetime).
            validate: If True, validate data for NaN/Inf values

        Returns:
            numpy array of shape (T, V) where T=timesteps, V=variables

        Raises:
            FileNotFoundError: If file does not exist
            KeyError: If requested variable not found in CSV
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)

        # Drop datetime column if present
        if 'datetime' in df.columns:
            df = df.drop(columns=['datetime'])

        if variables is not None:
            missing = set(variables) - set(df.columns)
            if missing:
                raise KeyError(f"Variables not found in {file_path}: {missing}")
            df = df[variables]

        data = df.values.astype(np.float32)

        if validate:
            CSVEventReader._validate_data(data, file_path)

        return data

    @staticmethod
    def _validate_data(data: np.ndarray, file_path: str) -> None:
        """Validate data for NaN and Inf values."""
        nan_count = np.sum(np.isnan(data))
        inf_count = np.sum(np.isinf(data))

        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in {file_path}")
        if inf_count > 0:
            logger.warning(f"Found {inf_count} Inf values in {file_path}")


# =============================================================================
# Normalizer
# =============================================================================

class Normalizer:
    """Multi-method normalizer for SDO images and OMNI variables.

    Supports different normalization methods per variable:
    - zscore: (x - mean) / std
    - log_zscore: log(x) -> z-score (positive values only)
    - log1p_zscore: log(1 + x) -> z-score (non-negative values)
    - minmax: (x - min) / (max - min)
    """

    VALID_METHODS = {'zscore', 'log_zscore', 'log1p_zscore', 'minmax'}

    def __init__(
        self,
        stat_dict: Optional[Dict[str, Dict[str, float]]] = None,
        method_config: Optional[Dict] = None
    ):
        """Initialize normalizer.

        Args:
            stat_dict: Dictionary of statistics for each variable
                       Format: {variable: {'mean', 'std', 'log_mean', 'log_std', ...}}
            method_config: Normalization method configuration from config.data.normalization
                          Format: {'default': str, 'methods': {variable: method}}
        """
        self.stat_dict = stat_dict or {}
        self.method_config = method_config or {}
        self.default_method = self.method_config.get('default', 'zscore')

    def get_method(self, variable: str) -> str:
        """Get normalization method for a variable.

        Args:
            variable: Variable name

        Returns:
            Normalization method name
        """
        methods = self.method_config.get('methods', {})
        return methods.get(variable, self.default_method)

    def normalize_sdo(self, data: np.ndarray) -> np.ndarray:
        """Normalize SDO image data from [0, 255] to [-1, 1].

        Args:
            data: 8-bit image data

        Returns:
            Normalized data in range [-1, 1]
        """
        return (data * (2.0 / 255.0)) - 1.0

    def normalize_omni(self, data: np.ndarray, variable: str) -> np.ndarray:
        """Normalize OMNI data using variable-specific method.

        Args:
            data: Raw OMNI data
            variable: Variable name for statistics and method lookup

        Returns:
            Normalized data

        Raises:
            KeyError: If statistics not found for variable
            ValueError: If unknown normalization method
        """
        if variable not in self.stat_dict:
            raise KeyError(f"Statistics not found for variable: {variable}")

        stats = self.stat_dict[variable]
        method = self.get_method(variable)

        if method == 'zscore':
            mean = stats['mean']
            std = stats['std']
            return (data - mean) / (std + 1e-8)

        elif method == 'log_zscore':
            # Clip to avoid log(0), use small epsilon
            data_clipped = np.maximum(data, 1e-6)
            log_data = np.log(data_clipped)
            log_mean = stats.get('log_mean', 0.0)
            log_std = stats.get('log_std', 1.0)
            return (log_data - log_mean) / (log_std + 1e-8)

        elif method == 'log1p_zscore':
            # Safe for 0 and positive values
            log1p_data = np.log1p(np.maximum(data, 0))
            log1p_mean = stats.get('log1p_mean', 0.0)
            log1p_std = stats.get('log1p_std', 1.0)
            return (log1p_data - log1p_mean) / (log1p_std + 1e-8)

        elif method == 'minmax':
            min_val = stats.get('min', 0.0)
            max_val = stats.get('max', 1.0)
            return (data - min_val) / (max_val - min_val + 1e-8)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def denormalize_omni(self, data: np.ndarray, variable: str) -> np.ndarray:
        """Denormalize OMNI data back to original scale.

        Args:
            data: Normalized data
            variable: Variable name for statistics and method lookup

        Returns:
            Data in original scale

        Raises:
            KeyError: If statistics not found for variable
            ValueError: If unknown normalization method
        """
        if variable not in self.stat_dict:
            raise KeyError(f"Statistics not found for variable: {variable}")

        stats = self.stat_dict[variable]
        method = self.get_method(variable)

        if method == 'zscore':
            mean = stats['mean']
            std = stats['std']
            return data * std + mean

        elif method == 'log_zscore':
            log_mean = stats.get('log_mean', 0.0)
            log_std = stats.get('log_std', 1.0)
            log_data = data * log_std + log_mean
            return np.exp(log_data)

        elif method == 'log1p_zscore':
            log1p_mean = stats.get('log1p_mean', 0.0)
            log1p_std = stats.get('log1p_std', 1.0)
            log1p_data = data * log1p_std + log1p_mean
            return np.expm1(log1p_data)  # exp(x) - 1

        elif method == 'minmax':
            min_val = stats.get('min', 0.0)
            max_val = stats.get('max', 1.0)
            return data * (max_val - min_val) + min_val

        else:
            raise ValueError(f"Unknown normalization method: {method}")


# =============================================================================
# Online Statistics Computation
# =============================================================================

class OnlineStatistics:
    """Compute statistics for multiple normalization methods using Welford's algorithm.

    Memory efficient - O(1) space complexity regardless of data size.
    Computes statistics for: zscore, log_zscore, log1p_zscore, minmax.
    """

    def __init__(self):
        """Initialize statistics counters for all methods."""
        # Z-score statistics
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

        # Log transform statistics (for positive values only)
        self.n_log = 0
        self.log_mean = 0.0
        self.log_M2 = 0.0

        # Log1p transform statistics (for non-negative values)
        self.n_log1p = 0
        self.log1p_mean = 0.0
        self.log1p_M2 = 0.0

        # Min-max statistics
        self.min_val = float('inf')
        self.max_val = float('-inf')

    def update(self, batch: np.ndarray) -> None:
        """Update statistics with a new batch of data.

        Args:
            batch: Data array of any shape
        """
        values = batch.flatten()
        valid_values = values[np.isfinite(values)]

        for x in valid_values:
            # Z-score: all values
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2

            # Min-max
            self.min_val = min(self.min_val, x)
            self.max_val = max(self.max_val, x)

            # Log transform: positive values only
            if x > 0:
                log_x = np.log(x)
                self.n_log += 1
                delta_log = log_x - self.log_mean
                self.log_mean += delta_log / self.n_log
                delta2_log = log_x - self.log_mean
                self.log_M2 += delta_log * delta2_log

            # Log1p transform: non-negative values
            if x >= 0:
                log1p_x = np.log1p(x)
                self.n_log1p += 1
                delta_log1p = log1p_x - self.log1p_mean
                self.log1p_mean += delta_log1p / self.n_log1p
                delta2_log1p = log1p_x - self.log1p_mean
                self.log1p_M2 += delta_log1p * delta2_log1p

    def _compute_std(self, M2: float, n: int) -> float:
        """Compute standard deviation from M2 and count."""
        if n < 2:
            return 1.0  # Fallback for insufficient data
        return float(np.sqrt(M2 / n))

    @property
    def std(self) -> float:
        """Return standard deviation for z-score."""
        return self._compute_std(self.M2, self.n)

    @property
    def log_std(self) -> float:
        """Return standard deviation for log transform."""
        return self._compute_std(self.log_M2, self.n_log)

    @property
    def log1p_std(self) -> float:
        """Return standard deviation for log1p transform."""
        return self._compute_std(self.log1p_M2, self.n_log1p)

    def get_stats(self) -> Dict[str, float]:
        """Return statistics as dictionary for all methods."""
        return {
            # Z-score
            'mean': float(self.mean),
            'std': self.std,
            # Log transform
            'log_mean': float(self.log_mean),
            'log_std': self.log_std,
            # Log1p transform
            'log1p_mean': float(self.log1p_mean),
            'log1p_std': self.log1p_std,
            # Min-max
            'min': float(self.min_val) if self.min_val != float('inf') else 0.0,
            'max': float(self.max_val) if self.max_val != float('-inf') else 1.0,
            # Sample counts
            'n': self.n,
            'n_log': self.n_log,
            'n_log1p': self.n_log1p,
        }


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


# =============================================================================
# Dynamic Undersampling Sampler
# =============================================================================

class BalancedUndersamplerSampler(Sampler):
    """Custom sampler that re-samples negatives each epoch for balanced training.

    Keeps ALL samples in the dataset. Each epoch, selects all positives
    plus an equal number of randomly chosen negatives (1:1 ratio).
    Compatible with persistent_workers since the dataset is unchanged.

    Args:
        labels: List of integer labels (0=negative, 1=positive) aligned
            with dataset indices.
        seed: Base random seed for reproducibility.
    """

    def __init__(self, labels: List[int], seed: int = 42):
        """Initialize sampler with label information.

        Args:
            labels: List of integer labels for each dataset sample.
            seed: Base random seed. Actual seed per epoch = seed + epoch.
        """
        self.positive_indices = [i for i, lbl in enumerate(labels) if lbl != 0]
        self.negative_indices = [i for i, lbl in enumerate(labels) if lbl == 0]
        self.num_negatives_per_epoch = len(self.positive_indices)  # 1:1 ratio
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set epoch number for deterministic resampling.

        Must be called before each epoch to get a new negative sample.

        Args:
            epoch: Current epoch number.
        """
        self.epoch = epoch

    def __iter__(self):
        """Yield shuffled indices: all positives + sampled negatives."""
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)

        # All positives always included
        pos = list(self.positive_indices)

        # Random subset of negatives (1:1 ratio with positives)
        n_neg = min(self.num_negatives_per_epoch, len(self.negative_indices))
        neg_perm = torch.randperm(len(self.negative_indices), generator=rng).tolist()
        neg = [self.negative_indices[i] for i in neg_perm[:n_neg]]

        # Combine and shuffle
        indices = pos + neg
        order = torch.randperm(len(indices), generator=rng).tolist()
        indices = [indices[i] for i in order]

        return iter(indices)

    def __len__(self) -> int:
        """Return total number of samples per epoch."""
        n_neg = min(self.num_negatives_per_epoch, len(self.negative_indices))
        return len(self.positive_indices) + n_neg


# =============================================================================
# Dataset Classes
# =============================================================================

class BaseDataset(Dataset):
    """Abstract base class for solar wind datasets.

    Handles common initialization including:
    - Configuration parsing
    - File list loading
    - Statistics computation
    - Normalizer setup
    """

    def __init__(self, config):
        """Initialize base dataset.

        Args:
            config: Hydra configuration object
        """
        self.config = config
        self.data_root = config.environment.data_root
        self.dataset_name = config.data.dataset_name
        self.dataset_suffix = config.data.dataset_suffix

        # SDO settings (with time-based support)
        self.sdo_wavelengths = get_sdo_wavelengths(config)
        self.sdo_start_index, self.sdo_end_index = get_sdo_indices(config)

        # OMNI input settings
        self.input_variables = get_input_variables(config)
        self.input_start_index, self.input_end_index = get_omni_input_indices(config)

        # OMNI target settings
        self.target_variables = get_target_variables(config)
        self.target_start_index, self.target_end_index = get_omni_target_indices(config)

        # Combined OMNI variables for reading
        self.omni_variables = list(set(
            self.input_variables + self.target_variables
        ))

        # Target days for label computation
        self.target_days = list(config.sampling.target_days)

        # Load file lists
        # CSV naming: {dataset_name}_{dataset_suffix}_{phase}.csv
        # Example: original_64_full_train.csv
        train_list_path = os.path.join(
            self.data_root,
            f"{self.dataset_name}_{self.dataset_suffix}_train.csv"
        )
        train_file_name, train_file_class = self._load_file_list(train_list_path)
        self.train_list = list(zip(train_file_name, train_file_class))

        validation_list_path = os.path.join(
            self.data_root,
            f"{self.dataset_name}_{self.dataset_suffix}_validation.csv"
        )
        validation_file_name, validation_file_class = self._load_file_list(validation_list_path)
        self.validation_list = list(zip(validation_file_name, validation_file_class))

        # Compute/load statistics
        statistics_file_path = os.path.join(
            self.data_root,
            f"{self.dataset_name}_{self.dataset_suffix}_stats.pkl"
        )
        self.stat_dict = compute_statistics(
            stat_file_path=statistics_file_path,
            data_root=self.data_root,
            data_dir_name=self.dataset_name,
            data_file_list=train_file_name,
            variables=self.omni_variables,
            overwrite=False
        )

        # Get normalization method config if available
        norm_config = None
        if hasattr(config.data, 'normalization'):
            norm_config = dict(config.data.normalization)
            if 'methods' in norm_config:
                norm_config['methods'] = dict(norm_config['methods'])

        self.normalizer = Normalizer(
            stat_dict=self.stat_dict,
            method_config=norm_config
        )

        # Input days for reference
        self.input_days = list(config.sampling.input_days)

        # Log configuration
        logger.info(f"Input days: {self.input_days}, Target days: {self.target_days}")
        logger.info(f"SDO: wavelengths={self.sdo_wavelengths}, "
                    f"indices=[{self.sdo_start_index}:{self.sdo_end_index}] "
                    f"({self.sdo_end_index - self.sdo_start_index} timesteps)")
        logger.info(f"OMNI input: variables={len(self.input_variables)}, "
                    f"indices=[{self.input_start_index}:{self.input_end_index}] "
                    f"({self.input_end_index - self.input_start_index} timesteps)")
        logger.info(f"OMNI target: variables={len(self.target_variables)}, "
                    f"indices=[{self.target_start_index}:{self.target_end_index}] "
                    f"({self.target_end_index - self.target_start_index} timesteps)")

    def _load_file_list(self, csv_path: str) -> Tuple[List[str], List[int]]:
        """Load file list from CSV.

        Args:
            csv_path: Path to CSV file

        Returns:
            Tuple of (file_names, labels)
        """
        df = pd.read_csv(csv_path)
        file_names = df['file_name'].tolist()

        # Compute labels as max across target days
        # CSV format: class_day_1, class_day_2, class_day_3
        list_labels = {}
        for day in self.target_days:
            key = f'class_day_{day}'
            list_labels[key] = df[key].tolist()

        labels = []
        for idx in range(len(file_names)):
            day_labels = [
                list_labels[f'class_day_{day}'][idx]
                for day in self.target_days
            ]
            labels.append(max(day_labels))

        return file_names, labels

    def process_sdo(self, sdo_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Process SDO image data.

        Args:
            sdo_data: Dictionary of wavelength -> image array
                      Each array has shape (T, H, W)

        Returns:
            Processed array of shape (C, T, H, W)
        """
        sdo_arrays = []
        for wavelength in self.sdo_wavelengths:
            # Shape: (T, H, W)
            data = sdo_data[wavelength]
            # Select timesteps first
            data = data[self.sdo_start_index:self.sdo_end_index]
            # Normalize to [-1, 1]
            data = self.normalizer.normalize_sdo(data)
            sdo_arrays.append(data)

        # Stack along new axis: (C, T, H, W)
        sdo_array = np.stack(sdo_arrays, axis=0)
        return sdo_array

    def process_omni_input(self, omni_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Process OMNI input data.

        Args:
            omni_data: Dictionary of variable -> data array

        Returns:
            Processed array of shape (T, V)
        """
        omni_arrays = []
        for variable in self.input_variables:
            data = omni_data[variable]
            data = self.normalizer.normalize_omni(data, variable)
            omni_arrays.append(data)

        # Stack: (T, V)
        omni_array = np.stack(omni_arrays, axis=-1)
        # Select timesteps
        omni_array = omni_array[self.input_start_index:self.input_end_index]
        return omni_array

    def process_omni_target(self, omni_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Process OMNI target data.

        Args:
            omni_data: Dictionary of variable -> data array

        Returns:
            Processed array of shape (T, V)
        """
        omni_arrays = []
        for variable in self.target_variables:
            data = omni_data[variable]
            data = self.normalizer.normalize_omni(data, variable)
            omni_arrays.append(data)

        # Stack: (T, V)
        omni_array = np.stack(omni_arrays, axis=-1)
        # Select timesteps
        omni_array = omni_array[self.target_start_index:self.target_end_index]
        return omni_array


class TrainDataset(BaseDataset):
    """Training dataset with undersampling support."""

    def __init__(self, config):
        """Initialize training dataset.

        Args:
            config: Hydra configuration object
        """
        super().__init__(config)

        self.enable_undersampling = config.sampling.enable_undersampling
        self.undersampling_mode = getattr(
            config.sampling, 'undersampling_mode', 'static'
        )
        self.num_subsample = config.sampling.num_subsamples
        self.subsample_index = config.sampling.subsample_index

        if self.enable_undersampling and self.undersampling_mode == 'static':
            # Static k-fold undersampling (legacy)
            seed = config.experiment.seed
            subsample, num_pos, num_neg = undersample(
                self.train_list,
                num_subsample=self.num_subsample,
                subsample_index=self.subsample_index,
                seed=seed
            )
            self.file_list = subsample
            logger.info(
                f"Static undersampling: {num_pos} positive, {num_neg} negative, "
                f"fold {self.subsample_index}/{self.num_subsample}, seed={seed}"
            )
        elif self.enable_undersampling and self.undersampling_mode == 'dynamic':
            # Dynamic undersampling: keep all data, sampler controls per-epoch
            self.file_list = self.train_list
            positive, negative = split_by_class(self.train_list)
            logger.info(
                f"Dynamic undersampling: {len(positive)} positive, "
                f"{len(negative)} negative (sampler balances each epoch)"
            )
        else:
            self.file_list = self.train_list

        self.num = len(self.file_list)
        logger.info(f"TrainDataset: {self.num} samples")

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        file_name, label = self.file_list[idx]
        file_path = os.path.join(
            self.data_root,
            self.dataset_name,
            file_name
        )

        sdo_data, omni_data = HDF5Reader.read(
            file_path,
            self.sdo_wavelengths,
            self.omni_variables
        )

        sdo_array = self.process_sdo(sdo_data)
        input_array = self.process_omni_input(omni_data)
        target_array = self.process_omni_target(omni_data)
        label_array = np.array([[label]], dtype=np.float32)

        return {
            'sdo': torch.tensor(sdo_array, dtype=torch.float32),
            'inputs': torch.tensor(input_array, dtype=torch.float32),
            'targets': torch.tensor(target_array, dtype=torch.float32),
            'labels': torch.tensor(label_array, dtype=torch.float32),
            'file_names': file_name
        }


class ValidationDataset(BaseDataset):
    """Validation dataset."""

    def __init__(self, config):
        """Initialize validation dataset.

        Args:
            config: Hydra configuration object
        """
        super().__init__(config)
        self.file_list = self.validation_list
        self.num = len(self.file_list)
        logger.info(f"ValidationDataset: {self.num} samples")

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        file_name, label = self.file_list[idx]
        file_path = os.path.join(
            self.data_root,
            self.dataset_name,
            file_name
        )

        sdo_data, omni_data = HDF5Reader.read(
            file_path,
            self.sdo_wavelengths,
            self.omni_variables
        )

        sdo_array = self.process_sdo(sdo_data)
        input_array = self.process_omni_input(omni_data)
        target_array = self.process_omni_target(omni_data)
        label_array = np.array([[label]], dtype=np.float32)

        return {
            'sdo': torch.tensor(sdo_array, dtype=torch.float32),
            'inputs': torch.tensor(input_array, dtype=torch.float32),
            'targets': torch.tensor(target_array, dtype=torch.float32),
            'labels': torch.tensor(label_array, dtype=torch.float32),
            'file_names': file_name
        }


class TestDataset(BaseDataset):
    """Test dataset with targets for evaluation."""

    def __init__(self, config):
        """Initialize test dataset.

        Args:
            config: Hydra configuration object
        """
        super().__init__(config)

        # Load test file list
        # CSV naming: {dataset_name}_{dataset_suffix}_{phase}.csv
        test_list_path = os.path.join(
            self.data_root,
            f"{self.dataset_name}_{self.dataset_suffix}_test.csv"
        )

        if os.path.exists(test_list_path):
            test_file_name, test_file_class = self._load_file_list(test_list_path)
            self.file_list = list(zip(test_file_name, test_file_class))
        else:
            # Fallback to validation list if test list doesn't exist
            logger.warning(f"Test list not found at {test_list_path}, using validation list")
            self.file_list = self.validation_list

        self.num = len(self.file_list)
        logger.info(f"TestDataset: {self.num} samples")

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        file_name, label = self.file_list[idx]
        file_path = os.path.join(
            self.data_root,
            self.dataset_name,
            file_name
        )

        sdo_data, omni_data = HDF5Reader.read(
            file_path,
            self.sdo_wavelengths,
            self.omni_variables
        )

        sdo_array = self.process_sdo(sdo_data)
        input_array = self.process_omni_input(omni_data)
        target_array = self.process_omni_target(omni_data)
        label_array = np.array([[label]], dtype=np.float32)

        return {
            'sdo': torch.tensor(sdo_array, dtype=torch.float32),
            'inputs': torch.tensor(input_array, dtype=torch.float32),
            'targets': torch.tensor(target_array, dtype=torch.float32),
            'labels': torch.tensor(label_array, dtype=torch.float32),
            'file_names': file_name
        }


class OperationDataset(Dataset):
    """Operational dataset for real-time inference without targets.

    This dataset is used for making predictions on new data where
    ground truth targets are not available.
    """

    def __init__(
        self,
        config,
        file_paths: Optional[List[str]] = None,
        data_dir: Optional[str] = None,
        stats_path: Optional[str] = None
    ):
        """Initialize operational dataset.

        Args:
            config: Hydra configuration object
            file_paths: List of HDF5 file paths (mutually exclusive with data_dir)
            data_dir: Directory containing HDF5 files (mutually exclusive with file_paths)
            stats_path: Path to statistics pickle file (uses default if None)
        """
        self.config = config

        # SDO settings
        self.sdo_wavelengths = get_sdo_wavelengths(config)
        self.sdo_start_index, self.sdo_end_index = get_sdo_indices(config)

        # OMNI input settings (no targets for operation)
        self.input_variables = get_input_variables(config)
        self.input_start_index, self.input_end_index = get_omni_input_indices(config)

        # Get file list
        if file_paths is not None:
            self.file_paths = file_paths
        elif data_dir is not None:
            self.file_paths = [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith('.h5')
            ]
            self.file_paths.sort()
        else:
            raise ValueError("Either file_paths or data_dir must be provided")

        self.num = len(self.file_paths)

        # Load statistics
        # Stats file naming: {dataset_name}_{dataset_suffix}_stats.pkl
        if stats_path is None:
            stats_path = os.path.join(
                config.environment.data_root,
                f"{config.data.dataset_name}_{config.data.dataset_suffix}_stats.pkl"
            )

        if os.path.exists(stats_path):
            with open(stats_path, 'rb') as f:
                self.stat_dict = pickle.load(f)
            logger.info(f"Loaded statistics from {stats_path}")
        else:
            raise FileNotFoundError(f"Statistics file not found: {stats_path}")

        # Get normalization method config if available
        norm_config = None
        if hasattr(config.data, 'normalization'):
            norm_config = dict(config.data.normalization)
            if 'methods' in norm_config:
                norm_config['methods'] = dict(norm_config['methods'])

        self.normalizer = Normalizer(
            stat_dict=self.stat_dict,
            method_config=norm_config
        )
        logger.info(f"OperationDataset: {self.num} samples")

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        file_name = os.path.basename(file_path)

        # Read only SDO and input variables (no targets)
        sdo_data = {}
        omni_data = {}

        with h5py.File(file_path, 'r') as f:
            for wavelength in self.sdo_wavelengths:
                dataset_name = f"sdo/{wavelength}"
                sdo_data[wavelength] = f[dataset_name][:]

            for variable in self.input_variables:
                dataset_name = f"omni/{variable}"
                omni_data[variable] = f[dataset_name][:]

        # Process SDO
        sdo_arrays = []
        for wavelength in self.sdo_wavelengths:
            # Shape: (T, H, W)
            data = sdo_data[wavelength]
            # Select timesteps first
            data = data[self.sdo_start_index:self.sdo_end_index]
            # Normalize to [-1, 1]
            data = self.normalizer.normalize_sdo(data)
            sdo_arrays.append(data)
        # Stack along new axis: (C, T, H, W)
        sdo_array = np.stack(sdo_arrays, axis=0)

        # Process OMNI input
        omni_arrays = []
        for variable in self.input_variables:
            data = self.normalizer.normalize_omni(omni_data[variable], variable)
            omni_arrays.append(data)
        input_array = np.stack(omni_arrays, axis=-1)
        input_array = input_array[self.input_start_index:self.input_end_index]

        # Extract timestamp from filename (assumed format: YYYYMMDDHH.h5)
        timestamp = file_name.replace('.h5', '')

        return {
            'sdo': torch.tensor(sdo_array, dtype=torch.float32),
            'inputs': torch.tensor(input_array, dtype=torch.float32),
            'file_names': file_name,
            'timestamps': timestamp
        }


# =============================================================================
# CSV Event Dataset Classes
# =============================================================================

class CSVBaseDataset(Dataset):
    """Base class for CSV event-based solar wind datasets.

    Reads individual CSV event files with 30-min interval time series data.
    Each file = one event, shape (T_total, V_all).
    Splits into input and target windows based on days_before/days_after config.
    """

    def __init__(self, config):
        """Initialize CSV base dataset.

        Args:
            config: Hydra configuration object with data.timeseries section
        """
        self.config = config
        self.data_root = config.environment.data_root
        ts_cfg = get_timeseries_config(config)

        self.dataset_dir = ts_cfg.dataset_dir
        self.file_list_suffix = ts_cfg.file_list_suffix
        self.points_per_day = ts_cfg.points_per_day
        self.days_before = ts_cfg.days_before
        self.days_after = ts_cfg.days_after

        # Compute split indices
        self.input_len = self.days_before * self.points_per_day
        self.target_len = self.days_after * self.points_per_day
        self.total_len = self.input_len + self.target_len

        # Variable lists
        self.input_variables = get_timeseries_input_variables(config)
        self.target_variables = get_timeseries_target_variables(config)
        self.all_variables = list(dict.fromkeys(
            self.input_variables + self.target_variables
        ))

        # Load file lists
        train_list_path = os.path.join(
            self.data_root,
            f"{self.dataset_dir}_{self.file_list_suffix}_train.csv"
        )
        self.train_list = self._load_file_list(train_list_path)

        validation_list_path = os.path.join(
            self.data_root,
            f"{self.dataset_dir}_{self.file_list_suffix}_validation.csv"
        )
        self.validation_list = self._load_file_list(validation_list_path)

        # Compute/load statistics from training data
        statistics_file_path = os.path.join(
            self.data_root,
            f"{self.dataset_dir}_{self.file_list_suffix}_stats.pkl"
        )
        train_csv_paths = [
            os.path.join(self.data_root, self.dataset_dir, fn)
            for fn, _ in self.train_list
        ]
        self.stat_dict = compute_statistics_csv(
            stat_file_path=statistics_file_path,
            csv_file_paths=train_csv_paths,
            variables=self.all_variables,
            overwrite=False
        )

        # Setup normalizer
        norm_config = get_timeseries_normalization_config(config)
        self.normalizer = Normalizer(
            stat_dict=self.stat_dict,
            method_config=norm_config
        )

        # Log configuration
        logger.info(f"CSV TimeSeries mode: days_before={self.days_before}, "
                    f"days_after={self.days_after}")
        logger.info(f"Input: {len(self.input_variables)} vars, "
                    f"{self.input_len} timesteps")
        logger.info(f"Target: {len(self.target_variables)} vars, "
                    f"{self.target_len} timesteps")

    def _load_file_list(self, csv_path: str) -> List[Tuple[str, int]]:
        """Load file list from CSV.

        Supports two formats:
        1. With class columns (class_day_1, class_day_2, ...): computes max label
        2. Simple format (file_name only or file_name,label): uses label or 0

        Args:
            csv_path: Path to file list CSV

        Returns:
            List of (filename, label) tuples
        """
        df = pd.read_csv(csv_path)

        if 'file_name' in df.columns:
            file_names = df['file_name'].tolist()
        else:
            # Assume first column is file names
            file_names = df.iloc[:, 0].tolist()

        # Try class_day columns first (legacy format)
        class_cols = [c for c in df.columns if c.startswith('class_day_')]
        if class_cols:
            labels = df[class_cols].max(axis=1).astype(int).tolist()
        elif 'label' in df.columns:
            labels = df['label'].astype(int).tolist()
        else:
            labels = [0] * len(file_names)

        return list(zip(file_names, labels))

    def _read_and_process(self, file_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read CSV file and split into normalized input/target arrays.

        Args:
            file_name: CSV filename

        Returns:
            Tuple of (input_array, target_array)
            input_array shape: (input_len, num_input_vars)
            target_array shape: (target_len, num_target_vars)
        """
        file_path = os.path.join(self.data_root, self.dataset_dir, file_name)
        raw = CSVEventReader.read(file_path, variables=self.all_variables)

        # Normalize each variable
        normalized_cols = []
        for i, var in enumerate(self.all_variables):
            normalized_cols.append(
                self.normalizer.normalize_omni(raw[:, i], var)
            )
        normalized = np.stack(normalized_cols, axis=-1)  # (T_total, V_all)

        # Build input array: select input variables, first input_len timesteps
        input_var_indices = [
            self.all_variables.index(v) for v in self.input_variables
        ]
        input_array = normalized[:self.input_len, :][:, input_var_indices]

        # Build target array: select target variables, last target_len timesteps
        target_var_indices = [
            self.all_variables.index(v) for v in self.target_variables
        ]
        target_array = normalized[self.input_len:, :][:, target_var_indices]

        return input_array.astype(np.float32), target_array.astype(np.float32)


class CSVTrainDataset(CSVBaseDataset):
    """Training dataset for CSV events with undersampling support."""

    def __init__(self, config):
        super().__init__(config)

        self.enable_undersampling = config.sampling.enable_undersampling
        self.undersampling_mode = getattr(
            config.sampling, 'undersampling_mode', 'static'
        )

        if self.enable_undersampling and self.undersampling_mode == 'static':
            seed = config.experiment.seed
            subsample, num_pos, num_neg = undersample(
                self.train_list,
                num_subsample=config.sampling.num_subsamples,
                subsample_index=config.sampling.subsample_index,
                seed=seed
            )
            self.file_list = subsample
            logger.info(
                f"Static undersampling: {num_pos} pos, {num_neg} neg, "
                f"fold {config.sampling.subsample_index}/{config.sampling.num_subsamples}"
            )
        elif self.enable_undersampling and self.undersampling_mode == 'dynamic':
            self.file_list = self.train_list
            positive, negative = split_by_class(self.train_list)
            logger.info(
                f"Dynamic undersampling: {len(positive)} pos, "
                f"{len(negative)} neg (sampler balances each epoch)"
            )
        else:
            self.file_list = self.train_list

        self.num = len(self.file_list)
        logger.info(f"CSVTrainDataset: {self.num} samples")

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        file_name, label = self.file_list[idx]
        input_array, target_array = self._read_and_process(file_name)
        label_array = np.array([[label]], dtype=np.float32)

        return {
            'inputs': torch.tensor(input_array, dtype=torch.float32),
            'targets': torch.tensor(target_array, dtype=torch.float32),
            'labels': torch.tensor(label_array, dtype=torch.float32),
            'file_names': file_name
        }


class CSVValidationDataset(CSVBaseDataset):
    """Validation dataset for CSV events."""

    def __init__(self, config):
        super().__init__(config)
        self.file_list = self.validation_list
        self.num = len(self.file_list)
        logger.info(f"CSVValidationDataset: {self.num} samples")

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        file_name, label = self.file_list[idx]
        input_array, target_array = self._read_and_process(file_name)
        label_array = np.array([[label]], dtype=np.float32)

        return {
            'inputs': torch.tensor(input_array, dtype=torch.float32),
            'targets': torch.tensor(target_array, dtype=torch.float32),
            'labels': torch.tensor(label_array, dtype=torch.float32),
            'file_names': file_name
        }


class CSVTestDataset(CSVBaseDataset):
    """Test dataset for CSV events."""

    def __init__(self, config):
        super().__init__(config)

        test_list_path = os.path.join(
            self.data_root,
            f"{self.dataset_dir}_{self.file_list_suffix}_test.csv"
        )

        if os.path.exists(test_list_path):
            self.file_list = self._load_file_list(test_list_path)
        else:
            logger.warning(f"Test list not found at {test_list_path}, using validation list")
            self.file_list = self.validation_list

        self.num = len(self.file_list)
        logger.info(f"CSVTestDataset: {self.num} samples")

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        file_name, label = self.file_list[idx]
        input_array, target_array = self._read_and_process(file_name)
        label_array = np.array([[label]], dtype=np.float32)

        return {
            'inputs': torch.tensor(input_array, dtype=torch.float32),
            'targets': torch.tensor(target_array, dtype=torch.float32),
            'labels': torch.tensor(label_array, dtype=torch.float32),
            'file_names': file_name
        }


# =============================================================================
# DataLoader Creation
# =============================================================================

def create_dataloader(config, phase: str = "train") -> DataLoader:
    """Create DataLoader for specified phase.

    Auto-selects CSV or HDF5 dataset classes based on config.data.modalities.

    Args:
        config: Hydra configuration object
        phase: One of "train", "validation", "test", "operation"

    Returns:
        DataLoader instance
    """
    phase = phase.upper()
    use_csv = is_timeseries_mode(config)

    if use_csv:
        if phase == "TRAIN":
            dataset = CSVTrainDataset(config)
        elif phase == "VALIDATION":
            dataset = CSVValidationDataset(config)
        elif phase == "TEST":
            dataset = CSVTestDataset(config)
        elif phase == "OPERATION":
            raise NotImplementedError(
                "CSV OperationDataset not yet implemented. "
                "Use HDF5 mode (modalities.timeseries=false, modalities.omni_hdf5=true)"
            )
        else:
            raise ValueError(f"Invalid phase: {phase}")
    else:
        if phase == "TRAIN":
            dataset = TrainDataset(config)
        elif phase == "VALIDATION":
            dataset = ValidationDataset(config)
        elif phase == "TEST":
            dataset = TestDataset(config)
        elif phase == "OPERATION":
            dataset = OperationDataset(config)
        else:
            raise ValueError(f"Invalid phase: {phase}")

    # Dynamic undersampling: use custom sampler instead of shuffle
    sampler = None
    shuffle = (phase == "TRAIN")

    if (phase == "TRAIN"
            and config.sampling.enable_undersampling
            and getattr(config.sampling, 'undersampling_mode', 'static') == 'dynamic'):
        labels = [label for _, label in dataset.file_list]
        sampler = BalancedUndersamplerSampler(
            labels=labels,
            seed=config.experiment.seed,
        )
        shuffle = False  # sampler and shuffle are mutually exclusive
        logger.info(
            f"BalancedUndersamplerSampler: "
            f"{len(sampler.positive_indices)} pos, "
            f"{len(sampler.negative_indices)} neg, "
            f"{sampler.num_negatives_per_epoch} neg/epoch (1:1 ratio)"
        )

    dataloader = DataLoader(
        dataset,
        batch_size=config.experiment.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.environment.num_workers,
        pin_memory=(config.environment.device == 'cuda'),
        drop_last=False,
        persistent_workers=(config.environment.num_workers > 0),
    )

    logger.info(f"{phase} dataloader created")
    logger.info(f"# of samples: {len(dataset)}, # of batches: {len(dataloader)}")

    return dataloader


# =============================================================================
# Verification Function
# =============================================================================

def verify_pipeline(config) -> None:
    """Verify pipeline configuration and data loading.

    Args:
        config: Hydra configuration object
    """
    print("=" * 60)
    print("Pipeline Verification")
    print("=" * 60)

    use_csv = is_timeseries_mode(config)
    print(f"\n[Active Modality] {'CSV TimeSeries' if use_csv else 'HDF5 (SDO + OMNI)'}")

    if use_csv:
        ts_cfg = get_timeseries_config(config)
        print(f"\n[CSV TimeSeries Configuration]")
        print(f"Dataset dir: {ts_cfg.dataset_dir}")
        print(f"Interval: {ts_cfg.interval_minutes} min")
        print(f"Points per day: {ts_cfg.points_per_day}")
        print(f"Days before: {ts_cfg.days_before} ({ts_cfg.days_before * ts_cfg.points_per_day} timesteps)")
        print(f"Days after: {ts_cfg.days_after} ({ts_cfg.days_after * ts_cfg.points_per_day} timesteps)")
        print(f"Input variables: {len(get_timeseries_input_variables(config))}")
        print(f"Target variables: {get_timeseries_target_variables(config)}")
    else:
        # Legacy HDF5 mode
        print("\n[Day-based Configuration]")
        if hasattr(config, 'sampling'):
            input_days = list(config.sampling.input_days)
            target_days = list(config.sampling.target_days)
            print(f"Input days: {input_days}")
            print(f"Target days: {target_days}")

        print("\n[Computed Indices]")
        sdo_start, sdo_end = get_sdo_indices(config)
        input_start, input_end = get_omni_input_indices(config)
        target_start, target_end = get_omni_target_indices(config)

        print(f"SDO indices: [{sdo_start}:{sdo_end}] = {sdo_end - sdo_start} timesteps")
        print(f"OMNI input indices: [{input_start}:{input_end}] = {input_end - input_start} timesteps")
        print(f"OMNI target indices: [{target_start}:{target_end}] = {target_end - target_start} timesteps")

        print("\n[days_to_indices Test]")
        test_cases_omni = [
            ([-7, -6, -5, -4, -3, -2, -1], "All input days"),
            ([1, 2, 3], "All target days"),
            ([-3, -2, -1], "Last 3 input days"),
            ([3], "Day 3 only"),
        ]
        for days, desc in test_cases_omni:
            start, end = days_to_indices_omni(days)
            print(f"  OMNI {desc}: days={days} -> [{start}:{end}]")

        test_cases_sdo = [
            ([-7, -6, -5, -4, -3, -2, -1], "All 7 days"),
            ([-4, -3, -2, -1], "Last 4 days"),
        ]
        for days, desc in test_cases_sdo:
            start, end = days_to_indices_sdo(days)
            print(f"  SDO {desc}: days={days} -> [{start}:{end}]")

    # Try to create train dataloader
    print("\n[DataLoader Test]")
    try:
        train_loader = create_dataloader(config, "train")

        print("\nSample batch:")
        for batch in train_loader:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  {key}: type={type(value).__name__}, len={len(value)}")
            break

        print("\nPipeline verification completed successfully!")
    except Exception as e:
        print(f"\nError during verification: {e}")
        print("This may be expected if data files are not available.")

    print("=" * 60)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    @hydra.main(config_path="../configs", config_name="local", version_base=None)
    def main(config: DictConfig):
        verify_pipeline(config)

    main()
