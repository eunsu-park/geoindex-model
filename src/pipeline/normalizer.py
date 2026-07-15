"""Normalization utilities for solar wind data.

Provides multi-method normalization (zscore, log_zscore, log1p_zscore, minmax)
and online statistics computation using Welford's algorithm.
"""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# Normalization methods whose transform is defined only on non-negative values.
# log/log1p map a non-negative domain; zscore/minmax are sign-preserving. This is
# the single source of truth for deciding whether a denormalized prediction may be
# clipped at 0 (ap30/hp30) or must keep its sign (e.g. Dst/SYM-H).
NONNEGATIVE_METHODS = frozenset({'log_zscore', 'log1p_zscore'})


def method_is_nonnegative(method: str) -> bool:
    """Return True if a normalization method implies a non-negative value range.

    Args:
        method: Normalization method name ('zscore', 'log_zscore',
            'log1p_zscore', 'minmax').

    Returns:
        True for log/log1p methods (non-negative domain), False otherwise.
    """
    return method in NONNEGATIVE_METHODS


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

    def is_nonnegative(self, variable: str) -> bool:
        """Return True if the variable's normalization implies a non-negative range.

        log/log1p transforms are defined on non-negative values; zscore/minmax are
        not. Used to decide whether a denormalized prediction may be clipped at 0
        (ap30/hp30) or must keep its sign (e.g. Dst/SYM-H).

        Args:
            variable: Variable name.

        Returns:
            True for log/log1p-normalized variables, False otherwise.
        """
        return method_is_nonnegative(self.get_method(variable))

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
