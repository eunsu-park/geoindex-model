"""Data readers for HDF5 and CSV event files.

Provides static reader classes for loading solar wind data from
HDF5 files (SDO + OMNI) and CSV event files (30-min time series).
"""

import os
import logging
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
