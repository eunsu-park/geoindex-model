"""HDF5-based dataset classes for solar wind prediction.

Provides dataset classes that load data from HDF5 files containing
SDO images and OMNI time series data.
"""

import os
import logging
import pickle
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

from .config_helpers import (
    get_sdo_wavelengths, get_sdo_indices, get_input_variables,
    get_omni_input_indices, get_target_variables, get_omni_target_indices
)
from .readers import HDF5Reader
from .normalizer import Normalizer
from .statistics import compute_statistics, split_by_class, undersample

logger = logging.getLogger(__name__)


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
