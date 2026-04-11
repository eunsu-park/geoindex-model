"""CSV event-based dataset classes for solar wind prediction.

Provides dataset classes that load data from individual CSV event files,
each containing a 30-min interval time series window.
"""

import os
import logging
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

from .config_helpers import (
    get_timeseries_config, get_timeseries_input_variables,
    get_timeseries_target_variables, get_timeseries_normalization_config,
    get_csv_window_indices
)
from .readers import CSVEventReader
from .normalizer import Normalizer
from .statistics import compute_statistics_csv, split_by_class, undersample

logger = logging.getLogger(__name__)


class CSVBaseDataset(Dataset):
    """Base class for CSV event-based solar wind datasets.

    Reads individual CSV event files with 30-min interval time series data.
    Each file = one event, shape (T_total, V_all).
    Splits into input and target windows based on days_before/days_after config.
    """

    def __init__(self, config):
        """Initialize CSV base dataset.

        Supports two data organization modes:
        1. Directory mode: train_dir/validation_dir point to directories of CSV files
        2. File list mode: dataset_dir + file_list_suffix for file list CSVs

        Args:
            config: Hydra configuration object with data.timeseries section
        """
        self.config = config
        self.data_root = config.environment.data_root
        ts_cfg = get_timeseries_config(config)

        self.points_per_day = ts_cfg.points_per_day
        self.days_before = ts_cfg.days_before
        self.days_after = ts_cfg.days_after

        # Compute split indices (supports timestep-based windowing)
        i_s, i_e, t_s, t_e = get_csv_window_indices(ts_cfg)
        self.input_start_idx = i_s
        self.input_end_idx = i_e
        self.target_start_idx = t_s
        self.target_end_idx = t_e
        self.input_len = i_e - i_s
        self.target_len = t_e - t_s

        # Variable lists
        self.input_variables = get_timeseries_input_variables(config)
        self.target_variables = get_timeseries_target_variables(config)
        self.all_variables = list(dict.fromkeys(
            self.input_variables + self.target_variables
        ))

        # Determine data loading mode
        self.train_dir = getattr(ts_cfg, 'train_dir', None)
        self.validation_dir = getattr(ts_cfg, 'validation_dir', None)
        self.test_dir = getattr(ts_cfg, 'test_dir', None)

        if self.train_dir is not None:
            # Directory mode: scan directories for CSV files
            self._use_directory_mode = True
            train_dir_path = self._resolve_path(self.train_dir)
            val_dir_path = self._resolve_path(self.validation_dir)

            self.train_list = self._scan_directory(train_dir_path)
            self.validation_list = self._scan_directory(val_dir_path)

            # For statistics computation
            train_csv_paths = [
                os.path.join(train_dir_path, fn) for fn, _ in self.train_list
            ]

            # Statistics file alongside data
            statistics_file_path = os.path.join(
                self.data_root, "stats.pkl"
            )

            logger.info(f"Directory mode: train={train_dir_path} ({len(self.train_list)} files), "
                        f"validation={val_dir_path} ({len(self.validation_list)} files)")
        else:
            # File list mode (legacy)
            self._use_directory_mode = False
            self.dataset_dir = ts_cfg.dataset_dir
            self.file_list_suffix = ts_cfg.file_list_suffix

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

            train_csv_paths = [
                os.path.join(self.data_root, self.dataset_dir, fn)
                for fn, _ in self.train_list
            ]

            statistics_file_path = os.path.join(
                self.data_root,
                f"{self.dataset_dir}_{self.file_list_suffix}_stats.pkl"
            )

        # Compute/load statistics from training data
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
        logger.info(f"CSV TimeSeries window: "
                    f"input=[{self.input_start_idx}:{self.input_end_idx}] "
                    f"({self.input_len} steps), "
                    f"target=[{self.target_start_idx}:{self.target_end_idx}] "
                    f"({self.target_len} steps)")
        logger.info(f"Input: {len(self.input_variables)} vars, "
                    f"{self.input_len} timesteps")
        logger.info(f"Target: {len(self.target_variables)} vars, "
                    f"{self.target_len} timesteps")

    def _resolve_path(self, path: str) -> str:
        """Resolve a path that may be absolute or relative to data_root."""
        if os.path.isabs(path):
            return path
        return os.path.join(self.data_root, path)

    def _scan_directory(self, dir_path: str) -> List[Tuple[str, int]]:
        """Scan directory for CSV files and return as file list.

        Args:
            dir_path: Absolute path to directory containing CSV files

        Returns:
            Sorted list of (filename, label=0) tuples
        """
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Data directory not found: {dir_path}")

        csv_files = sorted([
            f for f in os.listdir(dir_path) if f.endswith('.csv')
        ])

        if not csv_files:
            raise ValueError(f"No CSV files found in {dir_path}")

        return [(f, 0) for f in csv_files]

    def _get_file_path(self, file_name: str, phase: str = "train") -> str:
        """Get full file path based on loading mode and phase.

        Args:
            file_name: CSV filename
            phase: "train", "validation", or "test"
        """
        if self._use_directory_mode:
            if phase == "train":
                return os.path.join(self._resolve_path(self.train_dir), file_name)
            elif phase == "validation":
                return os.path.join(self._resolve_path(self.validation_dir), file_name)
            elif phase == "test":
                test_dir = self.test_dir or self.validation_dir
                return os.path.join(self._resolve_path(test_dir), file_name)
        return os.path.join(self.data_root, self.dataset_dir, file_name)

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

    def _read_and_process(self, file_name: str, phase: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        """Read CSV file and split into normalized input/target arrays.

        Args:
            file_name: CSV filename
            phase: "train", "validation", or "test" (for directory mode path resolution)

        Returns:
            Tuple of (input_array, target_array)
            input_array shape: (input_len, num_input_vars)
            target_array shape: (target_len, num_target_vars)
        """
        file_path = self._get_file_path(file_name, phase)
        raw = CSVEventReader.read(file_path, variables=self.all_variables)

        # Normalize each variable
        normalized_cols = []
        for i, var in enumerate(self.all_variables):
            normalized_cols.append(
                self.normalizer.normalize_omni(raw[:, i], var)
            )
        normalized = np.stack(normalized_cols, axis=-1)  # (T_total, V_all)

        # Build input array: select input variables within window
        input_var_indices = [
            self.all_variables.index(v) for v in self.input_variables
        ]
        input_array = normalized[self.input_start_idx:self.input_end_idx, :][:, input_var_indices]

        # Build target array: select target variables within window
        target_var_indices = [
            self.all_variables.index(v) for v in self.target_variables
        ]
        target_array = normalized[self.target_start_idx:self.target_end_idx, :][:, target_var_indices]

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
        input_array, target_array = self._read_and_process(file_name, phase="train")

        # Data augmentation (training only)
        noise_std = getattr(
            self.config.data.timeseries.augmentation, 'gaussian_noise_std', 0.0
        )
        if noise_std > 0:
            input_array = input_array + np.random.normal(
                0, noise_std, input_array.shape
            ).astype(np.float32)

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
        input_array, target_array = self._read_and_process(file_name, phase="validation")
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

        if self._use_directory_mode:
            test_dir = self.test_dir or self.validation_dir
            test_dir_path = self._resolve_path(test_dir)
            self.file_list = self._scan_directory(test_dir_path)
        else:
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
        input_array, target_array = self._read_and_process(file_name, phase="test")
        label_array = np.array([[label]], dtype=np.float32)

        return {
            'inputs': torch.tensor(input_array, dtype=torch.float32),
            'targets': torch.tensor(target_array, dtype=torch.float32),
            'labels': torch.tensor(label_array, dtype=torch.float32),
            'file_names': file_name
        }
