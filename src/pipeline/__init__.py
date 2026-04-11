"""Data loading and processing pipeline for solar wind prediction.

This package provides dataset classes, normalization, and data loading
utilities organized by data source type (HDF5, CSV, Table/Parquet).
Use create_dataloader(config, phase) to instantiate data loaders by config.
"""

# Re-export all public items for backward compatibility
from .config_helpers import (
    hours_to_index, days_to_indices_omni, days_to_indices_sdo,
    get_sdo_indices, get_omni_input_indices, get_omni_target_indices,
    get_csv_window_indices, get_sdo_wavelengths, get_input_variables,
    get_target_variables, get_sdo_image_size, is_timeseries_mode,
    get_timeseries_config, get_timeseries_input_variables,
    get_timeseries_target_variables, get_timeseries_normalization_config
)
from .readers import HDF5Reader, CSVEventReader
from .normalizer import Normalizer, OnlineStatistics
from .statistics import (
    compute_statistics, compute_statistics_csv, compute_statistics_table,
    split_by_class, undersample
)
from .sampler import BalancedUndersamplerSampler
from .datasets_hdf5 import BaseDataset, TrainDataset, ValidationDataset, TestDataset, OperationDataset
from .datasets_csv import CSVBaseDataset, CSVTrainDataset, CSVValidationDataset, CSVTestDataset
from .datasets_table import TableBaseDataset, TableTrainDataset, TableValidationDataset, TableTestDataset
from .factory import create_dataloader, verify_pipeline
