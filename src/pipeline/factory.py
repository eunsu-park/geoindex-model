"""Factory functions for creating data loaders and verifying pipeline configuration.

Provides create_dataloader() which auto-selects the appropriate dataset class
based on config modalities, and verify_pipeline() for configuration testing.
"""

import logging

import torch
from torch.utils.data import DataLoader

from .config_helpers import (
    is_timeseries_mode, get_timeseries_config, get_csv_window_indices,
    get_timeseries_input_variables, get_timeseries_target_variables,
    get_sdo_indices, get_omni_input_indices, get_omni_target_indices,
    days_to_indices_omni, days_to_indices_sdo
)
from .sampler import BalancedUndersamplerSampler
from .datasets_hdf5 import (
    TrainDataset, ValidationDataset, TestDataset, OperationDataset
)
from .datasets_csv import (
    CSVTrainDataset, CSVValidationDataset, CSVTestDataset
)
from .datasets_table import (
    TableTrainDataset, TableValidationDataset, TableTestDataset
)

logger = logging.getLogger(__name__)


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
        dataset_mode = getattr(
            get_timeseries_config(config), 'dataset_mode', 'csv')

        if dataset_mode == 'table':
            # Table mode: single Parquet + index files
            if phase == "TRAIN":
                dataset = TableTrainDataset(config)
            elif phase == "VALIDATION":
                dataset = TableValidationDataset(config)
            elif phase == "TEST":
                dataset = TableTestDataset(config)
            else:
                raise ValueError(f"Invalid phase for table mode: {phase}")
        elif phase == "TRAIN":
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
        dataset_mode = getattr(ts_cfg, 'dataset_mode', 'csv')
        print(f"\n[TimeSeries Configuration] (mode: {dataset_mode})")
        print(f"Interval: {ts_cfg.interval_minutes} min")
        print(f"Points per day: {ts_cfg.points_per_day}")
        print(f"Days before: {ts_cfg.days_before} ({ts_cfg.days_before * ts_cfg.points_per_day} timesteps)")
        print(f"Days after: {ts_cfg.days_after} ({ts_cfg.days_after * ts_cfg.points_per_day} timesteps)")
        i_s, i_e, t_s, t_e = get_csv_window_indices(ts_cfg)
        print(f"Window indices: input=[{i_s}:{i_e}] ({i_e - i_s} steps), "
              f"target=[{t_s}:{t_e}] ({t_e - t_s} steps)")
        print(f"Input variables: {len(get_timeseries_input_variables(config))}")
        print(f"Target variables: {get_timeseries_target_variables(config)}")
        if dataset_mode == 'table':
            print(f"\n[Table Mode]")
            print(f"Table file: {ts_cfg.table_file}")
            print(f"Train index: {ts_cfg.train_index}")
            print(f"Validation index: {ts_cfg.validation_index}")
            print(f"Test index: {ts_cfg.test_index}")
        else:
            print(f"Dataset dir: {ts_cfg.dataset_dir}")
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
