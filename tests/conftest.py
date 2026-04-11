"""Shared test fixtures for regression-sw test suite.

Provides reusable config objects, dummy data batches, and helper
fixtures used across multiple test modules.
"""

import pytest
import torch
from omegaconf import OmegaConf


@pytest.fixture
def minimal_config():
    """Minimal config for unit tests."""
    return OmegaConf.create({
        'experiment': {'name': 'test', 'seed': 42, 'batch_size': 2},
        'environment': {
            'device': 'cpu',
            'save_root': '/tmp/test_results',
            'num_workers': 0,
        },
        'model': {
            'model_type': 'linear',
            'd_model': 32,
            'transformer_nhead': 2,
            'transformer_num_layers': 1,
            'transformer_dim_feedforward': 64,
            'transformer_dropout': 0.1,
        },
        'data': {
            'modalities': {'timeseries': True, 'sdo': False},
            'timeseries': {
                'input_variables': ['v_avg', 'bz_avg'],
                'target_variables': ['ap30'],
                'points_per_day': 48,
                'input_start': -48,
                'input_end': 0,
                'target_start': 0,
                'target_end': 24,
            },
        },
        'training': {
            'regression_loss_type': 'mse',
            'contrastive_loss_type': 'none',
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'weight_decay': 0.0,
            'epochs': 2,
            'scheduler_factor': 0.5,
            'scheduler_patience': 3,
            'scheduler_type': 'reduce_on_plateau',
            'gradient_accumulation_steps': 1,
            'max_grad_norm': 1.0,
            'model_save_freq': 1,
            'report_freq': 1,
            'enable_plot': False,
        },
        'validation': {
            'output_dir': '/tmp/test_validation',
            'save_plots': False,
            'save_npz': False,
            'report_freq': 50,
            'epoch': 'best',
        },
    })


@pytest.fixture
def dummy_batch():
    """Create a dummy data batch for testing."""
    return {
        'inputs': torch.randn(2, 48, 2),   # (batch, seq_len, input_vars)
        'targets': torch.randn(2, 24, 1),  # (batch, target_len, target_vars)
        'file_names': ['test_001.csv', 'test_002.csv'],
    }
