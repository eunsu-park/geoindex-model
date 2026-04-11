"""Tests for src/utils.py — seed, device, path resolution, logging helpers."""

import os
import logging
import random
from io import StringIO
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.utils import (
    setup_seed,
    setup_device,
    setup_experiment,
    resolve_paths,
    save_model,
    load_model,
    _log_message,
)


# ---------------------------------------------------------------------------
# setup_seed
# ---------------------------------------------------------------------------
class TestSetupSeed:
    """Tests for setup_seed()."""

    def test_deterministic_random(self):
        """After seeding, random.random() produces the same sequence."""
        setup_seed(123)
        a = [random.random() for _ in range(5)]
        setup_seed(123)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_deterministic_numpy(self):
        """After seeding, np.random produces the same sequence."""
        setup_seed(123)
        a = np.random.rand(5).tolist()
        setup_seed(123)
        b = np.random.rand(5).tolist()
        assert a == b

    def test_deterministic_torch(self):
        """After seeding, torch.randn produces the same sequence."""
        setup_seed(123)
        a = torch.randn(5)
        setup_seed(123)
        b = torch.randn(5)
        assert torch.allclose(a, b)

    def test_sets_pythonhashseed(self):
        """PYTHONHASHSEED env var is set."""
        setup_seed(999)
        assert os.environ['PYTHONHASHSEED'] == '999'

    def test_default_seed(self):
        """Default seed value works without error."""
        setup_seed()  # uses default 250104


# ---------------------------------------------------------------------------
# setup_device
# ---------------------------------------------------------------------------
class TestSetupDevice:
    """Tests for setup_device()."""

    def test_cpu(self):
        """Requesting 'cpu' returns cpu device."""
        device = setup_device('cpu')
        assert device == torch.device('cpu')

    def test_unknown_falls_back_to_cpu(self):
        """Unknown device string falls back to cpu."""
        device = setup_device('tpu_v99')
        assert device == torch.device('cpu')

    def test_cuda_fallback(self):
        """If CUDA unavailable, falls back to cpu."""
        if not torch.cuda.is_available():
            device = setup_device('cuda')
            assert device == torch.device('cpu')

    def test_mps_returns_device(self):
        """MPS request returns either mps or cpu (platform-dependent)."""
        device = setup_device('mps')
        assert device.type in ('mps', 'cpu')


# ---------------------------------------------------------------------------
# setup_experiment
# ---------------------------------------------------------------------------
class TestSetupExperiment:
    """Tests for setup_experiment()."""

    def test_returns_device(self, minimal_config):
        """setup_experiment returns a torch.device."""
        device = setup_experiment(minimal_config)
        assert isinstance(device, torch.device)
        assert device == torch.device('cpu')


# ---------------------------------------------------------------------------
# resolve_paths
# ---------------------------------------------------------------------------
class TestResolvePaths:
    """Tests for resolve_paths()."""

    def test_epoch_best(self, minimal_config):
        """epoch='best' produces model_best.pth."""
        # Clear output_dir to test epoch-based auto-generation
        cfg = OmegaConf.merge(minimal_config, {'validation': {'output_dir': '', 'epoch': 'best'}})
        cp, od = resolve_paths(cfg, 'validation')
        assert cp.endswith('model_best.pth')
        assert od.endswith('validation/best')

    def test_epoch_final(self, minimal_config):
        """epoch='final' produces model_final.pth."""
        cfg = OmegaConf.merge(minimal_config, {'validation': {'output_dir': '', 'epoch': 'final'}})
        cp, od = resolve_paths(cfg, 'validation')
        assert cp.endswith('model_final.pth')
        assert od.endswith('validation/final')

    def test_epoch_integer(self, minimal_config):
        """Integer epoch produces zero-padded filename."""
        cfg = OmegaConf.merge(minimal_config, {'validation': {'output_dir': '', 'epoch': 10}})
        cp, od = resolve_paths(cfg, 'validation')
        assert 'model_epoch_0010.pth' in cp
        assert 'epoch_0010' in od

    def test_invalid_epoch_string(self, minimal_config):
        """Invalid epoch string raises ValueError."""
        cfg = OmegaConf.merge(minimal_config, {'validation': {'epoch': 'latest'}})
        with pytest.raises(ValueError, match="Invalid epoch string"):
            resolve_paths(cfg, 'validation')

    def test_explicit_checkpoint_path(self, minimal_config):
        """Explicit checkpoint_path takes priority."""
        cfg = OmegaConf.merge(minimal_config, {
            'validation': {
                'checkpoint_path': '/custom/path.pth',
                'epoch': 'best',
            }
        })
        cp, od = resolve_paths(cfg, 'validation')
        assert cp == '/custom/path.pth'

    def test_no_epoch_no_path_raises(self, minimal_config):
        """Missing both epoch and checkpoint_path raises ValueError."""
        cfg = OmegaConf.merge(minimal_config, {
            'validation': {
                'epoch': None,
                'output_dir': '',
            }
        })
        with pytest.raises(ValueError):
            resolve_paths(cfg, 'validation')


# ---------------------------------------------------------------------------
# save_model / load_model round-trip
# ---------------------------------------------------------------------------
class TestSaveLoadModel:
    """Tests for save_model() and load_model()."""

    def test_round_trip(self, tmp_path):
        """save_model -> load_model preserves weights."""
        model = torch.nn.Linear(4, 2)
        path = str(tmp_path / "ckpt" / "model.pth")

        save_model(model, path, epoch=1)
        assert os.path.exists(path)

        model2 = torch.nn.Linear(4, 2)
        model2 = load_model(model2, path, torch.device('cpu'))

        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_load_missing_raises(self):
        """Loading non-existent file raises FileNotFoundError."""
        model = torch.nn.Linear(4, 2)
        with pytest.raises(FileNotFoundError):
            load_model(model, '/nonexistent/model.pth', torch.device('cpu'))


# ---------------------------------------------------------------------------
# _log_message
# ---------------------------------------------------------------------------
class TestLogMessage:
    """Tests for _log_message()."""

    def test_with_logger(self):
        """Message is sent to the logger when provided."""
        logger = logging.getLogger('test_log_msg')
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(StringIO())
        logger.addHandler(handler)

        _log_message(logger, "hello", logging.WARNING)
        # No assertion on output; just ensure no exception.

    def test_without_logger(self, capsys):
        """Without logger, message is printed."""
        _log_message(None, "fallback message")
        captured = capsys.readouterr()
        assert "fallback message" in captured.out
