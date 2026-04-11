"""Tests for src/plotting.py — shared plotting utilities."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.plotting import extract_file_names, denormalize_arrays, plot_prediction_timeseries


# ---------------------------------------------------------------------------
# extract_file_names
# ---------------------------------------------------------------------------
class TestExtractFileNames:
    """Tests for extract_file_names()."""

    def test_with_list(self):
        """Returns file names when given a list."""
        data = {
            'inputs': torch.randn(3, 10, 2),
            'file_names': ['a.csv', 'b.csv', 'c.csv'],
        }
        names = extract_file_names(data, batch_idx=0)
        assert names == ['a.csv', 'b.csv', 'c.csv']

    def test_without_file_names_key(self):
        """Falls back to generated names when file_names is absent."""
        data = {'inputs': torch.randn(4, 10, 2)}
        names = extract_file_names(data, batch_idx=5)
        assert len(names) == 4
        assert names[0] == 'batch_5_sample_0'
        assert names[3] == 'batch_5_sample_3'

    def test_with_tensor(self):
        """Handles tensor file_names (e.g. integer ids)."""
        data = {
            'inputs': torch.randn(2, 10, 2),
            'file_names': torch.tensor([100, 200]),
        }
        names = extract_file_names(data, batch_idx=0)
        assert names == ['100', '200']

    def test_with_single_value(self):
        """Handles a single scalar file name."""
        data = {
            'inputs': torch.randn(1, 10, 2),
            'file_names': 'single.csv',
        }
        names = extract_file_names(data, batch_idx=0)
        assert names == ['single.csv']


# ---------------------------------------------------------------------------
# denormalize_arrays
# ---------------------------------------------------------------------------
class TestDenormalizeArrays:
    """Tests for denormalize_arrays()."""

    @staticmethod
    def _make_normalizer(offset: float = 10.0):
        """Create a mock normalizer that adds a fixed offset."""
        normalizer = MagicMock()
        normalizer.denormalize_omni = MagicMock(side_effect=lambda arr, name: arr + offset)
        return normalizer

    def test_basic_denormalization(self):
        """Inputs, predictions, and targets are all denormalized."""
        normalizer = self._make_normalizer(offset=5.0)
        inp = np.zeros((10, 2))
        pred = np.ones((5, 1))
        tgt = np.full((5, 1), 2.0)

        d_inp, d_tgt, d_pred = denormalize_arrays(
            inp, pred,
            input_variables=['v_avg', 'bz_avg'],
            target_variables=['ap30'],
            normalizer=normalizer,
            targets=tgt,
        )

        assert d_inp.shape == inp.shape
        np.testing.assert_allclose(d_inp, 5.0)
        np.testing.assert_allclose(d_pred, 6.0)
        np.testing.assert_allclose(d_tgt, 7.0)

    def test_no_targets(self):
        """Works when targets=None."""
        normalizer = self._make_normalizer(offset=1.0)
        inp = np.zeros((10, 1))
        pred = np.zeros((5, 1))

        d_inp, d_tgt, d_pred = denormalize_arrays(
            inp, pred,
            input_variables=['v_avg'],
            target_variables=['ap30'],
            normalizer=normalizer,
            targets=None,
        )

        assert d_tgt is None
        np.testing.assert_allclose(d_inp, 1.0)
        np.testing.assert_allclose(d_pred, 1.0)

    def test_does_not_mutate_originals(self):
        """Original arrays are not modified in place."""
        normalizer = self._make_normalizer(offset=100.0)
        inp = np.zeros((10, 1))
        pred = np.zeros((5, 1))

        denormalize_arrays(inp, pred, ['v_avg'], ['ap30'], normalizer)

        np.testing.assert_allclose(inp, 0.0)
        np.testing.assert_allclose(pred, 0.0)


# ---------------------------------------------------------------------------
# plot_prediction_timeseries
# ---------------------------------------------------------------------------
class TestPlotPredictionTimeseries:
    """Tests for plot_prediction_timeseries()."""

    def test_creates_file(self, tmp_path):
        """Plot is saved to the specified path."""
        save_path = tmp_path / "plot.png"

        inputs = np.random.randn(48, 2)
        predictions = np.random.randn(24, 1)
        targets = np.random.randn(24, 1)

        plot_prediction_timeseries(
            inputs=inputs,
            predictions=predictions,
            input_variables=['v_avg', 'bz_avg'],
            target_variables=['ap30'],
            save_path=save_path,
            targets=targets,
            title='Test Plot',
        )

        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_without_targets(self, tmp_path):
        """Plot works in inference mode (no targets)."""
        save_path = tmp_path / "plot_no_tgt.png"

        plot_prediction_timeseries(
            inputs=np.random.randn(48, 1),
            predictions=np.random.randn(24, 1),
            input_variables=['ap30'],
            target_variables=['ap30'],
            save_path=save_path,
        )

        assert save_path.exists()

    def test_with_normalizer(self, tmp_path):
        """Plot runs without error when normalizer is provided."""
        save_path = tmp_path / "plot_denorm.png"

        normalizer = MagicMock()
        normalizer.denormalize_omni = MagicMock(side_effect=lambda arr, name: arr)

        plot_prediction_timeseries(
            inputs=np.random.randn(48, 2),
            predictions=np.random.randn(24, 1),
            input_variables=['v_avg', 'bz_avg'],
            target_variables=['ap30'],
            save_path=save_path,
            targets=np.random.randn(24, 1),
            normalizer=normalizer,
        )

        assert save_path.exists()

    def test_multiple_target_vars(self, tmp_path):
        """Plot handles multiple target variables."""
        save_path = tmp_path / "plot_multi.png"

        plot_prediction_timeseries(
            inputs=np.random.randn(48, 3),
            predictions=np.random.randn(24, 2),
            input_variables=['v_avg', 'bz_avg', 'ap30'],
            target_variables=['ap30', 'hp30'],
            save_path=save_path,
            targets=np.random.randn(24, 2),
        )

        assert save_path.exists()
