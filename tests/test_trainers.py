"""Tests for src/trainers.py — MetricsTracker and CheckpointManager components."""

import torch
import numpy as np
import pytest

from src.trainers import MetricsTracker, CheckpointManager, EarlyStopping


# ---------------------------------------------------------------------------
# MetricsTracker
# ---------------------------------------------------------------------------
class TestMetricsTracker:
    """Tests for the MetricsTracker class."""

    def test_initial_state(self):
        """Freshly created tracker has empty metrics."""
        tracker = MetricsTracker()
        for values in tracker.metrics.values():
            assert values == []

    def test_update_single_batch(self):
        """Single update stores values correctly."""
        tracker = MetricsTracker()
        tracker.update({'loss': 1.0, 'reg_loss': 0.8, 'mae': 0.5, 'rmse': 0.6})

        assert tracker.metrics['loss'] == [1.0]
        assert tracker.metrics['reg_loss'] == [0.8]
        assert tracker.metrics['mae'] == [0.5]

    def test_update_ignores_unknown_keys(self):
        """Keys not in the predefined set are silently ignored."""
        tracker = MetricsTracker()
        tracker.update({'loss': 1.0, 'custom_metric': 99.0})
        assert 'custom_metric' not in tracker.metrics

    def test_update_multiple_batches(self):
        """Multiple updates accumulate values."""
        tracker = MetricsTracker()
        tracker.update({'loss': 1.0, 'mae': 0.3})
        tracker.update({'loss': 2.0, 'mae': 0.5})
        tracker.update({'loss': 3.0, 'mae': 0.7})

        assert len(tracker.metrics['loss']) == 3
        assert len(tracker.metrics['mae']) == 3

    def test_reset(self):
        """reset() clears all accumulated metrics."""
        tracker = MetricsTracker()
        tracker.update({'loss': 1.0, 'mae': 0.3})
        tracker.reset()

        for values in tracker.metrics.values():
            assert values == []

    def test_get_running_average_all(self):
        """Running average over all samples."""
        tracker = MetricsTracker()
        tracker.update({'loss': 1.0})
        tracker.update({'loss': 3.0})

        avg = tracker.get_running_average()
        assert avg['loss'] == pytest.approx(2.0)

    def test_get_running_average_last_n(self):
        """Running average over last N samples."""
        tracker = MetricsTracker()
        for val in [10.0, 20.0, 30.0, 40.0]:
            tracker.update({'loss': val})

        avg = tracker.get_running_average(last_n=2)
        assert avg['loss'] == pytest.approx(35.0)

    def test_get_epoch_summary(self):
        """Epoch summary returns mean, std, min, max."""
        tracker = MetricsTracker()
        tracker.update({'loss': 1.0})
        tracker.update({'loss': 3.0})
        tracker.update({'loss': 5.0})

        summary = tracker.get_epoch_summary()
        assert 'loss' in summary
        assert summary['loss']['mean'] == pytest.approx(3.0)
        assert summary['loss']['min'] == pytest.approx(1.0)
        assert summary['loss']['max'] == pytest.approx(5.0)

    def test_epoch_summary_empty_metrics_skipped(self):
        """Metrics with no values are excluded from summary."""
        tracker = MetricsTracker()
        tracker.update({'loss': 1.0})  # only loss, no cosine_sim

        summary = tracker.get_epoch_summary()
        assert 'loss' in summary
        assert 'cosine_sim' not in summary


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------
class TestCheckpointManager:
    """Tests for the CheckpointManager class."""

    @staticmethod
    def _make_model_and_optimizer():
        """Create a minimal model and optimizer."""
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        return model, optimizer

    def test_save_creates_file(self, tmp_path):
        """save() creates a checkpoint file."""
        mgr = CheckpointManager(str(tmp_path / "ckpts"))
        model, optimizer = self._make_model_and_optimizer()

        mgr.save(model, optimizer, epoch=1, loss=0.5)

        expected = tmp_path / "ckpts" / "model_epoch_0001.pth"
        assert expected.exists()

    def test_save_custom_filename(self, tmp_path):
        """save() with explicit filename."""
        mgr = CheckpointManager(str(tmp_path / "ckpts"))
        model, optimizer = self._make_model_and_optimizer()

        mgr.save(model, optimizer, epoch=1, loss=0.5, filename="custom.pth")

        assert (tmp_path / "ckpts" / "custom.pth").exists()

    def test_save_load_round_trip(self, tmp_path):
        """Saved checkpoint can be loaded and weights match."""
        mgr = CheckpointManager(str(tmp_path / "ckpts"))
        model, optimizer = self._make_model_and_optimizer()

        mgr.save(model, optimizer, epoch=5, loss=0.3)

        ckpt_path = tmp_path / "ckpts" / "model_epoch_0005.pth"
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        assert checkpoint['epoch'] == 5
        assert checkpoint['loss'] == pytest.approx(0.3)
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'timestamp' in checkpoint

    def test_save_if_best_updates(self, tmp_path):
        """save_if_best() saves when loss improves."""
        mgr = CheckpointManager(str(tmp_path / "ckpts"))
        model, optimizer = self._make_model_and_optimizer()

        mgr.save_if_best(model, optimizer, epoch=1, loss=1.0)
        assert (tmp_path / "ckpts" / "model_best.pth").exists()
        assert mgr.best_loss == pytest.approx(1.0)

    def test_save_if_best_no_update(self, tmp_path):
        """save_if_best() does not overwrite when loss is worse."""
        mgr = CheckpointManager(str(tmp_path / "ckpts"))
        model, optimizer = self._make_model_and_optimizer()

        mgr.save_if_best(model, optimizer, epoch=1, loss=1.0)
        mgr.save_if_best(model, optimizer, epoch=2, loss=2.0)

        assert mgr.best_loss == pytest.approx(1.0)

    def test_save_periodic(self, tmp_path):
        """save_periodic() only saves at multiples of save_freq."""
        mgr = CheckpointManager(str(tmp_path / "ckpts"), save_freq=3)
        model, optimizer = self._make_model_and_optimizer()

        for epoch in range(1, 7):
            mgr.save_periodic(model, optimizer, epoch, loss=0.5)

        assert (tmp_path / "ckpts" / "model_epoch_0003.pth").exists()
        assert (tmp_path / "ckpts" / "model_epoch_0006.pth").exists()
        assert not (tmp_path / "ckpts" / "model_epoch_0001.pth").exists()
        assert not (tmp_path / "ckpts" / "model_epoch_0002.pth").exists()


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------
class TestEarlyStopping:
    """Tests for the EarlyStopping class."""

    def test_no_stop_on_improvement(self):
        """Does not trigger when loss keeps improving."""
        es = EarlyStopping(patience=3)
        model = torch.nn.Linear(2, 1)

        assert es(1.0, model) is False
        assert es(0.9, model) is False
        assert es(0.8, model) is False
        assert es.early_stop is False

    def test_stops_after_patience(self):
        """Triggers after patience epochs without improvement."""
        es = EarlyStopping(patience=2)
        model = torch.nn.Linear(2, 1)

        es(1.0, model)   # best
        es(1.1, model)   # no improvement, counter=1
        result = es(1.2, model)  # counter=2 >= patience

        assert result is True
        assert es.early_stop is True

    def test_restore_best_model(self):
        """restore_best_model() restores the best weights."""
        es = EarlyStopping(patience=5, restore_best_weights=True)
        model = torch.nn.Linear(2, 1)

        # Record initial weights
        with torch.no_grad():
            model.weight.fill_(1.0)
        es(0.5, model)  # saves best

        with torch.no_grad():
            model.weight.fill_(99.0)

        es.restore_best_model(model)
        assert torch.allclose(model.weight, torch.ones_like(model.weight))
