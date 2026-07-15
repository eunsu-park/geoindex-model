"""Unit tests for src/uncertainty.py (MC-dropout sampling + calibration + recalibration)."""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.uncertainty import (
    mcd_sample_stats,
    uncertainty_metrics,
    fit_sigma_scale,
    recalibrate_cv,
    QUANTILE_KEYS,
)
from src.pipeline import Normalizer


def overconfident(n_anchors=400, horizon=8, true_sigma=10.0, reported_sigma=2.0, seed=0):
    """Long-form arrays where the reported std is 5x too small (overconfident)."""
    rng = np.random.default_rng(seed)
    base = np.datetime64("2022-01-01T00:00:00")
    anchor, hz, true, mean, std = [], [], [], [], []
    for a in range(n_anchors):
        ts = base + np.timedelta64(30 * a, "m")  # 30-min cadence
        mu = rng.normal(50.0, 20.0, size=horizon)
        obs = mu + rng.normal(0.0, true_sigma, size=horizon)
        anchor.append(np.full(horizon, ts, dtype="datetime64[ns]"))
        hz.append(np.arange(horizon))
        mean.append(mu)
        true.append(obs)
        std.append(np.full(horizon, reported_sigma))
    return (np.concatenate(anchor), np.concatenate(hz),
            np.concatenate(true), np.concatenate(mean), np.concatenate(std))


class _TinyDropoutModel(nn.Module):
    """Minimal model with a live Dropout, forward signature (inputs, sdo, return_features)."""

    def __init__(self, target_len=4, n_vars=1, p=0.5):
        super().__init__()
        self.drop = nn.Dropout(p)
        self.lin = nn.Linear(3, target_len * n_vars)
        self.target_len, self.n_vars = target_len, n_vars

    def forward(self, inputs, sdo=None, return_features=False):
        x = self.drop(inputs.mean(dim=1))          # (batch, 3)
        out = self.lin(x)                          # (batch, target_len*n_vars)
        return out.view(-1, self.target_len, self.n_vars)


class TestMcdSampleStats:
    def _normalizer(self):
        # zscore target -> linear denorm, sign-preserving (is_nonnegative False)
        return Normalizer(stat_dict={"ap30": {"mean": 0.0, "std": 1.0}},
                          method_config={"default": "zscore"})

    def test_shapes_and_keys(self):
        torch.manual_seed(0)
        model = _TinyDropoutModel(target_len=4, n_vars=1)
        inputs = torch.randn(5, 6, 3)  # (batch, seq, features)
        stats = mcd_sample_stats(model, inputs, None, self._normalizer(),
                                 ["ap30"], num_samples=32)
        expected = {"mcd_mean", "mcd_std", "mcd_min", "mcd_max", "n_samples", *QUANTILE_KEYS}
        assert set(stats) == expected
        assert stats["n_samples"] == 32
        for key in expected - {"n_samples"}:
            assert stats[key].shape == (5, 4, 1)

    def test_dropout_produces_variance_and_ordering(self):
        torch.manual_seed(0)
        model = _TinyDropoutModel(p=0.5)
        inputs = torch.randn(5, 6, 3)
        s = mcd_sample_stats(model, inputs, None, self._normalizer(), ["ap30"], num_samples=64)
        assert s["mcd_std"].mean() > 0.0                       # dropout active -> spread
        assert np.all(s["mcd_min"] <= s["mcd_mean"] + 1e-9)
        assert np.all(s["mcd_mean"] <= s["mcd_max"] + 1e-9)
        assert np.all(s["mcd_q025"] <= s["mcd_q975"] + 1e-9)
        assert np.all(s["mcd_q05"] <= s["mcd_q95"] + 1e-9)

    def test_model_left_deterministic(self):
        """mcd_sample_stats must restore eval (dropout off) so later use is deterministic."""
        model = _TinyDropoutModel()
        inputs = torch.randn(3, 6, 3)
        mcd_sample_stats(model, inputs, None, self._normalizer(), ["ap30"], num_samples=8)
        assert not model.drop.training

    def test_nonnegative_gate_clips_only_nonnegative_targets(self):
        """log1p target clips at 0; a signed zscore target keeps negatives."""
        model = _TinyDropoutModel()
        # force a large negative bias so raw samples go below 0
        with torch.no_grad():
            model.lin.bias.fill_(-100.0)
        inputs = torch.randn(4, 6, 3)
        signed = Normalizer(stat_dict={"dst": {"mean": 0.0, "std": 1.0}},
                            method_config={"default": "zscore"})
        nonneg = Normalizer(stat_dict={"ap30": {"log1p_mean": 0.0, "log1p_std": 1.0}},
                            method_config={"default": "log1p_zscore"})
        s_signed = mcd_sample_stats(model, inputs, None, signed, ["dst"], num_samples=8)
        s_nonneg = mcd_sample_stats(model, inputs, None, nonneg, ["ap30"], num_samples=8)
        assert s_signed["mcd_mean"].min() < 0.0     # signed kept its sign
        assert s_nonneg["mcd_min"].min() >= 0.0     # non-negative clipped at 0


class TestSigmaScale:
    def test_fit_recovers_true_dispersion(self):
        rng = np.random.default_rng(1)
        true_sigma, reported = 10.0, 2.0
        mean = np.zeros(20000)
        true = rng.normal(0.0, true_sigma, size=20000)
        std = np.full(20000, reported)
        s = fit_sigma_scale(true, mean, std, coverage=0.95, k=2.0)
        assert s == pytest.approx(true_sigma / reported * 1.959964 / 2.0, rel=0.05)

    def test_metrics_flag_overconfidence(self):
        _, _, true, mean, std = overconfident()
        m = uncertainty_metrics(true, mean, std)
        assert m["picp_2sigma"] < 0.6


class TestRecalibrateCV:
    def test_lifts_coverage_to_target(self):
        anchor, hz, true, mean, std = overconfident()
        r = recalibrate_cv(anchor, hz, true, mean, std, n_folds=5)
        assert r["picp_2sigma_raw"] < 0.6
        assert r["picp_2sigma_recal"] == pytest.approx(0.95, abs=0.03)
        assert r["sigma_scale"] > 1.0
        assert len(r["sigma_scale_folds"]) == 5

    def test_point_forecast_untouched(self):
        anchor, hz, true, mean, std = overconfident()
        mae_before = float(np.mean(np.abs(mean - true)))
        recalibrate_cv(anchor, hz, true, mean, std, n_folds=5)
        assert mae_before == pytest.approx(
            uncertainty_metrics(true, mean, std)["mae_mcd_mean"], rel=1e-12
        )
