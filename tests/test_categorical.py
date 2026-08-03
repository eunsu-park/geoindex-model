"""Unit tests for the contingency scoring in analysis/compare_loss_variants.py."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.compare_loss_variants import contingency


class TestContingency:
    def test_perfect_forecast(self):
        obs = np.array([10.0, 40.0, 80.0, 120.0])
        c = contingency(obs.copy(), obs, 50.0)
        assert (c["hits"], c["false_alarms"], c["misses"], c["correct_negatives"]) == (2, 0, 0, 2)
        assert c["pod"] == 1.0
        assert c["far"] == 0.0
        assert c["csi"] == 1.0
        assert c["bias"] == 1.0
        assert c["hss"] == pytest.approx(1.0)

    def test_forecast_that_never_fires(self):
        """A fully damped forecast: no hits, no false alarms, HSS 0."""
        obs = np.array([10.0, 40.0, 80.0, 120.0])
        pred = np.zeros_like(obs)
        c = contingency(pred, obs, 50.0)
        assert (c["hits"], c["false_alarms"], c["misses"]) == (0, 0, 2)
        assert c["pod"] == 0.0
        assert c["bias"] == 0.0
        assert c["hss"] == pytest.approx(0.0)
        assert np.isnan(c["far"])            # FAR is undefined with no forecast events

    def test_bias_separates_under_and_over_forecasting(self):
        """Frequency bias is forecast events / observed events."""
        obs = np.array([0.0, 0.0, 60.0, 60.0])
        under = np.array([0.0, 0.0, 0.0, 60.0])
        over = np.array([60.0, 60.0, 60.0, 60.0])
        assert contingency(under, obs, 50.0)["bias"] == pytest.approx(0.5)
        assert contingency(over, obs, 50.0)["bias"] == pytest.approx(2.0)

    def test_threshold_is_inclusive_on_the_observation_side(self):
        """A value exactly at the threshold counts as an event."""
        obs = np.array([50.0])
        assert contingency(np.array([50.0]), obs, 50.0)["hits"] == 1
        assert contingency(np.array([49.9]), obs, 50.0)["misses"] == 1

    def test_hss_is_zero_for_a_random_style_forecast(self):
        """HSS measures skill over chance, so a forecast matching the base rate scores ~0."""
        rng = np.random.default_rng(0)
        obs = (rng.random(20000) < 0.1) * 100.0
        pred = (rng.random(20000) < 0.1) * 100.0     # independent, same base rate
        assert abs(contingency(pred, obs, 50.0)["hss"]) < 0.03


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
