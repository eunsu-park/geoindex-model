"""Unit tests for analysis/calibrate_warning.py."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.calibrate_warning import (
    auc,
    contingency,
    fit_threshold,
    isotonic_apply,
    isotonic_fit,
)


def noisy_scores(n=4000, seed=0):
    """Scores that rank the event well but sit far below the event's own scale."""
    rng = np.random.default_rng(seed)
    label = rng.random(n) < 0.15
    score = rng.normal(8, 4, n) + label * 9.0        # separated, but never near 50
    return score, label


class TestAuc:
    def test_perfect_and_random_ranking(self):
        label = np.array([False, False, True, True])
        assert auc(np.array([1.0, 2.0, 3.0, 4.0]), label) == pytest.approx(1.0)
        assert auc(np.array([4.0, 3.0, 2.0, 1.0]), label) == pytest.approx(0.0)

    def test_ranking_survives_a_damped_scale(self):
        """AUC is scale-free: shrinking every score leaves discrimination untouched."""
        score, label = noisy_scores()
        assert auc(score * 0.2, label) == pytest.approx(auc(score, label))


class TestFitThreshold:
    def test_budget_is_respected_on_the_fitting_data(self):
        score, label = noisy_scores()
        for budget in (0.2, 0.3, 0.5):
            t = fit_threshold(score, label, budget)
            assert contingency(score >= t, label)["far"] <= budget + 1e-9

    def test_lower_budget_gives_a_higher_threshold(self):
        score, label = noisy_scores()
        assert fit_threshold(score, label, 0.2) >= fit_threshold(score, label, 0.5)

    def test_hss_optimal_beats_the_raw_index_scale(self):
        """The point of the exercise: the index-scale threshold throws skill away."""
        score, label = noisy_scores()
        t = fit_threshold(score, label, None)
        assert contingency(score >= t, label)["hss"] > contingency(score >= 50.0, label)["hss"]


class TestIsotonic:
    def test_fit_is_monotone(self):
        score, label = noisy_scores()
        _, y = isotonic_fit(score, label)
        assert np.all(np.diff(y) >= -1e-12)

    def test_apply_is_clamped_and_monotone(self):
        score, label = noisy_scores()
        x, y = isotonic_fit(score, label)
        probe = np.linspace(score.min() - 50, score.max() + 50, 400)
        p = isotonic_apply(x, y, probe)
        assert p.min() >= 0.0 and p.max() <= 1.0
        assert np.all(np.diff(p) >= -1e-12)

    def test_probabilities_track_the_base_rate(self):
        score, label = noisy_scores()
        x, y = isotonic_fit(score, label)
        p = isotonic_apply(x, y, score)
        assert p.mean() == pytest.approx(label.mean(), abs=0.02)

    def test_beats_climatology_on_brier(self):
        score, label = noisy_scores()
        cut = len(label) // 2
        x, y = isotonic_fit(score[:cut], label[:cut])
        p = isotonic_apply(x, y, score[cut:])
        brier = np.mean((p - label[cut:]) ** 2)
        clim = np.mean((label[:cut].mean() - label[cut:]) ** 2)
        assert brier < clim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
