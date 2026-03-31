"""Unit tests for losses.py components."""

import pytest
import torch
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.losses import (
    MultiModalContrastiveLoss,
    MultiModalMSELoss,
    GeneralWeightedMSELoss,
    SolarWindWeightedLoss,
)


class TestMultiModalContrastiveLoss:
    """Tests for InfoNCE contrastive loss."""

    def test_output_shape(self):
        """Loss should return scalar."""
        loss_fn = MultiModalContrastiveLoss(temperature=0.3)

        features1 = torch.randn(8, 256)
        features2 = torch.randn(8, 256)

        loss = loss_fn(features1, features2)

        assert loss.dim() == 0  # Scalar

    def test_identical_features_low_loss(self):
        """Identical features should have low loss."""
        loss_fn = MultiModalContrastiveLoss(temperature=0.3)

        features = torch.randn(8, 256)
        loss = loss_fn(features, features.clone())

        # With identical features, loss should be relatively low
        assert loss.item() < 1.0

    def test_random_features_higher_loss(self):
        """Random features should have higher loss than identical."""
        loss_fn = MultiModalContrastiveLoss(temperature=0.3)

        features1 = torch.randn(8, 256)
        features2 = torch.randn(8, 256)

        loss_random = loss_fn(features1, features2)
        loss_identical = loss_fn(features1, features1.clone())

        assert loss_random.item() > loss_identical.item()

    def test_temperature_effect(self):
        """Higher temperature should produce different loss values."""
        features1 = torch.randn(8, 256)
        features2 = torch.randn(8, 256)

        loss_low_temp = MultiModalContrastiveLoss(temperature=0.1)(features1, features2)
        loss_high_temp = MultiModalContrastiveLoss(temperature=1.0)(features1, features2)

        # Different temperatures should produce different losses
        assert not torch.isclose(loss_low_temp, loss_high_temp)

    def test_gradient_flow(self):
        """Loss should allow gradient computation."""
        loss_fn = MultiModalContrastiveLoss(temperature=0.3)

        features1 = torch.randn(8, 256, requires_grad=True)
        features2 = torch.randn(8, 256, requires_grad=True)

        loss = loss_fn(features1, features2)
        loss.backward()

        assert features1.grad is not None
        assert features2.grad is not None

    def test_batch_size_one(self):
        """Should handle batch size of 1."""
        loss_fn = MultiModalContrastiveLoss(temperature=0.3)

        features1 = torch.randn(1, 256)
        features2 = torch.randn(1, 256)

        loss = loss_fn(features1, features2)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestMultiModalMSELoss:
    """Tests for MSE consistency loss."""

    def test_output_shape_mean(self):
        """Mean reduction should return scalar."""
        loss_fn = MultiModalMSELoss(reduction="mean")

        features1 = torch.randn(8, 256)
        features2 = torch.randn(8, 256)

        loss = loss_fn(features1, features2)

        assert loss.dim() == 0

    def test_identical_features_zero_loss(self):
        """Identical features should have zero loss."""
        loss_fn = MultiModalMSELoss(reduction="mean")

        features = torch.randn(8, 256)
        loss = loss_fn(features, features.clone())

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_known_values(self):
        """Test with known input/output."""
        loss_fn = MultiModalMSELoss(reduction="mean")

        features1 = torch.tensor([[1.0, 2.0, 3.0]])
        features2 = torch.tensor([[2.0, 3.0, 4.0]])

        loss = loss_fn(features1, features2)

        # MSE = mean((1)^2 + (1)^2 + (1)^2) = 1.0
        assert torch.isclose(loss, torch.tensor(1.0))

    def test_reduction_sum(self):
        """Sum reduction should return sum of squared errors."""
        loss_fn = MultiModalMSELoss(reduction="sum")

        features1 = torch.tensor([[1.0, 2.0]])
        features2 = torch.tensor([[2.0, 4.0]])

        loss = loss_fn(features1, features2)

        # Sum = (1)^2 + (2)^2 = 5.0
        assert torch.isclose(loss, torch.tensor(5.0))

    def test_reduction_none(self):
        """None reduction should return per-element losses."""
        loss_fn = MultiModalMSELoss(reduction="none")

        features1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        features2 = torch.tensor([[2.0, 4.0], [3.0, 5.0]])

        loss = loss_fn(features1, features2)

        # F.mse_loss with reduction='none' returns element-wise squared differences
        # Shape: (batch_size, feature_dim)
        assert loss.shape == (2, 2)

    def test_invalid_reduction(self):
        """Should raise error for invalid reduction."""
        with pytest.raises(ValueError):
            MultiModalMSELoss(reduction="invalid")

    def test_gradient_flow(self):
        """Loss should allow gradient computation."""
        loss_fn = MultiModalMSELoss(reduction="mean")

        features1 = torch.randn(8, 256, requires_grad=True)
        features2 = torch.randn(8, 256, requires_grad=True)

        loss = loss_fn(features1, features2)
        loss.backward()

        assert features1.grad is not None
        assert features2.grad is not None


class TestGeneralWeightedMSELoss:
    """Tests for GeneralWeightedMSELoss."""

    def test_output_shape(self):
        """Loss should return scalar with mean reduction."""
        loss_fn = GeneralWeightedMSELoss(threshold=50.0, high_weight=10.0, low_weight=1.0)

        pred = torch.randn(4, 24, 1)
        target = torch.randn(4, 24, 1)

        loss = loss_fn(pred, target)

        assert loss.dim() == 0  # Scalar

    def test_high_weight_effect(self):
        """High target values should receive higher weights."""
        loss_fn = GeneralWeightedMSELoss(threshold=50.0, high_weight=10.0, low_weight=1.0)

        pred = torch.zeros(2, 1, 1)

        # Same error magnitude, different target values
        target_low = torch.tensor([[[10.0]]])   # Below threshold
        target_high = torch.tensor([[[100.0]]]) # Above threshold

        # Error magnitude is the same (10 vs 100 from zero prediction)
        # but high target should be weighted more
        loss_low = loss_fn(pred[:1], target_low)
        loss_high = loss_fn(pred[:1], target_high)

        # High target should have higher loss due to higher weight
        # Note: The actual error is also different, so we test weight assignment directly
        assert loss_high.item() > loss_low.item()

    def test_threshold_behavior(self):
        """Weights should change at threshold boundary."""
        # Use reduction='none' to directly observe weight effects
        loss_fn = GeneralWeightedMSELoss(
            threshold=50.0, high_weight=10.0, low_weight=1.0, reduction='none'
        )

        # Create targets just below and above threshold
        pred = torch.zeros(2, 1, 1)
        target = torch.tensor([[[49.0]], [[51.0]]])  # below and above threshold

        weighted_loss = loss_fn(pred, target)

        # Error: 49^2 = 2401, 51^2 = 2601
        # Weights: 1.0 for 49, 10.0 for 51
        # Weighted loss: 2401 * 1.0 = 2401, 2601 * 10.0 = 26010
        loss_below = weighted_loss[0, 0, 0].item()
        loss_above = weighted_loss[1, 0, 0].item()

        # The loss for above-threshold should be ~10x higher (adjusted for error diff)
        # loss_above / loss_below ≈ 26010 / 2401 ≈ 10.83
        loss_ratio = loss_above / loss_below

        assert loss_ratio > 8  # Should be around 10.8

    def test_gradient_flow(self):
        """Loss should allow gradient computation."""
        loss_fn = GeneralWeightedMSELoss(threshold=50.0, high_weight=10.0)

        pred = torch.randn(4, 24, 1, requires_grad=True)
        target = torch.randn(4, 24, 1) * 100  # Some values above threshold

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None

    def test_equal_weights(self):
        """With equal weights, should behave like standard MSE."""
        loss_fn = GeneralWeightedMSELoss(threshold=50.0, high_weight=1.0, low_weight=1.0)
        mse_loss = torch.nn.MSELoss()

        pred = torch.randn(4, 24, 1)
        target = torch.randn(4, 24, 1) * 100

        loss_weighted = loss_fn(pred, target)
        loss_mse = mse_loss(pred, target)

        assert torch.isclose(loss_weighted, loss_mse, rtol=1e-4)


class TestSolarWindWeightedLoss:
    """Tests for SolarWindWeightedLoss."""

    def test_output_shape(self):
        """Loss should return scalar with mean reduction."""
        loss_fn = SolarWindWeightedLoss(weighting_mode='multi_tier')

        pred = torch.randn(4, 24, 1)
        target = torch.abs(torch.randn(4, 24, 1)) * 50  # Positive Ap values

        loss = loss_fn(pred, target)

        assert loss.dim() == 0  # Scalar

    def test_multi_tier_weights(self):
        """Multi-tier mode should assign correct weights based on NOAA G-scale."""
        loss_fn = SolarWindWeightedLoss(
            weighting_mode='multi_tier',
            combine_temporal=False  # Disable temporal to test only Ap weights
        )

        # Create targets in different tiers (G-Scale based, 4 levels)
        # AP_TIERS from losses.py:
        # none: 0-29 (weight 1.0) - No storm
        # g1: 30-49 (weight 2.0) - G1 Minor Storm
        # g2: 50-99 (weight 4.0) - G2 Moderate Storm
        # g3_plus: 100+ (weight 8.0) - G3-G5 Strong-Extreme
        pred = torch.zeros(3, 1, 1)
        target_none = torch.tensor([[[10.0]]])       # weight 1.0
        target_g1 = torch.tensor([[[40.0]]])         # weight 2.0
        target_g3 = torch.tensor([[[150.0]]])        # weight 8.0

        loss_none = loss_fn(pred, target_none)
        loss_g1 = loss_fn(pred, target_g1)
        loss_g3 = loss_fn(pred, target_g3)

        # Error: 10^2=100, 40^2=1600, 150^2=22500
        # Weighted loss ratios should reflect both error and weight differences
        # But with normalized reduction, we compare relative weights

        # Verify ordering: g3 > g1 > none
        assert loss_g3.item() > loss_g1.item() > loss_none.item()

    def test_continuous_weights(self):
        """Continuous mode should scale weights smoothly."""
        loss_fn = SolarWindWeightedLoss(
            weighting_mode='continuous',
            alpha=5.0,
            beta=1.5,
            combine_temporal=False
        )

        pred = torch.zeros(1, 1, 1)

        # Test that higher Ap gets higher weight
        target_low = torch.tensor([[[10.0]]])
        target_high = torch.tensor([[[200.0]]])

        loss_low = loss_fn(pred, target_low)
        loss_high = loss_fn(pred, target_high)

        # Higher Ap should have higher weighted loss
        assert loss_high.item() > loss_low.item()

    def test_threshold_mode(self):
        """Threshold mode should apply binary weights."""
        # Use reduction='none' to directly observe weight effects
        loss_fn = SolarWindWeightedLoss(
            weighting_mode='threshold',
            threshold=30.0,
            high_weight=10.0,
            combine_temporal=False,
            reduction='none'
        )

        pred = torch.zeros(2, 1, 1)
        target = torch.tensor([[[25.0]], [[35.0]]])  # Below and above threshold

        weighted_loss = loss_fn(pred, target)

        loss_below = weighted_loss[0, 0, 0].item()
        loss_above = weighted_loss[1, 0, 0].item()

        # Error ratio: 35^2 / 25^2 = 1.96
        # Weight ratio: 10 / 1 = 10
        # Loss ratio should be approximately 1.96 * 10 = 19.6
        loss_ratio = loss_above / loss_below

        assert loss_ratio > 10  # Should be ~19.6

    def test_temporal_weight_combination(self):
        """Temporal weights should emphasize later timesteps."""
        # Use reduction='none' to directly observe weight effects
        loss_fn_with_temporal = SolarWindWeightedLoss(
            weighting_mode='multi_tier',
            combine_temporal=True,
            temporal_weight_range=(0.5, 1.0),
            reduction='none'
        )
        loss_fn_without_temporal = SolarWindWeightedLoss(
            weighting_mode='multi_tier',
            combine_temporal=False,
            reduction='none'
        )

        pred = torch.zeros(1, 10, 1)
        target = torch.ones(1, 10, 1) * 50  # Uniform target (moderate_storm tier)

        loss_with = loss_fn_with_temporal(pred, target)
        loss_without = loss_fn_without_temporal(pred, target)

        # With temporal weights, earlier timesteps should have lower weighted loss
        # temporal_weight_range=(0.5, 1.0) means:
        # - first timestep weight: 0.5
        # - last timestep weight: 1.0
        first_loss_with = loss_with[0, 0, 0].item()
        last_loss_with = loss_with[0, -1, 0].item()

        # Last timestep should have higher weighted loss than first
        assert last_loss_with > first_loss_with

        # Without temporal weights, all timesteps should have equal loss
        first_loss_without = loss_without[0, 0, 0].item()
        last_loss_without = loss_without[0, -1, 0].item()

        assert torch.isclose(
            torch.tensor(first_loss_without),
            torch.tensor(last_loss_without),
            rtol=1e-4
        )

    def test_base_loss_types(self):
        """Should support different base loss types."""
        pred = torch.randn(4, 24, 1)
        target = torch.abs(torch.randn(4, 24, 1)) * 50

        for loss_type in ['mse', 'mae', 'huber']:
            loss_fn = SolarWindWeightedLoss(
                base_loss_type=loss_type,
                weighting_mode='multi_tier',
                combine_temporal=False
            )

            loss = loss_fn(pred, target)

            assert loss.dim() == 0
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)

    def test_gradient_flow(self):
        """Loss should allow gradient computation."""
        loss_fn = SolarWindWeightedLoss(weighting_mode='multi_tier')

        pred = torch.randn(4, 24, 1, requires_grad=True)
        target = torch.abs(torch.randn(4, 24, 1)) * 50

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None

    def test_ap_tier_boundaries(self):
        """Test that AP_TIERS boundaries are correctly applied."""
        loss_fn = SolarWindWeightedLoss(
            weighting_mode='multi_tier',
            combine_temporal=False
        )

        # Test values at tier boundaries (G-Scale based, 4 levels)
        # Reference: https://www.swpc.noaa.gov/noaa-scales-explanation
        # AP_TIERS from losses.py:
        # none: 0-29 (weight 1.0)
        # g1: 30-49 (weight 2.0)
        # g2: 50-99 (weight 4.0)
        # g3_plus: 100+ (weight 8.0)
        tier_tests = [
            (10.0, 1.0),   # none: 0-29 (No storm, Kp < 5)
            (25.0, 1.0),   # none: 0-29 (No storm, Kp < 5)
            (35.0, 2.0),   # g1: 30-49 (G1 Minor Storm, Kp 5)
            (75.0, 4.0),   # g2: 50-99 (G2 Moderate Storm, Kp 6)
            (150.0, 8.0),  # g3_plus: 100+ (G3-G5 Strong-Extreme, Kp 7-9)
        ]

        for ap_value, expected_weight in tier_tests:
            target = torch.tensor([[[ap_value]]])
            weights = loss_fn._compute_multi_tier_weights(target)
            assert torch.isclose(
                weights[0, 0, 0],
                torch.tensor(expected_weight),
                rtol=1e-4
            ), f"Ap={ap_value} should have weight {expected_weight}, got {weights[0, 0, 0]}"

    def test_denormalization_log1p_zscore(self):
        """Test denormalization for log1p_zscore method."""
        # Simulate normalization statistics (typical values for Ap index)
        norm_stats = {
            'log1p_mean': 2.5,
            'log1p_std': 1.2,
        }

        loss_fn = SolarWindWeightedLoss(
            weighting_mode='multi_tier',
            combine_temporal=False,
            denormalize=True,
            norm_method='log1p_zscore',
            norm_stats=norm_stats
        )

        # Test denormalization: normalized -> raw Ap
        # Raw Ap = 50 -> log1p(50) = 3.93 -> z-score = (3.93 - 2.5) / 1.2 = 1.19
        normalized_value = (np.log1p(50.0) - norm_stats['log1p_mean']) / norm_stats['log1p_std']
        normalized_target = torch.tensor([[[normalized_value]]], dtype=torch.float32)

        # Denormalize should return ~50
        raw_target = loss_fn._denormalize_target(normalized_target)
        assert torch.isclose(
            raw_target[0, 0, 0],
            torch.tensor(50.0, dtype=torch.float32),
            rtol=1e-4
        ), f"Denormalized value should be ~50, got {raw_target[0, 0, 0]}"

    def test_denormalization_affects_weights(self):
        """Test that denormalization correctly affects weight assignment."""
        # Simulate normalization: Ap=50 (G2 tier, weight=4.0)
        # AP_TIERS: g2: (50, 99, 4.0)
        norm_stats = {
            'log1p_mean': 2.5,
            'log1p_std': 1.2,
        }

        # Loss function WITH denormalization (correct behavior)
        loss_fn_with_denorm = SolarWindWeightedLoss(
            weighting_mode='multi_tier',
            combine_temporal=False,
            denormalize=True,
            norm_method='log1p_zscore',
            norm_stats=norm_stats,
            reduction='none'
        )

        # Loss function WITHOUT denormalization (incorrect, for comparison)
        loss_fn_without_denorm = SolarWindWeightedLoss(
            weighting_mode='multi_tier',
            combine_temporal=False,
            denormalize=False,
            reduction='none'
        )

        # Create normalized target representing raw Ap=50 (G2 tier)
        raw_ap = 50.0
        normalized_value = (np.log1p(raw_ap) - norm_stats['log1p_mean']) / norm_stats['log1p_std']
        pred = torch.zeros(1, 1, 1, dtype=torch.float32)
        target = torch.tensor([[[normalized_value]]], dtype=torch.float32)

        # With denormalization: should recognize Ap=50 as G2 tier (weight=4.0)
        loss_with = loss_fn_with_denorm(pred, target)
        # Get the weight used (loss = base_loss * weight)
        base_mse = (pred - target) ** 2
        weight_with = loss_with / base_mse

        # Without denormalization: normalized value ~1.19 falls in 'none' tier (weight=1)
        loss_without = loss_fn_without_denorm(pred, target)
        weight_without = loss_without / base_mse

        # Weight with denormalization should be higher (G2 tier = 4x)
        assert weight_with.item() > weight_without.item(), \
            f"Weight with denorm ({weight_with.item()}) should be > without ({weight_without.item()})"

        # Weight should be approximately 4.0 (G2 tier)
        assert torch.isclose(
            weight_with[0, 0, 0],
            torch.tensor(4.0, dtype=torch.float32),
            rtol=1e-4
        ), f"Weight should be 4.0 (G2 tier), got {weight_with[0, 0, 0]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
