"""Loss functions for multi-modal solar wind prediction.

This module provides various loss functions including:
- Regression losses: MSE, MAE, Huber with various weighting strategies
- Contrastive losses: InfoNCE, Consistency for multimodal alignment
- Advanced losses: Adaptive, Gradient-based, Quantile, Multi-task

Classes:
    WeightedMSELoss: MSE with time-based weighting
    HuberMultiCriteriaLoss: Huber with temporal and gradient weighting
    MAEOutlierFocusedLoss: MAE with outlier detection
    AdaptiveWeightLoss: Dynamic error-based weighting
    GradientBasedWeightLoss: Emphasis on rapid changes
    QuantileLoss: Prediction intervals with uncertainty
    MultiTaskLoss: Combined regression and outlier detection
    MultiModalContrastiveLoss: InfoNCE-style alignment
    MultiModalMSELoss: MSE-based consistency

Functions:
    create_loss_functions: Factory function for creating losses from config
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pipeline.normalizer import method_is_nonnegative


# =============================================================================
# Weighted Regression Losses
# =============================================================================

class WeightedMSELoss(nn.Module):
    """Weighted MSE Loss with custom time-based weighting.

    Applies different weights based on prediction time horizons:
    - First 8 points: weight 0.5
    - Next 16 points (9-24): weight 0.3
    - Last 24 points would be: weight 0.2 (but sequence length is 24 total)

    Args:
        reduction: Reduction method ('mean', 'sum', 'none').
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction='none')

    def _compute_time_weights(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute time-based weights for the sequence.

        Args:
            seq_len: Length of the sequence.
            device: Device to create tensor on.

        Returns:
            Weight tensor of shape (seq_len,).
        """
        weights = torch.zeros(seq_len, device=device)

        # First 8 points: weight 0.5
        end_first = min(8, seq_len)
        weights[:end_first] = 0.5

        # Next 16 points (9-24): weight 0.3
        if seq_len > 8:
            end_second = min(24, seq_len)
            weights[8:end_second] = 0.3

        # Remaining points: weight 0.2
        if seq_len > 24:
            weights[24:] = 0.2

        return weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the weighted MSE loss function.

        Args:
            pred: Predicted values of shape (batch_size, seq_len, feature_dim).
            target: Target values of shape (batch_size, seq_len, feature_dim).

        Returns:
            Computed weighted loss value.
        """
        batch_size, seq_len, feature_dim = pred.shape

        # Compute base MSE loss
        base_loss = self.mse_loss(pred, target)  # Shape: (batch_size, seq_len, feature_dim)

        # Compute time-based weights
        time_weights = self._compute_time_weights(seq_len, pred.device)
        time_weights = time_weights.unsqueeze(0).unsqueeze(2)  # Shape: (1, seq_len, 1)

        # Apply weights to loss
        weighted_loss = base_loss * time_weights

        # Apply reduction
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class HuberMultiCriteriaLoss(nn.Module):
    """Huber Loss with Multi-criteria Weighting (Highest Priority).

    Combines Huber loss with temporal weighting (future emphasis) and
    gradient-based weighting (emphasizes rapid changes).

    Args:
        beta: Threshold for Huber loss transition between L2 and L1.
        temporal_weight_range: Tuple of (start_weight, end_weight) for temporal weighting.
        gradient_weight_scale: Scale factor for gradient-based weighting.
        reduction: Reduction method ('mean', 'sum', 'none').
    """

    def __init__(self,
                 beta: float = 0.3,
                 temporal_weight_range: Tuple[float, float] = (0.3, 1.0),
                 gradient_weight_scale: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.beta = beta
        self.temporal_weight_range = temporal_weight_range
        self.gradient_weight_scale = gradient_weight_scale
        self.reduction = reduction
        self.huber_loss = nn.SmoothL1Loss(beta=beta, reduction='none')

    def _compute_temporal_weights(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute weights that increase towards future timesteps.

        Args:
            seq_len: Length of the sequence.
            device: Device to create tensor on.

        Returns:
            Temporal weights tensor of shape (seq_len,).
        """
        start_weight, end_weight = self.temporal_weight_range
        return torch.linspace(start_weight, end_weight, seq_len, device=device)

    def _compute_gradient_weights(self, target: torch.Tensor) -> torch.Tensor:
        """Compute weights based on gradient magnitude (emphasizes rapid changes).

        Args:
            target: Target tensor of shape (batch_size, seq_len, feature_dim).

        Returns:
            Gradient-based weights tensor of shape (batch_size, seq_len).
        """
        # Compute temporal gradient (difference between consecutive timesteps)
        grad = torch.diff(target, dim=1)  # Shape: (batch_size, seq_len-1, feature_dim)
        grad_magnitude = torch.norm(grad, dim=2)  # Shape: (batch_size, seq_len-1)

        # Pad to match original sequence length
        grad_magnitude = F.pad(grad_magnitude, (1, 0), value=0)  # Shape: (batch_size, seq_len)

        # Apply exponential scaling to emphasize high gradients
        max_grad = grad_magnitude.max(dim=1, keepdim=True)[0] + 1e-8
        normalized_grad = grad_magnitude / max_grad
        grad_weights = torch.exp(normalized_grad * self.gradient_weight_scale)

        return torch.clamp(grad_weights, min=0.1, max=3.0)  # Prevent extreme weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the loss function.

        Args:
            pred: Predicted values of shape (batch_size, seq_len, feature_dim).
            target: Target values of shape (batch_size, seq_len, feature_dim).

        Returns:
            Computed loss value.
        """
        batch_size, seq_len, feature_dim = pred.shape

        # Compute base Huber loss
        base_loss = self.huber_loss(pred, target)  # Shape: (batch_size, seq_len, feature_dim)

        # Compute temporal weights (future emphasis)
        temporal_weights = self._compute_temporal_weights(seq_len, pred.device)
        temporal_weights = temporal_weights.unsqueeze(0).unsqueeze(2)  # Shape: (1, seq_len, 1)

        # Compute gradient weights (rapid change emphasis)
        gradient_weights = self._compute_gradient_weights(target)
        gradient_weights = gradient_weights.unsqueeze(2)  # Shape: (batch_size, seq_len, 1)

        # Combine weights
        combined_weights = temporal_weights * gradient_weights

        # Apply weights to loss
        weighted_loss = base_loss * combined_weights

        # Apply reduction
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class GeneralWeightedMSELoss(nn.Module):
    """General Target-value-based Weighted MSE Loss.

    Applies higher weights to samples with target values above a threshold.
    This is useful for regression problems with imbalanced target distributions,
    where rare high-value events need more emphasis during training.

    Args:
        threshold: Threshold value for weight assignment.
        high_weight: Weight for samples with target > threshold.
        low_weight: Weight for samples with target <= threshold.
        reduction: Reduction method ('mean', 'sum', 'none').
    """

    def __init__(
        self,
        threshold: float = 50.0,
        high_weight: float = 10.0,
        low_weight: float = 1.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.threshold = threshold
        self.high_weight = high_weight
        self.low_weight = low_weight
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the weighted MSE loss.

        Args:
            pred: Predicted values of shape (batch_size, seq_len, feature_dim).
            target: Target values of shape (batch_size, seq_len, feature_dim).

        Returns:
            Computed weighted loss value.
        """
        # Compute base MSE loss
        base_loss = self.mse_loss(pred, target)

        # Compute weights based on target values
        weights = torch.where(
            target > self.threshold,
            torch.tensor(self.high_weight, device=target.device, dtype=target.dtype),
            torch.tensor(self.low_weight, device=target.device, dtype=target.dtype)
        )

        # Apply weights
        weighted_loss = base_loss * weights

        # Apply reduction
        if self.reduction == 'mean':
            return weighted_loss.sum() / weights.sum()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class SolarWindWeightedLoss(nn.Module):
    """Weighted Loss for Solar Wind Prediction (Ap Index based).

    Applies weights based on NOAA geomagnetic activity scale for Ap Index.
    Supports three weighting modes:
    - 'threshold': Binary weighting based on a single threshold
    - 'continuous': Smooth power-scaled weighting
    - 'multi_tier': Multi-level weighting based on NOAA storm categories

    Optionally combines with temporal weighting to emphasize future predictions.

    Note on Normalization:
        The pipeline normalizes Ap index using log1p_zscore method. However, the
        tier boundaries (AP_TIERS) are defined in raw Ap units (0-400 range).
        When denormalize=True, targets are converted back to raw Ap values before
        weight calculation to ensure correct tier assignment.

    Args:
        base_loss_type: Type of base loss ('mse', 'mae', 'huber').
        weighting_mode: Weighting mode ('threshold', 'continuous', 'multi_tier').
        threshold: Threshold for binary weighting (weighting_mode='threshold').
        high_weight: Weight for high values (weighting_mode='threshold').
        alpha: Scale factor for continuous weighting (weighting_mode='continuous').
        beta: Power for continuous weighting (weighting_mode='continuous').
        temporal_weight_range: Tuple of (start_weight, end_weight) for temporal weighting.
        combine_temporal: Whether to combine with temporal weighting.
        huber_delta: Delta parameter for Huber loss.
        reduction: Reduction method ('mean', 'sum', 'none').
        denormalize: Whether to denormalize targets before weight computation.
        norm_method: Normalization method used in pipeline ('log1p_zscore', 'zscore').
        norm_stats: Normalization statistics dict with keys like 'log1p_mean', 'log1p_std'.
    """

    # NOAA Geomagnetic Activity Scale (Ap Index)
    # Reference:
    #   - NOAA Space Weather Scales: https://www.swpc.noaa.gov/noaa-scales-explanation
    #   - Kp to Ap conversion: https://www.ncei.noaa.gov/products/geomagnetic-indices
    #   - K-index Wikipedia: https://en.wikipedia.org/wiki/K-index
    #
    # Kp to Ap conversion table:
    #   Kp: 0   1   2   3   4   5   6   7   8   9
    #   Ap: 0   3   7  15  27  48  80 140 240 400
    #
    # G-Scale based tiers (simplified, 4 levels):
    AP_TIERS = {
        'none': (0, 29, 1.0),                   # No storm (Kp < 5)
        'g1': (30, 49, 2.0),                    # G1 Minor Storm (Kp 5)
        'g2': (50, 99, 4.0),                    # G2 Moderate Storm (Kp 6)
        'g3_plus': (100, float('inf'), 8.0),   # G3-G5 Strong-Extreme (Kp 7-9)
    }

    # Detailed tiers (6 levels, finer granularity):
    # AP_TIERS = {
    #     'quiet': (0, 7, 1.0),           # Kp 0-2
    #     'unsettled': (8, 15, 2.0),      # Kp 3
    #     'active': (16, 29, 4.0),        # Kp 4
    #     'minor_storm': (30, 49, 8.0),   # Kp 5, G1
    #     'moderate_storm': (50, 99, 12.0),   # Kp 6, G2
    #     'severe_storm': (100, float('inf'), 16.0),  # Kp 7-9, G3-G5
    # }

    def __init__(
        self,
        base_loss_type: str = 'mse',
        weighting_mode: str = 'multi_tier',
        threshold: float = 30.0,
        high_weight: float = 10.0,
        alpha: float = 5.0,
        beta: float = 1.5,
        temporal_weight_range: Tuple[float, float] = (0.5, 1.0),
        combine_temporal: bool = True,
        huber_delta: float = 1.0,
        reduction: str = 'mean',
        denormalize: bool = False,
        norm_method: str = 'log1p_zscore',
        norm_stats: Optional[dict] = None
    ):
        super().__init__()
        self.weighting_mode = weighting_mode
        self.threshold = threshold
        self.high_weight = high_weight
        self.alpha = alpha
        self.beta = beta
        self.temporal_weight_range = temporal_weight_range
        self.combine_temporal = combine_temporal
        self.reduction = reduction

        # Denormalization settings
        self.denormalize = denormalize
        self.norm_method = norm_method
        self.norm_stats = norm_stats or {}

        # Base loss function
        if base_loss_type == 'mse':
            self.base_loss_fn = nn.MSELoss(reduction='none')
        elif base_loss_type == 'mae':
            self.base_loss_fn = nn.L1Loss(reduction='none')
        elif base_loss_type == 'huber':
            self.base_loss_fn = nn.SmoothL1Loss(beta=huber_delta, reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {base_loss_type}")

    def _denormalize_target(self, target: torch.Tensor) -> torch.Tensor:
        """Denormalize target values back to original Ap scale.

        Converts normalized targets back to raw Ap values (0-400 range)
        for correct tier assignment.

        Args:
            target: Normalized target tensor.

        Returns:
            Denormalized target tensor in raw Ap units.
        """
        if not self.denormalize:
            return target

        if self.norm_method == 'log1p_zscore':
            log1p_mean = self.norm_stats.get('log1p_mean', 0.0)
            log1p_std = self.norm_stats.get('log1p_std', 1.0)
            # Reverse z-score: log1p_data = normalized * std + mean
            log1p_data = target * log1p_std + log1p_mean
            # Reverse log1p: raw = exp(log1p_data) - 1
            raw_target = torch.expm1(log1p_data)
            return self._clip_if_nonnegative(raw_target)

        elif self.norm_method == 'zscore':
            mean = self.norm_stats.get('mean', 0.0)
            std = self.norm_stats.get('std', 1.0)
            raw_target = target * std + mean
            return self._clip_if_nonnegative(raw_target)

        elif self.norm_method == 'log_zscore':
            log_mean = self.norm_stats.get('log_mean', 0.0)
            log_std = self.norm_stats.get('log_std', 1.0)
            log_data = target * log_std + log_mean
            return torch.exp(log_data)

        elif self.norm_method == 'minmax':
            min_val = self.norm_stats.get('min', 0.0)
            max_val = self.norm_stats.get('max', 1.0)
            raw_target = target * (max_val - min_val) + min_val
            return self._clip_if_nonnegative(raw_target)

        else:
            # Unknown method, return as-is
            return target

    def _clip_if_nonnegative(self, raw_target: torch.Tensor) -> torch.Tensor:
        """Clamp to non-negative only when the normalization implies a non-negative range.

        Non-negative targets (ap30/hp30 under log/log1p) are clamped at 0 for correct
        tier assignment. Signed targets (e.g. Dst under zscore) must keep their sign,
        otherwise negative storm values collapse to 0 and the tier weighting degrades
        to plain temporal-MSE.

        Args:
            raw_target: Denormalized target tensor in the original index scale.

        Returns:
            The tensor, clamped at 0 for non-negative methods, unchanged otherwise.
        """
        if method_is_nonnegative(self.norm_method):
            return torch.clamp(raw_target, min=0.0)
        return raw_target

    def _compute_threshold_weights(self, target: torch.Tensor) -> torch.Tensor:
        """Compute threshold-based binary weights.

        Args:
            target: Target tensor (in raw Ap units if denormalize=True).

        Returns:
            Weight tensor with same shape as target.
        """
        weights = torch.where(
            target > self.threshold,
            torch.tensor(self.high_weight, device=target.device, dtype=target.dtype),
            torch.tensor(1.0, device=target.device, dtype=target.dtype)
        )
        return weights

    def _compute_continuous_weights(self, target: torch.Tensor) -> torch.Tensor:
        """Compute continuous power-scaled weights.

        Args:
            target: Target tensor.

        Returns:
            Weight tensor with same shape as target.
        """
        # Normalize Ap values (typical max ~400)
        normalized = target / 400.0
        normalized = torch.clamp(normalized, 0, 1)

        # Power-scaled weighting: w = 1 + alpha * (Ap/Ap_max)^beta
        weights = 1.0 + self.alpha * torch.pow(normalized, self.beta)
        return weights

    def _compute_multi_tier_weights(self, target: torch.Tensor) -> torch.Tensor:
        """Compute multi-tier weights based on NOAA geomagnetic activity scale.

        Args:
            target: Target tensor.

        Returns:
            Weight tensor with same shape as target.
        """
        weights = torch.ones_like(target)

        for tier_name, (low, high, weight) in self.AP_TIERS.items():
            mask = (target >= low) & (target < high)
            weights = torch.where(
                mask,
                torch.tensor(weight, device=target.device, dtype=target.dtype),
                weights
            )

        return weights

    def _compute_ap_weights(self, target: torch.Tensor) -> torch.Tensor:
        """Compute Ap-value-based weights based on weighting mode.

        Args:
            target: Target tensor.

        Returns:
            Weight tensor with same shape as target.
        """
        if self.weighting_mode == 'threshold':
            return self._compute_threshold_weights(target)
        elif self.weighting_mode == 'continuous':
            return self._compute_continuous_weights(target)
        elif self.weighting_mode == 'multi_tier':
            return self._compute_multi_tier_weights(target)
        else:
            raise ValueError(f"Unknown weighting mode: {self.weighting_mode}")

    def _compute_temporal_weights(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute temporal weights that increase towards future timesteps.

        Args:
            seq_len: Length of the sequence.
            device: Device to create tensor on.

        Returns:
            Temporal weights tensor of shape (seq_len,).
        """
        start_weight, end_weight = self.temporal_weight_range
        return torch.linspace(start_weight, end_weight, seq_len, device=device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the solar wind weighted loss.

        Args:
            pred: Predicted values of shape (batch_size, seq_len, feature_dim).
            target: Target values of shape (batch_size, seq_len, feature_dim).
                   May be normalized (if denormalize=True, will be converted
                   back to raw Ap units for weight computation).

        Returns:
            Computed weighted loss value.
        """
        batch_size, seq_len, feature_dim = pred.shape

        # Compute base loss (using normalized values for consistent metrics)
        base_loss = self.base_loss_fn(pred, target)

        # Denormalize targets for weight computation if needed
        # This ensures tier boundaries are compared against raw Ap values
        target_for_weights = self._denormalize_target(target)

        # Compute Ap-value-based weights (using raw Ap values)
        ap_weights = self._compute_ap_weights(target_for_weights)

        # Optionally combine with temporal weights
        if self.combine_temporal:
            temporal_weights = self._compute_temporal_weights(seq_len, pred.device)
            temporal_weights = temporal_weights.view(1, seq_len, 1)
            combined_weights = ap_weights * temporal_weights
        else:
            combined_weights = ap_weights

        # Apply weights
        weighted_loss = base_loss * combined_weights

        # Apply reduction
        if self.reduction == 'mean':
            return weighted_loss.sum() / combined_weights.sum()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class MAEOutlierFocusedLoss(nn.Module):
    """MAE Loss with Outlier Detection and Future Temporal Weighting (Second Priority).

    Uses MAE loss (robust to outliers) with outlier-based weighting and future emphasis.

    Args:
        outlier_threshold: Z-score threshold for outlier detection.
        outlier_weight_multiplier: Multiplier for outlier regions.
        temporal_weight_range: Tuple of (start_weight, end_weight) for temporal weighting.
        reduction: Reduction method ('mean', 'sum', 'none').
    """

    def __init__(self,
                 outlier_threshold: float = 2.0,
                 outlier_weight_multiplier: float = 3.0,
                 temporal_weight_range: Tuple[float, float] = (0.3, 1.0),
                 reduction: str = 'mean'):
        super().__init__()
        self.outlier_threshold = outlier_threshold
        self.outlier_weight_multiplier = outlier_weight_multiplier
        self.temporal_weight_range = temporal_weight_range
        self.reduction = reduction
        self.mae_loss = nn.L1Loss(reduction='none')

    def _detect_outliers(self, target: torch.Tensor) -> torch.Tensor:
        """Detect outliers using Z-score method.

        Args:
            target: Target tensor of shape (batch_size, seq_len, feature_dim).

        Returns:
            Outlier weights tensor of shape (batch_size, seq_len).
        """
        # Compute statistics across feature dimensions
        mean = target.mean(dim=2, keepdim=True)  # Shape: (batch_size, seq_len, 1)
        std = target.std(dim=2, keepdim=True) + 1e-8  # Add epsilon for stability

        # Compute Z-scores
        z_scores = torch.abs((target - mean) / std)  # Shape: (batch_size, seq_len, feature_dim)
        max_z_score = z_scores.max(dim=2)[0]  # Shape: (batch_size, seq_len)

        # Create outlier weights
        outlier_mask = (max_z_score > self.outlier_threshold).float()
        outlier_weights = 1.0 + outlier_mask * (self.outlier_weight_multiplier - 1.0)

        return outlier_weights

    def _compute_temporal_weights(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute weights that increase towards future timesteps."""
        start_weight, end_weight = self.temporal_weight_range
        return torch.linspace(start_weight, end_weight, seq_len, device=device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the loss function."""
        batch_size, seq_len, feature_dim = pred.shape

        # Compute base MAE loss
        base_loss = self.mae_loss(pred, target)  # Shape: (batch_size, seq_len, feature_dim)

        # Compute temporal weights (future emphasis)
        temporal_weights = self._compute_temporal_weights(seq_len, pred.device)
        temporal_weights = temporal_weights.unsqueeze(0).unsqueeze(2)  # Shape: (1, seq_len, 1)

        # Compute outlier weights
        outlier_weights = self._detect_outliers(target)
        outlier_weights = outlier_weights.unsqueeze(2)  # Shape: (batch_size, seq_len, 1)

        # Combine weights
        combined_weights = temporal_weights * outlier_weights

        # Apply weights to loss
        weighted_loss = base_loss * combined_weights

        # Apply reduction
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


# =============================================================================
# Contrastive Losses for Multimodal Alignment
# =============================================================================

class MultiModalContrastiveLoss(nn.Module):
    """Contrastive loss for aligning multimodal representations.

    Uses InfoNCE-style loss to align transformer (solar wind) and ConvLSTM (image)
    features in the same embedding space. Features from the same sample are treated
    as positive pairs, while features from different samples in the batch serve as
    negative pairs (in-batch negatives).

    Args:
        temperature: Temperature parameter for scaling similarity scores.
                    Higher values (0.3-0.5) recommended for small batch sizes.
                    Lower values (0.07-0.1) for large batch sizes.
        normalize: Whether to L2-normalize features before computing similarity.
                  True is strongly recommended for stable training.
    """

    def __init__(self, temperature: float = 0.3, normalize: bool = True):
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")

        self.temperature = temperature
        self.normalize = normalize

    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss between two sets of features.

        Args:
            features1: Features from first modality of shape (batch_size, feature_dim).
            features2: Features from second modality of shape (batch_size, feature_dim).

        Returns:
            Contrastive loss scalar.

        Raises:
            ValueError: If input tensors have mismatched dimensions.
        """
        if features1.dim() != 2 or features2.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got {features1.dim()}D and {features2.dim()}D")

        if features1.size(0) != features2.size(0):
            raise ValueError(f"Batch sizes must match: {features1.size(0)} vs {features2.size(0)}")

        if features1.size(1) != features2.size(1):
            raise ValueError(f"Feature dimensions must match: {features1.size(1)} vs {features2.size(1)}")

        batch_size = features1.size(0)

        # L2 normalization for cosine similarity
        if self.normalize:
            features1 = F.normalize(features1, p=2, dim=1)
            features2 = F.normalize(features2, p=2, dim=1)

        # Concatenate features from both modalities
        # Shape: (2 * batch_size, feature_dim)
        features = torch.cat([features1, features2], dim=0)

        # Compute similarity matrix
        # Shape: (2 * batch_size, 2 * batch_size)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create labels for positive pairs
        # For each sample i in features1, its positive pair is sample i in features2
        # which is at index (i + batch_size) in the concatenated tensor
        labels = torch.arange(batch_size, device=features1.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        # Create mask to exclude self-similarity (diagonal elements)
        # We don't want to compare a sample with itself
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features1.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

        # Compute cross-entropy loss
        # This encourages high similarity with positive pairs and low similarity with negatives
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss


class MultiModalMSELoss(nn.Module):
    """MSE-based consistency loss for multimodal alignment.

    Directly minimizes the Euclidean distance between features from two modalities.
    This approach encourages the same sample's features from different modalities
    to have similar representations in the embedding space.

    Unlike contrastive learning approaches (InfoNCE), MSE loss does not use negative
    samples, making it more suitable for small batch sizes where negative samples
    would be limited.

    Args:
        reduction: Specifies the reduction to apply to the output.
                  'mean' (default): average of all elements
                  'sum': sum of all elements
                  'none': no reduction
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: {reduction}. Must be 'mean', 'sum', or 'none'")

        self.reduction = reduction

    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss between two sets of features.

        Args:
            features1: Features from first modality of shape (batch_size, feature_dim).
            features2: Features from second modality of shape (batch_size, feature_dim).

        Returns:
            MSE loss scalar (or tensor if reduction='none').

        Raises:
            ValueError: If input tensors have mismatched dimensions.
        """
        if features1.dim() != 2 or features2.dim() != 2:
            raise ValueError(f"Expected 2D tensors, got {features1.dim()}D and {features2.dim()}D")

        if features1.size(0) != features2.size(0):
            raise ValueError(f"Batch sizes must match: {features1.size(0)} vs {features2.size(0)}")

        if features1.size(1) != features2.size(1):
            raise ValueError(f"Feature dimensions must match: {features1.size(1)} vs {features2.size(1)}")

        # Compute MSE loss - directly aligns features from both modalities
        loss = F.mse_loss(features1, features2, reduction=self.reduction)

        return loss


# =============================================================================
# Advanced Losses
# =============================================================================

class AdaptiveWeightLoss(nn.Module):
    """Adaptive Weight Loss with Dynamic Error-based Weighting (Third Priority).

    Automatically assigns higher weights to larger errors (outliers/rapid changes).

    Args:
        base_loss_type: Type of base loss ('mse', 'mae', 'huber').
        beta: Beta parameter for Huber loss (if applicable).
        adaptive_power: Power for adaptive weighting (higher = more emphasis on large errors).
        temporal_weight_range: Tuple of (start_weight, end_weight) for temporal weighting.
        reduction: Reduction method ('mean', 'sum', 'none').
    """

    def __init__(self,
                 base_loss_type: str = 'huber',
                 beta: float = 0.5,
                 adaptive_power: float = 1.5,
                 temporal_weight_range: Tuple[float, float] = (0.3, 1.0),
                 reduction: str = 'mean'):
        super().__init__()
        self.base_loss_type = base_loss_type
        self.adaptive_power = adaptive_power
        self.temporal_weight_range = temporal_weight_range
        self.reduction = reduction

        if base_loss_type == 'mse':
            self.base_loss_fn = nn.MSELoss(reduction='none')
        elif base_loss_type == 'mae':
            self.base_loss_fn = nn.L1Loss(reduction='none')
        elif base_loss_type == 'huber':
            self.base_loss_fn = nn.SmoothL1Loss(beta=beta, reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {base_loss_type}")

    def _compute_adaptive_weights(self, errors: torch.Tensor) -> torch.Tensor:
        """Compute adaptive weights based on error magnitude.

        Args:
            errors: Error tensor of shape (batch_size, seq_len, feature_dim).

        Returns:
            Adaptive weights tensor of shape (batch_size, seq_len).
        """
        # Compute error magnitude across feature dimensions
        error_magnitude = torch.norm(errors, dim=2)  # Shape: (batch_size, seq_len)

        # Apply power scaling and normalization
        adaptive_weights = torch.pow(error_magnitude + 1e-8, self.adaptive_power)

        # Normalize weights per sequence to prevent extreme scaling
        adaptive_weights = adaptive_weights / (adaptive_weights.mean(dim=1, keepdim=True) + 1e-8)

        return torch.clamp(adaptive_weights, min=0.1, max=5.0)

    def _compute_temporal_weights(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute weights that increase towards future timesteps."""
        start_weight, end_weight = self.temporal_weight_range
        return torch.linspace(start_weight, end_weight, seq_len, device=device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the loss function."""
        batch_size, seq_len, feature_dim = pred.shape

        # Compute base loss
        base_loss = self.base_loss_fn(pred, target)  # Shape: (batch_size, seq_len, feature_dim)

        # Compute errors for adaptive weighting
        errors = torch.abs(pred - target)
        adaptive_weights = self._compute_adaptive_weights(errors)
        adaptive_weights = adaptive_weights.unsqueeze(2)  # Shape: (batch_size, seq_len, 1)

        # Compute temporal weights
        temporal_weights = self._compute_temporal_weights(seq_len, pred.device)
        temporal_weights = temporal_weights.unsqueeze(0).unsqueeze(2)  # Shape: (1, seq_len, 1)

        # Combine weights
        combined_weights = temporal_weights * adaptive_weights

        # Apply weights to loss
        weighted_loss = base_loss * combined_weights

        # Apply reduction
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class GradientBasedWeightLoss(nn.Module):
    """Gradient-based Weight Loss focusing on rapid changes (Fourth Priority).

    Emphasizes timesteps with high temporal gradients (rapid changes).

    Args:
        base_loss_type: Type of base loss ('mse', 'mae', 'huber').
        beta: Beta parameter for Huber loss (if applicable).
        gradient_weight_scale: Scale factor for gradient-based weighting.
        temporal_weight_range: Tuple of (start_weight, end_weight) for temporal weighting.
        reduction: Reduction method ('mean', 'sum', 'none').
    """

    def __init__(self,
                 base_loss_type: str = 'mae',
                 beta: float = 0.5,
                 gradient_weight_scale: float = 3.0,
                 temporal_weight_range: Tuple[float, float] = (0.3, 1.0),
                 reduction: str = 'mean'):
        super().__init__()
        self.base_loss_type = base_loss_type
        self.gradient_weight_scale = gradient_weight_scale
        self.temporal_weight_range = temporal_weight_range
        self.reduction = reduction

        if base_loss_type == 'mse':
            self.base_loss_fn = nn.MSELoss(reduction='none')
        elif base_loss_type == 'mae':
            self.base_loss_fn = nn.L1Loss(reduction='none')
        elif base_loss_type == 'huber':
            self.base_loss_fn = nn.SmoothL1Loss(beta=beta, reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {base_loss_type}")

    def _compute_gradient_weights(self, target: torch.Tensor) -> torch.Tensor:
        """Compute weights based on temporal gradient magnitude."""
        # Compute temporal gradients
        grad = torch.diff(target, dim=1)  # Shape: (batch_size, seq_len-1, feature_dim)
        grad_magnitude = torch.norm(grad, dim=2)  # Shape: (batch_size, seq_len-1)

        # Pad to match original sequence length
        grad_magnitude = F.pad(grad_magnitude, (1, 0), value=0)  # Shape: (batch_size, seq_len)

        # Apply exponential weighting to emphasize high gradients
        max_grad = grad_magnitude.max(dim=1, keepdim=True)[0] + 1e-8
        normalized_grad = grad_magnitude / max_grad
        gradient_weights = torch.exp(normalized_grad * self.gradient_weight_scale)

        return torch.clamp(gradient_weights, min=0.2, max=4.0)

    def _compute_temporal_weights(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute weights that increase towards future timesteps."""
        start_weight, end_weight = self.temporal_weight_range
        return torch.linspace(start_weight, end_weight, seq_len, device=device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the loss function."""
        batch_size, seq_len, feature_dim = pred.shape

        # Compute base loss
        base_loss = self.base_loss_fn(pred, target)  # Shape: (batch_size, seq_len, feature_dim)

        # Compute gradient weights
        gradient_weights = self._compute_gradient_weights(target)
        gradient_weights = gradient_weights.unsqueeze(2)  # Shape: (batch_size, seq_len, 1)

        # Compute temporal weights
        temporal_weights = self._compute_temporal_weights(seq_len, pred.device)
        temporal_weights = temporal_weights.unsqueeze(0).unsqueeze(2)  # Shape: (1, seq_len, 1)

        # Combine weights
        combined_weights = temporal_weights * gradient_weights

        # Apply weights to loss
        weighted_loss = base_loss * combined_weights

        # Apply reduction
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class QuantileLoss(nn.Module):
    """Quantile Loss with uncertainty-based weighting (Fifth Priority).

    Provides prediction intervals along with point estimates.

    Args:
        quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9]).
        temporal_weight_range: Tuple of (start_weight, end_weight) for temporal weighting.
        uncertainty_weight_scale: Scale factor for uncertainty-based weighting.
        reduction: Reduction method ('mean', 'sum', 'none').
    """

    def __init__(self,
                 quantiles: list = [0.1, 0.5, 0.9],
                 temporal_weight_range: Tuple[float, float] = (0.3, 1.0),
                 uncertainty_weight_scale: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.quantiles = quantiles
        self.temporal_weight_range = temporal_weight_range
        self.uncertainty_weight_scale = uncertainty_weight_scale
        self.reduction = reduction

    def _quantile_loss(self, pred: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
        """Compute quantile loss for a specific quantile.

        Args:
            pred: Predicted values.
            target: Target values.
            quantile: Quantile level.

        Returns:
            Quantile loss.
        """
        errors = target - pred
        loss = torch.max((quantile - 1) * errors, quantile * errors)
        return loss

    def _compute_uncertainty_weights(self, pred: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty weights based on prediction interval width.

        Args:
            pred: Predicted quantiles of shape (batch_size, seq_len, feature_dim, num_quantiles).

        Returns:
            Uncertainty weights of shape (batch_size, seq_len).
        """
        # Compute prediction interval width (difference between high and low quantiles)
        interval_width = pred[..., -1] - pred[..., 0]  # Shape: (batch_size, seq_len, feature_dim)
        interval_width = torch.mean(interval_width, dim=2)  # Shape: (batch_size, seq_len)

        # Normalize and apply exponential weighting
        max_width = interval_width.max(dim=1, keepdim=True)[0] + 1e-8
        normalized_width = interval_width / max_width
        uncertainty_weights = torch.exp(normalized_width * self.uncertainty_weight_scale)

        return torch.clamp(uncertainty_weights, min=0.3, max=3.0)

    def _compute_temporal_weights(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute weights that increase towards future timesteps."""
        start_weight, end_weight = self.temporal_weight_range
        return torch.linspace(start_weight, end_weight, seq_len, device=device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the quantile loss function.

        Args:
            pred: Predicted quantiles of shape (batch_size, seq_len, feature_dim, num_quantiles).
            target: Target values of shape (batch_size, seq_len, feature_dim).

        Returns:
            Computed loss value.
        """
        batch_size, seq_len, feature_dim = target.shape

        # Ensure pred has the correct number of quantiles
        if pred.shape[-1] != len(self.quantiles):
            raise ValueError(f"Expected {len(self.quantiles)} quantiles, got {pred.shape[-1]}")

        # Compute quantile losses
        total_loss = 0
        for i, quantile in enumerate(self.quantiles):
            pred_q = pred[..., i]  # Shape: (batch_size, seq_len, feature_dim)
            q_loss = self._quantile_loss(pred_q, target, quantile)
            total_loss += q_loss

        # Average over quantiles
        total_loss = total_loss / len(self.quantiles)  # Shape: (batch_size, seq_len, feature_dim)

        # Compute uncertainty weights
        uncertainty_weights = self._compute_uncertainty_weights(pred)
        uncertainty_weights = uncertainty_weights.unsqueeze(2)  # Shape: (batch_size, seq_len, 1)

        # Compute temporal weights
        temporal_weights = self._compute_temporal_weights(seq_len, pred.device)
        temporal_weights = temporal_weights.unsqueeze(0).unsqueeze(2)  # Shape: (1, seq_len, 1)

        # Combine weights
        combined_weights = temporal_weights * uncertainty_weights

        # Apply weights to loss
        weighted_loss = total_loss * combined_weights

        # Apply reduction
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class MultiTaskLoss(nn.Module):
    """Multi-task Learning Loss for simultaneous regression and outlier detection (Sixth Priority).

    Combines regression loss with auxiliary outlier detection task.

    Args:
        regression_loss_type: Type of regression loss ('mse', 'mae', 'huber').
        beta: Beta parameter for Huber loss (if applicable).
        outlier_loss_weight: Weight for outlier detection loss.
        temporal_weight_range: Tuple of (start_weight, end_weight) for temporal weighting.
        outlier_threshold: Z-score threshold for outlier labeling.
        reduction: Reduction method ('mean', 'sum', 'none').
    """

    def __init__(self,
                 regression_loss_type: str = 'huber',
                 beta: float = 0.5,
                 outlier_loss_weight: float = 0.3,
                 temporal_weight_range: Tuple[float, float] = (0.3, 1.0),
                 outlier_threshold: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.regression_loss_type = regression_loss_type
        self.outlier_loss_weight = outlier_loss_weight
        self.temporal_weight_range = temporal_weight_range
        self.outlier_threshold = outlier_threshold
        self.reduction = reduction

        # Regression loss
        if regression_loss_type == 'mse':
            self.regression_loss_fn = nn.MSELoss(reduction='none')
        elif regression_loss_type == 'mae':
            self.regression_loss_fn = nn.L1Loss(reduction='none')
        elif regression_loss_type == 'huber':
            self.regression_loss_fn = nn.SmoothL1Loss(beta=beta, reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {regression_loss_type}")

        # Outlier detection loss
        self.outlier_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def _generate_outlier_labels(self, target: torch.Tensor) -> torch.Tensor:
        """Generate outlier labels using Z-score method.

        Args:
            target: Target tensor of shape (batch_size, seq_len, feature_dim).

        Returns:
            Outlier labels tensor of shape (batch_size, seq_len).
        """
        # Compute statistics across feature dimensions
        mean = target.mean(dim=2, keepdim=True)
        std = target.std(dim=2, keepdim=True) + 1e-8

        # Compute Z-scores
        z_scores = torch.abs((target - mean) / std)
        max_z_score = z_scores.max(dim=2)[0]  # Shape: (batch_size, seq_len)

        # Create binary outlier labels
        outlier_labels = (max_z_score > self.outlier_threshold).float()

        return outlier_labels

    def _compute_temporal_weights(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute weights that increase towards future timesteps."""
        start_weight, end_weight = self.temporal_weight_range
        return torch.linspace(start_weight, end_weight, seq_len, device=device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                outlier_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the multi-task loss function.

        Args:
            pred: Predicted regression values of shape (batch_size, seq_len, feature_dim).
            target: Target values of shape (batch_size, seq_len, feature_dim).
            outlier_logits: Predicted outlier logits of shape (batch_size, seq_len).
                          If None, outlier detection loss is skipped.

        Returns:
            Combined loss value.
        """
        batch_size, seq_len, feature_dim = pred.shape

        # Compute regression loss
        regression_loss = self.regression_loss_fn(pred, target)  # Shape: (batch_size, seq_len, feature_dim)

        # Compute temporal weights
        temporal_weights = self._compute_temporal_weights(seq_len, pred.device)
        temporal_weights = temporal_weights.unsqueeze(0).unsqueeze(2)  # Shape: (1, seq_len, 1)

        # Apply temporal weights to regression loss
        weighted_regression_loss = regression_loss * temporal_weights

        # Compute total loss
        total_loss = weighted_regression_loss

        # Add outlier detection loss if outlier logits are provided
        if outlier_logits is not None:
            outlier_labels = self._generate_outlier_labels(target)  # Shape: (batch_size, seq_len)
            outlier_loss = self.outlier_loss_fn(outlier_logits, outlier_labels)  # Shape: (batch_size, seq_len)

            # Apply temporal weights to outlier loss
            temporal_weights_2d = temporal_weights.squeeze(2)  # Shape: (1, seq_len)
            weighted_outlier_loss = outlier_loss * temporal_weights_2d

            # Add to total loss
            total_loss = total_loss + self.outlier_loss_weight * weighted_outlier_loss.unsqueeze(2)

        # Apply reduction
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss


# =============================================================================
# Factory Function
# =============================================================================

def create_loss_functions(config, stat_dict: Optional[dict] = None):
    """Create loss functions from config.

    Args:
        config: Hydra configuration object
        stat_dict: Optional normalization statistics dictionary.
                   Required for SolarWindWeightedLoss with denormalize=True.
                   Format: {'ap_index_nt': {'log1p_mean': float, 'log1p_std': float, ...}}

    Returns:
        Tuple of (regression_criterion, contrastive_criterion)
    """
    regression_loss_type = config.training.regression_loss_type.lower()

    if regression_loss_type == "mse":
        regression_criterion = nn.MSELoss()
        regression_loss_name = "MSE"
    elif regression_loss_type == "mae":
        regression_criterion = nn.L1Loss()
        regression_loss_name = "MAE"
    elif regression_loss_type == "huber":
        regression_criterion = nn.HuberLoss(delta=config.training.huber_delta)
        regression_loss_name = "Huber"
    elif regression_loss_type == "weighted_mse":
        weighted_cfg = config.training.weighted_mse
        regression_criterion = GeneralWeightedMSELoss(
            threshold=weighted_cfg.threshold,
            high_weight=weighted_cfg.high_weight,
            low_weight=weighted_cfg.low_weight,
            reduction='mean'
        )
        regression_loss_name = "WeightedMSE"
    elif regression_loss_type == "solar_wind_weighted":
        sw_cfg = config.training.solar_wind_weighted

        # Determine target variable name dynamically (ap30 for CSV, ap_index_nt for HDF5)
        use_csv = getattr(config.data.modalities, 'timeseries', False)
        if use_csv:
            target_var = list(config.data.timeseries.target_variables)[0]
        elif hasattr(config.data, 'omni') and hasattr(config.data.omni, 'target'):
            target_var = list(config.data.omni.target.variables)[0]
        else:
            target_var = list(config.data.target_variables)[0]

        # Get normalization settings for target (Ap index)
        denormalize = getattr(sw_cfg, 'denormalize', True)
        norm_method = 'log1p_zscore'  # Default for Ap-type variables
        norm_stats = {}

        # Get normalization method from active config
        if use_csv and hasattr(config.data.timeseries, 'normalization'):
            norm_methods = config.data.timeseries.normalization.methods
            if hasattr(norm_methods, target_var):
                norm_method = getattr(norm_methods, target_var)
        elif hasattr(config.data, 'normalization') and hasattr(config.data.normalization, 'methods'):
            norm_methods = config.data.normalization.methods
            if hasattr(norm_methods, target_var):
                norm_method = getattr(norm_methods, target_var)

        # Get normalization statistics if provided
        if stat_dict is not None and target_var in stat_dict:
            norm_stats = stat_dict[target_var]
        elif denormalize:
            print(f"Warning: SolarWindWeightedLoss denormalize=True but no statistics for '{target_var}'.")
            print("         Weight calculation will use normalized values (may not match NOAA tiers).")
            denormalize = False

        regression_criterion = SolarWindWeightedLoss(
            base_loss_type=sw_cfg.base_loss,
            weighting_mode=sw_cfg.weighting_mode,
            threshold=sw_cfg.threshold,
            high_weight=sw_cfg.high_weight,
            alpha=sw_cfg.alpha,
            beta=sw_cfg.beta,
            temporal_weight_range=tuple(sw_cfg.temporal_weight_range),
            combine_temporal=sw_cfg.combine_temporal,
            huber_delta=config.training.huber_delta,
            reduction='mean',
            denormalize=denormalize,
            norm_method=norm_method,
            norm_stats=norm_stats
        )
        denorm_status = "enabled" if denormalize else "disabled"
        regression_loss_name = f"SolarWindWeighted({sw_cfg.weighting_mode}, denorm={denorm_status})"
    elif regression_loss_type == "none":
        # For two-stage training: Stage 1 uses only contrastive loss
        regression_criterion = None
        regression_loss_name = "None (contrastive only)"
    else:
        regression_criterion = nn.MSELoss()
        regression_loss_name = "MSE"
        print(f"Unknown regression loss type '{regression_loss_type}', using MSE")

    contrastive_loss_type = config.training.contrastive_loss_type.lower()
    if contrastive_loss_type == 'infonce':
        contrastive_criterion = MultiModalContrastiveLoss(
            temperature=config.training.contrastive_temperature,
            normalize=True
        )
        contrastive_loss_name = "InfoNCE"
    elif contrastive_loss_type == 'consistency':
        contrastive_criterion = MultiModalMSELoss(reduction='mean')
        contrastive_loss_name = "Consistency"
    else:
        contrastive_criterion = MultiModalMSELoss(reduction='mean')
        contrastive_loss_name = "Consistency"
        print(f"Unknown contrastive loss type '{contrastive_loss_type}', using consistency")

    print(f"Losses: Regression={regression_loss_name}, Contrastive={contrastive_loss_name}")

    return regression_criterion, contrastive_criterion


# =============================================================================
# Verification
# =============================================================================

def verify_losses(config) -> None:
    """Verify loss functions with config.

    Args:
        config: Hydra configuration object
    """
    from .pipeline import get_sdo_indices, get_omni_input_indices, get_omni_target_indices
    from .pipeline import get_sdo_wavelengths, get_input_variables, get_target_variables
    from .networks import create_model

    print("=" * 60)
    print("Loss Functions Verification")
    print("=" * 60)

    # Create model and losses
    model = create_model(config)
    regression_criterion, contrastive_criterion = create_loss_functions(config)

    print(f"\nRegression: {regression_criterion}")
    print(f"Contrastive: {contrastive_criterion}")

    # Create dummy data
    batch_size = config.experiment.batch_size
    sdo_start, sdo_end = get_sdo_indices(config)
    input_start, input_end = get_omni_input_indices(config)
    target_start, target_end = get_omni_target_indices(config)

    sdo = torch.randn(
        batch_size,
        len(get_sdo_wavelengths(config)),
        sdo_end - sdo_start,
        64, 64
    )
    inputs = torch.randn(
        batch_size,
        input_end - input_start,
        len(get_input_variables(config))
    )
    targets = torch.randn(
        batch_size,
        target_end - target_start,
        len(get_target_variables(config))
    )

    print(f"\nSDO: {sdo.shape}")
    print(f"Inputs: {inputs.shape}")
    print(f"Targets: {targets.shape}")

    # Forward pass
    outputs, tf_feat, cl_feat = model(inputs, sdo, return_features=True)
    print(f"Outputs: {outputs.shape}")

    # Compute losses
    reg_loss = regression_criterion(outputs, targets)
    print(f"Regression loss: {reg_loss.item():.4f}")

    if tf_feat is not None and cl_feat is not None:
        con_loss = contrastive_criterion(tf_feat, cl_feat)
        print(f"Contrastive loss: {con_loss.item():.4f}")

    print("\nLoss verification completed!")
    print("=" * 60)


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="../configs", config_name="local", version_base=None)
    def main(config: DictConfig):
        verify_losses(config)

    main()
