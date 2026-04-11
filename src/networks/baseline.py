"""Baseline model (Son et al. 2023 style - Conv3D + Linear)."""

from typing import Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import _get_model_dimensions
from ._registry import register_model
from .linear import LinearEncoder
from .convlstm import ConvLSTMModel


class Conv3DEncoder(nn.Module):
    """3D Convolutional encoder for SDO image sequences.

    Simplified version of Son et al. (2023) architecture.
    Uses Conv3D layers instead of Inception blocks for simplicity.

    Architecture:
        Conv3d(3->32) -> BN -> ReLU -> MaxPool(1x2x2)
        Conv3d(32->64) -> BN -> ReLU -> MaxPool(2x2x2)
        Conv3d(64->128) -> BN -> ReLU -> MaxPool(2x2x2)
        AdaptiveAvgPool3d -> Linear -> Dropout

    Args:
        input_channels: Number of input channels (e.g., 3 for wavelengths).
        output_dim: Output feature dimension.
        dropout: Dropout rate for regularization.
    """

    def __init__(self, input_channels: int, output_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()

        if input_channels <= 0:
            raise ValueError("Input channels must be positive")
        if output_dim <= 0:
            raise ValueError("Output dimension must be positive")

        self.input_channels = input_channels
        self.output_dim = output_dim

        # Conv3D layers: (B, C, T, H, W)
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2))  # Spatial pooling only
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2))  # Temporal + spatial pooling
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )

        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.projection = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, T, H, W) where
               B=batch, C=channels, T=timesteps, H=height, W=width

        Returns:
            Features of shape (B, output_dim)
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B, C, T, H, W), got {x.dim()}D")

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.global_pool(x)  # (B, 128, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 128)
        x = self.projection(x)  # (B, output_dim)

        return x


class BaselineModel(nn.Module):
    """Baseline fusion model with Conv3D + Linear encoders.

    Based on Son et al. (2023) "Three-day Forecasting of Solar Wind Speed
    Using SDO/AIA EUV Images", but simplified:
    - Conv3D instead of Inception blocks
    - Linear layers instead of LSTM for time series
    - Simple concatenation + dense fusion

    This model includes Dropout layers for MC Dropout uncertainty estimation.

    Args:
        num_input_variables: Number of OMNI input variables.
        input_sequence_length: Length of OMNI input sequence.
        num_target_variables: Number of target variables to predict.
        target_sequence_length: Length of prediction sequence.
        d_model: Feature dimension for encoders.
        input_channels: Number of SDO image channels.
        dropout: Dropout rate (used throughout for MC Dropout support).
    """

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        num_target_variables: int,
        target_sequence_length: int,
        d_model: int = 256,
        input_channels: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        if num_target_variables <= 0 or target_sequence_length <= 0:
            raise ValueError("Target variables and sequence length must be positive")

        self.num_target_variables = num_target_variables
        self.target_sequence_length = target_sequence_length
        self.d_model = d_model

        # Image encoder (Conv3D)
        self.image_encoder = Conv3DEncoder(
            input_channels=input_channels,
            output_dim=d_model,
            dropout=dropout
        )

        # Time series encoder (Linear)
        ts_input_size = input_sequence_length * num_input_variables
        self.ts_encoder = LinearEncoder(
            input_size=ts_input_size,
            output_dim=d_model,
            hidden_dim=d_model,
            dropout=dropout
        )

        # Fusion head (concatenate + dense)
        self.fusion_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, target_sequence_length * num_target_variables)
        )

    def forward(
        self,
        solar_wind_input: torch.Tensor,
        image_input: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            solar_wind_input: OMNI time series (B, seq_len, num_vars)
            image_input: SDO images (B, C, T, H, W)
            return_features: If True, return encoder features for analysis

        Returns:
            If return_features=False:
                Predictions of shape (B, target_seq_len, num_target_vars)
            If return_features=True:
                Tuple of (predictions, ts_features, img_features)
        """
        if solar_wind_input is None or image_input is None:
            raise ValueError("Both solar_wind_input and image_input must be provided")

        if solar_wind_input.size(0) != image_input.size(0):
            raise ValueError(
                f"Batch sizes must match: {solar_wind_input.size(0)} vs {image_input.size(0)}"
            )

        # Encode both modalities
        ts_features = self.ts_encoder(solar_wind_input)  # (B, d_model)
        img_features = self.image_encoder(image_input)   # (B, d_model)

        # Concatenate and fuse
        combined = torch.cat([ts_features, img_features], dim=1)  # (B, d_model*2)
        predictions = self.fusion_head(combined)

        # Reshape to output format
        output = predictions.reshape(
            predictions.size(0), self.target_sequence_length, self.num_target_variables
        )

        if return_features:
            return output, ts_features, img_features

        return output


@register_model("baseline")
def _create_baseline(config):
    """Factory function for Baseline model."""
    num_input_variables, input_sequence_length, \
        num_target_variables, target_sequence_length = _get_model_dimensions(config)

    print(f"Creating baseline model: Output shape (batch, {target_sequence_length}, {num_target_variables})")

    # Baseline model (Son et al. 2023 style: Conv3D + Linear)
    baseline_dropout = getattr(config.model, 'baseline_dropout', 0.1)
    return BaselineModel(
        num_input_variables=num_input_variables,
        input_sequence_length=input_sequence_length,
        num_target_variables=num_target_variables,
        target_sequence_length=target_sequence_length,
        d_model=config.model.d_model,
        input_channels=config.model.convlstm_input_channels,
        dropout=baseline_dropout
    )
