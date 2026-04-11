"""Temporal Convolutional Network (TCN) models for time series."""

from typing import Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import _get_model_dimensions
from ._registry import register_model


class TemporalBlock(nn.Module):
    """Temporal block with dilated causal convolution.

    A single residual block consisting of two dilated causal convolutions
    with weight normalization, dropout, and a residual connection.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        dilation: Dilation factor for causal convolution.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()

        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be positive and odd")
        if dilation <= 0:
            raise ValueError("Dilation must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Causal padding: (kernel_size - 1) * dilation on left side only
        self.padding = (kernel_size - 1) * dilation

        # First convolution with weight normalization
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                dilation=dilation, padding=self.padding
            )
        )

        # Second convolution with weight normalization
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                out_channels, out_channels, kernel_size,
                dilation=dilation, padding=self.padding
            )
        )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        # Residual connection (1x1 conv if channels differ)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, seq_len).

        Returns:
            Output tensor of shape (batch, out_channels, seq_len).
        """
        # First conv block
        out = self.conv1(x)
        out = out[:, :, :-self.padding]  # Remove future values (causal)
        out = self.relu(out)
        out = self.dropout(out)

        # Second conv block
        out = self.conv2(out)
        out = out[:, :, :-self.padding]  # Remove future values (causal)
        out = self.relu(out)
        out = self.dropout(out)

        # Residual connection
        res = self.residual(x)

        return self.relu(out + res)


class TCNEncoder(nn.Module):
    """Temporal Convolutional Network encoder for time series.

    Uses stacked TemporalBlocks with exponentially increasing dilation
    to capture long-range dependencies while maintaining causality.

    Receptive field = 2 * (kernel_size - 1) * sum(2^i for i in range(num_layers)) + 1

    Args:
        num_input_variables: Number of input variables per timestep.
        input_sequence_length: Length of input sequence.
        channels: List of channel sizes for each layer.
        kernel_size: Convolution kernel size.
        dropout: Dropout rate.
        output_dim: Output feature dimension.
    """

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        channels: list = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        output_dim: int = 128
    ):
        super().__init__()

        if num_input_variables <= 0:
            raise ValueError(f"Number of input variables must be positive, got {num_input_variables}")
        if input_sequence_length <= 0:
            raise ValueError(f"Input sequence length must be positive, got {input_sequence_length}")
        if output_dim <= 0:
            raise ValueError(f"Output dimension must be positive, got {output_dim}")

        # Default channel configuration
        if channels is None:
            channels = [64, 128, 256]

        self.num_input_variables = num_input_variables
        self.input_sequence_length = input_sequence_length
        self.output_dim = output_dim

        # Input projection
        self.input_projection = nn.Linear(num_input_variables, channels[0])

        # Build TCN layers with exponential dilation
        layers = []
        num_channels = [channels[0]] + list(channels)
        for i in range(len(channels)):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels=num_channels[i],
                    out_channels=num_channels[i + 1],
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )

        self.tcn = nn.Sequential(*layers)

        # Global pooling and output projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Linear(channels[-1], output_dim)

        # Calculate receptive field
        self._receptive_field = self._calculate_receptive_field(kernel_size, len(channels))

    def _calculate_receptive_field(self, kernel_size: int, num_layers: int) -> int:
        """Calculate the receptive field of the TCN."""
        # rf = 1 + 2 * (kernel_size - 1) * sum(2^i for i in range(num_layers))
        dilation_sum = sum(2 ** i for i in range(num_layers))
        return 1 + 2 * (kernel_size - 1) * dilation_sum

    @property
    def receptive_field(self) -> int:
        """Return the receptive field of the TCN."""
        return self._receptive_field

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, num_vars).

        Returns:
            Output features of shape (batch, output_dim).
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, seq_len, vars), got {x.dim()}D")

        batch_size, seq_len, num_vars = x.size()

        if seq_len != self.input_sequence_length:
            raise ValueError(f"Expected seq_len {self.input_sequence_length}, got {seq_len}")
        if num_vars != self.num_input_variables:
            raise ValueError(f"Expected {self.num_input_variables} vars, got {num_vars}")

        # Project input variables to channels
        x = self.input_projection(x)  # (batch, seq_len, channels[0])

        # Transpose for Conv1d: (batch, channels, seq_len)
        x = x.transpose(1, 2)

        # Apply TCN
        x = self.tcn(x)  # (batch, channels[-1], seq_len)

        # Global pooling
        x = self.global_pool(x).squeeze(-1)  # (batch, channels[-1])

        # Output projection
        x = self.output_projection(x)  # (batch, output_dim)

        return x


class TCNOnlyModel(nn.Module):
    """OMNI time series-only model using TCN encoder.

    Uses Temporal Convolutional Networks (TCN) for time series processing.
    TCN uses dilated causal convolutions to capture long-range dependencies
    while maintaining causality (no information leakage from future).

    Args:
        num_input_variables: Number of OMNI input variables.
        input_sequence_length: Length of OMNI input sequence.
        num_target_variables: Number of target variables to predict.
        target_sequence_length: Length of prediction sequence.
        d_model: Feature dimension for output.
        tcn_channels: List of channel sizes for TCN layers.
        tcn_kernel_size: Kernel size for TCN convolutions.
        dropout: Dropout rate for regularization.
    """

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        num_target_variables: int,
        target_sequence_length: int,
        d_model: int = 128,
        tcn_channels: list = None,
        tcn_kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        if num_target_variables <= 0 or target_sequence_length <= 0:
            raise ValueError("Target variables and sequence length must be positive")

        if tcn_channels is None:
            tcn_channels = [64, 128, 256]

        self.num_target_variables = num_target_variables
        self.target_sequence_length = target_sequence_length
        self.d_model = d_model

        # TCN encoder
        self.tcn_encoder = TCNEncoder(
            num_input_variables=num_input_variables,
            input_sequence_length=input_sequence_length,
            channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=dropout,
            output_dim=d_model
        )

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, target_sequence_length * num_target_variables)
        )

    @property
    def receptive_field(self) -> int:
        """Return the receptive field of the TCN encoder."""
        return self.tcn_encoder.receptive_field

    def forward(
        self,
        solar_wind_input: torch.Tensor,
        image_input: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, None]]:
        """Forward pass.

        Args:
            solar_wind_input: OMNI time series (B, seq_len, num_vars)
            image_input: Ignored (for API compatibility)
            return_features: Whether to return intermediate features

        Returns:
            Predictions of shape (B, target_seq_len, num_target_vars)
        """
        # Only use solar_wind_input, ignore image_input
        tcn_features = self.tcn_encoder(solar_wind_input)

        predictions = self.regression_head(tcn_features)
        output = predictions.reshape(
            predictions.size(0), self.target_sequence_length, self.num_target_variables
        )

        if return_features:
            return output, tcn_features, None
        return output


@register_model("tcn")
def _create_tcn(config):
    """Factory function for TCN model."""
    num_input_variables, input_sequence_length, \
        num_target_variables, target_sequence_length = _get_model_dimensions(config)

    print(f"Creating tcn model: Output shape (batch, {target_sequence_length}, {num_target_variables})")

    tcn_channels = getattr(config.model, 'tcn_channels', [64, 128, 256])
    tcn_kernel_size = getattr(config.model, 'tcn_kernel_size', 3)
    tcn_dropout = getattr(config.model, 'tcn_dropout', 0.1)

    # Convert OmegaConf list to Python list if necessary
    if hasattr(tcn_channels, '__iter__') and not isinstance(tcn_channels, list):
        tcn_channels = list(tcn_channels)

    model = TCNOnlyModel(
        num_input_variables=num_input_variables,
        input_sequence_length=input_sequence_length,
        num_target_variables=num_target_variables,
        target_sequence_length=target_sequence_length,
        d_model=config.model.d_model,
        tcn_channels=tcn_channels,
        tcn_kernel_size=tcn_kernel_size,
        dropout=tcn_dropout
    )
    print(f"  TCN receptive field: {model.receptive_field} timesteps")
    return model
