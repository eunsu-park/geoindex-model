"""ConvLSTM models for SDO image sequence processing."""

from typing import Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import _get_model_dimensions
from ._registry import register_model


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell for spatial-temporal processing.

    Args:
        input_channels: Number of input channels.
        hidden_channels: Number of hidden channels.
        kernel_size: Size of convolutional kernel (must be odd).
        bias: Whether to use bias in convolutions.
    """

    def __init__(self, input_channels: int, hidden_channels: int,
                 kernel_size: int = 3, bias: bool = True):
        super().__init__()

        if input_channels <= 0 or hidden_channels <= 0:
            raise ValueError("Input and hidden channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be positive and odd")

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv_ih = nn.Conv2d(
            input_channels, 4 * hidden_channels,
            kernel_size, padding=self.padding, bias=bias
        )
        self.conv_hh = nn.Conv2d(
            hidden_channels, 4 * hidden_channels,
            kernel_size, padding=self.padding, bias=bias
        )

    def forward(self, input_tensor: torch.Tensor,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = input_tensor.size()

        if hidden_state is None:
            device = input_tensor.device
            hidden = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
            cell = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        else:
            hidden, cell = hidden_state

        conv_ih = self.conv_ih(input_tensor)
        conv_hh = self.conv_hh(hidden)
        combined_conv = conv_ih + conv_hh

        i_gate, f_gate, o_gate, g_gate = torch.split(combined_conv, self.hidden_channels, dim=1)

        input_gate = torch.sigmoid(i_gate)
        forget_gate = torch.sigmoid(f_gate)
        output_gate = torch.sigmoid(o_gate)
        candidate_gate = torch.tanh(g_gate)

        new_cell = forget_gate * cell + input_gate * candidate_gate
        new_hidden = output_gate * torch.tanh(new_cell)

        return new_hidden, new_cell


class ConvLSTMModel(nn.Module):
    """Multi-layer ConvLSTM for image sequence processing.

    Args:
        input_channels: Number of input channels (e.g., 3 for RGB/wavelengths).
        hidden_channels: Number of hidden channels per layer.
        kernel_size: Convolutional kernel size.
        num_layers: Number of stacked ConvLSTM layers.
        output_dim: Output feature dimension.
    """

    def __init__(self, input_channels: int, hidden_channels: int = 64,
                 kernel_size: int = 3, num_layers: int = 2,
                 output_dim: int = 256):
        super().__init__()

        if input_channels <= 0 or hidden_channels <= 0:
            raise ValueError("Input and hidden channels must be positive")
        if num_layers <= 0:
            raise ValueError("Number of layers must be positive")
        if output_dim <= 0:
            raise ValueError("Output dimension must be positive")

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.convlstm_layers = nn.ModuleList()
        self.convlstm_layers.append(
            ConvLSTMCell(input_channels, hidden_channels, kernel_size)
        )
        for _ in range(1, num_layers):
            self.convlstm_layers.append(
                ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
            )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_channels, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (batch, channels, seq_len, H, W), got {x.dim()}D")

        batch_size, channels, seq_len, height, width = x.size()

        if channels != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {channels}")

        hidden_states = [None] * self.num_layers

        for t in range(seq_len):
            input_frame = x[:, :, t, :, :]
            for layer_idx, convlstm_layer in enumerate(self.convlstm_layers):
                if layer_idx == 0:
                    hidden_states[layer_idx] = convlstm_layer(input_frame, hidden_states[layer_idx])
                else:
                    hidden_states[layer_idx] = convlstm_layer(
                        hidden_states[layer_idx - 1][0], hidden_states[layer_idx]
                    )

        final_hidden = hidden_states[-1][0]
        pooled = self.global_pool(final_hidden).squeeze(-1).squeeze(-1)
        output = self.output_projection(pooled)

        return output


class ConvLSTMOnlyModel(nn.Module):
    """SDO image-only model using ConvLSTM.

    Uses only SDO image sequences for prediction, ignoring OMNI time series.
    """

    def __init__(
        self,
        num_target_variables: int,
        target_sequence_length: int,
        d_model: int,
        convlstm_input_channels: int,
        convlstm_hidden_channels: int,
        convlstm_kernel_size: int,
        convlstm_num_layers: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_target_variables = num_target_variables
        self.target_sequence_length = target_sequence_length

        self.convlstm_model = ConvLSTMModel(
            input_channels=convlstm_input_channels,
            hidden_channels=convlstm_hidden_channels,
            kernel_size=convlstm_kernel_size,
            num_layers=convlstm_num_layers,
            output_dim=d_model
        )

        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, target_sequence_length * num_target_variables)
        )

    def forward(
        self,
        solar_wind_input: Optional[torch.Tensor],
        image_input: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, None, torch.Tensor]]:
        # Only use image_input, ignore solar_wind_input
        convlstm_features = self.convlstm_model(image_input)

        predictions = self.regression_head(convlstm_features)
        output = predictions.reshape(
            predictions.size(0), self.target_sequence_length, self.num_target_variables
        )

        if return_features:
            return output, None, convlstm_features
        return output


@register_model("convlstm")
def _create_convlstm(config):
    """Factory function for ConvLSTM model."""
    num_input_variables, input_sequence_length, \
        num_target_variables, target_sequence_length = _get_model_dimensions(config)

    print(f"Creating convlstm model: Output shape (batch, {target_sequence_length}, {num_target_variables})")

    return ConvLSTMOnlyModel(
        num_target_variables=num_target_variables,
        target_sequence_length=target_sequence_length,
        d_model=config.model.d_model,
        convlstm_input_channels=config.model.convlstm_input_channels,
        convlstm_hidden_channels=config.model.convlstm_hidden_channels,
        convlstm_kernel_size=config.model.convlstm_kernel_size,
        convlstm_num_layers=config.model.convlstm_num_layers,
        dropout=config.model.fusion_dropout
    )
