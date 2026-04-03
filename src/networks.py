"""Model architectures for multi-modal solar wind prediction.

This module contains:
- ConvLSTMCell, ConvLSTMModel: For processing SDO image sequences
- PositionalEncoding, TransformerEncoderModel: For processing OMNI time series
- TemporalBlock, TCNEncoder: Temporal Convolutional Network for time series
- CrossModalAttention, CrossModalFusion: For fusing modalities
- ConvLSTMOnlyModel: SDO-only model
- TransformerOnlyModel: OMNI-only model (Transformer encoder)
- TCNOnlyModel: OMNI-only model (TCN encoder)
- LinearOnlyModel: OMNI-only model (Linear encoder)
- MultiModalModel: Fusion model combining both modalities
- Conv3DEncoder, LinearEncoder, BaselineModel: Baseline model (Son et al. 2023 style)
- create_model: Factory function to create model based on config
"""

from typing import Tuple, Optional, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model) with batch_first=True
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class TransformerEncoderModel(nn.Module):
    """Transformer encoder for time series processing.

    Args:
        num_input_variables: Number of input variables per timestep.
        input_sequence_length: Length of input sequence.
        d_model: Model dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer layers.
        dim_feedforward: Feedforward network dimension.
        dropout: Dropout rate.
    """

    def __init__(self, num_input_variables: int, input_sequence_length: int,
                 d_model: int = 256, nhead: int = 8, num_layers: int = 3,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()

        if num_input_variables <= 0:
            raise ValueError(f"Number of input variables must be positive, got {num_input_variables}")
        if input_sequence_length <= 0:
            raise ValueError(f"Input sequence length must be positive, got {input_sequence_length}")
        if d_model <= 0:
            raise ValueError(f"Model dimension must be positive, got {d_model}")
        if d_model % nhead != 0:
            raise ValueError(f"d_model {d_model} must be divisible by nhead {nhead}")
        if nhead <= 0 or num_layers <= 0:
            raise ValueError("Number of heads and layers must be positive")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError("Dropout must be between 0 and 1")

        self.d_model = d_model
        self.input_sequence_length = input_sequence_length
        self.num_input_variables = num_input_variables

        self.input_projection = nn.Linear(num_input_variables, d_model)
        self.pos_encoder = PositionalEncoding(d_model, input_sequence_length, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, seq_len, vars), got {x.dim()}D")

        batch_size, seq_len, num_vars = x.size()

        if seq_len != self.input_sequence_length:
            raise ValueError(f"Expected seq_len {self.input_sequence_length}, got {seq_len}")
        if num_vars != self.num_input_variables:
            raise ValueError(f"Expected {self.num_input_variables} vars, got {num_vars}")

        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        x = self.output_projection(x)

        return x


# =============================================================================
# Temporal Convolutional Network (TCN) for Time Series
# =============================================================================

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


class CrossModalAttention(nn.Module):
    """Multi-head cross-modal attention module."""

    def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        if feature_dim <= 0:
            raise ValueError("Feature dimension must be positive")
        if feature_dim % num_heads != 0:
            raise ValueError("Feature dimension must be divisible by number of heads")
        if num_heads <= 0:
            raise ValueError("Number of heads must be positive")

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)

        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, query_features: torch.Tensor,
                key_value_features: torch.Tensor) -> torch.Tensor:
        batch_size = query_features.size(0)

        Q = self.q_proj(query_features)
        K = self.k_proj(key_value_features)
        V = self.v_proj(key_value_features)

        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, self.feature_dim)

        output = self.out_proj(attended)
        output = self.norm(output + query_features)

        return output


class CrossModalFusion(nn.Module):
    """Bidirectional cross-modal fusion with gating mechanism."""

    def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.feature_dim = feature_dim

        self.transformer_to_convlstm = CrossModalAttention(feature_dim, num_heads, dropout)
        self.convlstm_to_transformer = CrossModalAttention(feature_dim, num_heads, dropout)

        self.feature_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )

        self.combination_layer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )

        self.final_norm = nn.LayerNorm(feature_dim)

    def forward(self, transformer_features: torch.Tensor,
                convlstm_features: torch.Tensor) -> torch.Tensor:
        transformer_attended = self.transformer_to_convlstm(transformer_features, convlstm_features)
        convlstm_attended = self.convlstm_to_transformer(convlstm_features, transformer_features)

        concatenated = torch.cat([transformer_attended, convlstm_attended], dim=1)
        gate_weights = self.feature_gate(concatenated)

        weighted_transformer = gate_weights * transformer_attended
        weighted_convlstm = (1 - gate_weights) * convlstm_attended

        combined = torch.cat([weighted_transformer, weighted_convlstm], dim=1)
        fused_features = self.combination_layer(combined)

        residual = (transformer_features + convlstm_features) / 2
        output = self.final_norm(fused_features + residual)

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


class TransformerOnlyModel(nn.Module):
    """OMNI time series-only model using Transformer.

    Uses only OMNI time series for prediction, ignoring SDO images.
    """

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        num_target_variables: int,
        target_sequence_length: int,
        d_model: int,
        transformer_nhead: int,
        transformer_num_layers: int,
        transformer_dim_feedforward: int,
        transformer_dropout: float
    ):
        super().__init__()

        self.num_target_variables = num_target_variables
        self.target_sequence_length = target_sequence_length

        self.transformer_model = TransformerEncoderModel(
            num_input_variables=num_input_variables,
            input_sequence_length=input_sequence_length,
            d_model=d_model,
            nhead=transformer_nhead,
            num_layers=transformer_num_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout
        )

        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(transformer_dropout),
            nn.Linear(d_model // 2, target_sequence_length * num_target_variables)
        )

    def forward(
        self,
        solar_wind_input: torch.Tensor,
        image_input: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, None]]:
        # Only use solar_wind_input, ignore image_input
        transformer_features = self.transformer_model(solar_wind_input)

        predictions = self.regression_head(transformer_features)
        output = predictions.reshape(
            predictions.size(0), self.target_sequence_length, self.num_target_variables
        )

        if return_features:
            return output, transformer_features, None
        return output


class MultiModalModel(nn.Module):
    """Fusion model combining both OMNI time series and SDO images.

    Uses Transformer for OMNI data, ConvLSTM for SDO images,
    and cross-modal fusion to combine both modalities.
    """

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        num_target_variables: int,
        target_sequence_length: int,
        d_model: int,
        transformer_nhead: int,
        transformer_num_layers: int,
        transformer_dim_feedforward: int,
        transformer_dropout: float,
        convlstm_input_channels: int,
        convlstm_hidden_channels: int,
        convlstm_kernel_size: int,
        convlstm_num_layers: int,
        fusion_num_heads: int = 4,
        fusion_dropout: float = 0.1
    ):
        super().__init__()

        if num_target_variables <= 0 or target_sequence_length <= 0:
            raise ValueError("Target variables and sequence length must be positive")

        self.transformer_model = TransformerEncoderModel(
            num_input_variables=num_input_variables,
            input_sequence_length=input_sequence_length,
            d_model=d_model,
            nhead=transformer_nhead,
            num_layers=transformer_num_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout
        )

        self.convlstm_model = ConvLSTMModel(
            input_channels=convlstm_input_channels,
            hidden_channels=convlstm_hidden_channels,
            kernel_size=convlstm_kernel_size,
            num_layers=convlstm_num_layers,
            output_dim=d_model
        )

        self.cross_modal_fusion = CrossModalFusion(
            feature_dim=d_model,
            num_heads=fusion_num_heads,
            dropout=fusion_dropout
        )

        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(d_model // 2, target_sequence_length * num_target_variables)
        )

        self.num_target_variables = num_target_variables
        self.target_sequence_length = target_sequence_length

    def forward(
        self,
        solar_wind_input: torch.Tensor,
        image_input: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if solar_wind_input is None or image_input is None:
            raise ValueError("Both solar_wind_input and image_input must be provided")

        if solar_wind_input.size(0) != image_input.size(0):
            raise ValueError(f"Batch sizes must match: {solar_wind_input.size(0)} vs {image_input.size(0)}")

        transformer_features = self.transformer_model(solar_wind_input)
        convlstm_features = self.convlstm_model(image_input)

        fused_features = self.cross_modal_fusion(transformer_features, convlstm_features)

        predictions = self.regression_head(fused_features)
        output = predictions.reshape(
            predictions.size(0), self.target_sequence_length, self.num_target_variables
        )

        if return_features:
            return output, transformer_features, convlstm_features

        return output


# =============================================================================
# Baseline Model (Son et al. 2023 style - Conv3D + Linear)
# =============================================================================

class Conv3DEncoder(nn.Module):
    """3D Convolutional encoder for SDO image sequences.

    Simplified version of Son et al. (2023) architecture.
    Uses Conv3D layers instead of Inception blocks for simplicity.

    Architecture:
        Conv3d(3→32) → BN → ReLU → MaxPool(1×2×2)
        Conv3d(32→64) → BN → ReLU → MaxPool(2×2×2)
        Conv3d(64→128) → BN → ReLU → MaxPool(2×2×2)
        AdaptiveAvgPool3d → Linear → Dropout

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


class LinearEncoder(nn.Module):
    """Simple linear encoder for time series data.

    Flattens the input and processes through dense layers.

    Architecture:
        Flatten → Linear(hidden_dim) → ReLU → Dropout
        Linear(output_dim) → ReLU → Dropout

    Args:
        input_size: Total input size after flattening (seq_len * num_vars).
        output_dim: Output feature dimension.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout rate for regularization.
    """

    def __init__(self, input_size: int, output_dim: int = 256,
                 hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        if input_size <= 0:
            raise ValueError("Input size must be positive")
        if output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Dimensions must be positive")

        self.input_size = input_size
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, seq_len, num_vars)

        Returns:
            Features of shape (B, output_dim)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (B, seq_len, num_vars), got {x.dim()}D")

        return self.encoder(x)


class LinearOnlyModel(nn.Module):
    """OMNI time series-only model using Linear encoder.

    Based on the ablation analysis showing that OMNI-only outperforms
    multi-modal approaches for Ap prediction. Uses simple linear layers
    instead of Transformer for efficiency.

    Args:
        num_input_variables: Number of OMNI input variables.
        input_sequence_length: Length of OMNI input sequence.
        num_target_variables: Number of target variables to predict.
        target_sequence_length: Length of prediction sequence.
        d_model: Feature dimension for encoder.
        dropout: Dropout rate for regularization.
    """

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        num_target_variables: int,
        target_sequence_length: int,
        d_model: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        if num_target_variables <= 0 or target_sequence_length <= 0:
            raise ValueError("Target variables and sequence length must be positive")

        self.num_target_variables = num_target_variables
        self.target_sequence_length = target_sequence_length
        self.d_model = d_model

        # Time series encoder (Linear)
        ts_input_size = input_sequence_length * num_input_variables
        self.ts_encoder = LinearEncoder(
            input_size=ts_input_size,
            output_dim=d_model,
            hidden_dim=d_model,
            dropout=dropout
        )

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, target_sequence_length * num_target_variables)
        )

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
        ts_features = self.ts_encoder(solar_wind_input)

        predictions = self.regression_head(ts_features)
        output = predictions.reshape(
            predictions.size(0), self.target_sequence_length, self.num_target_variables
        )

        if return_features:
            return output, ts_features, None
        return output


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


def _get_model_dimensions(config):
    """Compute input/output dimensions from active modality config.

    Returns:
        Tuple of (num_input_variables, input_sequence_length,
                  num_target_variables, target_sequence_length)
    """
    use_csv = getattr(config.data.modalities, 'timeseries', False)

    if use_csv:
        ts_cfg = config.data.timeseries
        num_input_variables = len(ts_cfg.input_variables)
        num_target_variables = len(ts_cfg.target_variables)
        ppd = ts_cfg.points_per_day

        input_start = getattr(ts_cfg, 'input_start', None)
        input_end = getattr(ts_cfg, 'input_end', None)
        target_start = getattr(ts_cfg, 'target_start', None)
        target_end = getattr(ts_cfg, 'target_end', None)

        if input_start is not None and input_end is not None:
            input_sequence_length = input_end - input_start
        else:
            input_sequence_length = ts_cfg.days_before * ppd

        if target_start is not None and target_end is not None:
            target_sequence_length = target_end - target_start
        else:
            target_sequence_length = ts_cfg.days_after * ppd
    else:
        num_input_variables = len(config.data.input_variables)
        input_sequence_length = config.data.input_end_index - config.data.input_start_index
        num_target_variables = len(config.data.target_variables)
        target_sequence_length = config.data.target_end_index - config.data.target_start_index

    # Allow manual override via model.output_seq_len
    if getattr(config.model, 'output_seq_len', None) is not None:
        target_sequence_length = config.model.output_seq_len

    return num_input_variables, input_sequence_length, num_target_variables, target_sequence_length


def create_model(config):
    """Create model based on configuration.

    Args:
        config: Hydra configuration object with model settings.

    Returns:
        Model instance based on config.model.model_type:
        - "convlstm": ConvLSTMOnlyModel (SDO only)
        - "transformer": TransformerOnlyModel (OMNI only)
        - "fusion": MultiModalModel (both modalities)
        - "baseline": BaselineModel (Conv3D + Linear, Son et al. 2023 style)
        - "linear": LinearOnlyModel (OMNI only, simple linear encoder)
        - "tcn": TCNOnlyModel (OMNI only, Temporal Convolutional Network)
    """
    model_type = config.model.model_type

    num_input_variables, input_sequence_length, \
        num_target_variables, target_sequence_length = _get_model_dimensions(config)

    print(f"Creating {model_type} model: Output shape (batch, {target_sequence_length}, {num_target_variables})")

    if model_type == "convlstm":
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

    elif model_type == "transformer":
        return TransformerOnlyModel(
            num_input_variables=num_input_variables,
            input_sequence_length=input_sequence_length,
            num_target_variables=num_target_variables,
            target_sequence_length=target_sequence_length,
            d_model=config.model.d_model,
            transformer_nhead=config.model.transformer_nhead,
            transformer_num_layers=config.model.transformer_num_layers,
            transformer_dim_feedforward=config.model.transformer_dim_feedforward,
            transformer_dropout=config.model.transformer_dropout
        )

    elif model_type == "fusion":
        return MultiModalModel(
            num_input_variables=num_input_variables,
            input_sequence_length=input_sequence_length,
            num_target_variables=num_target_variables,
            target_sequence_length=target_sequence_length,
            d_model=config.model.d_model,
            transformer_nhead=config.model.transformer_nhead,
            transformer_num_layers=config.model.transformer_num_layers,
            transformer_dim_feedforward=config.model.transformer_dim_feedforward,
            transformer_dropout=config.model.transformer_dropout,
            convlstm_input_channels=config.model.convlstm_input_channels,
            convlstm_hidden_channels=config.model.convlstm_hidden_channels,
            convlstm_kernel_size=config.model.convlstm_kernel_size,
            convlstm_num_layers=config.model.convlstm_num_layers,
            fusion_num_heads=config.model.fusion_num_heads,
            fusion_dropout=config.model.fusion_dropout
        )

    elif model_type == "baseline":
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

    elif model_type == "linear":
        # Linear-only model (OMNI only, no SDO)
        # Based on ablation analysis: OMNI-only outperforms multi-modal
        linear_dropout = getattr(config.model, 'baseline_dropout', 0.1)
        return LinearOnlyModel(
            num_input_variables=num_input_variables,
            input_sequence_length=input_sequence_length,
            num_target_variables=num_target_variables,
            target_sequence_length=target_sequence_length,
            d_model=config.model.d_model,
            dropout=linear_dropout
        )

    elif model_type == "tcn":
        # TCN-only model (OMNI only, Temporal Convolutional Network)
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

    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Choose from: convlstm, transformer, fusion, baseline, linear, tcn"
        )


def verify_model(config):
    """Verify model creation and forward pass with dummy inputs.

    Args:
        config: Hydra configuration object.

    Returns:
        Tuple of (model, outputs) for verification.
    """
    model = create_model(config)
    print(f"\nModel architecture:\n{model}\n")

    batch_size = config.experiment.batch_size
    model_type = config.model.model_type
    use_csv = getattr(config.data.modalities, 'timeseries', False)

    num_input_vars, input_seq_len, num_target_vars, target_seq_len = \
        _get_model_dimensions(config)

    # Create dummy inputs
    inputs = torch.randn(batch_size, input_seq_len, num_input_vars)
    print(f"Inputs shape: {inputs.shape}")

    sdo = None
    if not use_csv and model_type in ("fusion", "baseline", "convlstm"):
        sdo = torch.randn(
            batch_size,
            len(config.data.sdo.wavelengths),
            config.data.sdo_end_index - config.data.sdo_start_index,
            config.data.sdo.image_size,
            config.data.sdo.image_size
        )
        print(f"SDO shape: {sdo.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        if model_type in ("fusion", "baseline"):
            outputs, tf_feat, cl_feat = model(inputs, sdo, return_features=True)
            print(f"Transformer features: {tf_feat.shape if tf_feat is not None else None}")
            print(f"ConvLSTM features: {cl_feat.shape if cl_feat is not None else None}")
        elif model_type == "convlstm":
            outputs = model(inputs, sdo)
        else:
            outputs = model(inputs)

    print(f"Outputs shape: {outputs.shape}")
    print(f"Expected: (batch={batch_size}, seq={target_seq_len}, vars={num_target_vars})")

    return model, outputs


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from hydra import initialize, compose

    with initialize(config_path="../configs", version_base=None):
        config = compose(config_name="local")
        verify_model(config)
