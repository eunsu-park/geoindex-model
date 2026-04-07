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


# =============================================================================
# Graph Neural Network (GNN) Components
# =============================================================================

# Default variable-to-node grouping for solar wind data:
# 7 physical variable groups (avg/min/max triplet each) + 2 geomagnetic indices
VARIABLE_NODE_GROUPS = {
    'v': ['v_avg', 'v_min', 'v_max'],
    'np': ['np_avg', 'np_min', 'np_max'],
    't': ['t_avg', 't_min', 't_max'],
    'bx': ['bx_avg', 'bx_min', 'bx_max'],
    'by': ['by_avg', 'by_min', 'by_max'],
    'bz': ['bz_avg', 'bz_min', 'bz_max'],
    'bt': ['bt_avg', 'bt_min', 'bt_max'],
    'ap30': ['ap30'],
    'hp30': ['hp30'],
}


class GraphConvLayer(nn.Module):
    """Single graph convolution layer.

    Performs message passing on a graph with adaptive adjacency matrix.
    X' = sigma(A @ X @ W + b)

    Args:
        in_features: Input feature dimension per node.
        out_features: Output feature dimension per node.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features (batch, num_nodes, in_features).
            adj: Adjacency matrix (num_nodes, num_nodes), softmax-normalized.

        Returns:
            Updated node features (batch, num_nodes, out_features).
        """
        # Message passing: aggregate neighbor features via adjacency
        support = torch.matmul(adj, x)  # (batch, num_nodes, in_features)
        return self.weight(support)  # (batch, num_nodes, out_features)


class GNNEncoder(nn.Module):
    """Graph Neural Network encoder for multivariate time series.

    Maps input variables to graph nodes, learns adaptive inter-variable
    relationships via GCN, then applies a temporal encoder (Transformer,
    TCN, or BiLSTM) to capture time dynamics.

    Architecture:
        1. Variable grouping: 23 input vars → 9 graph nodes
        2. Per-timestep GCN: learns inter-variable relationships
        3. Temporal encoder: captures time dynamics
        4. Output projection: → d_model features

    Args:
        num_input_variables: Number of input variables (23).
        input_sequence_length: Length of input sequence.
        num_nodes: Number of graph nodes (default 9).
        node_feature_dim: Feature dimension per node after projection.
        gcn_hidden_dim: Hidden dimension in GCN layers.
        num_gcn_layers: Number of GCN layers.
        temporal_type: Temporal encoder type ("transformer", "tcn", "bilstm").
        d_model: Output feature dimension.
        dropout: Dropout rate.
        node_embed_dim: Dimension of node embeddings for adaptive adjacency.
        transformer_nhead: Number of attention heads (for transformer temporal).
        transformer_num_layers: Number of transformer layers.
        transformer_dim_feedforward: Feedforward dimension.
        tcn_channels: Channel list for TCN temporal encoder.
        tcn_kernel_size: Kernel size for TCN.
        bilstm_hidden_size: Hidden size for BiLSTM.
        bilstm_num_layers: Number of BiLSTM layers.
    """

    # Variable group sizes (hardcoded for 23-var solar wind input)
    _GROUP_SIZES = [3, 3, 3, 3, 3, 3, 3, 1, 1]  # v,np,t,bx,by,bz,bt,ap30,hp30
    _NUM_NODES = 9

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        num_nodes: int = 9,
        node_feature_dim: int = 32,
        gcn_hidden_dim: int = 64,
        num_gcn_layers: int = 2,
        temporal_type: str = "transformer",
        d_model: int = 128,
        dropout: float = 0.1,
        node_embed_dim: int = 16,
        # Transformer temporal params
        transformer_nhead: int = 4,
        transformer_num_layers: int = 2,
        transformer_dim_feedforward: int = 256,
        # TCN temporal params
        tcn_channels: list = None,
        tcn_kernel_size: int = 3,
        # BiLSTM temporal params
        bilstm_hidden_size: int = 128,
        bilstm_num_layers: int = 2,
    ):
        super().__init__()

        if num_input_variables != sum(self._GROUP_SIZES):
            raise ValueError(
                f"Expected {sum(self._GROUP_SIZES)} input variables, "
                f"got {num_input_variables}"
            )

        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.temporal_type = temporal_type
        self.d_model = d_model
        self.input_sequence_length = input_sequence_length

        # Per-node input projections (variable group → node_feature_dim)
        self.node_projections = nn.ModuleList()
        for size in self._GROUP_SIZES:
            self.node_projections.append(
                nn.Linear(size, node_feature_dim)
            )

        # Adaptive adjacency matrix via learnable node embeddings
        # A = softmax(relu(E1 @ E2^T))
        self.node_embed1 = nn.Parameter(
            torch.randn(num_nodes, node_embed_dim)
        )
        self.node_embed2 = nn.Parameter(
            torch.randn(num_nodes, node_embed_dim)
        )

        # GCN layers
        gcn_layers = []
        in_dim = node_feature_dim
        for i in range(num_gcn_layers):
            out_dim = gcn_hidden_dim if i < num_gcn_layers - 1 else gcn_hidden_dim
            gcn_layers.append(GraphConvLayer(in_dim, out_dim))
            in_dim = out_dim
        self.gcn_layers = nn.ModuleList(gcn_layers)
        self.gcn_activation = nn.ReLU()
        self.gcn_dropout = nn.Dropout(dropout)

        # Flatten GCN output: num_nodes * gcn_hidden_dim → temporal input dim
        temporal_input_dim = num_nodes * gcn_hidden_dim

        # Temporal encoder (processes GCN features across time)
        if temporal_type == "transformer":
            self.temporal_proj = nn.Linear(temporal_input_dim, d_model)
            self.pos_encoder = PositionalEncoding(
                d_model, input_sequence_length, dropout
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=transformer_nhead,
                dim_feedforward=transformer_dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.temporal_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=transformer_num_layers
            )
        elif temporal_type == "tcn":
            if tcn_channels is None:
                tcn_channels = [64, 128, 256]
            self.temporal_proj = nn.Linear(temporal_input_dim, tcn_channels[0])
            layers = []
            num_ch = [tcn_channels[0]] + list(tcn_channels)
            for i in range(len(tcn_channels)):
                layers.append(TemporalBlock(
                    num_ch[i], num_ch[i + 1], tcn_kernel_size,
                    dilation=2 ** i, dropout=dropout
                ))
            self.temporal_encoder = nn.Sequential(*layers)
            self._tcn_out_dim = tcn_channels[-1]
        elif temporal_type == "bilstm":
            self.temporal_proj = nn.Linear(temporal_input_dim, bilstm_hidden_size)
            self.temporal_encoder = nn.LSTM(
                input_size=bilstm_hidden_size,
                hidden_size=bilstm_hidden_size,
                num_layers=bilstm_num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if bilstm_num_layers > 1 else 0.0
            )
            self._bilstm_out_dim = bilstm_hidden_size * 2  # bidirectional
        else:
            raise ValueError(f"Unknown temporal_type: {temporal_type}")

        # Global pooling + output projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        if temporal_type == "transformer":
            self.output_projection = nn.Linear(d_model, d_model)
        elif temporal_type == "tcn":
            self.output_projection = nn.Linear(self._tcn_out_dim, d_model)
        elif temporal_type == "bilstm":
            self.output_projection = nn.Linear(self._bilstm_out_dim, d_model)

    def _compute_adaptive_adj(self) -> torch.Tensor:
        """Compute adaptive adjacency matrix from node embeddings."""
        adj = F.relu(torch.matmul(self.node_embed1, self.node_embed2.T))
        return F.softmax(adj, dim=1)

    def _split_to_nodes(self, x: torch.Tensor) -> list:
        """Split input variables into node groups.

        Args:
            x: Input tensor (batch, seq_len, 23).

        Returns:
            List of 9 tensors, each (batch, seq_len, group_size).
        """
        nodes = []
        idx = 0
        for size in self._GROUP_SIZES:
            nodes.append(x[:, :, idx:idx + size])
            idx += size
        return nodes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, num_vars).

        Returns:
            Output features (batch, d_model).
        """
        batch_size, seq_len, _ = x.size()

        # 1. Split variables into node groups and project
        node_groups = self._split_to_nodes(x)
        node_features = []
        for i, group in enumerate(node_groups):
            # (batch, seq_len, group_size) → (batch, seq_len, node_feature_dim)
            node_features.append(self.node_projections[i](group))
        # Stack: (batch, seq_len, num_nodes, node_feature_dim)
        node_features = torch.stack(node_features, dim=2)

        # 2. Compute adaptive adjacency matrix
        adj = self._compute_adaptive_adj()  # (num_nodes, num_nodes)

        # 3. Apply GCN per timestep
        # Reshape: (batch * seq_len, num_nodes, node_feature_dim)
        h = node_features.reshape(batch_size * seq_len, self.num_nodes, -1)
        for gcn_layer in self.gcn_layers:
            h = gcn_layer(h, adj)
            h = self.gcn_activation(h)
            h = self.gcn_dropout(h)
        # (batch * seq_len, num_nodes, gcn_hidden_dim)

        # 4. Flatten nodes: → (batch, seq_len, num_nodes * gcn_hidden_dim)
        h = h.reshape(batch_size, seq_len, -1)

        # 5. Temporal encoding
        if self.temporal_type == "transformer":
            h = self.temporal_proj(h)  # (batch, seq_len, d_model)
            h = self.pos_encoder(h)
            h = self.temporal_encoder(h)  # (batch, seq_len, d_model)
            h = h.transpose(1, 2)  # (batch, d_model, seq_len)
        elif self.temporal_type == "tcn":
            h = self.temporal_proj(h)  # (batch, seq_len, tcn_channels[0])
            h = h.transpose(1, 2)  # (batch, tcn_channels[0], seq_len)
            h = self.temporal_encoder(h)  # (batch, tcn_channels[-1], seq_len)
        elif self.temporal_type == "bilstm":
            h = self.temporal_proj(h)  # (batch, seq_len, hidden_size)
            h, _ = self.temporal_encoder(h)  # (batch, seq_len, hidden*2)
            h = h.transpose(1, 2)  # (batch, hidden*2, seq_len)

        # 6. Global pooling + output projection
        h = self.global_pool(h).squeeze(-1)  # (batch, feat_dim)
        h = self.output_projection(h)  # (batch, d_model)

        return h

    @property
    def adjacency_matrix(self) -> torch.Tensor:
        """Return the learned adjacency matrix (for visualization)."""
        with torch.no_grad():
            return self._compute_adaptive_adj()


class GNNOnlyModel(nn.Module):
    """Time series model using GNN encoder with pluggable temporal backend.

    Combines graph-based inter-variable relationship learning with
    temporal sequence modeling. The temporal encoder can be swapped
    between Transformer, TCN, and BiLSTM.

    Args:
        num_input_variables: Number of input variables.
        input_sequence_length: Length of input sequence.
        num_target_variables: Number of target variables.
        target_sequence_length: Length of prediction sequence.
        d_model: Feature dimension.
        gnn_node_feature_dim: Feature dim per graph node.
        gnn_gcn_hidden_dim: Hidden dim in GCN layers.
        gnn_num_gcn_layers: Number of GCN layers.
        gnn_temporal_type: Temporal encoder ("transformer", "tcn", "bilstm").
        gnn_dropout: Dropout rate.
        gnn_node_embed_dim: Dim of node embeddings for adaptive adjacency.
        transformer_nhead: Attention heads (for transformer temporal).
        transformer_num_layers: Transformer layers.
        transformer_dim_feedforward: Feedforward dim.
        tcn_channels: Channel list for TCN temporal.
        tcn_kernel_size: Kernel size for TCN.
        bilstm_hidden_size: Hidden size for BiLSTM.
        bilstm_num_layers: Number of BiLSTM layers.
    """

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        num_target_variables: int,
        target_sequence_length: int,
        d_model: int = 128,
        gnn_node_feature_dim: int = 32,
        gnn_gcn_hidden_dim: int = 64,
        gnn_num_gcn_layers: int = 2,
        gnn_temporal_type: str = "transformer",
        gnn_dropout: float = 0.1,
        gnn_node_embed_dim: int = 16,
        # Temporal encoder params (passed through)
        transformer_nhead: int = 4,
        transformer_num_layers: int = 2,
        transformer_dim_feedforward: int = 256,
        tcn_channels: list = None,
        tcn_kernel_size: int = 3,
        bilstm_hidden_size: int = 128,
        bilstm_num_layers: int = 2,
    ):
        super().__init__()

        if num_target_variables <= 0 or target_sequence_length <= 0:
            raise ValueError("Target variables and sequence length must be positive")

        self.num_target_variables = num_target_variables
        self.target_sequence_length = target_sequence_length

        self.gnn_encoder = GNNEncoder(
            num_input_variables=num_input_variables,
            input_sequence_length=input_sequence_length,
            node_feature_dim=gnn_node_feature_dim,
            gcn_hidden_dim=gnn_gcn_hidden_dim,
            num_gcn_layers=gnn_num_gcn_layers,
            temporal_type=gnn_temporal_type,
            d_model=d_model,
            dropout=gnn_dropout,
            node_embed_dim=gnn_node_embed_dim,
            transformer_nhead=transformer_nhead,
            transformer_num_layers=transformer_num_layers,
            transformer_dim_feedforward=transformer_dim_feedforward,
            tcn_channels=tcn_channels,
            tcn_kernel_size=tcn_kernel_size,
            bilstm_hidden_size=bilstm_hidden_size,
            bilstm_num_layers=bilstm_num_layers,
        )

        # Regression head (identical to Transformer/TCN models)
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(gnn_dropout),
            nn.Linear(d_model // 2, target_sequence_length * num_target_variables)
        )

    @property
    def adjacency_matrix(self) -> torch.Tensor:
        """Return the learned adjacency matrix (for visualization)."""
        return self.gnn_encoder.adjacency_matrix

    def forward(
        self,
        solar_wind_input: torch.Tensor,
        image_input: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, None]]:
        """Forward pass.

        Args:
            solar_wind_input: Input time series (B, seq_len, num_vars).
            image_input: Ignored (API compatibility).
            return_features: Whether to return intermediate features.

        Returns:
            Predictions (B, target_seq_len, num_target_vars), or
            tuple (predictions, features, None) if return_features=True.
        """
        gnn_features = self.gnn_encoder(solar_wind_input)  # (B, d_model)

        predictions = self.regression_head(gnn_features)
        output = predictions.reshape(
            predictions.size(0),
            self.target_sequence_length,
            self.num_target_variables
        )

        if return_features:
            return output, gnn_features, None
        return output


# =============================================================================
# TimesNet Components
# =============================================================================

class InceptionBlock(nn.Module):
    """Multi-scale 2D convolution block (Inception-style).

    Applies parallel 2D convolutions with different kernel sizes
    and sums the results for multi-scale feature extraction.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels per branch.
        num_kernels: Number of parallel conv branches (kernel sizes 1,3,5,...).
    """

    def __init__(self, in_channels: int, out_channels: int, num_kernels: int = 3):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=2 * k + 1, padding=k
            )
            for k in range(num_kernels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, channels, height, width).

        Returns:
            Output tensor (batch, out_channels, height, width).
        """
        out = sum(conv(x) for conv in self.convs)
        return out


class TimesBlock(nn.Module):
    """Single TimesBlock: FFT-based period detection + 2D convolution.

    Core building block of TimesNet. Converts 1D time series to 2D
    using detected periods, applies Inception-style 2D convolutions,
    then reshapes back to 1D with adaptive aggregation.

    Args:
        seq_len: Input sequence length.
        d_model: Feature dimension.
        d_ff: Hidden dimension in Inception blocks.
        top_k: Number of dominant periods to extract.
        num_kernels: Number of parallel conv branches in Inception.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        d_ff: int = 256,
        top_k: int = 3,
        num_kernels: int = 3
    ):
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k

        # Two Inception blocks (encoder-decoder style, as in SINet)
        self.inception1 = InceptionBlock(d_model, d_ff, num_kernels)
        self.inception2 = InceptionBlock(d_ff, d_model, num_kernels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, d_model).

        Returns:
            Output tensor (batch, seq_len, d_model) with residual.
        """
        batch_size, seq_len, d_model = x.size()

        # 1. FFT to find dominant periods
        # Compute amplitude spectrum (ignore DC component at index 0)
        x_freq = torch.fft.rfft(x, dim=1)
        amplitude = torch.abs(x_freq).mean(dim=-1)  # (batch, freq_bins)
        amplitude[:, 0] = 0  # Remove DC component

        # Select top-k periods by amplitude
        _, top_indices = torch.topk(amplitude, self.top_k, dim=1)
        # Convert frequency indices to periods
        # period = seq_len / frequency_index (clamped to avoid div by zero)
        top_indices = top_indices.detach()

        # Aggregate weights from amplitude (softmax over top-k)
        top_amplitudes = torch.gather(amplitude, 1, top_indices)
        weights = F.softmax(top_amplitudes, dim=1)  # (batch, top_k)

        # 2. For each top-k period, reshape to 2D, apply inception, reshape back
        results = []
        for k in range(self.top_k):
            # Get period for each sample in batch
            freq_idx = top_indices[:, k]  # (batch,)
            # Use the most common period in the batch for uniform reshaping
            period = int(seq_len / (freq_idx.float().mean().clamp(min=1)).item())
            period = max(period, 2)  # Minimum period of 2

            # Pad sequence to be divisible by period
            pad_len = (period - seq_len % period) % period
            if pad_len > 0:
                x_padded = F.pad(x, (0, 0, 0, pad_len))  # Pad seq_len dim
            else:
                x_padded = x
            padded_len = x_padded.size(1)

            # Reshape: (batch, padded_len, d_model) → (batch, d_model, period, padded_len//period)
            x_2d = x_padded.permute(0, 2, 1)  # (batch, d_model, padded_len)
            x_2d = x_2d.reshape(batch_size, d_model, period, padded_len // period)

            # Apply 2D Inception blocks
            x_2d = self.inception1(x_2d)
            x_2d = self.activation(x_2d)
            x_2d = self.inception2(x_2d)

            # Reshape back: (batch, d_model, period, n_periods) → (batch, padded_len, d_model)
            x_1d = x_2d.reshape(batch_size, d_model, padded_len)
            x_1d = x_1d.permute(0, 2, 1)  # (batch, padded_len, d_model)

            # Remove padding
            x_1d = x_1d[:, :seq_len, :]

            results.append(x_1d)

        # 3. Adaptive aggregation (weighted sum by FFT amplitudes)
        # Stack: (batch, top_k, seq_len, d_model)
        results = torch.stack(results, dim=1)
        # Weights: (batch, top_k) → (batch, top_k, 1, 1) for broadcasting
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        # Weighted sum → (batch, seq_len, d_model)
        output = (results * weights).sum(dim=1)

        # 4. Residual connection
        return output + x


class TimesNetEncoder(nn.Module):
    """TimesNet encoder for multivariate time series.

    Stacks multiple TimesBlocks with optional cross-variable mixing.

    Args:
        num_input_variables: Number of input variables.
        input_sequence_length: Length of input sequence.
        d_model: Feature dimension.
        d_ff: Hidden dimension in Inception blocks.
        num_blocks: Number of stacked TimesBlocks.
        top_k: Number of dominant periods per block.
        num_kernels: Number of Inception conv branches.
        dropout: Dropout rate.
        output_dim: Output feature dimension.
        enable_cross_variable: Enable cross-variable mixing layer.
    """

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        d_model: int = 64,
        d_ff: int = 128,
        num_blocks: int = 2,
        top_k: int = 3,
        num_kernels: int = 3,
        dropout: float = 0.1,
        output_dim: int = 128,
        enable_cross_variable: bool = True,
    ):
        super().__init__()
        self.num_input_variables = num_input_variables
        self.input_sequence_length = input_sequence_length

        # Input projection
        self.input_projection = nn.Linear(num_input_variables, d_model)

        # Stacked TimesBlocks
        self.blocks = nn.ModuleList([
            TimesBlock(
                seq_len=input_sequence_length,
                d_model=d_model,
                d_ff=d_ff,
                top_k=top_k,
                num_kernels=num_kernels
            )
            for _ in range(num_blocks)
        ])
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_blocks)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_blocks)
        ])

        # Cross-variable mixing (addresses channel-independent limitation)
        self.enable_cross_variable = enable_cross_variable
        if enable_cross_variable:
            self.cross_var_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=4,
                dropout=dropout, batch_first=True
            )
            self.cross_var_norm = nn.LayerNorm(d_model)

        # Global pooling + output projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, num_vars).

        Returns:
            Output features (batch, output_dim).
        """
        # Input projection: (batch, seq_len, num_vars) → (batch, seq_len, d_model)
        h = self.input_projection(x)

        # Stacked TimesBlocks with LayerNorm and dropout
        for block, dropout, norm in zip(self.blocks, self.dropouts, self.norms):
            h = norm(dropout(block(h)) + h)

        # Cross-variable mixing via self-attention over time dimension
        if self.enable_cross_variable:
            residual = h
            h, _ = self.cross_var_attn(h, h, h)
            h = self.cross_var_norm(h + residual)

        # Global pooling: (batch, seq_len, d_model) → (batch, d_model)
        h = h.transpose(1, 2)  # (batch, d_model, seq_len)
        h = self.global_pool(h).squeeze(-1)  # (batch, d_model)

        # Output projection
        h = self.output_projection(h)  # (batch, output_dim)

        return h


class TimesNetOnlyModel(nn.Module):
    """Time series model using TimesNet encoder.

    Uses FFT-based period detection and 2D convolutions to capture
    multi-periodic patterns in solar wind time series.

    Args:
        num_input_variables: Number of input variables.
        input_sequence_length: Length of input sequence.
        num_target_variables: Number of target variables.
        target_sequence_length: Length of prediction sequence.
        d_model: Internal feature dimension for TimesNet.
        d_ff: Hidden dimension in Inception blocks.
        output_d_model: Output feature dimension (for regression head).
        num_blocks: Number of stacked TimesBlocks.
        top_k: Number of dominant periods per block.
        num_kernels: Number of Inception conv branches.
        dropout: Dropout rate.
        enable_cross_variable: Enable cross-variable mixing.
    """

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        num_target_variables: int,
        target_sequence_length: int,
        d_model: int = 64,
        d_ff: int = 128,
        output_d_model: int = 128,
        num_blocks: int = 2,
        top_k: int = 3,
        num_kernels: int = 3,
        dropout: float = 0.1,
        enable_cross_variable: bool = True,
    ):
        super().__init__()

        if num_target_variables <= 0 or target_sequence_length <= 0:
            raise ValueError("Target variables and sequence length must be positive")

        self.num_target_variables = num_target_variables
        self.target_sequence_length = target_sequence_length

        self.timesnet_encoder = TimesNetEncoder(
            num_input_variables=num_input_variables,
            input_sequence_length=input_sequence_length,
            d_model=d_model,
            d_ff=d_ff,
            num_blocks=num_blocks,
            top_k=top_k,
            num_kernels=num_kernels,
            dropout=dropout,
            output_dim=output_d_model,
            enable_cross_variable=enable_cross_variable,
        )

        # Regression head (identical to other models)
        self.regression_head = nn.Sequential(
            nn.Linear(output_d_model, output_d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_d_model // 2, target_sequence_length * num_target_variables)
        )

    def forward(
        self,
        solar_wind_input: torch.Tensor,
        image_input: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, None]]:
        """Forward pass.

        Args:
            solar_wind_input: Input time series (B, seq_len, num_vars).
            image_input: Ignored (API compatibility).
            return_features: Whether to return intermediate features.

        Returns:
            Predictions (B, target_seq_len, num_target_vars), or
            tuple (predictions, features, None) if return_features=True.
        """
        features = self.timesnet_encoder(solar_wind_input)  # (B, output_d_model)

        predictions = self.regression_head(features)
        output = predictions.reshape(
            predictions.size(0),
            self.target_sequence_length,
            self.num_target_variables
        )

        if return_features:
            return output, features, None
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

    elif model_type == "gnn":
        # GNN model with pluggable temporal encoder
        gnn_temporal_type = getattr(config.model, 'gnn_temporal_type', 'transformer')
        gnn_node_feature_dim = getattr(config.model, 'gnn_node_feature_dim', 32)
        gnn_gcn_hidden_dim = getattr(config.model, 'gnn_gcn_hidden_dim', 64)
        gnn_num_gcn_layers = getattr(config.model, 'gnn_num_gcn_layers', 2)
        gnn_dropout = getattr(config.model, 'gnn_dropout', 0.1)
        gnn_node_embed_dim = getattr(config.model, 'gnn_node_embed_dim', 16)

        # Temporal encoder params (reuse existing config keys)
        tcn_channels = getattr(config.model, 'tcn_channels', [64, 128, 256])
        if hasattr(tcn_channels, '__iter__') and not isinstance(tcn_channels, list):
            tcn_channels = list(tcn_channels)
        tcn_kernel_size = getattr(config.model, 'tcn_kernel_size', 3)
        bilstm_hidden_size = getattr(config.model, 'bilstm_hidden_size', 128)
        bilstm_num_layers = getattr(config.model, 'bilstm_num_layers', 2)

        model = GNNOnlyModel(
            num_input_variables=num_input_variables,
            input_sequence_length=input_sequence_length,
            num_target_variables=num_target_variables,
            target_sequence_length=target_sequence_length,
            d_model=config.model.d_model,
            gnn_node_feature_dim=gnn_node_feature_dim,
            gnn_gcn_hidden_dim=gnn_gcn_hidden_dim,
            gnn_num_gcn_layers=gnn_num_gcn_layers,
            gnn_temporal_type=gnn_temporal_type,
            gnn_dropout=gnn_dropout,
            gnn_node_embed_dim=gnn_node_embed_dim,
            transformer_nhead=config.model.transformer_nhead,
            transformer_num_layers=config.model.transformer_num_layers,
            transformer_dim_feedforward=config.model.transformer_dim_feedforward,
            tcn_channels=tcn_channels,
            tcn_kernel_size=tcn_kernel_size,
            bilstm_hidden_size=bilstm_hidden_size,
            bilstm_num_layers=bilstm_num_layers,
        )
        print(f"  GNN temporal encoder: {gnn_temporal_type}")
        print(f"  GNN: {gnn_num_gcn_layers} GCN layers, "
              f"node_feat={gnn_node_feature_dim}, gcn_hidden={gnn_gcn_hidden_dim}")
        return model

    elif model_type == "timesnet":
        # TimesNet model (FFT-based period detection + 2D convolution)
        tn_d_model = getattr(config.model, 'timesnet_d_model', 64)
        tn_d_ff = getattr(config.model, 'timesnet_d_ff', 128)
        tn_num_blocks = getattr(config.model, 'timesnet_num_blocks', 2)
        tn_top_k = getattr(config.model, 'timesnet_top_k', 3)
        tn_num_kernels = getattr(config.model, 'timesnet_num_kernels', 3)
        tn_dropout = getattr(config.model, 'timesnet_dropout', 0.1)
        tn_cross_var = getattr(config.model, 'timesnet_cross_variable', True)

        model = TimesNetOnlyModel(
            num_input_variables=num_input_variables,
            input_sequence_length=input_sequence_length,
            num_target_variables=num_target_variables,
            target_sequence_length=target_sequence_length,
            d_model=tn_d_model,
            d_ff=tn_d_ff,
            output_d_model=config.model.d_model,
            num_blocks=tn_num_blocks,
            top_k=tn_top_k,
            num_kernels=tn_num_kernels,
            dropout=tn_dropout,
            enable_cross_variable=tn_cross_var,
        )
        print(f"  TimesNet: {tn_num_blocks} blocks, top_k={tn_top_k}, "
              f"d_model={tn_d_model}, d_ff={tn_d_ff}, "
              f"cross_variable={tn_cross_var}")
        return model

    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Choose from: convlstm, transformer, fusion, baseline, "
            f"linear, tcn, gnn, timesnet"
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
