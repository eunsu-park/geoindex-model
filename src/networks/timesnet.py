"""TimesNet models for time series with FFT-based period detection."""

from typing import Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import _get_model_dimensions
from ._registry import register_model


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
        max_k = seq_len // 2  # FFT produces seq_len//2+1 freq bins, minus DC
        if top_k > max_k:
            raise ValueError(
                f"top_k ({top_k}) must be <= seq_len//2 ({max_k})"
            )
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

            # Reshape: (batch, padded_len, d_model) -> (batch, d_model, period, padded_len//period)
            x_2d = x_padded.permute(0, 2, 1)  # (batch, d_model, padded_len)
            x_2d = x_2d.reshape(batch_size, d_model, period, padded_len // period)

            # Apply 2D Inception blocks
            x_2d = self.inception1(x_2d)
            x_2d = self.activation(x_2d)
            x_2d = self.inception2(x_2d)

            # Reshape back: (batch, d_model, period, n_periods) -> (batch, padded_len, d_model)
            x_1d = x_2d.reshape(batch_size, d_model, padded_len)
            x_1d = x_1d.permute(0, 2, 1)  # (batch, padded_len, d_model)

            # Remove padding
            x_1d = x_1d[:, :seq_len, :]

            results.append(x_1d)

        # 3. Adaptive aggregation (weighted sum by FFT amplitudes)
        # Stack: (batch, top_k, seq_len, d_model)
        results = torch.stack(results, dim=1)
        # Weights: (batch, top_k) -> (batch, top_k, 1, 1) for broadcasting
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        # Weighted sum -> (batch, seq_len, d_model)
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
        # Input projection: (batch, seq_len, num_vars) -> (batch, seq_len, d_model)
        h = self.input_projection(x)

        # Stacked TimesBlocks with LayerNorm and dropout
        for block, dropout, norm in zip(self.blocks, self.dropouts, self.norms):
            h = norm(dropout(block(h)) + h)

        # Cross-variable mixing via self-attention over time dimension
        if self.enable_cross_variable:
            residual = h
            h, _ = self.cross_var_attn(h, h, h)
            h = self.cross_var_norm(h + residual)

        # Global pooling: (batch, seq_len, d_model) -> (batch, d_model)
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


@register_model("timesnet")
def _create_timesnet(config):
    """Factory function for TimesNet model."""
    num_input_variables, input_sequence_length, \
        num_target_variables, target_sequence_length = _get_model_dimensions(config)

    print(f"Creating timesnet model: Output shape (batch, {target_sequence_length}, {num_target_variables})")

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
