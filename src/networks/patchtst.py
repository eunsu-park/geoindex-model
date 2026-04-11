"""PatchTST models for patch-based time series processing."""

from typing import Tuple, Optional, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import _get_model_dimensions
from ._registry import register_model


class PatchEmbedding(nn.Module):
    """Convert time series into patch tokens via sliding window.

    Divides a sequence into (possibly overlapping) patches and projects
    each patch to a d_model-dimensional embedding.

    Args:
        patch_len: Length of each patch (number of timesteps per patch).
        stride: Stride between consecutive patches.
        d_input: Input feature dimension per timestep.
        d_model: Output embedding dimension per patch token.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        patch_len: int = 16,
        stride: int = 8,
        d_input: int = 1,
        d_model: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

        # Linear projection: (patch_len * d_input) -> d_model
        self.projection = nn.Linear(patch_len * d_input, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create patch tokens from input sequence.

        Args:
            x: Input tensor (batch, seq_len, d_input).

        Returns:
            Patch tokens (batch, num_patches, d_model).
        """
        batch_size, seq_len, d_input = x.size()

        # Pad sequence if needed so all timesteps are covered
        pad_len = (self.stride - (seq_len - self.patch_len) % self.stride) % self.stride
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))  # Pad seq_len dimension

        # Unfold into patches: (batch, num_patches, patch_len, d_input)
        padded_len = x.size(1)
        num_patches = (padded_len - self.patch_len) // self.stride + 1
        patches = x.unfold(1, self.patch_len, self.stride)  # (batch, num_patches, d_input, patch_len)
        patches = patches.permute(0, 1, 3, 2)  # (batch, num_patches, patch_len, d_input)

        # Flatten and project: (batch, num_patches, patch_len * d_input) -> (batch, num_patches, d_model)
        patches = patches.reshape(batch_size, num_patches, -1)
        tokens = self.projection(patches)
        tokens = self.dropout(tokens)

        return tokens


class PatchTransformerEncoder(nn.Module):
    """PatchTST-style encoder for time series.

    Divides input into patches, applies Transformer encoder on patch tokens,
    then pools to a fixed-size representation.

    Args:
        num_input_variables: Number of input features per timestep.
        input_sequence_length: Length of input sequence.
        d_model: Transformer/output dimension.
        patch_len: Patch length (timesteps per patch).
        patch_stride: Stride between patches.
        nhead: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_feedforward: Feedforward dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        d_model: int = 128,
        patch_len: int = 16,
        patch_stride: int = 8,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_input_variables = num_input_variables
        self.input_sequence_length = input_sequence_length
        self.d_model = d_model

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            patch_len=patch_len,
            stride=patch_stride,
            d_input=num_input_variables,
            d_model=d_model,
            dropout=dropout
        )

        # Calculate number of patches for positional encoding
        pad_len = (patch_stride - (input_sequence_length - patch_len) % patch_stride) % patch_stride
        self.num_patches = (input_sequence_length + pad_len - patch_len) // patch_stride + 1

        # Learnable positional embedding (PatchTST style)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Global pooling + output projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, num_vars).

        Returns:
            Output features (batch, d_model).
        """
        # 1. Create patch tokens
        tokens = self.patch_embed(x)  # (batch, num_patches, d_model)

        # 2. Add positional embedding
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]

        # 3. Transformer encoder
        h = self.transformer_encoder(tokens)  # (batch, num_patches, d_model)

        # 4. Global pooling
        h = h.transpose(1, 2)  # (batch, d_model, num_patches)
        h = self.global_pool(h).squeeze(-1)  # (batch, d_model)
        h = self.output_projection(h)

        return h


class PatchTSTOnlyModel(nn.Module):
    """Time series model using PatchTST encoder.

    Divides input into subseries-level patches, applies Transformer
    on patch tokens for efficient long-range dependency modeling.

    Args:
        num_input_variables: Number of input variables.
        input_sequence_length: Length of input sequence.
        num_target_variables: Number of target variables.
        target_sequence_length: Length of prediction sequence.
        d_model: Feature dimension.
        patch_len: Patch length.
        patch_stride: Stride between patches.
        nhead: Attention heads.
        num_layers: Transformer layers.
        dim_feedforward: Feedforward dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        num_target_variables: int,
        target_sequence_length: int,
        d_model: int = 128,
        patch_len: int = 16,
        patch_stride: int = 8,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        if num_target_variables <= 0 or target_sequence_length <= 0:
            raise ValueError("Target variables and sequence length must be positive")

        self.num_target_variables = num_target_variables
        self.target_sequence_length = target_sequence_length

        self.encoder = PatchTransformerEncoder(
            num_input_variables=num_input_variables,
            input_sequence_length=input_sequence_length,
            d_model=d_model,
            patch_len=patch_len,
            patch_stride=patch_stride,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Regression head (identical to other models)
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, target_sequence_length * num_target_variables)
        )

    def forward(
        self,
        solar_wind_input: torch.Tensor,
        image_input: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, None]]:
        """Forward pass."""
        features = self.encoder(solar_wind_input)
        predictions = self.regression_head(features)
        output = predictions.reshape(
            predictions.size(0),
            self.target_sequence_length,
            self.num_target_variables
        )
        if return_features:
            return output, features, None
        return output


@register_model("patchtst")
def _create_patchtst(config):
    """Factory function for PatchTST model."""
    num_input_variables, input_sequence_length, \
        num_target_variables, target_sequence_length = _get_model_dimensions(config)

    print(f"Creating patchtst model: Output shape (batch, {target_sequence_length}, {num_target_variables})")

    # PatchTST model (patch-based Transformer)
    patch_len = getattr(config.model, 'patch_len', 16)
    patch_stride = getattr(config.model, 'patch_stride', 8)
    pt_dropout = getattr(config.model, 'patchtst_dropout', 0.1)

    model = PatchTSTOnlyModel(
        num_input_variables=num_input_variables,
        input_sequence_length=input_sequence_length,
        num_target_variables=num_target_variables,
        target_sequence_length=target_sequence_length,
        d_model=config.model.d_model,
        patch_len=patch_len,
        patch_stride=patch_stride,
        nhead=config.model.transformer_nhead,
        num_layers=config.model.transformer_num_layers,
        dim_feedforward=config.model.transformer_dim_feedforward,
        dropout=pt_dropout,
    )
    # Calculate num patches for info
    pad = (patch_stride - (input_sequence_length - patch_len) % patch_stride) % patch_stride
    n_patches = (input_sequence_length + pad - patch_len) // patch_stride + 1
    print(f"  PatchTST: patch_len={patch_len}, stride={patch_stride}, "
          f"num_patches={n_patches}")
    return model
