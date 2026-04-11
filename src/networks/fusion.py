"""Fusion/multi-modal models combining OMNI time series and SDO images."""

from typing import Tuple, Optional, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import _get_model_dimensions
from ._registry import register_model
from .convlstm import ConvLSTMModel
from .transformer import TransformerEncoderModel


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


@register_model("fusion")
def _create_fusion(config):
    """Factory function for Fusion model."""
    num_input_variables, input_sequence_length, \
        num_target_variables, target_sequence_length = _get_model_dimensions(config)

    print(f"Creating fusion model: Output shape (batch, {target_sequence_length}, {num_target_variables})")

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
