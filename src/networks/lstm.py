"""LSTM / BiLSTM models for time series processing."""

from typing import Tuple, Optional, Union

import torch
import torch.nn as nn

from ._base import _get_model_dimensions
from ._registry import register_model


class LSTMEncoder(nn.Module):
    """LSTM encoder for multivariate time series.

    Projects input variables, runs a (uni- or bi-directional) LSTM,
    then applies global average pooling over time.

    Args:
        num_input_variables: Number of input variables per timestep.
        hidden_size: Hidden size of the LSTM cells.
        num_layers: Number of stacked LSTM layers.
        d_model: Output feature dimension.
        dropout: Dropout rate between LSTM layers.
        bidirectional: If True, use a BiLSTM.
    """

    def __init__(
        self,
        num_input_variables: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        d_model: int = 128,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.input_projection = nn.Linear(num_input_variables, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Linear(lstm_out_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, num_input_variables).

        Returns:
            Output features (batch, d_model).
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (B, seq_len, num_vars), got {x.dim()}D")

        h = self.input_projection(x)            # (B, seq_len, hidden)
        h, _ = self.lstm(h)                     # (B, seq_len, hidden * dirs)
        h = h.transpose(1, 2)                   # (B, hidden * dirs, seq_len)
        h = self.global_pool(h).squeeze(-1)     # (B, hidden * dirs)
        return self.output_projection(h)        # (B, d_model)


class LSTMOnlyModel(nn.Module):
    """Time series-only model using an LSTM / BiLSTM encoder.

    Args:
        num_input_variables: Number of input variables.
        input_sequence_length: Length of input sequence (unused here, kept for API parity).
        num_target_variables: Number of target variables.
        target_sequence_length: Length of prediction sequence.
        d_model: Feature dimension.
        hidden_size: Hidden size of LSTM cells.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout rate.
        bidirectional: If True, use BiLSTM.
    """

    def __init__(
        self,
        num_input_variables: int,
        input_sequence_length: int,
        num_target_variables: int,
        target_sequence_length: int,
        d_model: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()

        if num_target_variables <= 0 or target_sequence_length <= 0:
            raise ValueError("Target variables and sequence length must be positive")

        self.num_target_variables = num_target_variables
        self.target_sequence_length = target_sequence_length

        self.ts_encoder = LSTMEncoder(
            num_input_variables=num_input_variables,
            hidden_size=hidden_size,
            num_layers=num_layers,
            d_model=d_model,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, target_sequence_length * num_target_variables),
        )

    def forward(
        self,
        solar_wind_input: torch.Tensor,
        image_input: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, None]]:
        """Forward pass.

        Args:
            solar_wind_input: Input time series (B, seq_len, num_vars).
            image_input: Ignored (API compatibility).
            return_features: If True, also return encoded features.

        Returns:
            Predictions (B, target_seq_len, num_target_vars), or tuple
            (predictions, features, None) if return_features=True.
        """
        features = self.ts_encoder(solar_wind_input)
        predictions = self.regression_head(features)
        output = predictions.reshape(
            predictions.size(0), self.target_sequence_length, self.num_target_variables
        )

        if return_features:
            return output, features, None
        return output


def _build_lstm_model(config, bidirectional: bool):
    """Shared factory for LSTM / BiLSTM."""
    num_input_variables, input_sequence_length, \
        num_target_variables, target_sequence_length = _get_model_dimensions(config)

    tag = "bilstm" if bidirectional else "lstm"
    print(f"Creating {tag} model: Output shape (batch, {target_sequence_length}, {num_target_variables})")

    hidden_size = getattr(config.model, 'bilstm_hidden_size', 128)
    num_layers = getattr(config.model, 'bilstm_num_layers', 2)
    dropout = getattr(config.model, 'gnn_dropout', 0.1)

    return LSTMOnlyModel(
        num_input_variables=num_input_variables,
        input_sequence_length=input_sequence_length,
        num_target_variables=num_target_variables,
        target_sequence_length=target_sequence_length,
        d_model=config.model.d_model,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )


@register_model("lstm")
def _create_lstm(config):
    """Factory function for unidirectional LSTM model."""
    return _build_lstm_model(config, bidirectional=False)


@register_model("bilstm")
def _create_bilstm(config):
    """Factory function for BiLSTM model."""
    return _build_lstm_model(config, bidirectional=True)
