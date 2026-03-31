"""Unit tests for networks.py components."""

import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.networks import (
    PositionalEncoding,
    TransformerEncoderModel,
    ConvLSTMCell,
    ConvLSTMModel,
    CrossModalAttention,
    CrossModalFusion,
    ConvLSTMOnlyModel,
    TransformerOnlyModel,
    MultiModalModel,
    TemporalBlock,
    TCNEncoder,
    TCNOnlyModel,
)


class TestPositionalEncoding:
    """Tests for PositionalEncoding module."""

    def test_output_shape(self):
        """Output shape should match input shape."""
        pe = PositionalEncoding(d_model=256, max_len=100)

        x = torch.randn(4, 50, 256)  # (batch, seq, d_model)
        out = pe(x)

        assert out.shape == x.shape

    def test_adds_position_info(self):
        """Output should differ from input (position info added)."""
        pe = PositionalEncoding(d_model=256, max_len=100, dropout=0.0)

        x = torch.randn(4, 50, 256)
        out = pe(x)

        assert not torch.allclose(x, out)

    def test_deterministic_without_dropout(self):
        """Same input should produce same output without dropout."""
        pe = PositionalEncoding(d_model=256, max_len=100, dropout=0.0)
        pe.eval()

        x = torch.randn(4, 50, 256)
        out1 = pe(x)
        out2 = pe(x)

        assert torch.allclose(out1, out2)


class TestTransformerEncoderModel:
    """Tests for TransformerEncoderModel."""

    def test_output_shape(self):
        """Output should have shape (batch, d_model)."""
        model = TransformerEncoderModel(
            num_input_variables=12,
            input_sequence_length=56,
            d_model=256,
            nhead=8,
            num_layers=2,
            dim_feedforward=512
        )

        x = torch.randn(4, 56, 12)  # (batch, seq, input_dim)
        out = model(x)

        assert out.shape == (4, 256)

    def test_different_sequence_lengths(self):
        """Should handle different sequence lengths (via input_sequence_length)."""
        for seq_len in [10, 50, 100]:
            model = TransformerEncoderModel(
                num_input_variables=12,
                input_sequence_length=seq_len,
                d_model=256,
                nhead=8,
                num_layers=2
            )

            x = torch.randn(4, seq_len, 12)
            out = model(x)
            assert out.shape == (4, 256)

    def test_gradient_flow(self):
        """Gradients should flow through model."""
        model = TransformerEncoderModel(
            num_input_variables=12,
            input_sequence_length=56,
            d_model=256,
            nhead=8,
            num_layers=2
        )

        x = torch.randn(4, 56, 12, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None


class TestConvLSTMCell:
    """Tests for ConvLSTMCell."""

    def test_output_shapes(self):
        """Hidden and cell states should have correct shapes."""
        cell = ConvLSTMCell(
            input_channels=3,
            hidden_channels=64,
            kernel_size=3
        )

        x = torch.randn(4, 3, 64, 64)
        h = torch.zeros(4, 64, 64, 64)
        c = torch.zeros(4, 64, 64, 64)

        h_new, c_new = cell(x, (h, c))

        assert h_new.shape == (4, 64, 64, 64)
        assert c_new.shape == (4, 64, 64, 64)

    def test_none_hidden_state(self):
        """Should initialize hidden state if None."""
        cell = ConvLSTMCell(
            input_channels=3,
            hidden_channels=64,
            kernel_size=3
        )

        x = torch.randn(4, 3, 64, 64)
        h_new, c_new = cell(x, None)

        assert h_new.shape == (4, 64, 64, 64)
        assert c_new.shape == (4, 64, 64, 64)


class TestConvLSTMModel:
    """Tests for ConvLSTMModel."""

    def test_output_shape(self):
        """Output should have shape (batch, output_dim)."""
        model = ConvLSTMModel(
            input_channels=3,
            hidden_channels=64,
            kernel_size=3,
            num_layers=2,
            output_dim=256
        )

        x = torch.randn(4, 3, 28, 64, 64)  # (batch, C, T, H, W)
        out = model(x)

        assert out.shape == (4, 256)

    def test_different_temporal_lengths(self):
        """Should handle different temporal lengths."""
        model = ConvLSTMModel(
            input_channels=3,
            hidden_channels=64,
            kernel_size=3,
            num_layers=2,
            output_dim=256
        )

        for T in [10, 28, 50]:
            x = torch.randn(4, 3, T, 64, 64)
            out = model(x)
            assert out.shape == (4, 256)

    def test_gradient_flow(self):
        """Gradients should flow through model."""
        model = ConvLSTMModel(
            input_channels=3,
            hidden_channels=64,
            kernel_size=3,
            num_layers=2,
            output_dim=256
        )

        x = torch.randn(4, 3, 28, 64, 64, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None


class TestCrossModalAttention:
    """Tests for CrossModalAttention."""

    def test_output_shape(self):
        """Output should match query shape."""
        attn = CrossModalAttention(feature_dim=256, num_heads=4)

        query = torch.randn(4, 256)
        key_value = torch.randn(4, 256)

        out = attn(query, key_value)

        assert out.shape == query.shape

    def test_attention_to_self(self):
        """Self-attention should work."""
        attn = CrossModalAttention(feature_dim=256, num_heads=4)

        x = torch.randn(4, 256)
        out = attn(x, x)

        assert out.shape == x.shape


class TestCrossModalFusion:
    """Tests for CrossModalFusion."""

    def test_output_shape(self):
        """Output should have shape (batch, feature_dim)."""
        fusion = CrossModalFusion(feature_dim=256, num_heads=4)

        transformer_features = torch.randn(4, 256)
        convlstm_features = torch.randn(4, 256)

        out = fusion(transformer_features, convlstm_features)

        assert out.shape == (4, 256)

    def test_gradient_flow(self):
        """Gradients should flow through fusion."""
        fusion = CrossModalFusion(feature_dim=256, num_heads=4)

        transformer_features = torch.randn(4, 256, requires_grad=True)
        convlstm_features = torch.randn(4, 256, requires_grad=True)

        out = fusion(transformer_features, convlstm_features)
        loss = out.sum()
        loss.backward()

        assert transformer_features.grad is not None
        assert convlstm_features.grad is not None


class TestTransformerOnlyModel:
    """Tests for TransformerOnlyModel."""

    def test_output_shape(self):
        """Output should have shape (batch, output_seq_len, num_target_vars)."""
        model = TransformerOnlyModel(
            num_input_variables=12,
            input_sequence_length=56,
            num_target_variables=1,
            target_sequence_length=24,
            d_model=256,
            transformer_nhead=8,
            transformer_num_layers=2,
            transformer_dim_feedforward=512,
            transformer_dropout=0.1
        )

        inputs = torch.randn(4, 56, 12)

        out = model(inputs)
        assert out.shape == (4, 24, 1)

    def test_return_features(self):
        """Should return features when requested."""
        model = TransformerOnlyModel(
            num_input_variables=12,
            input_sequence_length=56,
            num_target_variables=1,
            target_sequence_length=24,
            d_model=256,
            transformer_nhead=8,
            transformer_num_layers=2,
            transformer_dim_feedforward=512,
            transformer_dropout=0.1
        )

        inputs = torch.randn(4, 56, 12)

        out, tf_feat, cl_feat = model(inputs, return_features=True)

        assert out.shape == (4, 24, 1)
        assert tf_feat.shape == (4, 256)
        assert cl_feat is None

    def test_csv_timeseries_shape(self):
        """Should work with CSV timeseries shape (240 timesteps, 23 vars)."""
        model = TransformerOnlyModel(
            num_input_variables=23,
            input_sequence_length=240,
            num_target_variables=1,
            target_sequence_length=144,
            d_model=128,
            transformer_nhead=4,
            transformer_num_layers=2,
            transformer_dim_feedforward=256,
            transformer_dropout=0.1
        )

        inputs = torch.randn(4, 240, 23)
        out = model(inputs)
        assert out.shape == (4, 144, 1)


class TestConvLSTMOnlyModel:
    """Tests for ConvLSTMOnlyModel."""

    def test_output_shape(self):
        """Output should have shape (batch, output_seq_len, num_target_vars)."""
        model = ConvLSTMOnlyModel(
            num_target_variables=1,
            target_sequence_length=24,
            d_model=256,
            convlstm_input_channels=3,
            convlstm_hidden_channels=64,
            convlstm_kernel_size=3,
            convlstm_num_layers=2,
            dropout=0.1
        )

        inputs = torch.randn(4, 56, 12)  # Ignored
        sdo = torch.randn(4, 3, 28, 64, 64)

        out = model(inputs, sdo)
        assert out.shape == (4, 24, 1)

    def test_return_features(self):
        """Should return features when requested."""
        model = ConvLSTMOnlyModel(
            num_target_variables=1,
            target_sequence_length=24,
            d_model=256,
            convlstm_input_channels=3,
            convlstm_hidden_channels=64,
            convlstm_kernel_size=3,
            convlstm_num_layers=2,
            dropout=0.1
        )

        inputs = torch.randn(4, 56, 12)
        sdo = torch.randn(4, 3, 28, 64, 64)

        out, tf_feat, cl_feat = model(inputs, sdo, return_features=True)

        assert out.shape == (4, 24, 1)
        assert tf_feat is None
        assert cl_feat.shape == (4, 256)


class TestMultiModalModel:
    """Tests for MultiModalModel (fusion)."""

    def test_output_shape(self):
        """Output should have shape (batch, output_seq_len, num_target_vars)."""
        model = MultiModalModel(
            num_input_variables=12,
            input_sequence_length=56,
            num_target_variables=1,
            target_sequence_length=24,
            d_model=256,
            transformer_nhead=8,
            transformer_num_layers=2,
            transformer_dim_feedforward=512,
            transformer_dropout=0.1,
            convlstm_input_channels=3,
            convlstm_hidden_channels=64,
            convlstm_kernel_size=3,
            convlstm_num_layers=2,
            fusion_num_heads=4,
            fusion_dropout=0.1
        )

        inputs = torch.randn(4, 56, 12)
        sdo = torch.randn(4, 3, 28, 64, 64)

        out = model(inputs, sdo)
        assert out.shape == (4, 24, 1)

    def test_return_features(self):
        """Should return features when requested."""
        model = MultiModalModel(
            num_input_variables=12,
            input_sequence_length=56,
            num_target_variables=1,
            target_sequence_length=24,
            d_model=256,
            transformer_nhead=8,
            transformer_num_layers=2,
            transformer_dim_feedforward=512,
            transformer_dropout=0.1,
            convlstm_input_channels=3,
            convlstm_hidden_channels=64,
            convlstm_kernel_size=3,
            convlstm_num_layers=2,
            fusion_num_heads=4,
            fusion_dropout=0.1
        )

        inputs = torch.randn(4, 56, 12)
        sdo = torch.randn(4, 3, 28, 64, 64)

        out, tf_feat, cl_feat = model(inputs, sdo, return_features=True)

        assert out.shape == (4, 24, 1)
        assert tf_feat.shape == (4, 256)
        assert cl_feat.shape == (4, 256)


class TestTemporalBlock:
    """Tests for TemporalBlock (dilated causal convolution)."""

    def test_output_shape(self):
        """Output shape should match input shape (seq_len preserved)."""
        block = TemporalBlock(in_channels=64, out_channels=64, kernel_size=3, dilation=1)

        x = torch.randn(4, 64, 56)  # (batch, channels, seq_len)
        out = block(x)

        assert out.shape == (4, 64, 56)

    def test_channel_change(self):
        """Should handle channel size changes."""
        block = TemporalBlock(in_channels=32, out_channels=64, kernel_size=3, dilation=1)

        x = torch.randn(4, 32, 56)
        out = block(x)

        assert out.shape == (4, 64, 56)

    def test_dilation(self):
        """Should work with different dilation factors."""
        for dilation in [1, 2, 4, 8]:
            block = TemporalBlock(in_channels=64, out_channels=64, kernel_size=3, dilation=dilation)

            x = torch.randn(4, 64, 56)
            out = block(x)

            assert out.shape == (4, 64, 56)

    def test_causal(self):
        """Should be causal (no future information leakage)."""
        block = TemporalBlock(in_channels=64, out_channels=64, kernel_size=3, dilation=2)
        block.eval()

        x1 = torch.randn(1, 64, 10)
        x2 = x1.clone()
        x2[:, :, 5:] = torch.randn(1, 64, 5)

        out1 = block(x1)
        out2 = block(x2)

        assert torch.allclose(out1[:, :, :4], out2[:, :, :4], atol=1e-5)

    def test_gradient_flow(self):
        """Gradients should flow through block."""
        block = TemporalBlock(in_channels=64, out_channels=64, kernel_size=3)

        x = torch.randn(4, 64, 56, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None


class TestTCNEncoder:
    """Tests for TCNEncoder."""

    def test_output_shape(self):
        """Output should have shape (batch, output_dim)."""
        encoder = TCNEncoder(
            num_input_variables=12,
            input_sequence_length=56,
            channels=[64, 128, 256],
            output_dim=128
        )

        x = torch.randn(4, 56, 12)
        out = encoder(x)

        assert out.shape == (4, 128)

    def test_receptive_field(self):
        """Should calculate receptive field correctly."""
        encoder = TCNEncoder(
            num_input_variables=12,
            input_sequence_length=56,
            channels=[64, 128, 256],
            kernel_size=3
        )

        assert encoder.receptive_field == 29

    def test_different_channels(self):
        """Should work with different channel configurations."""
        for channels in [[32, 64], [64, 128, 256], [64, 64, 128, 128]]:
            encoder = TCNEncoder(
                num_input_variables=12,
                input_sequence_length=56,
                channels=channels,
                output_dim=128
            )

            x = torch.randn(4, 56, 12)
            out = encoder(x)

            assert out.shape == (4, 128)

    def test_gradient_flow(self):
        """Gradients should flow through encoder."""
        encoder = TCNEncoder(
            num_input_variables=12,
            input_sequence_length=56,
            channels=[64, 128],
            output_dim=128
        )

        x = torch.randn(4, 56, 12, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None

    def test_input_validation(self):
        """Should validate input dimensions."""
        encoder = TCNEncoder(
            num_input_variables=12,
            input_sequence_length=56,
            channels=[64, 128],
            output_dim=128
        )

        with pytest.raises(ValueError):
            encoder(torch.randn(4, 56))

        with pytest.raises(ValueError):
            encoder(torch.randn(4, 40, 12))

        with pytest.raises(ValueError):
            encoder(torch.randn(4, 56, 8))


class TestTCNOnlyModel:
    """Tests for TCNOnlyModel."""

    def test_output_shape(self):
        """Output should have shape (batch, target_seq_len, num_targets)."""
        model = TCNOnlyModel(
            num_input_variables=12,
            input_sequence_length=56,
            num_target_variables=1,
            target_sequence_length=24,
            d_model=128,
            tcn_channels=[64, 128, 256]
        )

        x = torch.randn(4, 56, 12)
        out = model(x)

        assert out.shape == (4, 24, 1)

    def test_return_features(self):
        """Should return features when requested."""
        model = TCNOnlyModel(
            num_input_variables=12,
            input_sequence_length=56,
            num_target_variables=1,
            target_sequence_length=24,
            d_model=128,
            tcn_channels=[64, 128]
        )

        x = torch.randn(4, 56, 12)
        out, tcn_feat, _ = model(x, return_features=True)

        assert out.shape == (4, 24, 1)
        assert tcn_feat.shape == (4, 128)

    def test_receptive_field_property(self):
        """Should expose receptive field property."""
        model = TCNOnlyModel(
            num_input_variables=12,
            input_sequence_length=56,
            num_target_variables=1,
            target_sequence_length=24,
            d_model=128,
            tcn_channels=[64, 128, 256],
            tcn_kernel_size=3
        )

        assert model.receptive_field == 29

    def test_ignores_image_input(self):
        """Should ignore image input (for API compatibility)."""
        model = TCNOnlyModel(
            num_input_variables=12,
            input_sequence_length=56,
            num_target_variables=1,
            target_sequence_length=24,
            d_model=128,
            tcn_channels=[64, 128]
        )

        x = torch.randn(4, 56, 12)
        sdo = torch.randn(4, 3, 28, 64, 64)

        model.eval()
        out1 = model(x)
        out2 = model(x, sdo)
        assert torch.allclose(out1, out2)

    def test_gradient_flow(self):
        """Gradients should flow through model."""
        model = TCNOnlyModel(
            num_input_variables=12,
            input_sequence_length=56,
            num_target_variables=1,
            target_sequence_length=24,
            d_model=128,
            tcn_channels=[64, 128]
        )

        x = torch.randn(4, 56, 12, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None

    def test_csv_timeseries_shape(self):
        """Should work with CSV timeseries shape (240 timesteps, 23 vars)."""
        model = TCNOnlyModel(
            num_input_variables=23,
            input_sequence_length=240,
            num_target_variables=1,
            target_sequence_length=144,
            d_model=128,
            tcn_channels=[64, 128, 256]
        )

        x = torch.randn(4, 240, 23)
        out = model(x)
        assert out.shape == (4, 144, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
