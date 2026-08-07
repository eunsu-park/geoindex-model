"""Tests for auxiliary channel forecasting (training.aux_forecast).

The auxiliary head predicts every INPUT channel over the target window. It is
a training-time device only: it must not alter the inference return contract,
and it must fail loudly rather than silently train without its term.
"""

import pytest
import torch
from omegaconf import OmegaConf

from src.losses import create_aux_criterion
from src.networks.gnn import GNNOnlyModel


def _config(enabled=True, model_type='gnn', dataset_mode='table',
            base_loss='mse', weight=1.0):
    """Minimal config exercising only what create_aux_criterion reads."""
    return OmegaConf.create({
        'model': {'model_type': model_type},
        'data': {'timeseries': {'dataset_mode': dataset_mode}},
        'training': {
            'huber_delta': 10.0,
            'aux_forecast': {'enabled': enabled, 'base_loss': base_loss,
                             'weight': weight},
        },
    })


def _model(num_aux_variables=0):
    return GNNOnlyModel(
        num_input_variables=6,
        input_sequence_length=8,
        num_target_variables=1,
        target_sequence_length=4,
        d_model=16,
        gnn_group_sizes=[3, 3],
        gnn_num_nodes=2,
        gnn_node_feature_dim=4,
        gnn_gcn_hidden_dim=4,
        gnn_num_gcn_layers=1,
        transformer_nhead=2,
        transformer_num_layers=1,
        transformer_dim_feedforward=16,
        num_aux_variables=num_aux_variables,
    )


class TestCreateAuxCriterion:
    """Tests for the criterion factory."""

    def test_disabled_returns_none(self):
        criterion, weight = create_aux_criterion(_config(enabled=False))
        assert criterion is None
        assert weight == 0.0

    def test_absent_block_returns_none(self):
        config = _config()
        del config.training.aux_forecast
        criterion, weight = create_aux_criterion(config)
        assert criterion is None
        assert weight == 0.0

    def test_enabled_returns_criterion_and_weight(self):
        criterion, weight = create_aux_criterion(_config(weight=0.5))
        assert isinstance(criterion, torch.nn.MSELoss)
        assert weight == pytest.approx(0.5)

    @pytest.mark.parametrize('base_loss,expected', [
        ('mse', torch.nn.MSELoss),
        ('mae', torch.nn.L1Loss),
        ('huber', torch.nn.HuberLoss),
    ])
    def test_base_loss_selection(self, base_loss, expected):
        criterion, _ = create_aux_criterion(_config(base_loss=base_loss))
        assert isinstance(criterion, expected)

    def test_unknown_base_loss_raises(self):
        with pytest.raises(ValueError, match='base_loss'):
            create_aux_criterion(_config(base_loss='quantile'))

    def test_csv_mode_raises(self):
        """CSV datasets do not emit auxiliary targets, so refuse up front."""
        with pytest.raises(ValueError, match='dataset_mode'):
            create_aux_criterion(_config(dataset_mode='csv'))

    def test_non_gnn_model_raises(self):
        """Only the gnn family builds the head."""
        with pytest.raises(ValueError, match='auxiliary head'):
            create_aux_criterion(_config(model_type='transformer'))


class TestAuxHead:
    """Tests for the model-side head."""

    def test_absent_by_default(self):
        assert _model().aux_head is None

    def test_return_contract_unchanged_without_aux(self):
        """A model with an aux head still returns a bare tensor by default."""
        model = _model(num_aux_variables=6)
        out = model(torch.randn(2, 8, 6))
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 4, 1)

    def test_return_aux_shape(self):
        model = _model(num_aux_variables=6)
        out, aux = model(torch.randn(2, 8, 6), return_aux=True)
        assert out.shape == (2, 4, 1)
        assert aux.shape == (2, 4, 6)

    def test_return_aux_with_features(self):
        model = _model(num_aux_variables=6)
        out, features, none_, aux = model(
            torch.randn(2, 8, 6), return_features=True, return_aux=True)
        assert out.shape == (2, 4, 1)
        assert features.shape == (2, 16)
        assert none_ is None
        assert aux.shape == (2, 4, 6)

    def test_return_aux_without_head_raises(self):
        model = _model(num_aux_variables=0)
        with pytest.raises(RuntimeError, match='without an auxiliary head'):
            model(torch.randn(2, 8, 6), return_aux=True)

    def test_aux_head_receives_gradient(self):
        """The auxiliary term must actually reach the shared trunk."""
        model = _model(num_aux_variables=6)
        _, aux = model(torch.randn(2, 8, 6), return_aux=True)
        aux.pow(2).mean().backward()
        trunk_grads = [p.grad for n, p in model.named_parameters()
                       if n.startswith('gnn_encoder') and p.grad is not None]
        assert trunk_grads, 'auxiliary loss did not reach the encoder'
        assert any(g.abs().sum() > 0 for g in trunk_grads)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
