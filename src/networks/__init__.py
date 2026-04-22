"""Model architectures for multi-modal solar wind prediction.

This package provides model architectures organized by family.
Use create_model(config) to instantiate models by config.
"""

# Import all submodules to trigger @register_model decorators
from . import convlstm, transformer, tcn, linear, fusion, baseline, gnn, timesnet, patchtst, lstm

# Registry
from ._registry import create_model, list_models

# Base utilities
from ._base import _get_model_dimensions, build_gnn_node_groups, DEFAULT_VARIABLE_NODE_GROUPS

# Re-export all model classes for backward compatibility
from .convlstm import ConvLSTMCell, ConvLSTMModel, ConvLSTMOnlyModel
from .transformer import PositionalEncoding, TransformerEncoderModel, TransformerOnlyModel
from .tcn import TemporalBlock, TCNEncoder, TCNOnlyModel
from .linear import LinearEncoder, LinearOnlyModel
from .fusion import CrossModalAttention, CrossModalFusion, MultiModalModel
from .baseline import Conv3DEncoder, BaselineModel
from .gnn import GraphConvLayer, GNNEncoder, GNNOnlyModel
from .timesnet import InceptionBlock, TimesBlock, TimesNetEncoder, TimesNetOnlyModel
from .patchtst import PatchEmbedding, PatchTransformerEncoder, PatchTSTOnlyModel
from .lstm import LSTMEncoder, LSTMOnlyModel


# verify_model function
def verify_model(config):
    """Verify model creation and forward pass with dummy inputs."""
    import torch
    model = create_model(config)
    print(f"\nModel architecture:\n{model}\n")

    batch_size = config.experiment.batch_size
    model_type = config.model.model_type

    num_input_vars, input_seq_len, num_target_vars, target_seq_len = _get_model_dimensions(config)

    inputs = torch.randn(batch_size, input_seq_len, num_input_vars)
    sdo = None
    use_sdo = getattr(config.data.modalities, 'sdo', False)
    if use_sdo:
        sdo = torch.randn(batch_size, getattr(config.data.sdo, 'num_images', 12), 1, 128, 128)

    model.eval()
    with torch.no_grad():
        if model_type in ['fusion', 'baseline']:
            outputs, f1, f2 = model(inputs, sdo, return_features=True)
        else:
            outputs = model(inputs, sdo, return_features=False)

    print(f"Input shape:  {inputs.shape}")
    if sdo is not None:
        print(f"SDO shape:    {sdo.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Expected:     (batch={batch_size}, seq={target_seq_len}, vars={num_target_vars})")

    return model, outputs
