"""Core modules for multi-modal solar wind prediction.

Modules:
- networks: Model architectures (ConvLSTM, Transformer, Fusion)
- losses: Loss functions (MSE, Huber, Contrastive)
- pipeline: Data loading and processing
- trainers: Training components
- validators: Validation components
- testers: Testing/inference components
- utils: Utility functions
"""

from .networks import (
    ConvLSTMCell,
    ConvLSTMModel,
    PositionalEncoding,
    TransformerEncoderModel,
    CrossModalAttention,
    CrossModalFusion,
    ConvLSTMOnlyModel,
    TransformerOnlyModel,
    MultiModalModel,
    create_model,
    verify_model,
)

__all__ = [
    # Network components
    "ConvLSTMCell",
    "ConvLSTMModel",
    "PositionalEncoding",
    "TransformerEncoderModel",
    "CrossModalAttention",
    "CrossModalFusion",
    # Complete models
    "ConvLSTMOnlyModel",
    "TransformerOnlyModel",
    "MultiModalModel",
    # Factory functions
    "create_model",
    "verify_model",
]
