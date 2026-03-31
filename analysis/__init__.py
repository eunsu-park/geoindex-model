"""
Analysis and Visualization Tools

This package provides tools for analyzing trained models:
- SaliencyExtractor: Grad-CAM, Integrated Gradients, Occlusion Sensitivity
- AttentionExtractor: Transformer attention weight analysis
- monte_carlo_dropout: MC Dropout for uncertainty estimation
"""

from .saliency_maps import SaliencyExtractor
from .attention_analysis import AttentionExtractor

__all__ = [
    'SaliencyExtractor',
    'AttentionExtractor',
]
