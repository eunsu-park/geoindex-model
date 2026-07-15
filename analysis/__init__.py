"""
Analysis and Visualization Tools

This package provides tools for analyzing trained models:
- SaliencyExtractor: Grad-CAM, Integrated Gradients, Occlusion Sensitivity
- AttentionExtractor: Transformer attention weight analysis
- evaluate_mcd / recalibrate_mcd: calibration + post-hoc recalibration of the
  MC-dropout intervals folded into the validation pass (see src/uncertainty.py)
"""

from .saliency_maps import SaliencyExtractor
from .attention_analysis import AttentionExtractor

__all__ = [
    'SaliencyExtractor',
    'AttentionExtractor',
]
