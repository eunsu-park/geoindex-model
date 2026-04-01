"""
Attention Analysis for Transformer Model

Extracts and analyzes attention weights from the trained Transformer encoder
to understand which temporal features the model focuses on.

Similar workflow to saliency_maps.py but without gradients (forward-only).

Features:
1. Extract attention weights from all transformer layers
2. Compute temporal importance from attention patterns
3. Visualize attention heatmaps (similar to paper Figure 6)
4. Compare attention patterns across different predictions
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict
from pathlib import Path


class AttentionExtractor:
    """
    Extract and analyze attention weights from trained Transformer model.

    Unlike SaliencyExtractor which uses gradients, this class extracts
    attention weights directly from the forward pass (much faster).

    Supports model types:
    - fusion: Full attention analysis with cross-modal fusion
    - transformer: Attention analysis on OMNI input only
    - convlstm: NOT SUPPORTED (no Transformer layers)
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize AttentionExtractor.

        Args:
            model: Trained model (fusion, transformer, or convlstm)
            device: 'cuda', 'mps', or 'cpu'

        Raises:
            ValueError: If model type is 'convlstm' (no Transformer layers)
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # Detect model type
        self.model_type = self._detect_model_type()

        if self.model_type == 'convlstm':
            raise ValueError(
                "AttentionExtractor does not support ConvLSTM-only models. "
                "ConvLSTM models do not have Transformer layers, so there are no "
                "attention weights to extract. Use SaliencyExtractor instead for "
                "SDO image analysis."
            )

        print(f"AttentionExtractor initialized on {device}")
        print(f"  Model type: {self.model_type}")

    def _detect_model_type(self) -> str:
        """Detect model type based on available components.

        Returns:
            'fusion': Has both transformer_model and convlstm_model
            'transformer': Has only transformer_model
            'convlstm': Has only convlstm_model
        """
        has_transformer = hasattr(self.model, 'transformer_model')
        has_convlstm = hasattr(self.model, 'convlstm_model')

        if has_transformer and has_convlstm:
            return 'fusion'
        elif has_transformer and not has_convlstm:
            return 'transformer'
        elif has_convlstm and not has_transformer:
            return 'convlstm'
        else:
            return 'unknown'

    def extract_attention_manual_forward(
        self,
        solar_wind_input: torch.Tensor,
        image_input: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Extract attention weights without modifying networks.py.

        Manually forward through Transformer encoder layers with need_weights=True.
        This is the core method that replaces gradient-based saliency.

        Args:
            solar_wind_input: (batch, seq_len, num_vars) - OMNI time series
            image_input: (batch, channels, seq_len, H, W) - SDO images
                         Optional for transformer-only models

        Returns:
            attention_weights: List of (batch, num_heads, seq_len, seq_len) per layer
            predictions: (batch, target_seq_len, num_vars) - Model predictions
        """
        solar_wind_input = solar_wind_input.to(self.device)
        if image_input is not None:
            image_input = image_input.to(self.device)

        transformer = self.model.transformer_model

        with torch.no_grad():
            # ================================================================
            # 1. Input Projection & Positional Encoding
            # ================================================================
            x = transformer.input_projection(solar_wind_input)  # (batch, seq_len, d_model)
            x = transformer.pos_encoder(x)  # Add positional encoding

            # ================================================================
            # 2. Transformer Encoder Layers - MANUAL FORWARD
            # ================================================================
            attention_weights = []

            for layer_idx, layer in enumerate(transformer.transformer_encoder.layers):
                # Self-attention with need_weights=True
                attn_output, attn_weight = layer.self_attn(
                    x, x, x,
                    need_weights=True,
                    average_attn_weights=False  # Keep per-head weights
                )
                # attn_weight shape: (batch, num_heads, seq_len, seq_len)
                attention_weights.append(attn_weight.detach())

                # Residual connection + Layer Norm
                x = layer.norm1(x + layer.dropout1(attn_output))

                # Feed-forward network
                x2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                x = layer.norm2(x + layer.dropout2(x2))

            # ================================================================
            # 3. Global Pooling & Output Projection
            # ================================================================
            x = x.transpose(1, 2)  # (batch, d_model, seq_len)
            x = transformer.global_pool(x).squeeze(-1)  # (batch, d_model)
            transformer_features = transformer.output_projection(x)

            # ================================================================
            # 4. Rest of the model - depends on model type
            # ================================================================
            if self.model_type == 'fusion':
                # Fusion model: ConvLSTM + Cross-modal fusion
                convlstm_features = self.model.convlstm_model(image_input)
                fused_features = self.model.cross_modal_fusion(
                    transformer_features, convlstm_features
                )
                predictions = self.model.regression_head(fused_features)
            elif self.model_type == 'transformer':
                # Transformer-only: Direct regression from transformer features
                predictions = self.model.regression_head(transformer_features)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            # Reshape output
            output = predictions.reshape(
                predictions.size(0),
                self.model.target_sequence_length,
                self.model.num_target_variables
            )

        return attention_weights, output

    def compute_temporal_importance(
        self,
        attention_weights: torch.Tensor,
        method: str = 'incoming'
    ) -> np.ndarray:
        """
        Compute temporal importance from attention weights.

        Args:
            attention_weights: (num_heads, seq_len, seq_len) from one layer
            method: 'incoming' (default) or 'outgoing'
                - incoming: sum of attention received by each timestep
                - outgoing: sum of attention given by each timestep

        Returns:
            temporal_imp: (seq_len,) - Importance score per timestep
        """
        if method == 'incoming':
            # How much attention does each timestep RECEIVE?
            # Sum over query dimension (axis=-2)
            temporal_imp = attention_weights.sum(dim=-2)  # (num_heads, seq_len)
            temporal_imp = temporal_imp.mean(dim=0)  # Average over heads
        elif method == 'outgoing':
            # How much attention does each timestep GIVE?
            # Sum over key dimension (axis=-1)
            temporal_imp = attention_weights.sum(dim=-1)  # (num_heads, seq_len)
            temporal_imp = temporal_imp.mean(dim=0)  # Average over heads
        else:
            raise ValueError(f"Unknown method: {method}")

        return temporal_imp.cpu().numpy()

    def visualize_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        save_path: Optional[Path] = None,
        title: str = "Attention Heatmap"
    ):
        """
        Visualize attention heatmap (similar to paper Figure 6).

        Args:
            attention_weights: (num_heads, seq_len, seq_len)
            save_path: Where to save the figure
            title: Plot title
        """
        # Average over heads
        avg_attention = attention_weights.mean(dim=0).cpu().numpy()  # (seq_len, seq_len)

        # Compute incoming attention for each timestep
        temporal_imp = avg_attention.sum(axis=0)  # (seq_len,)

        # Normalize for visualization
        temporal_imp = temporal_imp / temporal_imp.max()

        # Create heatmap (similar to paper style)
        fig, ax = plt.subplots(figsize=(12, 2))

        # Plot as image with colorbar
        im = ax.imshow(
            temporal_imp[np.newaxis, :],
            cmap='hot',
            aspect='auto',
            interpolation='nearest'
        )

        ax.set_xlabel('Input Timestep', fontsize=12)
        ax.set_yticks([])
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Attention Score (Normalized)', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path.name}")
            plt.close()
        else:
            plt.show()

    def visualize_attention_matrix(
        self,
        attention_weights: torch.Tensor,
        save_path: Optional[Path] = None,
        title: str = "Attention Matrix"
    ):
        """
        Visualize full attention matrix for all heads.

        Args:
            attention_weights: (num_heads, seq_len, seq_len)
            save_path: Where to save the figure
            title: Plot title
        """
        num_heads = attention_weights.shape[0]

        # Create subplot grid
        nrows = 2
        ncols = (num_heads + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))

        if num_heads == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)

        for head_idx in range(num_heads):
            row = head_idx // ncols
            col = head_idx % ncols
            ax = axes[row, col]

            attn_matrix = attention_weights[head_idx].cpu().numpy()

            im = ax.imshow(attn_matrix, cmap='viridis', aspect='auto')
            ax.set_title(f'Head {head_idx}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Key Position', fontsize=10)
            ax.set_ylabel('Query Position', fontsize=10)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide unused subplots
        for idx in range(num_heads, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].axis('off')

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path.name}")
            plt.close()
        else:
            plt.show()

    def visualize_layer_comparison(
        self,
        all_attention_weights: List[torch.Tensor],
        save_path: Optional[Path] = None
    ):
        """
        Compare attention patterns across different transformer layers.

        Args:
            all_attention_weights: List of (num_heads, seq_len, seq_len) per layer
            save_path: Where to save the figure
        """
        num_layers = len(all_attention_weights)

        fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 4))

        if num_layers == 1:
            axes = [axes]

        for layer_idx, attn_weights in enumerate(all_attention_weights):
            # Average over heads
            avg_attn = attn_weights.mean(dim=0).cpu().numpy()

            im = axes[layer_idx].imshow(avg_attn, cmap='viridis', aspect='auto')
            axes[layer_idx].set_title(f'Layer {layer_idx}', fontsize=12, fontweight='bold')
            axes[layer_idx].set_xlabel('Key Position', fontsize=10)
            axes[layer_idx].set_ylabel('Query Position', fontsize=10)

            plt.colorbar(im, ax=axes[layer_idx], fraction=0.046, pad=0.04)

        plt.suptitle('Attention Patterns Across Layers', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path.name}")
            plt.close()
        else:
            plt.show()

    def analyze_attention_statistics(
        self,
        attention_weights: List[torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """
        Compute various statistics from attention weights.

        Args:
            attention_weights: List of (num_heads, seq_len, seq_len) per layer

        Returns:
            stats: Dictionary containing various statistics
        """
        stats = {}

        for layer_idx, attn in enumerate(attention_weights):
            attn_np = attn.cpu().numpy()

            # Average attention per head
            avg_attn = attn_np.mean(axis=0)  # (num_heads, seq_len, seq_len)

            # Temporal importance (incoming attention)
            temporal_imp = avg_attn.sum(axis=-2).mean(axis=0)  # (seq_len,)

            # Attention entropy (how focused vs. uniform)
            # Higher entropy = more uniform attention
            epsilon = 1e-10
            entropy = -(attn_np * np.log(attn_np + epsilon)).sum(axis=-1).mean()

            # Diagonal dominance (self-attention strength)
            diagonal = np.diagonal(avg_attn, axis1=-2, axis2=-1).mean()

            stats[f'layer_{layer_idx}'] = {
                'temporal_importance': temporal_imp,
                'entropy': entropy,
                'diagonal_strength': diagonal,
                'max_attention': attn_np.max(),
                'mean_attention': attn_np.mean()
            }

        return stats


if __name__ == '__main__':
    print("Attention Analysis Module")
    print("Use this module to extract attention weights from trained Transformer models")
    print("\nKey differences from SaliencyExtractor:")
    print("  - No gradients needed (forward-only)")
    print("  - Much faster (~50-100x)")
    print("  - Directly shows what the model 'attends to'")
