"""
Saliency/Attribution Maps for AIA (SDO) Image Inputs

Analyze which parts of SDO images are important for model predictions.

Supported methods:
1. Grad-CAM: ConvLSTM spatial activation + gradient
2. Integrated Gradients: Per-pixel contribution
3. Occlusion Sensitivity: Region masking experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional, List, Tuple, Dict
from pathlib import Path


class SaliencyExtractor:
    """Extract saliency/attribution maps from trained model.

    Supports model types:
    - fusion: Full saliency analysis (Grad-CAM + IG on SDO images)
    - convlstm: Full saliency analysis (Grad-CAM + IG on SDO images)
    - transformer: Limited support (no SDO images, only OMNI gradient analysis)
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # Storage for gradients and activations
        self.gradients = None
        self.activations = None

        # Detect model type
        self.model_type = self._detect_model_type()
        print(f"SaliencyExtractor initialized with model type: {self.model_type}")

        if self.model_type == 'transformer':
            print("  WARNING: Transformer-only model detected.")
            print("  SDO image saliency methods (Grad-CAM) are NOT available.")
            print("  Use Integrated Gradients on OMNI input instead.")

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

    def supports_sdo_saliency(self) -> bool:
        """Check if model supports SDO image saliency analysis."""
        return self.model_type in ('fusion', 'convlstm')

    # ================================================================
    # Grad-CAM for ConvLSTM
    # ================================================================

    def _register_hooks(self) -> bool:
        """Register hooks on the last ConvLSTM layer.

        Returns:
            True if hooks registered successfully, False otherwise.
        """
        if not self.supports_sdo_saliency():
            print(f"WARNING: Model type '{self.model_type}' does not support Grad-CAM.")
            print("  Grad-CAM requires ConvLSTM layers (convlstm or fusion model).")
            return False

        def forward_hook(module, input, output):
            # ConvLSTMCell output: (hidden, cell)
            self.activations = output[0].detach()  # hidden state

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0]: gradient for hidden
            self.gradients = grad_output[0].detach()

        # Register hook on last ConvLSTM layer
        target_layer = self.model.convlstm_model.convlstm_layers[-1]
        self.forward_handle = target_layer.register_forward_hook(forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(backward_hook)
        return True

    def _remove_hooks(self):
        """Remove hooks."""
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()

    def grad_cam(
        self,
        solar_wind_input: torch.Tensor,
        image_input: torch.Tensor,
        target_index: int = 0,
        target_variable: int = 0
    ) -> Optional[np.ndarray]:
        """
        Generate spatial saliency map with Grad-CAM.

        Args:
            solar_wind_input: (batch, seq_len, num_vars)
            image_input: (batch, channels, seq_len, H, W)
            target_index: Which prediction timestep to analyze
            target_variable: Which variable to analyze (e.g., ap_index)

        Returns:
            saliency_maps: (seq_len, H, W) - saliency map per timestep
            None if model does not support Grad-CAM (transformer-only)
        """
        if not self._register_hooks():
            return None

        solar_wind_input = solar_wind_input.to(self.device)
        image_input = image_input.to(self.device)
        image_input.requires_grad = True

        batch_size, channels, seq_len, H, W = image_input.shape
        saliency_maps = []

        # Compute Grad-CAM for each timestep
        for t in range(seq_len):
            self.model.zero_grad()

            # Forward pass
            output = self.model(solar_wind_input, image_input)

            # Target: prediction at specific timestep and variable
            target = output[0, target_index, target_variable]

            # Backward
            target.backward(retain_graph=(t < seq_len - 1))

            # Compute Grad-CAM
            if self.gradients is not None and self.activations is not None:
                # Global average pooling of gradients
                weights = self.gradients[0].mean(dim=[1, 2], keepdim=True)  # (channels, 1, 1)

                # Weighted sum of activations
                cam = (weights * self.activations[0]).sum(dim=0)  # (H, W)

                # ReLU
                cam = F.relu(cam)

                # Normalize
                cam = cam.cpu().numpy()
                if cam.max() > 0:
                    cam = cam / cam.max()

                saliency_maps.append(cam)
            else:
                saliency_maps.append(np.zeros((H, W)))

        self._remove_hooks()

        return np.array(saliency_maps)  # (seq_len, H, W)

    # ================================================================
    # Integrated Gradients
    # ================================================================

    def integrated_gradients(
        self,
        solar_wind_input: torch.Tensor,
        image_input: torch.Tensor,
        target_index: int = 0,
        target_variable: int = 0,
        n_steps: int = 50,
        baseline: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Compute per-pixel contribution with Integrated Gradients.

        Args:
            solar_wind_input: (batch, seq_len, num_vars)
            image_input: (batch, channels, seq_len, H, W)
            target_index: Prediction timestep
            target_variable: Prediction variable
            n_steps: Interpolation steps
            baseline: Baseline image (zeros if None)

        Returns:
            attributions: (channels, seq_len, H, W) - per-pixel contribution
        """
        solar_wind_input = solar_wind_input.to(self.device)
        image_input = image_input.to(self.device)

        # Baseline (black image)
        if baseline is None:
            baseline = torch.zeros_like(image_input)
        else:
            baseline = baseline.to(self.device)

        # Compute integrated gradients
        attributions = torch.zeros_like(image_input)

        for step in range(n_steps):
            # Interpolation - use clone() and detach()
            alpha = step / n_steps
            interpolated = (baseline + alpha * (image_input - baseline)).clone().detach()
            interpolated.requires_grad = True  # Make it a leaf variable

            # Forward
            self.model.zero_grad()
            output = self.model(solar_wind_input, interpolated)
            target = output[0, target_index, target_variable]

            # Backward
            target.backward()

            # Accumulate gradients
            if interpolated.grad is not None:
                attributions += interpolated.grad.detach() / n_steps

        # Scale by (input - baseline)
        attributions = attributions * (image_input - baseline)

        return attributions[0].detach().cpu().numpy()  # (channels, seq_len, H, W)

    def integrated_gradients_batch_targets(
        self,
        solar_wind_input: torch.Tensor,
        image_input: torch.Tensor,
        target_variable: int = 0,
        n_steps: int = 50,
        baseline: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Integrated Gradients - compute all targets at once (Batch version).

        Approximately 5-6x faster than sequential version:
        - Sequential: n_targets * n_steps forward/backward passes
        - Batch: only n_steps (all targets computed simultaneously)

        Args:
            solar_wind_input: (batch, seq_len, num_vars)
            image_input: (batch, channels, seq_len, H, W)
            target_variable: Prediction variable
            n_steps: Interpolation steps
            baseline: Baseline image (zeros if None)

        Returns:
            attributions: (n_targets, channels, seq_len, H, W) - per-target pixel contribution
        """
        solar_wind_input = solar_wind_input.to(self.device)
        image_input = image_input.to(self.device)

        # Baseline (black image)
        if baseline is None:
            baseline = torch.zeros_like(image_input)
        else:
            baseline = baseline.to(self.device)

        # Check output shape (number of targets)
        with torch.no_grad():
            test_output = self.model(solar_wind_input, image_input)
        n_targets = test_output.shape[1]  # (batch, n_targets, n_variables)

        batch_size, channels, seq_len, H, W = image_input.shape

        # Store attributions for all targets
        # Shape: (n_targets, channels, seq_len, H, W)
        all_attributions = torch.zeros(n_targets, channels, seq_len, H, W, device=self.device)

        print(f"Computing IG for {n_targets} targets with {n_steps} steps...")
        print("Using batch mode (all targets simultaneously)...")

        # Compute all targets per step
        for step in range(n_steps):
            if (step + 1) % 10 == 0 or step == 0:
                print(f"  Step {step+1}/{n_steps}...", end='\r')

            # Interpolation
            alpha = step / n_steps
            interpolated = (baseline + alpha * (image_input - baseline)).clone().detach()
            interpolated.requires_grad = True

            # Forward
            self.model.zero_grad()
            output = self.model(solar_wind_input, interpolated)
            # output shape: (batch, n_targets, n_variables)

            # Compute gradient for each target
            for target_idx in range(n_targets):
                target = output[0, target_idx, target_variable]

                # Backward (retain graph except for last target)
                target.backward(retain_graph=(target_idx < n_targets - 1))

                # Store gradient
                if interpolated.grad is not None:
                    all_attributions[target_idx] += interpolated.grad[0].detach() / n_steps

                    # Reset gradient (for next target)
                    interpolated.grad.zero_()

        print()  # newline

        # Scale by (input - baseline) for each target
        input_diff = (image_input - baseline)[0]  # (channels, seq_len, H, W)

        # Broadcasting: (n_targets, channels, seq_len, H, W) * (channels, seq_len, H, W)
        all_attributions = all_attributions * input_diff.unsqueeze(0)

        print("Batch IG computation complete!")

        return all_attributions.detach().cpu().numpy()  # (n_targets, channels, seq_len, H, W)

    # ================================================================
    # Occlusion Sensitivity
    # ================================================================

    def occlusion_sensitivity(
        self,
        solar_wind_input: torch.Tensor,
        image_input: torch.Tensor,
        target_index: int = 0,
        target_variable: int = 0,
        patch_size: int = 16,
        stride: int = 8
    ) -> np.ndarray:
        """
        Find important regions with Occlusion Sensitivity.

        Args:
            solar_wind_input: (batch, seq_len, num_vars)
            image_input: (batch, channels, seq_len, H, W)
            target_index: Prediction timestep
            target_variable: Prediction variable
            patch_size: Size of occluding patch
            stride: Sliding interval

        Returns:
            sensitivity_maps: (seq_len, H, W) - importance per region
        """
        solar_wind_input = solar_wind_input.to(self.device)
        image_input = image_input.to(self.device)

        batch_size, channels, seq_len, H, W = image_input.shape

        # Baseline prediction
        with torch.no_grad():
            baseline_output = self.model(solar_wind_input, image_input)
            baseline_pred = baseline_output[0, target_index, target_variable].item()

        sensitivity_maps = []

        # For each timestep
        for t in range(seq_len):
            sensitivity_map = np.zeros((H, W))

            # Sliding window
            for i in range(0, H - patch_size + 1, stride):
                for j in range(0, W - patch_size + 1, stride):
                    # Occlude patch with zeros
                    occluded = image_input.clone()
                    occluded[:, :, t, i:i+patch_size, j:j+patch_size] = 0

                    # Predict
                    with torch.no_grad():
                        output = self.model(solar_wind_input, occluded)
                        occluded_pred = output[0, target_index, target_variable].item()

                    # Prediction change = importance
                    importance = abs(baseline_pred - occluded_pred)
                    sensitivity_map[i:i+patch_size, j:j+patch_size] = np.maximum(
                        sensitivity_map[i:i+patch_size, j:j+patch_size],
                        importance
                    )

            # Normalize
            if sensitivity_map.max() > 0:
                sensitivity_map = sensitivity_map / sensitivity_map.max()

            sensitivity_maps.append(sensitivity_map)

        return np.array(sensitivity_maps)  # (seq_len, H, W)

    # ================================================================
    # Temporal Importance
    # ================================================================

    def temporal_importance(
        self,
        solar_wind_input: torch.Tensor,
        image_input: torch.Tensor,
        target_index: int = 0,
        target_variable: int = 0
    ) -> np.ndarray:
        """
        Compute importance of each timestep (prediction change when removed).

        Args:
            solar_wind_input: (batch, seq_len, num_vars)
            image_input: (batch, channels, seq_len, H, W)
            target_index: Prediction timestep
            target_variable: Prediction variable

        Returns:
            importance: (seq_len,) - importance per timestep
        """
        solar_wind_input = solar_wind_input.to(self.device)
        image_input = image_input.to(self.device)

        seq_len = image_input.shape[2]

        # Baseline prediction
        with torch.no_grad():
            baseline_output = self.model(solar_wind_input, image_input)
            baseline_pred = baseline_output[0, target_index, target_variable].item()

        importance = []

        for t in range(seq_len):
            # Zero out t-th timestep
            masked = image_input.clone()
            masked[:, :, t, :, :] = 0

            with torch.no_grad():
                output = self.model(solar_wind_input, masked)
                masked_pred = output[0, target_index, target_variable].item()

            # Prediction change
            importance.append(abs(baseline_pred - masked_pred))

        importance = np.array(importance)
        if importance.max() > 0:
            importance = importance / importance.max()

        return importance

    # ================================================================
    # Visualization
    # ================================================================

    def visualize_grad_cam(
        self,
        saliency_maps: np.ndarray,
        original_images: torch.Tensor,
        channel_idx: int = 0,
        time_steps: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize Grad-CAM results overlaid on original images.

        Args:
            saliency_maps: (seq_len, H, W)
            original_images: (batch, channels, seq_len, H, W)
            channel_idx: Channel to display
            time_steps: Timesteps to display (None = select a few)
            save_path: Save path
        """
        seq_len = saliency_maps.shape[0]

        if time_steps is None:
            # First, middle, last
            time_steps = [0, seq_len // 2, seq_len - 1]

        num_steps = len(time_steps)
        fig, axes = plt.subplots(3, num_steps, figsize=(4*num_steps, 10))

        if num_steps == 1:
            axes = axes[:, np.newaxis]

        for idx, t in enumerate(time_steps):
            # Original image
            orig_img = original_images[0, channel_idx, t].detach().cpu().numpy()
            orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)

            # Saliency map
            sal_map = saliency_maps[t]

            # Resize saliency to match image
            if sal_map.shape != orig_img.shape:
                sal_map = cv2.resize(sal_map, (orig_img.shape[1], orig_img.shape[0]))

            # Generate heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * sal_map), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

            # 1. Original image
            axes[0, idx].imshow(orig_img, cmap='gray')
            axes[0, idx].set_title(f'Original (t={t})')
            axes[0, idx].axis('off')

            # 2. Saliency map
            im = axes[1, idx].imshow(sal_map, cmap='hot')
            axes[1, idx].set_title(f'Saliency Map (t={t})')
            axes[1, idx].axis('off')
            plt.colorbar(im, ax=axes[1, idx], fraction=0.046)

            # 3. Overlay
            overlay = orig_img[..., np.newaxis] * 0.6 + heatmap * 0.4
            axes[2, idx].imshow(overlay)
            axes[2, idx].set_title(f'Overlay (t={t})')
            axes[2, idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_temporal_importance(
        self,
        importance: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Visualize temporal importance.

        Args:
            importance: (seq_len,)
            save_path: Save path
        """
        fig, ax = plt.subplots(figsize=(12, 4))

        ax.bar(range(len(importance)), importance, color='steelblue', alpha=0.7)
        ax.plot(range(len(importance)), importance, 'r-', linewidth=2, marker='o')
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Importance', fontsize=12)
        ax.set_title('Temporal Importance of Each Time Step', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Mark most important timestep
        max_idx = np.argmax(importance)
        ax.axvline(max_idx, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax.text(max_idx, importance[max_idx], f'  Most Important\n  t={max_idx}',
                verticalalignment='bottom', fontsize=10, color='red')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def create_comprehensive_saliency_map(
        self,
        solar_wind_input: torch.Tensor,
        image_input: torch.Tensor,
        target_index: int = 0,
        target_variable: int = 0,
        channel_idx: int = 0,
        channel_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Combine all saliency methods in one visualization.

        Args:
            solar_wind_input: (batch, seq_len, num_vars)
            image_input: (batch, channels, seq_len, H, W)
            target_index: Prediction timestep
            target_variable: Prediction variable
            channel_idx: Channel to display
            channel_names: List of channel names for labeling
            save_path: Save path
        """
        print("Computing saliency maps...")

        # 1. Grad-CAM
        print("  [1/3] Grad-CAM...")
        grad_cam_maps = self.grad_cam(
            solar_wind_input, image_input,
            target_index, target_variable
        )

        # 2. Temporal Importance
        print("  [2/3] Temporal Importance...")
        temporal_imp = self.temporal_importance(
            solar_wind_input, image_input,
            target_index, target_variable
        )

        # 3. Integrated Gradients (simplified)
        print("  [3/3] Integrated Gradients...")
        ig_maps = self.integrated_gradients(
            solar_wind_input, image_input,
            target_index, target_variable,
            n_steps=20  # Fast
        )

        # Visualize
        seq_len = grad_cam_maps.shape[0]
        time_steps = [0, seq_len // 2, seq_len - 1]

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # Row 1: Grad-CAM
        for idx, t in enumerate(time_steps):
            ax = fig.add_subplot(gs[0, idx])
            orig_img = image_input[0, channel_idx, t].detach().cpu().numpy()
            sal_map = grad_cam_maps[t]

            # Resize
            if sal_map.shape != orig_img.shape:
                sal_map = cv2.resize(sal_map, (orig_img.shape[1], orig_img.shape[0]))

            # Overlay
            orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
            heatmap = cv2.applyColorMap(np.uint8(255 * sal_map), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
            overlay = orig_img[..., np.newaxis] * 0.6 + heatmap * 0.4

            ax.imshow(overlay)
            ax.set_title(f'Grad-CAM (t={t})', fontsize=12)
            ax.axis('off')

        # Row 1, Col 4: Temporal Importance
        ax = fig.add_subplot(gs[0, 3])
        ax.bar(range(len(temporal_imp)), temporal_imp, color='steelblue', alpha=0.7)
        ax.plot(range(len(temporal_imp)), temporal_imp, 'r-', linewidth=2)
        ax.set_title('Temporal Importance', fontsize=12)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Importance')
        ax.grid(True, alpha=0.3)

        # Row 2: Integrated Gradients (spatial avg)
        for idx, t in enumerate(time_steps):
            ax = fig.add_subplot(gs[1, idx])
            orig_img = image_input[0, channel_idx, t].detach().cpu().numpy()

            # IG map: average over channels
            ig_map = np.abs(ig_maps[:, t, :, :]).mean(axis=0)
            ig_map = (ig_map - ig_map.min()) / (ig_map.max() - ig_map.min() + 1e-8)

            # Overlay
            orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
            heatmap = cv2.applyColorMap(np.uint8(255 * ig_map), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
            overlay = orig_img[..., np.newaxis] * 0.6 + heatmap * 0.4

            ax.imshow(overlay)
            ax.set_title(f'Integrated Gradients (t={t})', fontsize=12)
            ax.axis('off')

        # Row 2, Col 4: Channel Importance
        ax = fig.add_subplot(gs[1, 3])
        channel_imp = np.abs(ig_maps).sum(axis=(1, 2, 3))
        channel_imp = channel_imp / channel_imp.max()
        # Use provided channel names or default
        ch_labels = channel_names if channel_names else [f'Ch{i}' for i in range(image_input.shape[1])]
        ax.bar(ch_labels, channel_imp, color=['red', 'green', 'blue'][:len(ch_labels)])
        ax.set_title('Channel Importance', fontsize=12)
        ax.set_ylabel('Importance')
        ax.grid(True, alpha=0.3)

        # Row 3: Original images
        for idx, t in enumerate(time_steps):
            ax = fig.add_subplot(gs[2, idx])
            orig_img = image_input[0, channel_idx, t].detach().cpu().numpy()
            orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
            ax.imshow(orig_img, cmap='gray')
            ax.set_title(f'Original (t={t})', fontsize=12)
            ax.axis('off')

        # Row 3, Col 4: Prediction info
        ax = fig.add_subplot(gs[2, 3])
        ax.axis('off')

        with torch.no_grad():
            output = self.model(solar_wind_input.to(self.device), image_input.to(self.device))
            pred = output[0, target_index, target_variable].item()

        info_text = f"""
        Target Prediction:
        - Time Index: {target_index}
        - Variable: {target_variable}
        - Value: {pred:.2f}

        Most Important:
        - Time Step: {np.argmax(temporal_imp)}
        - Channel: {ch_labels[np.argmax(channel_imp)]}
        """
        ax.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.suptitle('Comprehensive Saliency Analysis', fontsize=16, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nSaved: {save_path}")
        else:
            plt.show()

        plt.close()


    def visualize_full_sequence_analysis(
        self,
        solar_wind_input: torch.Tensor,
        image_input: torch.Tensor,
        target_index: int = 0,
        target_variable: int = 0,
        channel_idx: int = 0,
        save_path: Optional[str] = None
    ):
        """
        Comprehensive analysis plot showing entire input sequence.

        For all timesteps:
        1. Original image sequence (small thumbnails)
        2. Grad-CAM sequence
        3. Integrated Gradients sequence
        4. Temporal importance curve
        5. Prediction output

        Args:
            solar_wind_input: (batch, seq_len, num_vars)
            image_input: (batch, channels, seq_len, H, W)
            target_index: Prediction timestep
            target_variable: Prediction variable
            channel_idx: Channel to display
            save_path: Save path
        """
        print("\nGenerating full sequence analysis...")

        # 1. Compute Grad-CAM
        print("  Computing Grad-CAM for all time steps...")
        grad_cam_maps = self.grad_cam(
            solar_wind_input, image_input,
            target_index, target_variable
        )

        # 2. Compute Integrated Gradients
        print("  Computing Integrated Gradients...")
        ig_maps = self.integrated_gradients(
            solar_wind_input, image_input,
            target_index, target_variable,
            n_steps=50
        )

        # 3. Compute Temporal importance
        print("  Computing temporal importance...")
        temporal_imp = self.temporal_importance(
            solar_wind_input, image_input,
            target_index, target_variable
        )

        # 4. Compute predictions
        with torch.no_grad():
            output = self.model(
                solar_wind_input.to(self.device),
                image_input.to(self.device)
            )
            predictions = output[0, :, target_variable].cpu().numpy()

        # Sequence length
        seq_len = grad_cam_maps.shape[0]

        # Create figure (5 rows)
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(5, 1, height_ratios=[2, 2, 2, 1.5, 1], hspace=0.3)

        # ================================================================
        # Row 1: Original image sequence
        # ================================================================
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title('Original SDO Image Sequence', fontsize=14, fontweight='bold', pad=10)

        # Concatenate all timestep images horizontally
        image_sequence = []
        for t in range(seq_len):
            img = image_input[0, channel_idx, t].detach().cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            image_sequence.append(img)

        concat_images = np.concatenate(image_sequence, axis=1)
        ax1.imshow(concat_images, cmap='gray', aspect='auto')
        ax1.set_ylabel('Spatial\nDimension', fontsize=10)
        ax1.set_xlabel('')

        # X-axis: timestep labels
        img_width = image_sequence[0].shape[1]
        tick_positions = [img_width * t + img_width // 2 for t in range(seq_len)]
        ax1.set_xticks(tick_positions[::max(1, seq_len // 10)])  # Show only 10 labels
        ax1.set_xticklabels([f't={t}' for t in range(seq_len)][::max(1, seq_len // 10)])
        ax1.tick_params(axis='x', labelsize=8)

        # Highlight most important timestep
        most_important_t = np.argmax(temporal_imp)
        ax1.axvline(x=most_important_t * img_width + img_width // 2,
                   color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax1.text(most_important_t * img_width + img_width // 2,
                ax1.get_ylim()[0] * 0.95,
                f'Most Important\nt={most_important_t}',
                ha='center', va='top', color='red', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # ================================================================
        # Row 2: Grad-CAM sequence
        # ================================================================
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_title('Grad-CAM Saliency Sequence', fontsize=14, fontweight='bold', pad=10)

        # Overlay Grad-CAM on original images
        saliency_sequence = []
        for t in range(seq_len):
            # Original image
            orig = image_input[0, channel_idx, t].detach().cpu().numpy()
            orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)

            # Saliency map
            sal = grad_cam_maps[t]
            if sal.shape != orig.shape:
                sal = cv2.resize(sal, (orig.shape[1], orig.shape[0]))

            # Generate heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * sal), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

            # Overlay
            overlay = orig[..., np.newaxis] * 0.5 + heatmap * 0.5
            saliency_sequence.append(overlay)

        concat_saliency = np.concatenate(saliency_sequence, axis=1)
        ax2.imshow(concat_saliency, aspect='auto')
        ax2.set_ylabel('Spatial\nDimension', fontsize=10)
        ax2.set_xlabel('')
        ax2.set_xticks(tick_positions[::max(1, seq_len // 10)])
        ax2.set_xticklabels([f't={t}' for t in range(seq_len)][::max(1, seq_len // 10)])
        ax2.tick_params(axis='x', labelsize=8)

        # Highlight most important timestep
        ax2.axvline(x=most_important_t * img_width + img_width // 2,
                   color='red', linestyle='--', linewidth=2, alpha=0.7)

        # Colorbar
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm = ScalarMappable(cmap='jet', norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2, orientation='vertical', pad=0.01, fraction=0.02)
        cbar.set_label('Grad-CAM\nIntensity', fontsize=9)

        # ================================================================
        # Row 3: Integrated Gradients sequence
        # ================================================================
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.set_title('Integrated Gradients Sequence', fontsize=14, fontweight='bold', pad=10)

        # Overlay IG on original images
        ig_sequence = []
        for t in range(seq_len):
            # Original image
            orig = image_input[0, channel_idx, t].detach().cpu().numpy()
            orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)

            # IG map: select channel + absolute value
            ig_map = np.abs(ig_maps[channel_idx, t, :, :])

            # Normalize
            ig_map = (ig_map - ig_map.min()) / (ig_map.max() - ig_map.min() + 1e-8)

            # Resize if needed
            if ig_map.shape != orig.shape:
                ig_map = cv2.resize(ig_map, (orig.shape[1], orig.shape[0]))

            # Generate heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * ig_map), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

            # Overlay
            overlay = orig[..., np.newaxis] * 0.5 + heatmap * 0.5
            ig_sequence.append(overlay)

        concat_ig = np.concatenate(ig_sequence, axis=1)
        ax3.imshow(concat_ig, aspect='auto')
        ax3.set_ylabel('Spatial\nDimension', fontsize=10)
        ax3.set_xlabel('')
        ax3.set_xticks(tick_positions[::max(1, seq_len // 10)])
        ax3.set_xticklabels([f't={t}' for t in range(seq_len)][::max(1, seq_len // 10)])
        ax3.tick_params(axis='x', labelsize=8)

        # Highlight most important timestep
        ax3.axvline(x=most_important_t * img_width + img_width // 2,
                   color='red', linestyle='--', linewidth=2, alpha=0.7)

        # Colorbar
        sm_ig = ScalarMappable(cmap='jet', norm=Normalize(vmin=0, vmax=1))
        sm_ig.set_array([])
        cbar_ig = plt.colorbar(sm_ig, ax=ax3, orientation='vertical', pad=0.01, fraction=0.02)
        cbar_ig.set_label('IG\nIntensity', fontsize=9)

        # ================================================================
        # Row 4: Temporal Importance
        # ================================================================
        ax4 = fig.add_subplot(gs[3, 0])
        ax4.set_title('Temporal Importance', fontsize=14, fontweight='bold', pad=10)

        # Bar plot
        colors = ['red' if i == most_important_t else 'steelblue' for i in range(seq_len)]
        bars = ax4.bar(range(seq_len), temporal_imp, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Line plot
        ax4.plot(range(seq_len), temporal_imp, 'k-', linewidth=2, marker='o', markersize=4)

        ax4.set_xlabel('Time Step', fontsize=11)
        ax4.set_ylabel('Importance', fontsize=11)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_xlim(-0.5, seq_len - 0.5)

        # Mark maximum
        ax4.axvline(most_important_t, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax4.text(most_important_t, temporal_imp[most_important_t] * 1.05,
                f'Peak\nt={most_important_t}\n({temporal_imp[most_important_t]:.3f})',
                ha='center', va='bottom', fontsize=9, color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Adjust x-axis ticks
        if seq_len > 20:
            ax4.set_xticks(range(0, seq_len, max(1, seq_len // 10)))

        # ================================================================
        # Row 5: Prediction Output
        # ================================================================
        ax5 = fig.add_subplot(gs[4, 0])
        ax5.set_title('Model Predictions', fontsize=14, fontweight='bold', pad=10)

        time_axis = range(len(predictions))
        ax5.plot(time_axis, predictions, 'b-', linewidth=2, marker='s', markersize=6, label='Predicted')

        # Highlight target timestep
        ax5.axvline(target_index, color='green', linestyle='--', linewidth=2, alpha=0.5)
        ax5.plot(target_index, predictions[target_index], 'go', markersize=10, label=f'Target (t={target_index})')

        ax5.set_xlabel('Prediction Time Step', fontsize=11)
        ax5.set_ylabel('Prediction Value', fontsize=11)
        ax5.grid(True, alpha=0.3, linestyle='--')
        ax5.legend(loc='upper right', fontsize=9)

        # Display prediction statistics
        stats_text = f'Mean: {predictions.mean():.3f}\nStd: {predictions.std():.3f}\nMin: {predictions.min():.3f}\nMax: {predictions.max():.3f}'
        ax5.text(0.02, 0.98, stats_text, transform=ax5.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5))

        # ================================================================
        # Overall title
        # ================================================================
        fig.suptitle(
            f'Full Sequence Analysis (Grad-CAM + IG) - Channel {channel_idx} (Target: t={target_index}, var={target_variable})',
            fontsize=16, fontweight='bold', y=0.995
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()

        plt.close()

        # Print statistics summary
        print("\n" + "="*60)
        print("SEQUENCE ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total time steps: {seq_len}")
        print(f"Most important time step: t={most_important_t} (importance: {temporal_imp[most_important_t]:.4f})")
        print(f"\nPrediction statistics:")
        print(f"  Mean: {predictions.mean():.4f}")
        print(f"  Std:  {predictions.std():.4f}")
        print(f"  Min:  {predictions.min():.4f}")
        print(f"  Max:  {predictions.max():.4f}")
        print(f"\nGrad-CAM statistics:")
        print(f"  Mean saliency: {grad_cam_maps.mean():.4f}")
        print(f"  Max saliency:  {grad_cam_maps.max():.4f}")
        print(f"\nIntegrated Gradients statistics:")
        print(f"  Mean |attribution|: {np.abs(ig_maps).mean():.4f}")
        print(f"  Max |attribution|:  {np.abs(ig_maps).max():.4f}")
        print("="*60)


    def visualize_all_channels_analysis(
        self,
        solar_wind_input: torch.Tensor,
        image_input: torch.Tensor,
        target_index: int = 0,
        target_variable: int = 0,
        channel_names: Optional[List[str]] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Perform full analysis for all channels and save.

        For each channel:
        1. Grad-CAM
        2. Full Sequence Analysis
        3. Comprehensive Saliency

        Args:
            solar_wind_input: (batch, seq_len, num_vars)
            image_input: (batch, channels, seq_len, H, W)
            target_index: Prediction timestep
            target_variable: Prediction variable
            channel_names: Channel name list (e.g., ['193A', '211A', '304A'])
            output_dir: Output directory
        """
        num_channels = image_input.shape[1]

        if channel_names is None:
            channel_names = [f'Channel_{i}' for i in range(num_channels)]

        if output_dir is None:
            output_dir = Path('.')
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)

        print("\n" + "=" * 70)
        print(f"MULTI-CHANNEL ANALYSIS: {num_channels} channels")
        print("=" * 70)

        for ch_idx, ch_name in enumerate(channel_names[:num_channels]):
            print(f"\n{'-' * 70}")
            print(f"Processing Channel {ch_idx}: {ch_name}")
            print('-' * 70)

            ch_dir = output_dir / f"channel_{ch_idx}_{ch_name.replace('A', 'A')}"
            ch_dir.mkdir(exist_ok=True, parents=True)

            # 1. Grad-CAM
            print(f"\n[1/4] Grad-CAM for {ch_name}...")
            try:
                grad_cam_maps = self.grad_cam(
                    solar_wind_input, image_input,
                    target_index, target_variable
                )

                self.visualize_grad_cam(
                    grad_cam_maps, image_input,
                    channel_idx=ch_idx,
                    save_path=ch_dir / f"grad_cam_{ch_name}.png"
                )
                print(f"  Saved: grad_cam_{ch_name}.png")
            except Exception as e:
                print(f"  Error: {e}")

            # 2. Full Sequence Analysis
            print(f"\n[2/4] Full Sequence for {ch_name}...")
            try:
                self.visualize_full_sequence_analysis(
                    solar_wind_input, image_input,
                    target_index, target_variable,
                    channel_idx=ch_idx,
                    save_path=ch_dir / f"full_sequence_{ch_name}.png"
                )
                print(f"  Saved: full_sequence_{ch_name}.png")
            except Exception as e:
                print(f"  Error: {e}")

            # 3. Comprehensive Saliency
            print(f"\n[3/4] Comprehensive Saliency for {ch_name}...")
            try:
                self.create_comprehensive_saliency_map(
                    solar_wind_input, image_input,
                    target_index, target_variable,
                    channel_idx=ch_idx,
                    channel_names=channel_names,
                    save_path=ch_dir / f"comprehensive_{ch_name}.png"
                )
                print(f"  Saved: comprehensive_{ch_name}.png")
            except Exception as e:
                print(f"  Error: {e}")

            # 4. Temporal Importance (common for all channels)
            if ch_idx == 0:  # Calculate only for first channel (same for all)
                print(f"\n[4/4] Temporal Importance (all channels)...")
                try:
                    temporal_imp = self.temporal_importance(
                        solar_wind_input, image_input,
                        target_index, target_variable
                    )

                    self.visualize_temporal_importance(
                        temporal_imp,
                        save_path=output_dir / "temporal_importance_all_channels.png"
                    )
                    print(f"  Saved: temporal_importance_all_channels.png")
                except Exception as e:
                    print(f"  Error: {e}")

        # Create channel comparison plot
        print(f"\n{'-' * 70}")
        print("Creating Channel Comparison Plot...")
        print('-' * 70)

        try:
            self._create_channel_comparison_plot(
                solar_wind_input, image_input,
                target_index, target_variable,
                channel_names, output_dir
            )
            print(f"  Saved: channel_comparison.png")
        except Exception as e:
            print(f"  Error: {e}")

        print("\n" + "=" * 70)
        print("MULTI-CHANNEL ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"\nAll results saved to: {output_dir}")
        print(f"\nGenerated {num_channels} channel directories:")
        for ch_idx, ch_name in enumerate(channel_names[:num_channels]):
            ch_dir_name = f"channel_{ch_idx}_{ch_name.replace('A', 'A')}"
            print(f"  - {ch_dir_name}/")

    def _create_channel_comparison_plot(
        self,
        solar_wind_input: torch.Tensor,
        image_input: torch.Tensor,
        target_index: int,
        target_variable: int,
        channel_names: List[str],
        output_dir: Path
    ):
        """
        Create channel comparison plot.

        Show representative timestep images and Grad-CAM side by side.
        """
        num_channels = image_input.shape[1]
        seq_len = image_input.shape[2]

        # Compute Grad-CAM
        grad_cam_maps = self.grad_cam(
            solar_wind_input, image_input,
            target_index, target_variable
        )

        # Temporal importance
        temporal_imp = self.temporal_importance(
            solar_wind_input, image_input,
            target_index, target_variable
        )
        most_important_t = np.argmax(temporal_imp)

        # Select 3 timesteps: first, most important, last
        time_points = [0, most_important_t, seq_len - 1]

        # Create figure
        fig, axes = plt.subplots(
            num_channels, len(time_points) * 2,  # Original + Grad-CAM per timestep
            figsize=(4 * len(time_points) * 2, 4 * num_channels)
        )

        # Handle single channel
        if num_channels == 1:
            axes = axes.reshape(1, -1)

        for ch_idx in range(num_channels):
            ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f'Ch{ch_idx}'

            for t_idx, t in enumerate(time_points):
                col_orig = t_idx * 2
                col_sal = t_idx * 2 + 1

                # Original image
                orig_img = image_input[0, ch_idx, t].detach().cpu().numpy()
                orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)

                axes[ch_idx, col_orig].imshow(orig_img, cmap='gray')
                axes[ch_idx, col_orig].set_title(
                    f'{ch_name} - t={t}\n{"(Most Important)" if t == most_important_t else ""}',
                    fontsize=10, fontweight='bold' if t == most_important_t else 'normal'
                )
                axes[ch_idx, col_orig].axis('off')

                # Grad-CAM overlay
                sal_map = grad_cam_maps[t]
                if sal_map.shape != orig_img.shape:
                    sal_map = cv2.resize(sal_map, (orig_img.shape[1], orig_img.shape[0]))

                heatmap = cv2.applyColorMap(np.uint8(255 * sal_map), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
                overlay = orig_img[..., np.newaxis] * 0.5 + heatmap * 0.5

                axes[ch_idx, col_sal].imshow(overlay)
                axes[ch_idx, col_sal].set_title(f'Grad-CAM', fontsize=10)
                axes[ch_idx, col_sal].axis('off')

        plt.suptitle(
            f'Channel Comparison - Target: t={target_index}, var={target_variable}',
            fontsize=14, fontweight='bold'
        )

        plt.tight_layout()
        plt.savefig(output_dir / 'channel_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    print("Saliency Maps Module")
    print("Use this module to extract saliency/attribution maps from trained models")
