"""Training components for solar wind prediction model.

Contains Trainer, MetricsTracker, and CheckpointManager classes.

Classes:
    MetricsTracker: Track and aggregate training metrics across batches.
    CheckpointManager: Manage model checkpoint saving.
    Trainer: Main training loop for multimodal solar wind prediction.

Functions:
    save_training_history: Save training history to JSON file.
    plot_training_curves: Plot and save training curves.

Example:
    >>> from src.trainers import Trainer, MetricsTracker, CheckpointManager
    >>> trainer = Trainer(config, model, optimizer, scheduler, criterion,
    ...                   contrastive_criterion, device)
    >>> history = trainer.fit(dataloader, num_epochs=100)
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class MetricsTracker:
    """Track and aggregate training metrics.

    Attributes:
        metrics: Dictionary of metric lists (loss, reg_loss, cont_loss, etc.).
    """

    def __init__(self):
        """Initialize the metrics tracker."""
        self.reset()

    def reset(self):
        """Reset all metrics for new epoch."""
        self.metrics = {
            'loss': [],
            'reg_loss': [],
            'cont_loss': [],
            'mae': [],
            'rmse': [],
            'cosine_sim': []
        }

    def update(self, batch_metrics: Dict[str, float]):
        """Update metrics with batch results.

        Args:
            batch_metrics: Dictionary containing metric values from a batch.
        """
        for key in ['loss', 'reg_loss', 'cont_loss', 'mae', 'rmse', 'cosine_sim']:
            if key in batch_metrics:
                self.metrics[key].append(batch_metrics[key])

    def get_running_average(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """Get average of last N samples.

        Args:
            last_n: Number of recent samples to average. If None, use all.

        Returns:
            Dictionary of averaged metrics.
        """
        avg = {}
        for key, values in self.metrics.items():
            if values:
                data = values[-last_n:] if last_n else values
                avg[key] = float(np.mean(data))
        return avg

    def get_epoch_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for the epoch.

        Returns:
            Dictionary of statistics (mean, std, min, max) for each metric.
        """
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        return summary


class EarlyStopping:
    """Early stopping to terminate training when validation loss stops improving.

    Attributes:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change in loss to qualify as an improvement.
        counter: Number of epochs since last improvement.
        best_loss: Best loss seen so far.
        early_stop: Flag indicating whether to stop training.
        best_model_state: State dict of the best model.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True
    ):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change in loss to qualify as an improvement.
            restore_best_weights: Whether to restore best model weights when stopping.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Current validation loss.
            model: PyTorch model (for saving state).

        Returns:
            True if training should stop, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def restore_best_model(self, model: nn.Module):
        """Restore the best model weights.

        Args:
            model: PyTorch model to restore weights to.
        """
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


class CheckpointManager:
    """Manage model checkpoints.

    Attributes:
        checkpoint_dir: Path to checkpoint directory.
        best_loss: Best loss seen so far.
        save_freq: Frequency (in epochs) to save periodic checkpoints.
    """

    def __init__(self, checkpoint_dir: str, logger=None, save_freq: int = 1):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints.
            logger: Optional logger for output.
            save_freq: Frequency (in epochs) to save periodic checkpoints.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.save_freq = save_freq
        self.best_loss = float('inf')

    def save(self, model: nn.Module, optimizer: optim.Optimizer,
             epoch: int, loss: float, filename: Optional[str] = None):
        """Save checkpoint.

        Args:
            model: PyTorch model.
            optimizer: Optimizer.
            epoch: Current epoch number.
            loss: Current loss value.
            filename: Optional filename. If None, uses 'model_epoch_{epoch}.pth'.
        """
        if filename is None:
            filename = f"model_epoch_{epoch:04d}.pth"

        filepath = self.checkpoint_dir / filename

        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'timestamp': datetime.now().isoformat()
            }

            torch.save(checkpoint, filepath)

            message = f"Checkpoint saved: {filepath}"
            if self.logger:
                self.logger.info(message)
            else:
                print(message)

        except Exception as e:
            error_msg = f"Failed to save checkpoint {filepath}: {e}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(f"Error: {error_msg}")

    def save_if_best(self, model: nn.Module, optimizer: optim.Optimizer,
                     epoch: int, loss: float):
        """Save checkpoint if current loss is best.

        Args:
            model: PyTorch model.
            optimizer: Optimizer.
            epoch: Current epoch number.
            loss: Current loss value.
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.save(model, optimizer, epoch, loss, 'model_best.pth')

            message = f"New best model saved with loss: {loss:.6f}"
            if self.logger:
                self.logger.info(message)
            else:
                print(message)

    def save_periodic(self, model: nn.Module, optimizer: optim.Optimizer,
                      epoch: int, loss: float):
        """Save periodic checkpoint based on save frequency.

        Args:
            model: PyTorch model.
            optimizer: Optimizer.
            epoch: Current epoch number.
            loss: Current loss value.
        """
        if epoch % self.save_freq == 0:
            self.save(model, optimizer, epoch, loss)


class Trainer:
    """Trainer for multimodal solar wind prediction model.

    Supports three model types:
    - "fusion": Full multimodal model with cross-attention (SDO + OMNI)
    - "transformer": OMNI time series only
    - "convlstm": SDO images only

    Attributes:
        config: Configuration object.
        model: PyTorch model.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        criterion: Regression loss function.
        contrastive_criterion: Contrastive loss function (fusion mode only).
        device: Device for computation.
        metrics_tracker: MetricsTracker instance.
        checkpoint_manager: CheckpointManager instance.
    """

    def __init__(
        self,
        config,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        criterion: nn.Module,
        contrastive_criterion: nn.Module,
        device: torch.device,
        logger=None
    ):
        """Initialize trainer.

        Args:
            config: Configuration object.
            model: PyTorch model.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            criterion: Regression loss function.
            contrastive_criterion: Contrastive loss function.
            device: Device for computation.
            logger: Optional logger for output.
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.contrastive_criterion = contrastive_criterion
        self.device = device
        self.logger = logger

        # Model type determines training behavior
        self.model_type = getattr(config.model, 'model_type', 'fusion')

        # Components
        self.metrics_tracker = MetricsTracker()
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=f"{config.environment.save_root}/{config.experiment.name}/checkpoint",
            logger=logger,
            save_freq=config.training.model_save_freq
        )

        # Early stopping (optional, configured via training config)
        early_stopping_patience = getattr(config.training, 'early_stopping_patience', None)
        if early_stopping_patience is not None and early_stopping_patience > 0:
            self.early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                min_delta=getattr(config.training, 'early_stopping_min_delta', 0.0),
                restore_best_weights=True
            )
        else:
            self.early_stopping = None

        # Training state
        self.current_epoch = 0
        self.total_iterations = 0
        self.training_history = []

        # Lambda for contrastive loss (only used in fusion mode)
        self.lambda_contrastive_base = config.training.lambda_contrastive
        self.lambda_contrastive = self.lambda_contrastive_base

        # Contrastive warmup settings
        warmup_cfg = getattr(config.training, 'contrastive_warmup', None)
        if warmup_cfg and getattr(warmup_cfg, 'enable', False):
            self.warmup_enable = True
            self.warmup_epochs = getattr(warmup_cfg, 'warmup_epochs', 5)
            self.lambda_start = getattr(warmup_cfg, 'lambda_start', 1.0)
            self.lambda_end = getattr(warmup_cfg, 'lambda_end', 0.2)
        else:
            self.warmup_enable = False

        # =====================================================================
        # LR Warmup Settings
        # Meaning: 초기 학습률을 낮게 시작하여 점진적으로 증가
        #          → 학습 초기 불안정성 해소, 더 나은 수렴
        # =====================================================================
        lr_warmup_cfg = getattr(config.training, 'lr_warmup', None)
        if lr_warmup_cfg and getattr(lr_warmup_cfg, 'enable', False):
            self.lr_warmup_enable = True
            self.lr_warmup_epochs = getattr(lr_warmup_cfg, 'warmup_epochs', 5)
            self.lr_warmup_start_factor = getattr(lr_warmup_cfg, 'warmup_start_factor', 0.1)
            self.base_lr = config.training.learning_rate
        else:
            self.lr_warmup_enable = False

        # =====================================================================
        # Gradient Accumulation Settings
        # Meaning: 여러 미니배치의 gradient 누적 후 한 번에 업데이트
        #          → 효과적 배치 크기 증가, 학습 안정성 향상
        # =====================================================================
        self.gradient_accumulation_steps = getattr(
            config.training, 'gradient_accumulation_steps', 1
        )
        self.accumulation_counter = 0  # Track accumulated steps

        # =====================================================================
        # Scheduler Type Setting
        # Meaning: 스케줄러 종류에 따라 step() 호출 방식이 다름
        #          - reduce_on_plateau: step(loss) - loss 기반
        #          - cosine_annealing: step(epoch) - epoch 기반
        # =====================================================================
        self.scheduler_type = getattr(
            config.training, 'scheduler_type', 'reduce_on_plateau'
        )

        # Plot settings
        self.enable_plot = getattr(config.training, 'enable_plot', True)

        if self.enable_plot:
            self.plot_dir = Path(f"{config.environment.save_root}/{config.experiment.name}/plots")
            self.plot_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.plot_dir = None

        # Last batch cache for plotting
        self.last_batch_data = None
        self.last_batch_output = None

        # Normalizer for denormalization (set in fit())
        self.normalizer = None

        # Get variable info for plotting
        self._setup_variable_info()

    def _update_lambda_contrastive(self, epoch: int) -> None:
        """Update lambda_contrastive based on warmup schedule.

        Args:
            epoch: Current epoch (1-indexed).
        """
        if not self.warmup_enable:
            return

        if epoch <= self.warmup_epochs:
            # Linear decay from lambda_start to lambda_end
            progress = (epoch - 1) / max(self.warmup_epochs - 1, 1)
            self.lambda_contrastive = self.lambda_start - progress * (self.lambda_start - self.lambda_end)
        else:
            self.lambda_contrastive = self.lambda_end

    def _apply_lr_warmup(self, epoch: int) -> None:
        """Apply learning rate warmup.

        Meaning: 학습 초기에 낮은 LR로 시작하여 안정적으로 증가
                 → 초기 gradient가 큰 경우에도 안정적 학습 가능
                 → best epoch = 1 문제 해결에 기여

        Args:
            epoch: Current epoch (1-indexed).
        """
        if not self.lr_warmup_enable:
            return

        if epoch <= self.lr_warmup_epochs:
            # Linear warmup from start_factor to 1.0
            progress = epoch / self.lr_warmup_epochs
            warmup_factor = self.lr_warmup_start_factor + (1.0 - self.lr_warmup_start_factor) * progress
            new_lr = self.base_lr * warmup_factor

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            if self.logger:
                self.logger.info(f"LR Warmup: epoch {epoch}/{self.lr_warmup_epochs}, lr={new_lr:.6f}")
            else:
                print(f"LR Warmup: epoch {epoch}/{self.lr_warmup_epochs}, lr={new_lr:.6f}")

    def _setup_variable_info(self):
        """Setup variable information for plotting target variables."""
        # Get input and target variable lists based on active modality
        use_csv = getattr(self.config.data.modalities, 'timeseries', False)

        if use_csv:
            self.input_variables = list(self.config.data.timeseries.input_variables)
            self.target_variables = list(self.config.data.timeseries.target_variables)
        elif hasattr(self.config.data, 'omni') and hasattr(self.config.data.omni, 'input'):
            self.input_variables = list(self.config.data.omni.input.variables)
            self.target_variables = list(self.config.data.omni.target.variables)
        else:
            self.input_variables = list(self.config.data.input_variables)
            self.target_variables = list(self.config.data.target_variables)

        # Find which target variables are also in input
        self.target_in_input = {}
        for target_var in self.target_variables:
            if target_var in self.input_variables:
                self.target_in_input[target_var] = self.input_variables.index(target_var)

    def plot_prediction_sample(self, batch_idx: int):
        """Plot input, target, and prediction for target variables.

        Creates a time series plot showing:
        - Input (if target variable exists in input)
        - Target (ground truth)
        - Prediction (model output)

        Args:
            batch_idx: Current batch index for filename.
        """
        if not self.enable_plot:
            return

        if self.last_batch_data is None or self.last_batch_output is None:
            return

        try:
            # Get data from cache (first sample in batch)
            inputs = self.last_batch_data['inputs'][0].cpu().numpy()  # (seq_len, num_vars)
            targets = self.last_batch_data['targets'][0].cpu().numpy()  # (target_len, num_target_vars)
            outputs = self.last_batch_output[0].cpu().numpy()  # (target_len, num_target_vars)

            # Denormalize data if normalizer is available
            if self.normalizer is not None:
                # Denormalize inputs (variable by variable)
                inputs = inputs.copy()
                for var_idx, var_name in enumerate(self.input_variables):
                    inputs[:, var_idx] = self.normalizer.denormalize_omni(
                        inputs[:, var_idx], var_name
                    )

                # Denormalize targets and outputs (target variables)
                targets = targets.copy()
                outputs = outputs.copy()
                for var_idx, var_name in enumerate(self.target_variables):
                    targets[:, var_idx] = self.normalizer.denormalize_omni(
                        targets[:, var_idx], var_name
                    )
                    outputs[:, var_idx] = self.normalizer.denormalize_omni(
                        outputs[:, var_idx], var_name
                    )

            input_len = inputs.shape[0]
            target_len = targets.shape[0]
            num_target_vars = len(self.target_variables)

            # Create figure with subplot for each target variable
            fig, axes = plt.subplots(num_target_vars, 1, figsize=(14, 4 * num_target_vars))
            if num_target_vars == 1:
                axes = [axes]

            for var_idx, target_var in enumerate(self.target_variables):
                ax = axes[var_idx]

                # Time axis
                # Input: negative time steps, Target/Pred: positive time steps
                input_time = np.arange(-input_len, 0)
                target_time = np.arange(0, target_len)

                # Plot input if target variable is in input
                if target_var in self.target_in_input:
                    input_var_idx = self.target_in_input[target_var]
                    input_values = inputs[:, input_var_idx]
                    ax.plot(input_time, input_values, 'b-', linewidth=1.5,
                            label=f'Input ({target_var})', alpha=0.7)

                # Plot target and prediction
                target_values = targets[:, var_idx]
                pred_values = outputs[:, var_idx]

                ax.plot(target_time, target_values, 'g-', linewidth=2,
                        label='Target (Ground Truth)', marker='o', markersize=3)
                ax.plot(target_time, pred_values, 'r--', linewidth=2,
                        label='Prediction', marker='x', markersize=4)

                # Formatting
                ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='Reference Time')
                ax.set_xlabel('Time Step (relative)', fontsize=10)
                ax.set_ylabel(f'{target_var}', fontsize=10)
                ax.set_title(f'Epoch {self.current_epoch + 1}, Batch {batch_idx + 1} - {target_var}',
                            fontsize=12, fontweight='bold')
                ax.legend(loc='upper left', fontsize=9)
                ax.grid(True, alpha=0.3)

                # Add metrics
                mae = np.abs(target_values - pred_values).mean()
                rmse = np.sqrt(((target_values - pred_values) ** 2).mean())
                ax.text(0.98, 0.95, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}',
                        transform=ax.transAxes, fontsize=9, verticalalignment='top',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()

            # Save plot
            filename = f"pred_epoch{self.current_epoch + 1:04d}_batch{batch_idx + 1:04d}.png"
            save_path = self.plot_dir / filename
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to create prediction plot: {e}")
            else:
                print(f"Warning: Failed to create prediction plot: {e}")

    def train_step(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Perform a single training step with optional gradient accumulation.

        Gradient Accumulation:
        - 여러 미니배치의 gradient를 누적한 후 한 번에 업데이트
        - 효과적 배치 크기 = batch_size × gradient_accumulation_steps
        - 메모리 제한을 우회하면서 큰 배치 효과 달성
        - 학습 안정성 향상, gradient 추정 분산 감소

        Args:
            data_dict: Dictionary containing input data.

        Returns:
            Dictionary containing losses and metrics.
        """
        self.model.train()

        # Move data to device (sdo is optional — absent in CSV timeseries mode)
        sdo = data_dict["sdo"].to(self.device) if "sdo" in data_dict else None
        inputs = data_dict["inputs"].to(self.device)
        targets = data_dict["targets"].to(self.device)

        # Zero gradients only at the start of accumulation cycle
        # (첫 번째 미니배치에서만 gradient 초기화)
        if self.accumulation_counter == 0:
            self.optimizer.zero_grad()

        # Forward pass depends on model type
        if self.model_type == "fusion":
            # Full multimodal model with feature extraction
            outputs, transformer_features, convlstm_features = self.model(
                inputs, sdo, return_features=True
            )
            cont_loss = self.contrastive_criterion(transformer_features, convlstm_features)
            cosine_sim = F.cosine_similarity(
                transformer_features, convlstm_features, dim=1
            ).mean().item()
        elif self.model_type == "baseline":
            # Baseline model (Conv3D + Linear) with feature extraction
            outputs, ts_features, img_features = self.model(
                inputs, sdo, return_features=True
            )
            cont_loss = self.contrastive_criterion(ts_features, img_features)
            cosine_sim = F.cosine_similarity(
                ts_features, img_features, dim=1
            ).mean().item()
        elif self.model_type in ("transformer", "linear", "tcn", "convlstm",
                                   "gnn", "timesnet"):
            outputs = self.model(inputs, sdo, return_features=False)
            cont_loss = torch.tensor(0.0, device=self.device)
            cosine_sim = 0.0
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # Compute losses
        if self.criterion is not None:
            reg_loss = self.criterion(outputs, targets)
            if self.model_type in ["fusion", "baseline"]:
                total_loss = reg_loss + self.lambda_contrastive * cont_loss
            else:
                total_loss = reg_loss
        else:
            # Two-stage training: Stage 1 uses only contrastive loss
            reg_loss = torch.tensor(0.0, device=self.device)
            total_loss = cont_loss

        # Backward pass with gradient accumulation
        # Loss를 accumulation steps로 나누어 gradient scale 조정
        # (최종 gradient = 평균 gradient가 되도록)
        scaled_loss = total_loss / self.gradient_accumulation_steps
        scaled_loss.backward()

        # Increment accumulation counter
        self.accumulation_counter += 1

        # Optimizer step only after accumulating all gradients
        # (누적 완료 시에만 파라미터 업데이트)
        if self.accumulation_counter >= self.gradient_accumulation_steps:
            # Gradient clipping (누적된 전체 gradient에 적용)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.training.gradient_clip_max_norm
            )

            # Optimizer step
            self.optimizer.step()

            # Reset accumulation counter for next cycle
            self.accumulation_counter = 0

        # Calculate metrics
        with torch.no_grad():
            mae = F.l1_loss(outputs, targets).item()
            mse = F.mse_loss(outputs, targets).item()
            rmse = np.sqrt(mse)

        # Cache for plotting (detach to avoid memory leak)
        self.last_batch_data = {
            'inputs': data_dict['inputs'].detach(),
            'targets': data_dict['targets'].detach(),
        }
        if 'sdo' in data_dict:
            self.last_batch_data['sdo'] = data_dict['sdo'].detach()
        self.last_batch_output = outputs.detach()

        return {
            'loss': total_loss.item(),
            'reg_loss': reg_loss.item(),
            'cont_loss': cont_loss.item() if isinstance(cont_loss, torch.Tensor) else cont_loss,
            'mae': mae,
            'rmse': rmse,
            'cosine_sim': cosine_sim
        }

    def validate_step(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Perform a single validation step (no gradient computation).

        Args:
            data_dict: Dictionary containing input data.

        Returns:
            Dictionary containing losses and metrics.
        """
        # Move data to device (sdo is optional — absent in CSV timeseries mode)
        sdo = data_dict["sdo"].to(self.device) if "sdo" in data_dict else None
        inputs = data_dict["inputs"].to(self.device)
        targets = data_dict["targets"].to(self.device)

        # Forward pass depends on model type
        if self.model_type == "fusion":
            outputs, transformer_features, convlstm_features = self.model(
                inputs, sdo, return_features=True
            )
            cont_loss = self.contrastive_criterion(transformer_features, convlstm_features)
            cosine_sim = F.cosine_similarity(
                transformer_features, convlstm_features, dim=1
            ).mean().item()
        elif self.model_type == "baseline":
            outputs, ts_features, img_features = self.model(
                inputs, sdo, return_features=True
            )
            cont_loss = self.contrastive_criterion(ts_features, img_features)
            cosine_sim = F.cosine_similarity(
                ts_features, img_features, dim=1
            ).mean().item()
        elif self.model_type in ("transformer", "linear", "tcn", "convlstm",
                                   "gnn", "timesnet"):
            outputs = self.model(inputs, sdo, return_features=False)
            cont_loss = torch.tensor(0.0, device=self.device)
            cosine_sim = 0.0
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # Compute losses
        if self.criterion is not None:
            reg_loss = self.criterion(outputs, targets)
            if self.model_type in ["fusion", "baseline"]:
                total_loss = reg_loss + self.lambda_contrastive * cont_loss
            else:
                total_loss = reg_loss
        else:
            # Two-stage training: Stage 1 uses only contrastive loss
            reg_loss = torch.tensor(0.0, device=self.device)
            total_loss = cont_loss

        # Calculate metrics
        mae = F.l1_loss(outputs, targets).item()
        mse = F.mse_loss(outputs, targets).item()
        rmse = np.sqrt(mse)

        return {
            'loss': total_loss.item(),
            'reg_loss': reg_loss.item(),
            'cont_loss': cont_loss.item() if isinstance(cont_loss, torch.Tensor) else cont_loss,
            'mae': mae,
            'rmse': rmse,
            'cosine_sim': cosine_sim
        }

    def validate_epoch(self, dataloader) -> Dict[str, float]:
        """Validate one epoch.

        Args:
            dataloader: Validation data loader.

        Returns:
            Dictionary of epoch-averaged validation metrics.
        """
        self.model.eval()
        val_metrics_tracker = MetricsTracker()

        with torch.no_grad():
            for batch_idx, data_dict in enumerate(dataloader):
                try:
                    batch_metrics = self.validate_step(data_dict)
                    val_metrics_tracker.update(batch_metrics)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Validation step failed for batch {batch_idx}: {e}")
                    continue

        # Get epoch summary
        epoch_summary = val_metrics_tracker.get_epoch_summary()
        epoch_metrics = {f"val_{key}": stats['mean'] for key, stats in epoch_summary.items()}

        return epoch_metrics

    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train one epoch.

        Args:
            dataloader: Training data loader.

        Returns:
            Dictionary of epoch-averaged metrics.
        """
        # Update lambda_contrastive based on warmup schedule
        self._update_lambda_contrastive(self.current_epoch)

        self.metrics_tracker.reset()
        epoch_start_time = time.time()

        for batch_idx, data_dict in enumerate(dataloader):
            try:
                # Train step
                batch_metrics = self.train_step(data_dict)
                self.metrics_tracker.update(batch_metrics)
                self.total_iterations += 1

                # Log progress
                if (batch_idx + 1) % self.config.training.report_freq == 0:
                    self.log_progress(batch_idx, len(dataloader), epoch_start_time)

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Training step failed for batch {batch_idx}: {e}")
                else:
                    print(f"Warning: Batch {batch_idx} failed: {e}")
                continue

        # Handle remaining accumulated gradients at epoch end
        # (에포크 끝에서 남은 누적 gradient 처리 - 배치 수가 accumulation_steps로 나누어 떨어지지 않는 경우)
        if self.accumulation_counter > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.training.gradient_clip_max_norm
            )
            self.optimizer.step()
            self.accumulation_counter = 0

        # Get epoch summary
        epoch_summary = self.metrics_tracker.get_epoch_summary()
        epoch_metrics = {key: stats['mean'] for key, stats in epoch_summary.items()}

        # NOTE: LR scheduler stepping moved to fit() method
        # to use val_loss (selection_loss) for ReduceOnPlateau.

        # Log epoch summary
        self.log_epoch_summary(epoch_metrics, time.time() - epoch_start_time)

        return epoch_metrics

    def log_progress(self, batch_idx: int, total_batches: int, epoch_start_time: float):
        """Log training progress.

        Args:
            batch_idx: Current batch index.
            total_batches: Total number of batches.
            epoch_start_time: Start time of epoch.
        """
        avg_metrics = self.metrics_tracker.get_running_average(
            last_n=self.config.training.report_freq
        )

        elapsed_time = time.time() - epoch_start_time
        progress = (batch_idx + 1) / total_batches * 100

        message = (
            f"[Epoch {self.current_epoch + 1}, "
            f"Batch {batch_idx + 1}/{total_batches}, "
            f"Iter {self.total_iterations}] "
            f"total_loss: {avg_metrics.get('loss', 0):.6f} | "
            f"reg_loss: {avg_metrics.get('reg_loss', 0):.6f} | "
            f"cont_loss: {avg_metrics.get('cont_loss', 0):.6f} | "
            f"cosine_sim: {avg_metrics.get('cosine_sim', 0):.4f} | "
            f"MAE: {avg_metrics.get('mae', 0):.4f} | "
            f"RMSE: {avg_metrics.get('rmse', 0):.4f} | "
            f"Time: {elapsed_time:.2f}s | Progress: {progress:.1f}%"
        )

        if self.logger:
            self.logger.info(message)
        else:
            print(message)

        # Generate prediction plot at each report_freq
        self.plot_prediction_sample(batch_idx)

    def log_epoch_summary(self, metrics: Dict[str, float], duration: float):
        """Log epoch summary.

        Args:
            metrics: Dictionary of epoch metrics.
            duration: Epoch duration in seconds.
        """
        current_lr = self.optimizer.param_groups[0]['lr']

        message = (
            f"Epoch {self.current_epoch + 1} completed in {duration:.2f}s | "
            f"Total loss: {metrics.get('loss', 0):.6f} | "
            f"Reg loss: {metrics.get('reg_loss', 0):.6f} | "
            f"Cont loss: {metrics.get('cont_loss', 0):.6f} | "
            f"Cosine sim: {metrics.get('cosine_sim', 0):.4f} | "
            f"MAE: {metrics.get('mae', 0):.4f} | "
            f"RMSE: {metrics.get('rmse', 0):.4f} | "
            f"LR: {current_lr:.8f}"
        )

        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def fit(
        self,
        dataloader,
        num_epochs: int,
        val_dataloader=None
    ) -> List[Dict[str, Any]]:
        """Full training loop with optional validation.

        Args:
            dataloader: Training data loader.
            num_epochs: Number of epochs to train.
            val_dataloader: Optional validation data loader. If provided,
                validation is run after each epoch and val_loss is used
                for best model selection.

        Returns:
            Training history (list of epoch metrics).
        """
        # Store normalizer for denormalization in plots
        if hasattr(dataloader.dataset, 'normalizer'):
            self.normalizer = dataloader.dataset.normalizer

        use_validation = val_dataloader is not None

        if self.logger:
            self.logger.info("=" * 50)
            self.logger.info(f"Starting Training (model_type: {self.model_type})")
            if use_validation:
                self.logger.info("Validation enabled: using val_loss for best model selection")
            if self.early_stopping is not None:
                self.logger.info(f"Early stopping enabled: patience={self.early_stopping.patience}")
            self.logger.info("=" * 50)
        else:
            print("=" * 50)
            print(f"Starting Training (model_type: {self.model_type})")
            if use_validation:
                print("Validation enabled: using val_loss for best model selection")
            if self.early_stopping is not None:
                print(f"Early stopping enabled: patience={self.early_stopping.patience}")
            print("=" * 50)

        start_time = datetime.now()
        stopped_early = False

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Dynamic undersampling: resample negatives for this epoch
            if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
                dataloader.sampler.set_epoch(epoch)

            # Train epoch
            epoch_metrics = self.train_epoch(dataloader)

            # Validation epoch (if val_dataloader provided)
            if use_validation:
                val_metrics = self.validate_epoch(val_dataloader)
                epoch_metrics.update(val_metrics)
                # Log validation metrics
                self._log_validation_summary(val_metrics)

            # Save to history
            self.training_history.append({
                'epoch': epoch + 1,
                **epoch_metrics,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'timestamp': datetime.now().isoformat()
            })

            # Use validation loss for best model selection if available
            selection_loss = epoch_metrics.get('val_loss', epoch_metrics.get('loss', 0))

            # Learning rate scheduling (uses val_loss via selection_loss)
            if self.scheduler:
                if self.scheduler_type == "cosine_annealing":
                    self.scheduler.step(epoch + 1)
                else:
                    # ReduceLROnPlateau: step with val_loss (or train_loss fallback)
                    self.scheduler.step(selection_loss)

            # Save best model
            self.checkpoint_manager.save_if_best(
                self.model, self.optimizer, epoch + 1, selection_loss
            )

            # Save periodic checkpoint
            self.checkpoint_manager.save_periodic(
                self.model, self.optimizer, epoch + 1, selection_loss
            )

            # Early stopping check
            if self.early_stopping is not None:
                if self.early_stopping(selection_loss, self.model):
                    message = (
                        f"Early stopping triggered at epoch {epoch + 1}. "
                        f"No improvement for {self.early_stopping.patience} epochs."
                    )
                    if self.logger:
                        self.logger.info(message)
                    else:
                        print(message)
                    stopped_early = True
                    break

        # Restore best model weights if early stopping was used
        if stopped_early and self.early_stopping is not None:
            self.early_stopping.restore_best_model(self.model)
            message = "Best model weights restored from early stopping"
            if self.logger:
                self.logger.info(message)
            else:
                print(message)

        # Final model save
        actual_epochs = epoch + 1 if stopped_early else num_epochs
        self.checkpoint_manager.save(
            self.model, self.optimizer, actual_epochs,
            self.training_history[-1].get('val_loss',
                self.training_history[-1].get('loss', 0)) if self.training_history else 0,
            'model_final.pth'
        )

        # Training completion
        end_time = datetime.now()
        total_duration = end_time - start_time

        if self.logger:
            self.logger.info("=" * 50)
            if stopped_early:
                self.logger.info("Training Completed (Early Stopped)")
            else:
                self.logger.info("Training Completed")
            self.logger.info(f"Total epochs: {actual_epochs}/{num_epochs}")
            self.logger.info(f"Total iterations: {self.total_iterations}")
            self.logger.info(f"Best loss: {self.checkpoint_manager.best_loss:.6f}")
            if use_validation:
                self.logger.info("(Best loss based on validation loss)")
            self.logger.info(f"Total training time: {total_duration}")
            self.logger.info("=" * 50)
        else:
            print("=" * 50)
            if stopped_early:
                print("Training Completed (Early Stopped)")
            else:
                print("Training Completed")
            print(f"Total epochs: {actual_epochs}/{num_epochs}")
            print(f"Total iterations: {self.total_iterations}")
            print(f"Best loss: {self.checkpoint_manager.best_loss:.6f}")
            if use_validation:
                print("(Best loss based on validation loss)")
            print(f"Total training time: {total_duration}")
            print("=" * 50)

        return self.training_history

    def _log_validation_summary(self, val_metrics: Dict[str, float]):
        """Log validation metrics summary.

        Args:
            val_metrics: Dictionary of validation metrics.
        """
        message = (
            f"  [Validation] "
            f"val_loss: {val_metrics.get('val_loss', 0):.6f} | "
            f"val_reg_loss: {val_metrics.get('val_reg_loss', 0):.6f} | "
            f"val_MAE: {val_metrics.get('val_mae', 0):.4f} | "
            f"val_RMSE: {val_metrics.get('val_rmse', 0):.4f}"
        )

        if self.logger:
            self.logger.info(message)
        else:
            print(message)


def save_training_history(history: List[Dict[str, Any]], config, logger=None):
    """Save training history to JSON file.

    Args:
        history: List of epoch metrics.
        config: Configuration object.
        logger: Optional logger for output.
    """
    log_dir = f"{config.environment.save_root}/{config.experiment.name}/log"
    os.makedirs(log_dir, exist_ok=True)

    history_path = f"{log_dir}/training_history.json"

    try:
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        message = f"Training history saved: {history_path}"
        if logger:
            logger.info(message)
        else:
            print(message)

    except Exception as e:
        error_msg = f"Failed to save training history: {e}"
        if logger:
            logger.warning(error_msg)
        else:
            print(f"Warning: {error_msg}")


def plot_training_curves(history: List[Dict[str, Any]], config, logger=None):
    """Plot and save training curves.

    Args:
        history: List of epoch metrics.
        config: Configuration object.
        logger: Optional logger for output.
    """
    if not history:
        return

    log_dir = f"{config.environment.save_root}/{config.experiment.name}/log"
    os.makedirs(log_dir, exist_ok=True)

    try:
        # Extract training metrics
        epochs = [h['epoch'] for h in history]
        losses = [h.get('loss', 0) for h in history]
        reg_losses = [h.get('reg_loss', 0) for h in history]
        cont_losses = [h.get('cont_loss', 0) for h in history]
        cosine_sims = [h.get('cosine_sim', 0) for h in history]
        maes = [h.get('mae', 0) for h in history]
        rmses = [h.get('rmse', 0) for h in history]

        # Check if validation metrics are available
        has_validation = 'val_loss' in history[0]
        if has_validation:
            val_losses = [h.get('val_loss', 0) for h in history]
            val_maes = [h.get('val_mae', 0) for h in history]
            val_rmses = [h.get('val_rmse', 0) for h in history]

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Total loss (with validation if available)
        axes[0, 0].plot(epochs, losses, 'b-', linewidth=2, label='Train Loss')
        if has_validation:
            axes[0, 0].plot(epochs, val_losses, 'b--', linewidth=2, label='Val Loss')
        axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Regression vs Contrastive Loss
        axes[0, 1].plot(epochs, reg_losses, 'r-', linewidth=2, label='Regression Loss')
        axes[0, 1].plot(epochs, cont_losses, 'g-', linewidth=2, label='Contrastive Loss')
        axes[0, 1].set_title('Loss Components', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Cosine Similarity
        axes[1, 0].plot(epochs, cosine_sims, 'm-', linewidth=2)
        axes[1, 0].set_title('Feature Alignment (Cosine Similarity)',
                            fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Cosine Similarity')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Target (0.7)')
        axes[1, 0].legend()

        # MAE and RMSE (with validation if available)
        axes[1, 1].plot(epochs, maes, 'c-', linewidth=2, label='Train MAE')
        axes[1, 1].plot(epochs, rmses, 'orange', linewidth=2, label='Train RMSE')
        if has_validation:
            axes[1, 1].plot(epochs, val_maes, 'c--', linewidth=2, label='Val MAE')
            axes[1, 1].plot(epochs, val_rmses, 'orange', linewidth=2, linestyle='--', label='Val RMSE')
        axes[1, 1].set_title('Regression Metrics', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Error')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        curve_path = f"{log_dir}/training_curves.png"
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.close()

        message = f"Training curves saved: {curve_path}"
        if logger:
            logger.info(message)
        else:
            print(message)

    except Exception as e:
        error_msg = f"Failed to save training curves: {e}"
        if logger:
            logger.warning(error_msg)
        else:
            print(f"Warning: {error_msg}")
