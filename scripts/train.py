#!/usr/bin/env python
"""Training script for solar wind prediction model.

Usage:
    python scripts/train.py --config-name=local
    python scripts/train.py --config-name=local model.model_type=transformer
    python scripts/train.py --config-name=local model.model_type=tcn
"""

import os
import sys
from multiprocessing import freeze_support

import torch.optim as optim
import hydra

# Add parent directory to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import create_dataloader
from src.networks import create_model
from src.losses import create_loss_functions
from src.utils import setup_experiment
from src.trainers import Trainer, save_training_history, plot_training_curves


def create_optimizer(config, model):
    """Create optimizer from config.

    Args:
        config: Configuration object.
        model: PyTorch model.

    Returns:
        Optimizer instance.
    """
    weight_decay = getattr(config.training, 'weight_decay', 0.0)

    if config.training.optimizer == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        return optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=weight_decay
        )


def create_scheduler(config, optimizer):
    """Create learning rate scheduler.

    Supports multiple scheduler types:
    - "reduce_on_plateau": ReduceLROnPlateau (default)
      → validation loss가 정체되면 LR 감소
    - "cosine_annealing": CosineAnnealingWarmRestarts
      → 주기적으로 LR을 cosine 형태로 감소 후 재시작
      → 더 나은 local minima 탐색, 오버피팅 방지

    Args:
        config: Configuration object.
        optimizer: Optimizer instance.

    Returns:
        Scheduler instance.
    """
    scheduler_type = getattr(config.training, 'scheduler_type', 'reduce_on_plateau')

    if scheduler_type == "cosine_annealing":
        # CosineAnnealingWarmRestarts: 주기적으로 LR 재시작
        # → 학습 후반부에도 높은 LR로 새로운 minima 탐색 가능
        cosine_cfg = getattr(config.training, 'cosine_annealing', None)
        T_0 = cosine_cfg.T_0 if cosine_cfg else 10
        T_mult = cosine_cfg.T_mult if cosine_cfg else 2
        eta_min = cosine_cfg.eta_min if cosine_cfg else 1e-6

        print(f"Scheduler: CosineAnnealingWarmRestarts (T_0={T_0}, T_mult={T_mult})")
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )
    else:
        # ReduceLROnPlateau (default): loss 정체 시 LR 감소
        print(f"Scheduler: ReduceLROnPlateau (factor={config.training.scheduler_factor})")
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience
        )


@hydra.main(config_path="../configs", version_base=None)
def main(config):
    """Main training function.

    Args:
        config: Hydra configuration object.
    """
    # Setup experiment (seed, device)
    device = setup_experiment(config)

    # Create directories
    save_root = config.environment.save_root
    experiment_name = config.experiment.name
    experiment_dir = os.path.join(save_root, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    checkpoint_dir = os.path.join(experiment_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_dir = os.path.join(experiment_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    # Print configuration summary
    model_type = config.model.model_type
    print(f"Experiment: {experiment_name}")
    print(f"Model type: {model_type}")
    print(f"Device: {device}")

    # Create dataloaders
    train_dataloader = create_dataloader(config, phase="train")
    val_dataloader = create_dataloader(config, phase="validation")
    print(f"Training dataloader: {len(train_dataloader.dataset)} samples, {len(train_dataloader)} batches")
    print(f"Validation dataloader: {len(val_dataloader.dataset)} samples, {len(val_dataloader)} batches")

    # Create model
    model = create_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} total params, {trainable_params:,} trainable")

    # Load pretrained checkpoint if specified (for two-stage training)
    pretrained_path = getattr(config.training, 'pretrained_checkpoint', None)
    if pretrained_path:
        import torch
        full_path = os.path.join(save_root, pretrained_path)
        if os.path.exists(full_path):
            checkpoint = torch.load(full_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded pretrained checkpoint: {full_path}")
        else:
            print(f"Warning: Pretrained checkpoint not found: {full_path}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer)
    weight_decay = getattr(config.training, 'weight_decay', 0.0)
    print(f"Optimizer: {config.training.optimizer.upper()}, LR: {config.training.learning_rate}, WD: {weight_decay}")

    # Create loss functions (pass statistics for weighted loss denormalization)
    stat_dict = getattr(train_dataloader.dataset, 'stat_dict', None)
    criterion, contrastive_criterion = create_loss_functions(config, stat_dict=stat_dict)
    print(f"Loss: {config.training.regression_loss_type}, Contrastive: {config.training.contrastive_loss_type}")

    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        contrastive_criterion=contrastive_criterion,
        device=device,
        logger=None
    )

    try:
        # Run training with validation
        history = trainer.fit(
            train_dataloader,
            config.training.epochs,
            val_dataloader=val_dataloader
        )

        # Save results
        save_training_history(history, config, None)
        plot_training_curves(history, config, None)

        print("Training completed successfully")

    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    freeze_support()
    main()
