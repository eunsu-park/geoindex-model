"""Custom samplers for balanced training.

Provides a dynamic undersampling sampler that re-samples negatives each epoch
to maintain a balanced training set without modifying the underlying dataset.
"""

import logging
from typing import List

import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class BalancedUndersamplerSampler(Sampler):
    """Custom sampler that re-samples negatives each epoch for balanced training.

    Keeps ALL samples in the dataset. Each epoch, selects all positives
    plus an equal number of randomly chosen negatives (1:1 ratio).
    Compatible with persistent_workers since the dataset is unchanged.

    Args:
        labels: List of integer labels (0=negative, 1=positive) aligned
            with dataset indices.
        seed: Base random seed for reproducibility.
    """

    def __init__(self, labels: List[int], seed: int = 42):
        """Initialize sampler with label information.

        Args:
            labels: List of integer labels for each dataset sample.
            seed: Base random seed. Actual seed per epoch = seed + epoch.
        """
        self.positive_indices = [i for i, lbl in enumerate(labels) if lbl != 0]
        self.negative_indices = [i for i, lbl in enumerate(labels) if lbl == 0]
        self.num_negatives_per_epoch = len(self.positive_indices)  # 1:1 ratio
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set epoch number for deterministic resampling.

        Must be called before each epoch to get a new negative sample.

        Args:
            epoch: Current epoch number.
        """
        self.epoch = epoch

    def __iter__(self):
        """Yield shuffled indices: all positives + sampled negatives."""
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)

        # All positives always included
        pos = list(self.positive_indices)

        # Random subset of negatives (1:1 ratio with positives)
        n_neg = min(self.num_negatives_per_epoch, len(self.negative_indices))
        neg_perm = torch.randperm(len(self.negative_indices), generator=rng).tolist()
        neg = [self.negative_indices[i] for i in neg_perm[:n_neg]]

        # Combine and shuffle
        indices = pos + neg
        order = torch.randperm(len(indices), generator=rng).tolist()
        indices = [indices[i] for i in order]

        return iter(indices)

    def __len__(self) -> int:
        """Return total number of samples per epoch."""
        n_neg = min(self.num_negatives_per_epoch, len(self.negative_indices))
        return len(self.positive_indices) + n_neg
