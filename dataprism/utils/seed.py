"""Determinism utilities for reproducible experiments."""

import random
import os

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed for all random number generators.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU

    # Deterministic operations (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    get_logger().info("Random seed set to %d (deterministic mode)", seed)


def get_logger():
    from dataprism.utils.logging_utils import get_logger as _get
    return _get("seed")
