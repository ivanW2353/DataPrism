"""Logging and experiment tracking utilities for DataPrism."""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logging(
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    experiment_name: str = "dataprism",
) -> logging.Logger:
    """Configure logging to both console and file.

    Args:
        log_dir: Directory for log files (created if doesn't exist).
                If None, logs only to console.
        level: Logging level.
        experiment_name: Name for the log file.

    Returns:
        Configured root logger.
    """
    logger = logging.getLogger("dataprism")
    logger.setLevel(level)

    # Console handler
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        console.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(console)

    # File handler
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ))
        logger.addHandler(file_handler)
        logger.info("Logging to %s", log_path)

    return logger


def get_logger(name: str = "dataprism") -> logging.Logger:
    """Get a child logger for a specific module."""
    return logging.getLogger(f"dataprism.{name}")


def log_config_summary(config, logger: Optional[logging.Logger] = None):
    """Log a summary of the configuration."""
    if logger is None:
        logger = get_logger("config")

    logger.info("=" * 60)
    logger.info("DataPrism Configuration Summary")
    logger.info("=" * 60)
    logger.info("Model: %s", config.model.name)
    logger.info("LoRA: r=%d, alpha=%d", config.lora.r, config.lora.alpha)
    logger.info("Training: %d epochs, lr=%.2e, batch=%d",
                config.training.num_epochs,
                config.training.learning_rate,
                config.training.per_device_train_batch_size)
    logger.info("Data: %s", config.data.name)
    logger.info("Phase 1 (TracInCP): %s", "✓" if config.phase1.enabled else "✗")
    logger.info("Phase 2 (Importance): %s", "✓" if config.phase2.enabled else "✗")
    logger.info("Phase 3 (Multi-Eval): %s", "✓" if config.phase3.enabled else "✗")
    logger.info("Output: %s", config.output_dir)
    logger.info("=" * 60)
