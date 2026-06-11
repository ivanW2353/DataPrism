"""Logging and experiment tracking utilities for DataPrism."""

import logging
import os
import sys
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, TextIO


class TeeOutput:
    """Duplicates stdout/stderr to both terminal and a log file.

    Usage:
        with TeeOutput("outputs/logs/run.log"):
            # all print/stdout/stderr goes to both terminal and file
            run_experiment()
    """

    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        self.log_file: Optional[TextIO] = None
        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def __enter__(self):
        self.log_file = open(self.log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = _TeeStream(self._stdout, self.log_file)
        sys.stderr = _TeeStream(self._stderr, self.log_file)
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        if self.log_file:
            self.log_file.close()
        return False


class _TeeStream:
    """Stream that writes to both a primary stream and a log file."""

    def __init__(self, primary: TextIO, log_file: TextIO):
        self.primary = primary
        self.log_file = log_file

    def write(self, data: str):
        self.primary.write(data)
        self.log_file.write(data)

    def flush(self):
        self.primary.flush()
        self.log_file.flush()

    def isatty(self):
        return self.primary.isatty()

    def fileno(self):
        return self.primary.fileno()


def setup_logging(
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    experiment_name: str = "dataprism",
    capture_output: bool = True,
) -> tuple[logging.Logger, str]:
    """Configure logging to both console and file.

    Args:
        log_dir: Directory for log files (created if doesn't exist).
                If None, logs only to console.
        level: Logging level.
        experiment_name: Name for the log file.
        capture_output: If True, capture all stdout/stderr to the log file
                       (including tqdm progress bars, print statements).

    Returns:
        Tuple of (configured root logger, log file path).
    """
    logger = logging.getLogger("dataprism")
    logger.setLevel(level)

    # Build log path
    log_path = ""
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")

    # Console handler
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        console.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(console)

    # File handler for logging
    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ))
        logger.addHandler(file_handler)
        logger.info("Logging to %s", log_path)

    # Capture all stdout/stderr to file (includes tqdm, print, training metrics)
    if capture_output and log_path:
        # Append .full to distinguish from structured logs
        full_path = log_path.replace(".log", "_full.log")
        tee = TeeOutput(full_path)
        tee.__enter__()
        # Store reference so cleanup is possible (atexit handles exit)
        import atexit
        atexit.register(tee.__exit__)
        logger.info("Full terminal output: %s", full_path)

    return logger, log_path


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
