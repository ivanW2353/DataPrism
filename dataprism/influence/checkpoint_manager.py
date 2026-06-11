"""
Manages LoRA checkpoint lifecycle for TracInCP computation.

Saves only LoRA adapter weights (not full model) at regular intervals,
maintains a sliding window, and provides efficient checkpoint loading.
"""

import logging
import os
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

from peft import PeftModel

logger = logging.getLogger("dataprism.influence.checkpoint")


class CheckpointManager:
    """Manages LoRA checkpoints for TracIn influence computation.

    Implements a sliding window: when max_checkpoints is reached,
    the oldest checkpoint is pruned before saving a new one.

    Usage:
        manager = CheckpointManager("outputs/checkpoints", max_checkpoints=20)
        for step in range(num_steps):
            if step % 50 == 0:
                manager.save(model, step)
        ckpts = manager.list_checkpoints()
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 20,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Root directory for storing checkpoints.
            max_checkpoints: Maximum number of checkpoints to retain.
        """
        self._checkpoint_dir = Path(checkpoint_dir)
        self._max_checkpoints = max_checkpoints

        # Ordered dict: step → checkpoint path (FIFO for pruning)
        self._registry: OrderedDict[int, Path] = OrderedDict()

        os.makedirs(self._checkpoint_dir, exist_ok=True)

        # Scan for existing checkpoints
        self._scan_existing()

        logger.info(
            "CheckpointManager ready: dir=%s, max=%d, existing=%d",
            self._checkpoint_dir, max_checkpoints, len(self._registry),
        )

    def save(self, model: PeftModel, step: int) -> str:
        """Save LoRA adapter weights as a checkpoint.

        Args:
            model: PeftModel with trained LoRA adapter.
            step: Training step number.

        Returns:
            Path to the saved checkpoint directory.
        """
        checkpoint_path = self._checkpoint_dir / f"checkpoint-{step}"
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save only LoRA adapter weights
        model.save_pretrained(str(checkpoint_path))

        # Register
        self._registry[step] = checkpoint_path

        # Prune oldest if over limit
        if len(self._registry) > self._max_checkpoints:
            oldest_step, oldest_path = self._registry.popitem(last=False)
            self._prune(oldest_step, oldest_path)

        logger.info("Checkpoint saved at step %d (%d total)", step, len(self._registry))
        return str(checkpoint_path)

    def load(self, step: int, peft_model: PeftModel) -> PeftModel:
        """Load a LoRA checkpoint by applying its state_dict to the given PeftModel.

        This modifies the model in-place and returns it. More efficient than
        creating a new PeftModel instance (avoids adapter name conflicts).

        Args:
            step: Training step of the checkpoint.
            peft_model: The current PeftModel to apply checkpoint weights to.

        Returns:
            The same PeftModel with checkpoint weights applied.
        """
        import torch
        from safetensors.torch import load_file

        if step not in self._registry:
            available = sorted(self._registry.keys())
            raise KeyError(
                f"Checkpoint at step {step} not found. Available: {available}"
            )

        checkpoint_path = self._registry[step]
        adapter_path = os.path.join(str(checkpoint_path), "adapter_model.safetensors")

        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter weights not found at {adapter_path}")

        state_dict = load_file(adapter_path)

        # Map checkpoint keys to model keys.
        # Checkpoint uses format: ...lora_A.weight
        # Model state_dict uses:  ...lora_A.default.weight
        # Normalize by stripping '.default' from model keys for comparison.
        model_state = peft_model.state_dict()
        model_key_map = {k.replace(".default", ""): k for k in model_state}

        mapped = {}
        for ckpt_key, tensor in state_dict.items():
            # First try normalized match
            if ckpt_key in model_key_map:
                mapped[model_key_map[ckpt_key]] = tensor
            elif ckpt_key in model_state:
                mapped[ckpt_key] = tensor
            else:
                # Fallback: suffix match
                for norm_key, real_key in model_key_map.items():
                    if norm_key.endswith(ckpt_key) or ckpt_key.endswith(norm_key):
                        mapped[real_key] = tensor
                        break

        # Load mapped weights
        missing, unexpected = peft_model.load_state_dict(mapped, strict=False)
        if missing:
            logger.warning("Missing keys when loading checkpoint-%d: %d", step, len(missing))
        if unexpected:
            logger.warning("Unexpected keys when loading checkpoint-%d: %d", step, len(unexpected))

        logger.info("Checkpoint loaded into model: step=%d", step)
        return peft_model

    def list_checkpoints(self) -> list[int]:
        """Return sorted list of saved checkpoint steps."""
        return sorted(self._registry.keys())

    def get_checkpoint_path(self, step: int) -> Path:
        """Get the filesystem path for a checkpoint."""
        if step not in self._registry:
            raise KeyError(f"Checkpoint at step {step} not found")
        return self._registry[step]

    def prune_all(self) -> None:
        """Remove all checkpoints from disk."""
        for step, path in list(self._registry.items()):
            self._prune(step, path)
        self._registry.clear()
        logger.info("All checkpoints pruned")

    def _prune(self, step: int, path: Path) -> None:
        """Remove a checkpoint from disk."""
        if path.exists():
            shutil.rmtree(path)
            logger.debug("Pruned checkpoint step=%d at %s", step, path)

    def _scan_existing(self) -> None:
        """Scan checkpoint directory for existing checkpoints and register them."""
        if not self._checkpoint_dir.exists():
            return

        for entry in sorted(self._checkpoint_dir.iterdir()):
            if entry.is_dir() and entry.name.startswith("checkpoint-"):
                try:
                    step = int(entry.name.split("checkpoint-")[1])
                    # Verify it has adapter weights
                    if (entry / "adapter_config.json").exists():
                        self._registry[step] = entry
                except (ValueError, IndexError):
                    logger.warning("Skipping malformed checkpoint dir: %s", entry.name)

        logger.info("Found %d existing checkpoints", len(self._registry))

    @property
    def num_checkpoints(self) -> int:
        """Current number of saved checkpoints."""
        return len(self._registry)

    @property
    def checkpoint_dir(self) -> str:
        """Root checkpoint directory."""
        return str(self._checkpoint_dir)
