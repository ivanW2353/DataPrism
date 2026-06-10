"""
Custom callbacks for DataPrism training.

- CheckpointCallback: Saves LoRA checkpoints for TracInCP
- InfluenceCallback: Triggers influence computation between epochs (Phase 3)
"""

import logging
from typing import Optional

from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_callback import TrainingArguments

logger = logging.getLogger("dataprism.training.callbacks")


class CheckpointCallback(TrainerCallback):
    """Save additional LoRA checkpoints for TracInCP influence computation.

    Extends the standard checkpointing to ensure we capture enough
    intermediate training states for accurate influence estimation.
    """

    def __init__(self, checkpoint_manager, save_every_n_steps: int = 50):
        self._manager = checkpoint_manager
        self._save_every_n_steps = save_every_n_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Save checkpoint at configured intervals."""
        if state.global_step % self._save_every_n_steps == 0 and state.global_step > 0:
            model = kwargs.get("model")
            if model is not None:
                self._manager.save(model, state.global_step)


class InfluenceEpochCallback(TrainerCallback):
    """Trigger influence computation between epochs (Phase 3).

    This hooks into the training loop to compute TracInVS scores
    after each epoch and update per-sample sampling weights.
    """

    def __init__(self, phase3_pipeline):
        self._phase3 = phase3_pipeline

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Compute influence and update weights after each epoch."""
        model = kwargs.get("model")
        if model is not None and hasattr(self._phase3, "update_weights"):
            logger.info(
                "Epoch %d complete — computing influence scores...",
                int(state.epoch),
            )
            self._phase3.update_weights(model, int(state.epoch))
