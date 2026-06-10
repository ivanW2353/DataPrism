"""
Standalone training loop for maximum control (fallback if HF Trainer is too constrained).

Used when gradient capture per-sample or fine-grained sampling control is needed.
"""

import logging
import os
from typing import Optional

import torch
from datasets import Dataset
from peft import PeftModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from dataprism.training.data_collator import DataCollatorForLM
from dataprism.influence.checkpoint_manager import CheckpointManager

logger = logging.getLogger("dataprism.training.loop")


def train_loop(
    model: PeftModel,
    train_dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    gradient_accumulation_steps: int = 4,
    max_grad_norm: float = 1.0,
    checkpoint_manager: Optional[CheckpointManager] = None,
    checkpoint_every_n_steps: int = 50,
    importance_sampler=None,
    data_weights: Optional[dict[int, float]] = None,
    output_dir: str = "./outputs",
    log_every_n_steps: int = 10,
    device: str = "cuda",
):
    """Run a customizable LoRA fine-tuning loop.

    Args:
        model: PeftModel with LoRA adapter.
        train_dataset: Tokenized training dataset.
        tokenizer: Tokenizer for data collation.
        num_epochs: Number of training epochs.
        batch_size: Per-device batch size.
        learning_rate: Peak learning rate.
        gradient_accumulation_steps: Gradient accumulation steps.
        max_grad_norm: Max gradient norm for clipping.
        checkpoint_manager: Optional CheckpointManager for TracInCP.
        checkpoint_every_n_steps: Save checkpoints every N steps.
        importance_sampler: Optional Phase 2 importance sampler.
        data_weights: Optional Phase 3 per-sample weights.
        output_dir: Directory for outputs.
        log_every_n_steps: Logging frequency.
        device: Training device.
    """
    model = model.to(device)
    model.train()

    # Optimizer and scheduler
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
    )
    total_steps = (len(train_dataset) // batch_size) * num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    data_collator = DataCollatorForLM(tokenizer=tokenizer)
    global_step = 0

    logger.info("Starting training: %d epochs, %d steps/epoch",
                num_epochs, len(train_dataset) // batch_size)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress = tqdm(
            range(0, len(train_dataset), batch_size),
            desc=f"Epoch {epoch+1}/{num_epochs}",
        )

        for start_idx in progress:
            end_idx = min(start_idx + batch_size, len(train_dataset))
            batch_data = train_dataset[start_idx:end_idx]
            batch = data_collator(batch_data)

            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward + backward
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            epoch_loss += loss.item() * gradient_accumulation_steps

            # Gradient accumulation
            if (global_step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Logging
            if global_step % log_every_n_steps == 0:
                progress.set_postfix({
                    "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

            # Checkpoint saving
            if (checkpoint_manager is not None
                    and global_step % checkpoint_every_n_steps == 0
                    and global_step > 0):
                checkpoint_manager.save(model, global_step)

            global_step += 1

        avg_loss = epoch_loss / (len(train_dataset) // batch_size)
        logger.info("Epoch %d complete — avg loss: %.4f", epoch + 1, avg_loss)

        # Phase 3: Update data weights after each epoch
        if data_weights is not None and hasattr(data_weights, 'update'):
            data_weights.update(model, epoch)

    # Final save
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    logger.info("Training complete — model saved to %s/final_model", output_dir)
