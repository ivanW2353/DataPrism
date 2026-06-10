"""
Custom HuggingFace Trainer subclass for DataPrism.

Integrates:
- LoRA checkpoint saving for TracInCP (Phase 1 & 3)
- Importance-sampled data loading (Phase 2)
- Weighted sampling (Phase 3)
"""

import logging
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from datasets import Dataset

from dataprism.training.data_collator import DataCollatorForLM, ImportanceWeightedCollator

logger = logging.getLogger("dataprism.training")


class DataPrismTrainer(Trainer):
    """Extended HF Trainer with DataPrism-specific capabilities.

    Key extensions:
    1. Custom data collator with importance weight support
    2. LoRA checkpoint saving for TracInCP
    3. Support for weighted sampling (Phase 3)
    """

    def __init__(
        self,
        model=None,
        args: TrainingArguments = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer=None,
        data_collator=None,
        importance_sampler=None,
        data_weights: Optional[dict[int, float]] = None,
        checkpoint_manager=None,
        *pos_args,
        **kwargs,
    ):
        """Initialize the DataPrism trainer.

        Args:
            importance_sampler: ImportanceSampler for Phase 2.
            data_weights: Per-sample weights for Phase 3 weighted sampling.
            checkpoint_manager: CheckpointManager for TracInCP.
        """
        self._importance_sampler = importance_sampler
        self._data_weights = data_weights
        self._checkpoint_manager = checkpoint_manager
        self._tokenizer = tokenizer

        # Default data collator
        if data_collator is None and tokenizer is not None:
            if importance_sampler is not None:
                data_collator = ImportanceWeightedCollator(tokenizer=tokenizer)
            else:
                data_collator = DataCollatorForLM(tokenizer=tokenizer)

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            *pos_args,
            **kwargs,
        )

    def get_train_dataloader(self) -> DataLoader:
        """Return a custom DataLoader supporting importance sampling and weights.

        For Phase 2: uses ImportanceDataLoader with live candidate sampling.
        For Phase 3: uses WeightedDataLoader based on per-sample weights.
        Otherwise: falls back to standard HF DataLoader.
        """
        if self._importance_sampler is not None:
            return self._get_importance_dataloader()
        elif self._data_weights is not None:
            return self._get_weighted_dataloader()
        else:
            return super().get_train_dataloader()

    def _get_importance_dataloader(self) -> DataLoader:
        """Create a DataLoader with importance-sampled batches (Phase 2)."""
        from torch.utils.data import DataLoader as TorchDataLoader

        class ImportanceDataset(torch.utils.data.Dataset):
            def __init__(self, base_dataset, sampler, model, batch_size, collator):
                self.base = base_dataset
                self.sampler = sampler
                self.model = model
                self.batch_size = batch_size
                self.collator = collator
                self._step = 0

            def __len__(self):
                # Approximate length — actual batches vary due to sampling
                return len(self.base) // self.batch_size

            def __getitem__(self, idx):
                # Per-step sampling happens in the DataLoader's iteration
                # This returns a dummy; actual sampling is in the collator
                return idx

        dataset = ImportanceDataset(
            base_dataset=self.train_dataset,
            sampler=self._importance_sampler,
            model=self.model,
            batch_size=self.args.per_device_train_batch_size,
            collator=self.data_collator,
        )

        return TorchDataLoader(
            dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,  # Importance sampling handles selection
            num_workers=0,  # Importance sampler needs model access
            collate_fn=self._importance_collate_fn,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _importance_collate_fn(self, batch_indices: list[int]) -> dict:
        """Custom collate that runs importance sampling for each batch."""
        batch_size = self.args.per_device_train_batch_size

        selected_batch, importance_weights, global_indices = (
            self._importance_sampler.sample_batch(
                model=self.model,
                batch_size=batch_size,
                current_step=self.state.global_step,
            )
        )

        # Convert to feature dicts
        features = []
        for i, idx in enumerate(global_indices):
            item = selected_batch[i]
            item["importance_weight"] = importance_weights[i].item()
            item["global_index"] = idx
            features.append(item)

        return self.data_collator(features)

    def _get_weighted_dataloader(self) -> DataLoader:
        """Create a DataLoader with weighted sampling (Phase 3)."""
        from torch.utils.data import WeightedRandomSampler
        from torch.utils.data import DataLoader as TorchDataLoader

        # Build sampling weights
        indices = list(range(len(self.train_dataset)))
        weights = [self._data_weights.get(i, 1.0) for i in indices]

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(self.train_dataset),
            replacement=True,
        )

        return TorchDataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override loss computation to apply importance weights (Phase 2)."""
        importance_weights = inputs.pop("importance_weight", None)
        global_indices = inputs.pop("global_index", None)

        outputs = model(**inputs)
        loss = outputs.loss

        # Apply importance weights if present
        if importance_weights is not None:
            # Reshape loss to per-token and apply weights
            # loss is already averaged over tokens by HF
            # Multiply by importance weights for the re-weighting
            weighted_loss = (loss * importance_weights).mean()
            return (weighted_loss, outputs) if return_outputs else weighted_loss

        return (loss, outputs) if return_outputs else loss

    def save_lorA_checkpoint(self, step: int) -> str:
        """Save a LoRA checkpoint for TracInCP computation.

        Args:
            step: Current training step.

        Returns:
            Path to saved checkpoint.
        """
        if self._checkpoint_manager is not None:
            return self._checkpoint_manager.save(self.model, step)
        else:
            logger.warning("No CheckpointManager configured — skipping checkpoint save")
            return ""
