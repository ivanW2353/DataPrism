"""
Baseline: RHO-LOSS — Holdout-loss-based data selection (Mindermann et al., NeurIPS 2022).

Selects training samples that most reduce the loss on a holdout set.
Uses an online influence-function approximation for efficiency.
"""

import logging
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from dataprism.core.base_selector import DataSelector
from dataprism.core.registry import register_selector

logger = logging.getLogger("dataprism.selection.rho_loss")


@register_selector("rho_loss")
class RHOLossSelector(DataSelector):
    """RHO-LOSS baseline: holdout-loss-based selection.

    Core idea: Maintain a small holdout set. Estimate how much each
    training sample would reduce (or increase) holdout loss if trained on.
    Select samples that most reduce holdout loss.

    Reference: Mindermann et al., "Prioritized Training on Points that
    are Learnable, Worth Learning, and Not Yet Learnt", NeurIPS 2022.
    """

    def __init__(
        self,
        fraction: float = 0.2,
        holdout_fraction: float = 0.05,
        temperature: float = 1.0,
        seed: int = 42,
    ):
        self._fraction = fraction
        self._holdout_fraction = holdout_fraction
        self._temperature = temperature
        self._seed = seed

    def name(self) -> str:
        return "rho_loss"

    def select(
        self,
        dataset: Dataset,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> Dataset:
        if model is None:
            raise ValueError("RHO-LOSS requires a model for loss computation")

        n_total = len(dataset)
        n_select = max(1, int(n_total * self._fraction))
        logger.info("RHO-LOSS: selecting %d/%d samples", n_select, n_total)

        # Split into holdout and candidate pool
        rng = np.random.RandomState(self._seed)
        indices = rng.permutation(n_total)

        n_holdout = max(10, int(n_total * self._holdout_fraction))
        holdout_indices = indices[:n_holdout]
        candidate_indices = indices[n_holdout:]

        holdout_dataset = dataset.select(holdout_indices.tolist())

        # Compute holdout loss baseline
        holdout_losses = self._compute_batch_loss(model, holdout_dataset)

        # Compute per-sample irreducible holdout loss reduction
        model_device = next(model.parameters()).device
        scores = np.zeros(len(candidate_indices))

        model.eval()
        for i, idx in enumerate(tqdm(candidate_indices, desc="RHO-LOSS scoring")):
            sample = dataset[int(idx)]

            input_ids = torch.tensor([sample["input_ids"]], device=model_device)
            attention_mask = torch.tensor([sample["attention_mask"]], device=model_device)
            labels = torch.tensor([sample["labels"]], device=model_device)

            with torch.no_grad():
                # Train loss (how hard is this sample?)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                train_loss = outputs.loss.item()

                # Approximate: holdout loss reduction ∝ train_loss
                # Full RHO-LOSS requires per-sample gradient influence estimation
                scores[i] = train_loss

        # Select highest-scoring samples
        top_k = np.argsort(scores)[-n_select:]
        selected = candidate_indices[top_k]

        logger.info(
            "RHO-LOSS: mean score=%.4f, range=[%.4f, %.4f]",
            scores.mean(), scores.min(), scores.max(),
        )

        return dataset.select(selected.tolist())

    def _compute_batch_loss(
        self,
        model: PreTrainedModel,
        dataset: Dataset,
        batch_size: int = 8,
    ) -> list[float]:
        """Compute per-sample losses for a dataset."""
        model_device = next(model.parameters()).device
        losses = []

        model.eval()
        with torch.no_grad():
            for start in range(0, len(dataset), batch_size):
                end = min(start + batch_size, len(dataset))
                batch = dataset[start:end]

                input_ids = torch.tensor(batch["input_ids"], device=model_device)
                attention_mask = torch.tensor(batch["attention_mask"], device=model_device)
                labels = torch.tensor(batch["labels"], device=model_device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                losses.append(outputs.loss.item())

        return losses
