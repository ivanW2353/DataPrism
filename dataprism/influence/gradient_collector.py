"""
Collects and computes LoRA-space gradients for TracInCP influence estimation.

This is the computational engine for Phase 1 (self-influence) and Phase 3
(validation influence). It captures per-sample LoRA gradients efficiently
by restricting all operations to the low-rank parameter space.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from peft import PeftModel

from dataprism.models.gradient_hooks import (
    LoRAGradientCapture,
    compute_per_sample_lora_gradients,
)

logger = logging.getLogger("dataprism.influence.gradients")


class GradientCollector:
    """Collects and manages LoRA-space gradients for TracInCP.

    This wraps a PeftModel and provides methods to:
    1. Compute per-sample LoRA gradients
    2. Compute batch-averaged LoRA gradients (for validation sets)
    3. Normalize and compare gradient vectors

    All gradient operations are restricted to LoRA parameters only.
    For a model with r=64 LoRA on attention projections, this is ~2M
    parameters vs the full model's 8B — a 4000x reduction.
    """

    def __init__(self, model: PeftModel):
        """Initialize gradient collector.

        Args:
            model: PeftModel with LoRA adapter applied.
        """
        self._model = model
        self._capture = LoRAGradientCapture(model)
        self._total_dim = self._capture.total_lora_dim

        logger.info(
            "GradientCollector: LoRA gradient dim = %d (~%.1fM parameters)",
            self._total_dim, self._total_dim / 1_000_000,
        )

    def compute_sample_gradient(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the LoRA gradient for a single training sample.

        Args:
            input_ids: (1, seq_len) token ids.
            attention_mask: (1, seq_len) attention mask.
            labels: (1, seq_len) labels.

        Returns:
            1D tensor of shape (total_lorA_params,) with flattened gradient.
        """
        self._model.zero_grad()

        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()

        grads = self._capture.get_flattened_gradients()
        return grads.clone()

    def compute_batch_gradients(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        per_sample: bool = False,
    ) -> torch.Tensor:
        """Compute LoRA gradients for a batch.

        Args:
            input_ids: (batch_size, seq_len) token ids.
            attention_mask: (batch_size, seq_len) attention mask.
            labels: (batch_size, seq_len) labels.
            per_sample: If True, return per-sample gradients (list of tensors).
                       If False, return the batch-average gradient.

        Returns:
            If per_sample: list of 1D tensors (one per sample).
            If not per_sample: 1D tensor of averaged gradient.
        """
        if per_sample:
            return compute_per_sample_lora_gradients(
                self._model, input_ids, attention_mask, labels, self._capture,
            )

        # Batch-average gradient
        self._model.zero_grad()
        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()

        grads = self._capture.get_flattened_gradients()
        return grads.clone()

    def compute_validation_average_gradient(
        self,
        validation_dataset,
        num_samples: Optional[int] = None,
        batch_size: int = 4,
    ) -> torch.Tensor:
        """Compute the average LoRA gradient over a validation dataset.

        Used in Phase 3 for TracInVS: Score(z, V_j) uses the average
        gradient of validation set V_j.

        Args:
            validation_dataset: HuggingFace Dataset (tokenized).
            num_samples: Limit to this many samples (None = all).
            batch_size: Batch size for gradient computation.

        Returns:
            1D tensor of average LoRA gradient over the validation set.
        """
        if num_samples is not None and num_samples < len(validation_dataset):
            validation_dataset = validation_dataset.select(range(num_samples))

        total_grad = None
        n_processed = 0

        for start_idx in range(0, len(validation_dataset), batch_size):
            end_idx = min(start_idx + batch_size, len(validation_dataset))
            batch = validation_dataset[start_idx:end_idx]

            input_ids = torch.tensor(batch["input_ids"], device=self._model.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=self._model.device)
            labels = torch.tensor(batch["labels"], device=self._model.device)

            batch_grad = self.compute_batch_gradients(
                input_ids, attention_mask, labels, per_sample=False,
            )

            if total_grad is None:
                total_grad = batch_grad * (end_idx - start_idx)
            else:
                total_grad += batch_grad * (end_idx - start_idx)

            n_processed += (end_idx - start_idx)

        if total_grad is None:
            raise RuntimeError("No validation samples processed")

        return total_grad / n_processed

    @staticmethod
    def gradient_dot_product(
        grad_a: torch.Tensor,
        grad_b: torch.Tensor,
        normalize: bool = True,
    ) -> float:
        """Compute the dot product (similarity) between two gradient vectors.

        Args:
            grad_a: 1D gradient tensor.
            grad_b: 1D gradient tensor.
            normalize: If True, L2-normalize both vectors before dot product.

        Returns:
            Scalar similarity score.
        """
        if normalize:
            grad_a = torch.nn.functional.normalize(grad_a.unsqueeze(0), p=2, dim=1).squeeze(0)
            grad_b = torch.nn.functional.normalize(grad_b.unsqueeze(0), p=2, dim=1).squeeze(0)

        return (grad_a * grad_b).sum().item()

    @staticmethod
    def gradient_cosine_similarity(
        grad_a: torch.Tensor,
        grad_b: torch.Tensor,
    ) -> float:
        """Compute cosine similarity between two gradient vectors.

        Args:
            grad_a: 1D gradient tensor.
            grad_b: 1D gradient tensor.

        Returns:
            Cosine similarity in [-1, 1].
        """
        return torch.nn.functional.cosine_similarity(
            grad_a.unsqueeze(0), grad_b.unsqueeze(0),
        ).item()

    @staticmethod
    def gradient_l2_norm(grad: torch.Tensor) -> float:
        """Compute the L2 norm of a gradient vector.

        Args:
            grad: 1D gradient tensor.

        Returns:
            Scalar L2 norm.
        """
        return grad.norm(p=2).item()

    @property
    def total_dim(self) -> int:
        """Total dimension of concatenated LoRA gradient vectors."""
        return self._total_dim
