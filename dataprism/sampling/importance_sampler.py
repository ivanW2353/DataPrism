"""
Online importance sampling for LLM fine-tuning (Phase 2).

At each training step:
1. Pre-sample 4× batch_size candidates from the data pool
2. Forward pass → per-sample token-level cross-entropy loss
3. Sample target batch with probability ∝ softmax(loss / τ)
4. Down-weight selected samples by 1/prob for unbiased gradient estimation
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import PreTrainedModel

from dataprism.data.streaming_pool import StreamingDataPool
from dataprism.sampling.temperature_scheduler import TemperatureScheduler
from dataprism.sampling.variance_tracker import VarianceTracker

logger = logging.getLogger("dataprism.sampling")


class ImportanceSampler:
    """Online importance sampler for LLM training.

    Selects samples that would produce the highest loss — focusing the
    model on "difficult" examples each step to accelerate convergence.

    The key insight from Katharopoulos & Fleuret (ICML 2018):
    Sampling proportional to the gradient norm upper bound (which is
    proportional to per-sample loss for cross-entropy) provides an
    unbiased gradient estimate with reduced variance.
    """

    def __init__(
        self,
        pool: StreamingDataPool,
        temperature_scheduler: TemperatureScheduler,
        variance_tracker: VarianceTracker,
        candidate_multiplier: int = 4,
        importance_weight_clip: Optional[float] = 10.0,
    ):
        """Initialize the importance sampler.

        Args:
            pool: StreamingDataPool for fast candidate access.
            temperature_scheduler: Tau annealing schedule.
            variance_tracker: Variance reduction monitor.
            candidate_multiplier: Pre-sample multiplier × batch_size.
            importance_weight_clip: Cap on importance weights (None = no cap).
        """
        self._pool = pool
        self._tau_scheduler = temperature_scheduler
        self._variance_tracker = variance_tracker
        self._candidate_multiplier = candidate_multiplier
        self._importance_weight_clip = importance_weight_clip

        # Internal state
        self._current_step = 0
        self._batch_stats: list[dict] = []

    def sample_batch(
        self,
        model: PreTrainedModel,
        batch_size: int,
        current_step: int,
        tokenizer=None,
    ) -> Tuple[Dataset, torch.Tensor, list[int]]:
        """Sample a batch using importance-weighted selection.

        Args:
            model: The model for computing per-sample losses.
            batch_size: Target batch size.
            current_step: Current training step (for tau schedule).
            tokenizer: Optional tokenizer (not used; dataset is pre-tokenized).

        Returns:
            Tuple of (selected_batch, importance_weights, global_indices).
            - selected_batch: Dataset of B items
            - importance_weights: (B,) tensor for loss re-weighting
            - global_indices: Global pool indices of selected samples
        """
        self._current_step = current_step

        # Check if we should fall back to uniform
        if (self._variance_tracker.variance_reduction is not None
                and self._variance_tracker.should_use_uniform):
            self._variance_tracker.set_uniform_mode(True)
            return self._uniform_sample(batch_size)

        # Step 1: Pre-sample candidates
        n_candidates = batch_size * self._candidate_multiplier
        candidate_indices = self._pool.sample_candidates(batch_size, self._candidate_multiplier)
        candidate_batch = self._pool.get_batch(candidate_indices)

        # Step 2: Compute per-sample losses (forward only, no grad)
        losses = self._compute_losses(model, candidate_batch)

        # Step 3: Get current tau
        tau = self._tau_scheduler.get_tau(current_step)

        # Step 4: Compute sampling probabilities
        sampling_probs = self._compute_probs(losses, tau)

        # Step 5: Sample target batch
        n_select = min(batch_size, len(candidate_indices))
        selected_local = np.random.choice(
            len(candidate_indices),
            size=n_select,
            replace=False,
            p=sampling_probs.numpy(),
        )
        selected_indices = [candidate_indices[i] for i in selected_local]

        # Step 6: Compute importance weights (1 / prob)
        selected_probs = sampling_probs[selected_local]
        importance_weights = 1.0 / (selected_probs * len(candidate_indices))

        if self._importance_weight_clip is not None:
            importance_weights = torch.clamp(
                importance_weights,
                max=self._importance_weight_clip,
            )

        # Step 7: Update variance tracker
        selected_losses = losses[selected_local]
        self._variance_tracker.update_from_losses(
            importance_losses=selected_losses,
            sampling_probs=selected_probs,
        )
        self._variance_tracker.set_uniform_mode(False)

        # Step 8: Retrieve actual data
        selected_batch = self._pool.get_batch(selected_indices)

        # Log stats
        self._batch_stats.append({
            "step": current_step,
            "tau": tau,
            "mean_loss": losses.mean().item(),
            "selected_mean_loss": selected_losses.mean().item(),
            "variance_reduction": self._variance_tracker.variance_reduction,
            "n_candidates": n_candidates,
        })
        if len(self._batch_stats) > 1000:
            self._batch_stats = self._batch_stats[-500:]

        return selected_batch, importance_weights, selected_indices

    def _compute_losses(
        self,
        model: PreTrainedModel,
        batch: Dataset,
    ) -> torch.Tensor:
        """Compute per-sample cross-entropy loss without gradient tracking.

        Args:
            model: The model.
            batch: A batch of tokenized samples.

        Returns:
            (n_samples,) tensor of per-sample average token losses.
        """
        losses = []
        model_device = next(model.parameters()).device

        with torch.no_grad():
            for i in range(len(batch)):
                sample = batch[i]

                input_ids = torch.tensor([sample["input_ids"]], device=model_device)
                attention_mask = torch.tensor([sample["attention_mask"]], device=model_device)
                labels = torch.tensor([sample["labels"]], device=model_device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                # Average token loss for this sample
                loss = outputs.loss.item()
                losses.append(loss)

        return torch.tensor(losses, device=model_device)

    def _compute_probs(
        self,
        losses: torch.Tensor,
        tau: float,
    ) -> torch.Tensor:
        """Compute sampling probabilities via softmax(loss / tau).

        Args:
            losses: (n_candidates,) tensor of per-sample losses.
            tau: Temperature parameter.

        Returns:
            (n_candidates,) tensor of normalized probabilities.
        """
        # Numerical stability: subtract max
        logits = losses / tau
        logits = logits - logits.max()
        probs = F.softmax(logits, dim=0)
        return probs

    def _uniform_sample(self, batch_size: int) -> Tuple[Dataset, torch.Tensor, list[int]]:
        """Fall back to uniform random sampling.

        Returns:
            Same format as sample_batch().
        """
        indices = self._pool.sample_candidates(batch_size, multiplier=1)
        batch = self._pool.get_batch(indices)
        weights = torch.ones(batch_size)  # All weights = 1
        return batch, weights, indices

    def get_batch_stats(self) -> list[dict]:
        """Get recent batch statistics for logging/analysis."""
        return self._batch_stats.copy()

    def reset_stats(self) -> None:
        """Clear batch statistics."""
        self._batch_stats = []

    @property
    def current_step(self) -> int:
        return self._current_step
