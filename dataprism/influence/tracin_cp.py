"""
TracInCP (Checkpoint-based) influence computation for LoRA-space gradients.

Implements:
- SelfInfluence(z) = Σ_i η_i · ∇ℓ(z, θ_i) · ∇ℓ(z, θ_i)
- TracInVS: Score(z, V_j) = Σ_i ∇ℓ(z, θ_i) · ∇ℓ(V_j, θ_i)

All operations are restricted to LoRA parameter space (~2M dims).
Only scalar scores are stored — never full gradient vectors.
"""

import logging
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from peft import PeftModel
from tqdm import tqdm

from dataprism.influence.checkpoint_manager import CheckpointManager
from dataprism.influence.gradient_collector import GradientCollector

logger = logging.getLogger("dataprism.influence.tracin")


def _collate_and_pad(batch: dict, pad_token_id: int = 0) -> dict:
    """Pad sequences in a batch to the same length.

    Args:
        batch: Dict with 'input_ids', 'attention_mask', 'labels' as lists.
        pad_token_id: Token id used for padding (default: 0).

    Returns:
        Dict with padded tensors of shape (batch_size, max_len).
    """
    import torch

    input_ids = [torch.tensor(x) for x in batch["input_ids"]]
    attention_mask = [torch.tensor(x) for x in batch["attention_mask"]]
    labels = [torch.tensor(x) for x in batch["labels"]]

    max_len = max(x.size(0) for x in input_ids)

    def pad(tensors, value):
        padded = torch.full((len(tensors), max_len), value, dtype=tensors[0].dtype)
        for i, t in enumerate(tensors):
            padded[i, :t.size(0)] = t
        return padded

    return {
        "input_ids": pad(input_ids, pad_token_id),
        "attention_mask": pad(attention_mask, 0),
        "labels": pad(labels, -100),
    }


class TracInCP:
    """TracInCP influence computation adapted to LoRA parameter space.

    This is the mathematical core of DataPrism — it computes how much
    each training sample "influences" itself (self-influence, Phase 1)
    or a validation set (cross-influence, Phase 3).
    """

    def __init__(
        self,
        model: PeftModel,
        checkpoint_manager: CheckpointManager,
        collector: Optional[GradientCollector] = None,
    ):
        """Initialize TracInCP.

        Args:
            model: PeftModel with LoRA adapter.
            checkpoint_manager: Manages saved LoRA checkpoints.
            collector: GradientCollector (created if None).
        """
        self._model = model
        self._checkpoint_manager = checkpoint_manager
        self._collector = collector or GradientCollector(model)

    def compute_self_influence(
        self,
        dataset: Dataset,
        learning_rates: Optional[list[float]] = None,
        normalize_gradients: bool = True,
        max_checkpoints: int = 6,
        max_samples: Optional[int] = None,
        **kwargs,
    ) -> tuple[np.ndarray, list[int]]:
        """Compute self-influence for each sample in the dataset.

        SelfInfluence(z) = Σ_i η_i · ∇ℓ(z, θ_i) · ∇ℓ(z, θ_i)

        High self-influence → model must "memorize" this sample
        → potentially mislabeled or extremely atypical.

        Args:
            dataset: Tokenized training dataset.
            learning_rates: Learning rates for each checkpoint (η_i).
                           If None, assumes equal weight 1/K.
            normalize_gradients: L2-normalize gradients before dot product.
            max_checkpoints: Max number of checkpoints to use (subsample if more).
            max_samples: Max number of samples to process (None = all).

        Returns:
            Tuple of (scores_array, sample_indices).
            scores_array shape: (len(dataset),)
        """
        all_checkpoints = self._checkpoint_manager.list_checkpoints()

        if not all_checkpoints:
            raise RuntimeError(
                "No checkpoints available. Run initial SFT training first."
            )

        # Subsample checkpoints if too many (first + last + evenly spaced)
        if len(all_checkpoints) > max_checkpoints:
            indices = [0] + [
                int(i * (len(all_checkpoints) - 1) / (max_checkpoints - 1))
                for i in range(1, max_checkpoints)
            ]
            checkpoints = [all_checkpoints[i] for i in indices]
            logger.info("Using %d/%d checkpoints: %s", len(checkpoints), len(all_checkpoints),
                        [str(c) for c in checkpoints])
        else:
            checkpoints = all_checkpoints

        K = len(checkpoints)
        if learning_rates is None:
            learning_rates = [1.0 / K] * K
        else:
            lr_sum = sum(learning_rates)
            learning_rates = [lr / lr_sum for lr in learning_rates]

        # Save only LoRA weights to CPU for later restore
        original_lora = {
            k: v.detach().cpu().clone()
            for k, v in self._model.state_dict().items()
            if "lora_" in k
        }

        if max_samples is not None and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))

        n_samples = len(dataset)
        scores = np.zeros(n_samples, dtype=np.float32)

        logger.info(
            "Computing self-influence for %d samples across %d checkpoints",
            n_samples, K,
        )

        # For each checkpoint, compute and accumulate self-influence
        for ckpt_idx, (step, lr) in enumerate(
            zip(checkpoints, learning_rates)
        ):
            logger.info(
                "Checkpoint %d/%d (step=%d, lr=%.6f)",
                ckpt_idx + 1, K, step, lr,
            )

            # Load checkpoint weights into model
            self._checkpoint_manager.load(step, self._model)

            # Per-sample forward+backward (batched data access for efficiency)
            batch_size = 4
            for batch_start in tqdm(range(0, n_samples, batch_size), desc=f"CKPT {step}", leave=False):
                batch_end = min(batch_start + batch_size, n_samples)
                batch = dataset[batch_start:batch_end]

                # Pad to max length in batch
                collated = _collate_and_pad(batch, pad_token_id=0)
                input_ids = collated["input_ids"].to(self._model.device)
                attention_mask = collated["attention_mask"].to(self._model.device)
                labels = collated["labels"].to(self._model.device)

                per_sample_grads = self._collector.compute_batch_gradients(
                    input_ids, attention_mask, labels, per_sample=True,
                )

                for j, grad in enumerate(per_sample_grads):
                    sample_idx = batch_start + j
                    grad_norm = grad.norm(p=2).item()
                    scores[sample_idx] += lr * (grad_norm ** 2)
                    del grad

                del per_sample_grads

        # Restore original LoRA weights from CPU
        self._model.load_state_dict(
            {k: v.to(self._model.device) for k, v in original_lora.items()},
            strict=False,
        )
        logger.info(
            "Self-influence computed: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
            scores.mean(), scores.std(), scores.min(), scores.max(),
        )

        return scores, list(range(n_samples))

    def compute_validation_influence(
        self,
        training_dataset: Dataset,
        validation_dataset: Dataset,
        learning_rates: Optional[list[float]] = None,
        normalize_gradients: bool = True,
        max_samples: Optional[int] = None,
    ) -> np.ndarray:
        """Compute influence of each training sample on a validation set.

        Score(z, V) = Σ_i η_i · ∇ℓ(z, θ_i) · ∇ℓ(V, θ_i)

        Positive score → training sample helps validation performance
        Negative score → training sample hurts validation performance

        Args:
            training_dataset: Tokenized training dataset.
            validation_dataset: Tokenized validation dataset.
            learning_rates: Learning rates for each checkpoint.
            normalize_gradients: L2-normalize gradient vectors.
            max_samples: Limit training samples (None = all).

        Returns:
            scores array of shape (n_train_samples,).
        """
        checkpoints = self._checkpoint_manager.list_checkpoints()

        if not checkpoints:
            raise RuntimeError("No checkpoints available.")

        K = len(checkpoints)
        if learning_rates is None:
            learning_rates = [1.0 / K] * K
        else:
            lr_sum = sum(learning_rates)
            learning_rates = [lr / lr_sum for lr in learning_rates]

        n_train = len(training_dataset) if max_samples is None else max_samples
        if max_samples is not None:
            training_dataset = training_dataset.select(range(max_samples))

        # Save only LoRA weights to CPU for later restore
        original_lora_vs = {
            k: v.detach().cpu().clone()
            for k, v in self._model.state_dict().items()
            if "lora_" in k
        }

        logger.info(
            "Computing TracInVS: %d train samples × %d val samples × %d checkpoints",
            n_train, len(validation_dataset), K,
        )

        scores = np.zeros(n_train, dtype=np.float32)

        for ckpt_idx, (step, lr) in enumerate(zip(checkpoints, learning_rates)):
            logger.info("Checkpoint %d/%d (step=%d)", ckpt_idx + 1, K, step)

            # Load checkpoint weights into model
            self._checkpoint_manager.load(step, self._model)

            # Compute average validation gradient (reused for all training samples)
            logger.info("  Computing validation set average gradient...")
            val_avg_grad = self._collector.compute_validation_average_gradient(
                validation_dataset,
                batch_size=4,
            )

            if normalize_gradients:
                val_avg_grad = torch.nn.functional.normalize(
                    val_avg_grad.unsqueeze(0), p=2, dim=1
                ).squeeze(0)

            # Batch forward + per-sample backward
            batch_size = 4
            for batch_start in tqdm(range(0, n_train, batch_size), desc=f"CKPT {step}", leave=False):
                batch_end = min(batch_start + batch_size, n_train)
                batch = training_dataset[batch_start:batch_end]

                collated = _collate_and_pad(batch, pad_token_id=0)
                input_ids = collated["input_ids"].to(self._model.device)
                attention_mask = collated["attention_mask"].to(self._model.device)
                labels = collated["labels"].to(self._model.device)

                per_sample_grads = self._collector.compute_batch_gradients(
                    input_ids, attention_mask, labels, per_sample=True,
                )

                for j, train_grad in enumerate(per_sample_grads):
                    sample_idx = batch_start + j
                    influence = self._collector.gradient_dot_product(
                        train_grad, val_avg_grad, normalize=normalize_gradients,
                    )
                    scores[sample_idx] += lr * influence
                    del train_grad

                del per_sample_grads

            torch.cuda.empty_cache()

        # Restore original LoRA weights from CPU
        self._model.load_state_dict(
            {k: v.to(self._model.device) for k, v in original_lora_vs.items()},
            strict=False,
        )
        logger.info(
            "TracInVS computed: mean=%.4f, std=%.4f, pos=%.1f%%, neg=%.1f%%",
            scores.mean(), scores.std(),
            100 * (scores > 0).mean(), 100 * (scores < 0).mean(),
        )

        return scores

    def compute_multi_objective_influence(
        self,
        training_dataset: Dataset,
        validation_sets: dict[str, Dataset],
        lambda_weights: dict[str, float],
        learning_rates: Optional[list[float]] = None,
        normalize_gradients: bool = True,
        max_samples: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        """Compute influence for multiple validation dimensions (Phase 3).

        For each dimension j, computes:
        Score(z, V_j) = Σ_i ∇ℓ(z, θ_i) · ∇ℓ(V_j, θ_i)

        Then aggregates:
        TotalScore(z) = Σ_j λ_j · Score(z, V_j)

        Args:
            training_dataset: Training data.
            validation_sets: Dict of {dim_name: validation_dataset}.
            lambda_weights: Weight per dimension (should sum to ~1).
            learning_rates: Per-checkpoint learning rates.
            normalize_gradients: L2-normalize gradients.
            max_samples: Limit training samples.

        Returns:
            Dict with keys:
                '{dim_name}_score': per-dimension influence scores
                'total_score': weighted aggregate scores
        """
        results = {}
        total_score = None

        for dim_name, val_dataset in validation_sets.items():
            if len(val_dataset) == 0:
                logger.warning("Skipping empty validation dim: %s", dim_name)
                continue

            logger.info("Computing influence for dimension: %s", dim_name)
            dim_scores = self.compute_validation_influence(
                training_dataset=training_dataset,
                validation_dataset=val_dataset,
                learning_rates=learning_rates,
                normalize_gradients=normalize_gradients,
                max_samples=max_samples,
            )
            results[f"{dim_name}_score"] = dim_scores

            weight = lambda_weights.get(dim_name, 0.0)
            if total_score is None:
                total_score = weight * dim_scores
            else:
                total_score += weight * dim_scores

        if total_score is not None:
            results["total_score"] = total_score

        return results
