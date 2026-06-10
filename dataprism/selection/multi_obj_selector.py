"""
Phase 3: Multi-objective validation-driven iterative data reselection.

For each training sample, computes its influence on multiple validation
dimensions (reasoning, safety, chat, factual). Aggregates scores with
configurable weights, then updates per-sample sampling weights for the
next training epoch.
"""

import logging
from typing import Optional

import numpy as np
from datasets import Dataset
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer

from dataprism.config.dataclass import Phase3MultiEvalConfig
from dataprism.core.base_selector import DataSelector
from dataprism.core.registry import register_selector
from dataprism.core.types import InfluenceLabel, SelectorMode
from dataprism.influence.tracin_cp import TracInCP
from dataprism.influence.influence_store import InfluenceStore

logger = logging.getLogger("dataprism.selection.multi_obj")


@register_selector("multi_obj_tracin")
class MultiObjectiveSelector(DataSelector):
    """Phase 3 selector: Multi-objective TracInVS iterative reselection.

    This is an ITERATIVE selector — it updates per-sample weights
    after each training epoch based on validation set influence.

    Pipeline:
    1. Compute TracInVS: influence of each training sample on each validation set
    2. Aggregate: TotalScore(z) = Σ λ_j · Score(z, V_j)
    3. Classify: proponents (positive), opponents (negative), neutral (near-zero)
    4. Update per-sample weights for next epoch
    """

    def __init__(
        self,
        config: Phase3MultiEvalConfig,
        tracin: Optional[TracInCP] = None,
        store: Optional[InfluenceStore] = None,
    ):
        """Initialize multi-objective selector.

        Args:
            config: Phase 3 configuration.
            tracin: TracInCP for computing influence (set later if None).
            store: InfluenceStore for persisting scores.
        """
        self._config = config
        self._tracin = tracin
        self._store = store or InfluenceStore("outputs/influences/phase3_multi")

        # Per-sample weights (updated each round)
        self._sample_weights: dict[int, float] = {}
        # Track influence labels
        self._sample_labels: dict[int, str] = {}
        # Influence scores per dimension
        self._dimension_scores: dict[str, np.ndarray] = {}
        self._total_scores: Optional[np.ndarray] = None
        # Round counter
        self._current_round: int = 0

    def set_tracin(self, tracin: TracInCP) -> None:
        """Set the TracInCP instance."""
        self._tracin = tracin

    @property
    def mode(self) -> SelectorMode:
        return SelectorMode.ITERATIVE

    def name(self) -> str:
        return "multi_obj_tracin"

    def select(
        self,
        dataset: Dataset,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> Dataset:
        """Run multi-objective influence analysis and label samples.

        This is called once to initialize weights. Subsequent updates
        happen via update_weights() between epochs.

        Args:
            dataset: Training dataset.
            model: PeftModel (required for gradient computation).
            tokenizer: Not used.

        Returns:
            Dataset with influence scores and labels added.
        """
        if self._tracin is None:
            raise RuntimeError(
                "TracInCP not set. Call set_tracin() before select()."
            )

        n_total = len(dataset)
        logger.info("=" * 50)
        logger.info("Phase 3: Multi-Objective Validation-Driven Reselection")
        logger.info("  Dimensions: %s", list(self._config.validation_sets.keys()))
        logger.info("  Lambda weights: %s", self._config.lambda_weights)
        logger.info("  Selection fraction: %.2f", self._config.selection_fraction)
        logger.info("=" * 50)

        # Step 1: Compute influence scores for each dimension
        # This requires validation sets
        from dataprism.data.validation_sets import ValidationSetBuilder
        builder = ValidationSetBuilder(
            config=self._config,
            tokenizer=tokenizer,
        )
        validation_sets = builder.build()

        if not validation_sets:
            logger.warning("No validation sets built — skipping Phase 3")
            return dataset

        scores_dict = self._tracin.compute_multi_objective_influence(
            training_dataset=dataset,
            validation_sets=validation_sets,
            lambda_weights=self._config.lambda_weights,
            normalize_gradients=True,
        )

        self._dimension_scores = {
            k: v for k, v in scores_dict.items() if k.endswith("_score")
        }
        self._total_scores = scores_dict.get("total_score")

        if self._total_scores is None:
            logger.error("No total scores computed — Phase 3 skipped")
            return dataset

        # Step 2: Classify samples
        labels = self._classify_samples(self._total_scores)

        # Step 3: Initialize per-sample weights
        self._sample_weights = self._compute_initial_weights(
            self._total_scores, labels, n_total,
        )

        # Step 4: Attach metadata to dataset
        result_dataset = dataset.add_column(
            "influence_total_score", self._total_scores.tolist(),
        )
        result_dataset = result_dataset.add_column(
            "influence_label", [labels[i] for i in range(n_total)],
        )

        # Log distribution
        n_proponent = sum(1 for v in labels.values() if v == InfluenceLabel.PROPONENT.value)
        n_opponent = sum(1 for v in labels.values() if v == InfluenceLabel.OPPONENT.value)
        n_neutral = sum(1 for v in labels.values() if v == InfluenceLabel.NEUTRAL.value)

        logger.info("Phase 3 classification:")
        logger.info("  Proponents: %d (%.1f%%)", n_proponent, 100 * n_proponent / n_total)
        logger.info("  Opponents: %d (%.1f%%)", n_opponent, 100 * n_opponent / n_total)
        logger.info("  Neutral: %d (%.1f%%)", n_neutral, 100 * n_neutral / n_total)

        self._current_round = 1

        # Save scores
        self._store.save_validation_influence(
            scores=scores_dict,
            sample_indices=list(range(n_total)),
            metadata={
                "round": 1,
                "lambda_weights": self._config.lambda_weights,
                "dimensions": list(self._config.validation_sets.keys()),
            },
        )

        return result_dataset

    def update_weights(
        self,
        model: PeftModel,
        epoch: int,
        dataset: Optional[Dataset] = None,
    ) -> dict[int, float]:
        """Update per-sample weights after an epoch (Phase 3 iterative update).

        Called between epochs to re-weight samples based on their influence
        on validation performance.

        Args:
            model: Current PeftModel state.
            epoch: Current epoch number.
            dataset: Training dataset (needed for re-computing influence).

        Returns:
            Updated per-sample weights dict.
        """
        if self._current_round >= self._config.max_rounds:
            logger.info("Max rounds (%d) reached — weights frozen", self._config.max_rounds)
            return self._sample_weights

        self._current_round += 1
        logger.info("Phase 3: Round %d/%d — updating sample weights",
                    self._current_round, self._config.max_rounds)

        # Boost proponents, decay opponents
        for idx, label in self._sample_labels.items():
            if label == InfluenceLabel.PROPONENT.value:
                self._sample_weights[idx] *= self._config.proponent_weight_boost
            elif label == InfluenceLabel.OPPONENT.value:
                self._sample_weights[idx] *= self._config.opponent_weight_decay
            # Neutral: keep current weight

        # Normalize weights
        total = sum(self._sample_weights.values())
        if total > 0:
            for idx in self._sample_weights:
                self._sample_weights[idx] /= total

        logger.info("Weights updated for round %d", self._current_round)
        logger.info(
            "  Weight range: [%.4f, %.4f]",
            min(self._sample_weights.values()),
            max(self._sample_weights.values()),
        )

        return self._sample_weights

    def _classify_samples(self, scores: np.ndarray) -> dict[int, str]:
        """Classify samples based on their total influence score.

        Args:
            scores: (n_samples,) array of total influence scores.

        Returns:
            Dict mapping sample index → label string.
        """
        threshold = self._config.neutral_label_threshold
        labels: dict[int, str] = {}

        for i, score in enumerate(scores):
            if score > threshold:
                labels[i] = InfluenceLabel.PROPONENT.value
            elif score < -threshold:
                labels[i] = InfluenceLabel.OPPONENT.value
            else:
                labels[i] = InfluenceLabel.NEUTRAL.value

        self._sample_labels = labels
        return labels

    def _compute_initial_weights(
        self,
        scores: np.ndarray,
        labels: dict[int, str],
        n_total: int,
    ) -> dict[int, float]:
        """Compute initial sampling weights based on influence scores.

        Proponents get higher initial weights; opponents get lower.
        """
        weights = {}
        for i in range(n_total):
            label = labels.get(i, InfluenceLabel.NEUTRAL.value)
            if label == InfluenceLabel.PROPONENT.value:
                # Weight proportional to positive influence
                weights[i] = 1.0 + float(scores[i])
            elif label == InfluenceLabel.OPPONENT.value:
                # Low weight for harmful samples
                weights[i] = max(0.01, 1.0 + float(scores[i]))  # scores are negative
            else:
                weights[i] = 1.0

        # Normalize to mean = 1.0
        mean_w = np.mean(list(weights.values()))
        if mean_w > 0:
            for idx in weights:
                weights[idx] /= mean_w

        return weights

    @property
    def sample_weights(self) -> dict[int, float]:
        """Current per-sample sampling weights."""
        return self._sample_weights

    @property
    def sample_labels(self) -> dict[int, str]:
        """Per-sample influence labels."""
        return self._sample_labels

    @property
    def total_scores(self) -> Optional[np.ndarray]:
        """Total aggregated influence scores."""
        return self._total_scores

    def get_influence_report(self) -> dict:
        """Generate a summary report of influence analysis."""
        return {
            "round": self._current_round,
            "n_samples": len(self._sample_labels),
            "n_proponent": sum(
                1 for v in self._sample_labels.values()
                if v == InfluenceLabel.PROPONENT.value
            ),
            "n_opponent": sum(
                1 for v in self._sample_labels.values()
                if v == InfluenceLabel.OPPONENT.value
            ),
            "n_neutral": sum(
                1 for v in self._sample_labels.values()
                if v == InfluenceLabel.NEUTRAL.value
            ),
            "lambda_weights": self._config.lambda_weights,
            "dimension_scores_available": list(self._dimension_scores.keys()),
            "weight_stats": {
                "min": min(self._sample_weights.values()) if self._sample_weights else None,
                "max": max(self._sample_weights.values()) if self._sample_weights else None,
                "mean": np.mean(list(self._sample_weights.values())) if self._sample_weights else None,
            },
        }
