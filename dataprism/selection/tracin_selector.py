"""
Phase 1: TracInCP-based offline data quality selector.

Identifies and removes:
1. OUTLIERS: Samples with abnormally high self-influence (potentially mislabeled)
2. REDUNDANT: Samples with nearly identical influence patterns
3. Keeps CLEAN and REPRESENTATIVE samples

Outputs a filtered dataset with influence metadata attached.
"""

import logging
from typing import Optional

import numpy as np
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from dataprism.config.dataclass import Phase1TracInConfig
from dataprism.core.base_selector import DataSelector
from dataprism.core.registry import register_selector
from dataprism.core.types import InfluenceLabel, SelectorMode
from dataprism.influence.tracin_cp import TracInCP
from dataprism.influence.influence_store import InfluenceStore
from dataprism.utils.cluster_utils import cluster_by_similarity

logger = logging.getLogger("dataprism.selection.tracin")


@register_selector("tracin_cp")
class TracInSelector(DataSelector):
    """Phase 1 selector: TracInCP self-influence → denoise + deduplicate.

    This is an OFFLINE selector — it pre-processes the dataset before
    the main training loop begins.

    Pipeline:
    1. Run initial LoRA SFT for 1-3 epochs, saving checkpoints
    2. Compute self-influence for each training sample
    3. Flag top percentile as OUTLIERS
    4. Cluster remaining samples, keep REPRESENTATIVE centroids
    5. Return filtered dataset
    """

    def __init__(
        self,
        config: Phase1TracInConfig,
        tracin: Optional[TracInCP] = None,
        store: Optional[InfluenceStore] = None,
    ):
        """Initialize the TracIn selector.

        Args:
            config: Phase 1 configuration.
            tracin: Pre-initialized TracInCP (optional, set later).
            store: InfluenceStore for persisting scores.
        """
        self._config = config
        self._tracin = tracin
        self._store = store or InfluenceStore(config.influence_store_path)

    def set_tracin(self, tracin: TracInCP) -> None:
        """Set the TracInCP instance after initialization."""
        self._tracin = tracin

    @property
    def mode(self) -> SelectorMode:
        return SelectorMode.OFFLINE

    def name(self) -> str:
        return "tracin_cp"

    def select(
        self,
        dataset: Dataset,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> Dataset:
        """Select high-quality data by filtering outliers and redundancy.

        Args:
            dataset: Full tokenized training dataset.
            model: Required for TracInCP — PeftModel with LoRA adapter.
            tokenizer: Not used directly (data already tokenized).

        Returns:
            Filtered dataset with added columns:
                - influence_score: Self-influence value
                - influence_label: One of {CLEAN, OUTLIER, REDUNDANT, REPRESENTATIVE}
        """
        if self._tracin is None:
            raise RuntimeError(
                "TracInCP not set. Call set_tracin() or ensure Phase 1 "
                "SFT training has completed before calling select()."
            )

        n_total = len(dataset)
        logger.info("=" * 50)
        logger.info("Phase 1: TracInCP Data Quality Screening")
        logger.info("  Input samples: %d", n_total)
        logger.info("  Outlier percentile: %.1f", self._config.outlier_percentile)
        logger.info("  Redundancy threshold: %.2f", self._config.redundancy_similarity_threshold)
        logger.info("=" * 50)

        # Step 1: Compute self-influence
        scores, _ = self._tracin.compute_self_influence(
            dataset,
            normalize_gradients=self._config.normalize_gradients,
        )

        # Step 2: Detect outliers
        outlier_threshold = np.percentile(scores, self._config.outlier_percentile)
        outlier_mask = scores >= outlier_threshold

        n_outliers = outlier_mask.sum()
        logger.info(
            "Outlier detection: %d samples above %.4f (%.1f%%)",
            n_outliers, outlier_threshold, 100 * n_outliers / n_total,
        )

        # Step 3: Assign initial labels
        labels = np.array([InfluenceLabel.CLEAN.value] * n_total, dtype=object)
        labels[outlier_mask] = InfluenceLabel.OUTLIER.value

        # Step 4: Cluster non-outlier samples for redundancy removal
        if self._config.redundancy_method != "none" and n_total - n_outliers > 10:
            clean_indices = np.where(~outlier_mask)[0]
            clean_scores = scores[~outlier_mask]

            # Use self-influence scores as the feature for clustering
            # For richer clustering, we could use per-checkpoint scores
            clean_features = clean_scores.reshape(-1, 1)

            if clean_features.shape[0] > 1:
                centroid_indices, cluster_ids = cluster_by_similarity(
                    clean_features,
                    threshold=self._config.redundancy_similarity_threshold,
                    method=self._config.redundancy_method,
                    num_clusters=self._config.redundancy_clusters,
                )

                # Map back to original indices
                original_centroids = [clean_indices[c] for c in centroid_indices]

                # Mark non-centroid clean samples as redundant
                centroid_set = set(original_centroids)
                for i in clean_indices:
                    if i not in centroid_set:
                        labels[i] = InfluenceLabel.REDUNDANT.value

                # Mark centroids as representative
                for i in original_centroids:
                    labels[i] = InfluenceLabel.REPRESENTATIVE.value

                n_redundant = (labels == InfluenceLabel.REDUNDANT.value).sum()
                n_representative = (labels == InfluenceLabel.REPRESENTATIVE.value).sum()
                logger.info(
                    "Redundancy: %d representatives, %d redundant removed",
                    n_representative, n_redundant,
                )
        else:
            # No clustering — all clean samples are kept
            labels[~outlier_mask] = InfluenceLabel.REPRESENTATIVE.value

        # Step 5: Filter dataset
        keep_mask = np.isin(
            labels,
            [InfluenceLabel.CLEAN.value, InfluenceLabel.REPRESENTATIVE.value],
        )
        keep_indices = np.where(keep_mask)[0]

        # Enforce minimum samples
        if len(keep_indices) < self._config.min_samples_after_filter:
            logger.warning(
                "Too few samples after filtering (%d < %d). Keeping all non-outlier samples.",
                len(keep_indices), self._config.min_samples_after_filter,
            )
            keep_mask = ~outlier_mask
            keep_indices = np.where(keep_mask)[0]

        # Enforce maximum samples (keep highest-influence samples)
        if (self._config.max_samples_after_redundancy is not None
                and len(keep_indices) > self._config.max_samples_after_redundancy):
            # Keep samples with highest absolute influence
            top_k = np.argsort(np.abs(scores[keep_indices]))[-self._config.max_samples_after_redundancy:]
            keep_indices = keep_indices[top_k]
            logger.info("Truncated to %d samples (max limit)", len(keep_indices))

        # Step 6: Build filtered dataset
        filtered_dataset = dataset.select(keep_indices.tolist())

        # Attach metadata
        filtered_dataset = filtered_dataset.add_column(
            "influence_score", scores[keep_indices].tolist(),
        )
        filtered_dataset = filtered_dataset.add_column(
            "influence_label", labels[keep_indices].tolist(),
        )
        filtered_dataset = filtered_dataset.add_column(
            "original_index", keep_indices.tolist(),
        )

        logger.info(
            "Phase 1 complete: %d → %d samples (%.1f%% retained)",
            n_total, len(filtered_dataset), 100 * len(filtered_dataset) / n_total,
        )
        logger.info(
            "  Clean/Rep: %d | Outlier: %d | Redundant: %d",
            len(keep_indices), n_outliers,
            (labels == InfluenceLabel.REDUNDANT.value).sum(),
        )

        # Save influence scores
        self._store.save_self_influence(
            scores=scores,
            sample_indices=list(range(n_total)),
            metadata={
                "outlier_threshold": float(outlier_threshold),
                "n_outliers": int(n_outliers),
                "n_kept": len(keep_indices),
            },
        )

        return filtered_dataset
