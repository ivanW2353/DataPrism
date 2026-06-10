"""
Baseline: DSIR — Data Selection via Importance Resampling (Xie et al., NeurIPS 2023).

Selects training samples by matching n-gram feature distributions between
the raw data pool and a target (high-quality) dataset.
"""

import logging
from collections import Counter
from typing import Optional

import numpy as np
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from dataprism.core.base_selector import DataSelector
from dataprism.core.registry import register_selector

logger = logging.getLogger("dataprism.selection.dsir")


@register_selector("dsir")
class DSIRSelector(DataSelector):
    """DSIR baseline: n-gram distribution matching importance resampling.

    Core idea: Compute n-gram feature distributions for the source
    (raw data) and target (high-quality reference). Importance-weight
    each source sample by P_target(n-grams) / P_source(n-grams).

    Reference: Xie et al., "Data Selection for Language Models via
    Importance Resampling", NeurIPS 2023.
    """

    def __init__(
        self,
        fraction: float = 0.2,
        ngram_size: int = 3,
        num_features: int = 10000,
        importance_smoothing: float = 0.01,
        seed: int = 42,
    ):
        self._fraction = fraction
        self._ngram_size = ngram_size
        self._num_features = num_features
        self._importance_smoothing = importance_smoothing
        self._seed = seed

    def name(self) -> str:
        return "dsir"

    def select(
        self,
        dataset: Dataset,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> Dataset:
        n_total = len(dataset)
        n_select = max(1, int(n_total * self._fraction))
        logger.info("DSIR: selecting %d/%d samples", n_select, n_total)

        # Extract text from dataset
        texts = self._extract_texts(dataset)

        # Build n-gram distributions
        logger.info("Building %d-gram features...", self._ngram_size)
        source_counts = self._build_ngram_counts(texts, self._num_features)

        # Use the dataset itself as "target" for simplicity
        # (in practice, use a curated high-quality dataset)
        target_texts = texts[:max(1, n_total // 10)]  # Top 10% as "target"
        target_counts = self._build_ngram_counts(target_texts, self._num_features)

        # Compute importance weights
        importance_weights = self._compute_importance_weights(
            texts, source_counts, target_counts,
        )

        # Select samples with highest importance weights
        top_k = np.argsort(importance_weights)[-n_select:]
        top_k.sort()

        logger.info(
            "DSIR selection: mean weight=%.4f, range=[%.4f, %.4f]",
            importance_weights.mean(), importance_weights.min(), importance_weights.max(),
        )

        return dataset.select(top_k.tolist())

    def _extract_texts(self, dataset: Dataset) -> list[str]:
        """Extract raw text from tokenized dataset."""
        texts = []
        for i in range(len(dataset)):
            sample = dataset[i]
            # Use prompt or text field
            text = sample.get("prompt", sample.get("text", ""))
            texts.append(text)
        return texts

    def _build_ngram_counts(
        self,
        texts: list[str],
        max_features: int,
    ) -> Counter:
        """Build n-gram frequency counter.

        Args:
            texts: List of text strings.
            max_features: Maximum number of n-grams to keep.

        Returns:
            Counter of n-gram frequencies.
        """
        counts: Counter = Counter()

        for text in tqdm(texts, desc="Building n-grams", leave=False):
            # Simple character n-grams (word n-grams also possible)
            chars = text.lower()
            for i in range(len(chars) - self._ngram_size + 1):
                ngram = chars[i:i + self._ngram_size]
                counts[ngram] += 1

        # Keep only top max_features
        top_ngrams = counts.most_common(max_features)
        return Counter(dict(top_ngrams))

    def _compute_importance_weights(
        self,
        texts: list[str],
        source_counts: Counter,
        target_counts: Counter,
    ) -> np.ndarray:
        """Compute per-sample importance weights.

        weight(sample) = Σ_{ngram in sample} log(P_target(ngram) / P_source(ngram))

        Args:
            texts: List of text strings for each sample.
            source_counts: Source n-gram frequency counter.
            target_counts: Target n-gram frequency counter.

        Returns:
            (n_samples,) array of importance weights.
        """
        # Convert to probabilities with smoothing
        source_total = sum(source_counts.values())
        target_total = sum(target_counts.values())
        all_ngrams = set(source_counts.keys()) | set(target_counts.keys())
        smoothing = self._importance_smoothing * source_total

        # Precompute log-ratios for each ngram
        log_ratios = {}
        for ngram in all_ngrams:
            p_source = (source_counts.get(ngram, 0) + smoothing) / (source_total + smoothing * len(all_ngrams))
            p_target = (target_counts.get(ngram, 0) + smoothing) / (target_total + smoothing * len(all_ngrams))
            log_ratios[ngram] = np.log(max(p_target, 1e-10) / max(p_source, 1e-10))

        # Compute per-sample weight
        weights = np.zeros(len(texts))
        for i, text in enumerate(tqdm(texts, desc="Computing weights", leave=False)):
            chars = text.lower()
            sample_weight = 0.0
            n_ngrams = 0
            for j in range(len(chars) - self._ngram_size + 1):
                ngram = chars[j:j + self._ngram_size]
                if ngram in log_ratios:
                    sample_weight += log_ratios[ngram]
                    n_ngrams += 1
            # Average over n-grams in the sample
            if n_ngrams > 0:
                weights[i] = sample_weight / n_ngrams
            else:
                weights[i] = 0.0

        # Normalize to [0, 1]
        weights = weights - weights.min()
        if weights.max() > 0:
            weights = weights / weights.max()

        return weights
