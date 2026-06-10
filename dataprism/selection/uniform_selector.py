"""
Baseline: Uniform random sampling.
Lower bound — no intelligent selection, just random.
"""

import logging
from typing import Optional

import numpy as np
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from dataprism.core.base_selector import DataSelector
from dataprism.core.registry import register_selector

logger = logging.getLogger("dataprism.selection.uniform")


@register_selector("uniform")
class UniformSelector(DataSelector):
    """Random uniform sampling baseline.

    Selects a random fraction of the dataset without any intelligence.
    Serves as the lower bound for all other methods.
    """

    def __init__(self, fraction: float = 0.2, seed: int = 42):
        self._fraction = fraction
        self._seed = seed

    def name(self) -> str:
        return f"uniform_{self._fraction:.0%}"

    def select(
        self,
        dataset: Dataset,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> Dataset:
        n_total = len(dataset)
        n_select = max(1, int(n_total * self._fraction))

        rng = np.random.RandomState(self._seed)
        selected = rng.choice(n_total, n_select, replace=False)
        selected.sort()

        logger.info(
            "Uniform sampling: %d/%d samples (%.1f%%)",
            n_select, n_total, 100 * n_select / n_total,
        )

        return dataset.select(selected.tolist())
