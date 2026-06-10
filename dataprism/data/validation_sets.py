"""
Multi-dimensional validation set builder for Phase 3.

Builds separate validation sets for different capability dimensions:
reasoning, safety, chat quality, and factual accuracy.
"""

import logging
from typing import Optional

from datasets import Dataset
from transformers import PreTrainedTokenizer

from dataprism.config.dataclass import Phase3MultiEvalConfig
from dataprism.data.dataset import load_validation_set

logger = logging.getLogger("dataprism.data.validation")


# Default validation set sources
DEFAULT_VALIDATION_SOURCES = {
    "reasoning": {
        "source": "gsm8k",
        "split": "test",
        "num_samples": 500,
        "description": "Grade-school math word problems (GSM8K) — tests reasoning ability",
    },
    "safety": {
        "source": "PKU-Alignment/PKU-SafeRLHF",
        "split": "test",
        "num_samples": 500,
        "description": "Safety-aligned responses (PKU-SafeRLHF) — tests harmlessness",
    },
    "chat": {
        "source": "lmsys/mt_bench",
        "split": "train",
        "num_samples": 500,
        "description": "Multi-turn chat benchmark (MT-Bench) — tests conversation quality",
    },
    "factual": {
        "source": "hotpot_qa",
        "split": "validation",
        "num_samples": 500,
        "description": "Multi-hop QA (HotpotQA) — tests factual accuracy",
    },
}


class ValidationSetBuilder:
    """Builds and caches multi-dimensional validation sets for Phase 3.

    Each validation dimension (reasoning, safety, chat, factual) gets its
    own tokenized Dataset, enabling TracInVS computation of per-sample
    influence on each capability dimension independently.
    """

    def __init__(
        self,
        config: Phase3MultiEvalConfig,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
    ):
        """Initialize the builder.

        Args:
            config: Phase 3 configuration specifying validation set sources.
            tokenizer: Model tokenizer for preprocessing.
            max_seq_length: Maximum sequence length for tokenization.
        """
        self._config = config
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._validation_sets: dict[str, Dataset] = {}
        self._built = False

    def build(self) -> dict[str, Dataset]:
        """Build all validation sets.

        Returns:
            Dict mapping dimension name → tokenized Dataset.
        """
        if self._built:
            return self._validation_sets

        logger.info("Building multi-dimensional validation sets...")
        sources = self._config.validation_sets

        for dim_name, dim_config in sources.items():
            source = dim_config.get("source", "")
            split = dim_config.get("split", "test")
            num_samples = dim_config.get("num_samples", 500)

            logger.info("  Loading '%s': %s (split=%s, n=%d)",
                        dim_name, source, split, num_samples)

            try:
                dataset = load_validation_set(
                    source=source,
                    split=split,
                    num_samples=num_samples,
                    tokenizer=self._tokenizer,
                    max_length=self._max_seq_length,
                )
                self._validation_sets[dim_name] = dataset
                logger.info("    Loaded %d samples", len(dataset))
            except Exception as e:
                logger.error("    Failed to load '%s': %s", dim_name, e)
                # Create empty dataset as fallback
                self._validation_sets[dim_name] = Dataset.from_list([])

        self._built = True
        logger.info("Built %d validation sets", len(self._validation_sets))
        return self._validation_sets

    def get(self, dim_name: str) -> Optional[Dataset]:
        """Get a specific validation set by dimension name.

        Args:
            dim_name: Dimension name (e.g., 'reasoning', 'safety').

        Returns:
            Tokenized Dataset or None if not found.
        """
        if not self._built:
            self.build()
        return self._validation_sets.get(dim_name)

    def get_all(self) -> dict[str, Dataset]:
        """Get all validation sets.

        Returns:
            Dict of dimension name → Dataset.
        """
        if not self._built:
            self.build()
        return self._validation_sets

    @property
    def dimensions(self) -> list[str]:
        """List of validation dimension names."""
        return list(self._config.validation_sets.keys())

    @property
    def lambda_weights(self) -> dict[str, float]:
        """Objective importance weights."""
        return self._config.lambda_weights
