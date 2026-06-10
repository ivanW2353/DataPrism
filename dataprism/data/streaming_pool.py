"""
Memory-efficient data pool with fast indexed random access.

Used by Phase 2 (importance sampling) for candidate pre-sampling each step.
Leverages HuggingFace Dataset's memory-mapped backing for zero-copy access.
"""

import logging
from typing import Optional

import numpy as np
from datasets import Dataset

logger = logging.getLogger("dataprism.data.pool")


class StreamingDataPool:
    """Manages a large dataset pool with efficient indexed access.

    Rather than loading the full dataset into memory, this uses the
    HuggingFace Dataset's memory-mapped Apache Arrow backing for O(1)
    indexed access with minimal memory overhead.
    """

    def __init__(
        self,
        dataset: Dataset,
        seed: int = 42,
    ):
        """Initialize the pool.

        Args:
            dataset: HuggingFace Dataset (handles memory mapping internally).
            seed: Random seed for reproducible sampling.
        """
        self._dataset = dataset
        self._rng = np.random.RandomState(seed)
        self._size = len(dataset)
        logger.info("StreamingDataPool initialized: %d samples", self._size)

    def sample_candidates(
        self,
        batch_size: int,
        multiplier: int = 4,
        exclude_indices: Optional[set[int]] = None,
    ) -> list[int]:
        """Sample candidate indices for importance-based selection.

        Args:
            batch_size: Target batch size.
            multiplier: Sample multiplier * batch_size candidates.
            exclude_indices: Indices to exclude from sampling.

        Returns:
            List of candidate indices.
        """
        n_candidates = batch_size * multiplier
        n_candidates = min(n_candidates, self._size)

        # Build sampling pool excluding specified indices
        if exclude_indices:
            available = [i for i in range(self._size) if i not in exclude_indices]
            if len(available) < n_candidates:
                n_candidates = len(available)
            indices = self._rng.choice(available, n_candidates, replace=False)
        else:
            indices = self._rng.choice(self._size, n_candidates, replace=False)

        return indices.tolist()

    def get_batch(self, indices: list[int]) -> Dataset:
        """Retrieve a batch by indices.

        Args:
            indices: List of sample indices.

        Returns:
            Dataset containing only the specified samples.
        """
        return self._dataset.select(indices)

    def get_item(self, index: int) -> dict:
        """Get a single item by index.

        Args:
            index: Sample index.

        Returns:
            Dict of the sample's fields.
        """
        return self._dataset[int(index)]

    def shuffle(self) -> None:
        """Reshuffle the internal index mapping."""
        # HuggingFace Dataset handles shuffling via its own shuffle() method
        self._dataset = self._dataset.shuffle(seed=int(self._rng.randint(0, 2**31)))

    @property
    def size(self) -> int:
        """Total number of samples in the pool."""
        return self._size

    @property
    def dataset(self) -> Dataset:
        """Access the underlying HuggingFace Dataset."""
        return self._dataset

    def resample_with_weights(
        self,
        weights: dict[int, float],
        batch_size: int,
    ) -> list[int]:
        """Sample indices with weighted probabilities.

        Args:
            weights: Dict mapping global index → sampling weight.
            batch_size: Number of samples to draw.

        Returns:
            List of sampled indices.
        """
        indices = list(weights.keys())
        probs = np.array([weights[i] for i in indices])
        probs = probs / probs.sum()  # Normalize

        sampled = self._rng.choice(
            indices,
            size=min(batch_size, len(indices)),
            p=probs,
            replace=False,
        )
        return sampled.tolist()
