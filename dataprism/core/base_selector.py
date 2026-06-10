"""
Abstract base class for all data selection strategies.

Every selector — whether a DataPrism phase or a baseline method —
implements this interface, enabling drop-in substitution in experiments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizer

from dataprism.core.types import SelectorMode


class DataSelector(ABC):
    """Abstract base for all data selection strategies.

    Each selector takes a dataset and optionally a model/tokenizer,
    and returns a (potentially smaller, reweighted) dataset.
    """

    @abstractmethod
    def select(
        self,
        dataset: Dataset,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> Dataset:
        """Select a subset of the input dataset.

        Args:
            dataset: The raw training dataset (HuggingFace Dataset format).
            model: Optional pretrained/LoRA model. Required for gradient-based
                   selectors (TracIn, importance sampling), optional for
                   feature-based selectors (DSIR, LESS).
            tokenizer: Optional tokenizer for the model.

        Returns:
            A Dataset containing the selected subset, with metadata columns
            added for traceability (e.g., 'influence_score', 'label').
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable selector name for logging."""
        ...

    @property
    def mode(self) -> SelectorMode:
        """How this selector operates (offline, online, iterative)."""
        return SelectorMode.OFFLINE

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name()}')"
