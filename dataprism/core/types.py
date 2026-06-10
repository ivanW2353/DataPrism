"""Shared type aliases and enums for DataPrism."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from datasets import Dataset


class InfluenceLabel(str, Enum):
    """Label assigned to each training sample by DataPrism."""
    CLEAN = "clean"
    OUTLIER = "outlier"        # High self-influence → possibly mislabeled
    REDUNDANT = "redundant"    # Similar influence vector to another sample
    REPRESENTATIVE = "representative"  # Chosen as cluster centroid
    PROPONENT = "proponent"    # Helps validation performance
    OPPONENT = "opponent"      # Hurts validation performance
    NEUTRAL = "neutral"        # No significant influence


class PhaseStatus(str, Enum):
    """Status of a pipeline phase."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class SelectorMode(str, Enum):
    """How a selector operates."""
    OFFLINE = "offline"    # Pre-process data before training
    ONLINE = "online"      # Select data during training
    ITERATIVE = "iterative"  # Reselect data between epochs


# Type aliases (use Any as fallback when torch not available)
try:
    import torch
    GradientDict = dict[str, torch.Tensor]
    InfluenceVector = torch.Tensor
except ImportError:
    GradientDict = dict  # type: ignore
    InfluenceVector = Any  # type: ignore

SampleIndex = int  # Index into the dataset
WeightsDict = dict[SampleIndex, float]  # Sample index → weight/sampling probability
