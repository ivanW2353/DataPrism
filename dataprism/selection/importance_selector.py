"""
Phase 2: Online importance sampling selector.

Unlike Phase 1 (offline filtering), this operates ONLINE — it selects
samples dynamically during training rather than pre-processing.
"""

import logging
from typing import Optional

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from dataprism.config.dataclass import Phase2ImportanceConfig
from dataprism.core.base_selector import DataSelector
from dataprism.core.registry import register_selector
from dataprism.core.types import SelectorMode
from dataprism.data.streaming_pool import StreamingDataPool
from dataprism.sampling.importance_sampler import ImportanceSampler
from dataprism.sampling.temperature_scheduler import TemperatureScheduler
from dataprism.sampling.variance_tracker import VarianceTracker

logger = logging.getLogger("dataprism.selection.importance")


@register_selector("importance_sampling")
class ImportanceSelector(DataSelector):
    """Phase 2 selector: Online importance sampling.

    This is an ONLINE selector — it wraps the dataset for dynamic
    selection during each training step. The actual selection logic
    is in ImportanceSampler, integrated via DataPrismTrainer.

    The select() method returns the full dataset with a flag indicating
    that Phase 2 sampling should be active during training.
    """

    def __init__(self, config: Phase2ImportanceConfig):
        """Initialize importance selector.

        Args:
            config: Phase 2 configuration.
        """
        self._config = config
        self._pool: Optional[StreamingDataPool] = None
        self._sampler: Optional[ImportanceSampler] = None

    @property
    def mode(self) -> SelectorMode:
        return SelectorMode.ONLINE

    def name(self) -> str:
        return "importance_sampling"

    def select(
        self,
        dataset: Dataset,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> Dataset:
        """Initialize the importance sampling infrastructure.

        Unlike offline selectors, this doesn't filter the dataset.
        Instead, it sets up the StreamingDataPool and ImportanceSampler
        for dynamic selection during training.

        Args:
            dataset: Full training dataset.
            model: Model reference (may be None at init time).
            tokenizer: Not used.

        Returns:
            The original dataset (unfiltered), with importance sampling
            to be applied during training.
        """
        n_total = len(dataset)
        logger.info("=" * 50)
        logger.info("Phase 2: Online Importance Sampling")
        logger.info("  Candidate multiplier: %dx", self._config.candidate_multiplier)
        logger.info("  Initial tau: %.2f", self._config.initial_tau)
        logger.info("  Tau annealing: %s (→ %.2f over %d steps)",
                     self._config.tau_annealing, self._config.tau_min,
                     self._config.tau_schedule_length)
        logger.info("  Variance threshold: %.2f", self._config.variance_reduction_threshold)
        logger.info("=" * 50)

        # Build the streaming pool for fast candidate access
        self._pool = StreamingDataPool(dataset, seed=self._config.seed if hasattr(self._config, 'seed') else 42)

        # Build temperature scheduler
        tau_scheduler = TemperatureScheduler(
            strategy=self._config.tau_annealing,
            initial_tau=self._config.initial_tau,
            tau_min=self._config.tau_min,
            schedule_length=self._config.tau_schedule_length,
        )

        # Build variance tracker
        variance_tracker = VarianceTracker(
            alpha=self._config.variance_ema_alpha,
            threshold=self._config.variance_reduction_threshold,
        )

        # Build importance sampler
        self._sampler = ImportanceSampler(
            pool=self._pool,
            temperature_scheduler=tau_scheduler,
            variance_tracker=variance_tracker,
            candidate_multiplier=self._config.candidate_multiplier,
            importance_weight_clip=self._config.importance_weight_clip,
        )

        logger.info("Importance sampler initialized — %d samples in pool", self._pool.size)

        # Return the full dataset (selection happens online)
        return dataset

    @property
    def sampler(self) -> Optional[ImportanceSampler]:
        """Get the ImportanceSampler for integration with the trainer."""
        return self._sampler

    @property
    def pool(self) -> Optional[StreamingDataPool]:
        """Get the StreamingDataPool."""
        return self._pool
