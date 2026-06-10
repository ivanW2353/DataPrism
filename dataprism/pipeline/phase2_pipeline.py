"""
Phase 2 Pipeline: Online Importance Sampling Training.

Orchestrates importance-sampled training:
1. Initialize streaming pool and importance sampler
2. Train with dynamic sample selection each step
3. Track variance reduction, fall back to uniform when needed
"""

import logging
from typing import Optional

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import PreTrainedTokenizer, TrainingArguments

from dataprism.config.dataclass import DataPrismConfig
from dataprism.data.streaming_pool import StreamingDataPool
from dataprism.sampling.importance_sampler import ImportanceSampler
from dataprism.sampling.temperature_scheduler import TemperatureScheduler
from dataprism.sampling.variance_tracker import VarianceTracker
from dataprism.training.trainer import DataPrismTrainer

logger = logging.getLogger("dataprism.pipeline.phase2")


class Phase2Pipeline:
    """Orchestrates Phase 2: Online importance sampling training.

    Usage:
        pipeline = Phase2Pipeline(config)
        trained_model = pipeline.run(peft_model, tokenizer, dataset)
    """

    def __init__(self, config: DataPrismConfig):
        """Initialize Phase 2 pipeline.

        Args:
            config: Full DataPrism configuration.
        """
        self._config = config
        self._phase_config = config.phase2
        self._pool: Optional[StreamingDataPool] = None
        self._sampler: Optional[ImportanceSampler] = None
        self._variance_tracker: Optional[VarianceTracker] = None

    def run(
        self,
        model: PeftModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Dataset,
    ) -> PeftModel:
        """Run Phase 2: Importance-sampled training.

        Args:
            model: PeftModel (typically from Phase 1).
            tokenizer: Model tokenizer.
            dataset: Training dataset (filtered from Phase 1 or raw).

        Returns:
            Trained PeftModel.
        """
        logger.info("=" * 60)
        logger.info("Phase 2: Online Importance Sampling Training")
        logger.info("=" * 60)
        logger.info("Data pool: %d samples", len(dataset))

        # Step 1: Build streaming pool for fast candidate access
        self._pool = StreamingDataPool(dataset, seed=self._config.seed)

        # Step 2: Build temperature scheduler
        tau_scheduler = TemperatureScheduler(
            strategy=self._phase_config.tau_annealing,
            initial_tau=self._phase_config.initial_tau,
            tau_min=self._phase_config.tau_min,
            schedule_length=self._phase_config.tau_schedule_length,
        )

        # Step 3: Build variance tracker
        self._variance_tracker = VarianceTracker(
            alpha=self._phase_config.variance_ema_alpha,
            threshold=self._phase_config.variance_reduction_threshold,
        )

        # Step 4: Build importance sampler
        self._sampler = ImportanceSampler(
            pool=self._pool,
            temperature_scheduler=tau_scheduler,
            variance_tracker=self._variance_tracker,
            candidate_multiplier=self._phase_config.candidate_multiplier,
            importance_weight_clip=self._phase_config.importance_weight_clip,
        )

        # Step 5: Train with importance sampling
        training_args = TrainingArguments(
            output_dir=f"{self._config.output_dir}/phase2_training",
            num_train_epochs=self._config.training.num_epochs,
            per_device_train_batch_size=self._config.training.per_device_train_batch_size,
            gradient_accumulation_steps=self._config.training.gradient_accumulation_steps,
            learning_rate=self._config.training.learning_rate,
            warmup_ratio=self._config.training.warmup_ratio,
            logging_steps=self._config.training.logging_steps,
            save_steps=self._config.training.save_steps,
            fp16=(self._config.model.torch_dtype == "float16"),
            bf16=(self._config.model.torch_dtype == "bfloat16"),
            remove_unused_columns=False,
            report_to="none",
            run_name=f"{self._config.experiment_name}_phase2",
            seed=self._config.seed,
        )

        trainer = DataPrismTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            importance_sampler=self._sampler,
        )

        logger.info("Starting importance-sampled training...")
        trainer.train()

        # Log final stats
        stats = self._sampler.get_batch_stats()
        if stats:
            import numpy as np
            mean_tau = np.mean([s["tau"] for s in stats])
            mean_loss = np.mean([s["mean_loss"] for s in stats])
            logger.info("Phase 2 complete:")
            logger.info("  Mean tau: %.3f", mean_tau)
            logger.info("  Mean candidate loss: %.3f", mean_loss)
            logger.info("  Variance tracker stats: %s", self._variance_tracker.get_stats())

        return model

    @property
    def sampler(self) -> Optional[ImportanceSampler]:
        """Get the importance sampler."""
        return self._sampler

    @property
    def variance_tracker(self) -> Optional[VarianceTracker]:
        """Get the variance tracker."""
        return self._variance_tracker
