"""
Phase 3 Pipeline: Multi-Objective Validation-Driven Iterative Reselection.

Orchestrates the Phase 3 workflow:
1. Build multi-dimensional validation sets
2. Compute TracInVS influence scores
3. Classify samples (proponent/opponent/neutral)
4. Iteratively re-weight samples between epochs
5. Train with evolving data weights
"""

import logging
from typing import Optional

import numpy as np
from datasets import Dataset
from peft import PeftModel
from transformers import PreTrainedTokenizer, TrainingArguments

from dataprism.config.dataclass import DataPrismConfig
from dataprism.data.validation_sets import ValidationSetBuilder
from dataprism.influence.checkpoint_manager import CheckpointManager
from dataprism.influence.gradient_collector import GradientCollector
from dataprism.influence.tracin_cp import TracInCP
from dataprism.influence.influence_store import InfluenceStore
from dataprism.selection.multi_obj_selector import MultiObjectiveSelector
from dataprism.training.trainer import DataPrismTrainer
from dataprism.training.callbacks import InfluenceEpochCallback

logger = logging.getLogger("dataprism.pipeline.phase3")


class Phase3Pipeline:
    """Orchestrates Phase 3: Multi-objective iterative data reselection.

    Usage:
        pipeline = Phase3Pipeline(config, checkpoint_manager)
        final_model = pipeline.run(model, tokenizer, dataset)
    """

    def __init__(
        self,
        config: DataPrismConfig,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        """Initialize Phase 3 pipeline.

        Args:
            config: Full DataPrism configuration.
            checkpoint_manager: CheckpointManager from Phase 1 (reused).
        """
        self._config = config
        self._phase_config = config.phase3
        self._checkpoint_manager = checkpoint_manager
        self._tracin: Optional[TracInCP] = None
        self._selector: Optional[MultiObjectiveSelector] = None
        self._validation_sets: dict[str, Dataset] = {}

    def run(
        self,
        model: PeftModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Dataset,
    ) -> PeftModel:
        """Run Phase 3: Iterative validation-driven data reselection.

        Args:
            model: PeftModel from Phase 2 (or Phase 1).
            tokenizer: Model tokenizer.
            dataset: Training dataset.

        Returns:
            Trained PeftModel after iterative reselection.
        """
        logger.info("=" * 60)
        logger.info("Phase 3: Multi-Objective Validation-Driven Reselection")
        logger.info("=" * 60)

        # Step 1: Build validation sets
        builder = ValidationSetBuilder(
            config=self._phase_config,
            tokenizer=tokenizer,
        )
        self._validation_sets = builder.build()

        if not self._validation_sets:
            logger.warning("No validation sets available — skipping Phase 3")
            return model

        logger.info("Built %d validation sets: %s",
                    len(self._validation_sets),
                    list(self._validation_sets.keys()))

        # Step 2: Initialize or reuse checkpoint manager
        if self._checkpoint_manager is None:
            self._checkpoint_manager = CheckpointManager(
                checkpoint_dir=self._phase_config.tracin_vs_checkpoint_dir,
                max_checkpoints=self._config.phase1.max_checkpoints,
            )

        # Step 3: Initialize TracInCP
        gradient_collector = GradientCollector(model)
        self._tracin = TracInCP(
            model=model,
            checkpoint_manager=self._checkpoint_manager,
            collector=gradient_collector,
        )

        # Step 4: Initialize multi-objective selector
        self._selector = MultiObjectiveSelector(
            config=self._phase_config,
            tracin=self._tracin,
            store=InfluenceStore("outputs/influences/phase3_multi"),
        )

        # Step 5: Compute initial influence scores and classify
        enriched_dataset = self._selector.select(dataset, model=model, tokenizer=tokenizer)

        # Step 6: Train with iterative weight updates
        for round_num in range(1, self._phase_config.max_rounds + 1):
            logger.info("Phase 3 - Round %d/%d", round_num, self._phase_config.max_rounds)

            training_args = TrainingArguments(
                output_dir=f"{self._config.output_dir}/phase3_round{round_num}",
                num_train_epochs=1,  # One epoch per round
                per_device_train_batch_size=self._config.training.per_device_train_batch_size,
                gradient_accumulation_steps=self._config.training.gradient_accumulation_steps,
                learning_rate=self._config.training.learning_rate * (0.5 ** (round_num - 1)),
                warmup_ratio=0.0,  # No warmup for subsequent rounds
                logging_steps=self._config.training.logging_steps,
                save_steps=self._config.training.save_steps,
                fp16=(self._config.model.torch_dtype == "float16"),
                bf16=(self._config.model.torch_dtype == "bfloat16"),
                remove_unused_columns=False,
                report_to="none",
                run_name=f"{self._config.experiment_name}_phase3_r{round_num}",
                seed=self._config.seed + round_num,
            )

            trainer = DataPrismTrainer(
                model=model,
                args=training_args,
                train_dataset=enriched_dataset,
                tokenizer=tokenizer,
                data_weights=self._selector.sample_weights,
                checkpoint_manager=self._checkpoint_manager,
            )

            # Save initial checkpoints for the first round
            if round_num == 1:
                for step in range(
                    0,
                    len(dataset) // training_args.per_device_train_batch_size,
                    self._config.phase1.checkpoint_every_n_steps,
                ):
                    if step > 0:
                        self._checkpoint_manager.save(model, step)

            logger.info("Training round %d with weighted sampling...", round_num)
            trainer.train()

            # Update weights for next round (unless it's the last)
            if round_num < self._phase_config.max_rounds:
                self._selector.update_weights(model, round_num)

                # Log the influence report
                report = self._selector.get_influence_report()
                logger.info("Influence report: %s", report)

        # Final report
        report = self._selector.get_influence_report()
        logger.info("Phase 3 complete. Final influence report:")
        logger.info("  Total samples: %d", report["n_samples"])
        logger.info("  Proponents: %d", report["n_proponent"])
        logger.info("  Opponents: %d", report["n_opponent"])
        logger.info("  Neutral: %d", report["n_neutral"])

        return model

    @property
    def selector(self) -> Optional[MultiObjectiveSelector]:
        """Get the multi-objective selector."""
        return self._selector

    @property
    def tracin(self) -> Optional[TracInCP]:
        """Get the TracInCP instance."""
        return self._tracin
