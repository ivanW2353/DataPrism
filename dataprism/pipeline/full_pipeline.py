"""
Full DataPrism Pipeline — integrates all three phases.

This is the main entry point for running the complete DataPrism framework.
Phases can be individually enabled/disabled via configuration.
"""

import logging
import os
from typing import Optional, Tuple

from datasets import Dataset
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer

from dataprism.config.dataclass import DataPrismConfig
from dataprism.utils.seed import seed_everything
from dataprism.utils.logging_utils import setup_logging, log_config_summary
from dataprism.pipeline.phase1_pipeline import Phase1Pipeline
from dataprism.pipeline.phase2_pipeline import Phase2Pipeline
from dataprism.pipeline.phase3_pipeline import Phase3Pipeline

logger = logging.getLogger("dataprism.pipeline")


class DataPrismPipeline:
    """Complete DataPrism pipeline orchestrating all three phases.

    Phase 0: Load model and data
    Phase 1: Offline TracInCP quality screening
    Phase 2: Online importance sampling training
    Phase 3: Multi-objective iterative reselection
    """

    def __init__(self, config: DataPrismConfig):
        """Initialize the full pipeline.

        Args:
            config: Validated DataPrismConfig.
        """
        self._config = config

        # Setup logging
        log_dir = os.path.join(config.output_dir, "logs")
        setup_logging(log_dir=log_dir, experiment_name=config.experiment_name)

        # Set seeds
        seed_everything(config.seed)

        log_config_summary(config)

    def run(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        dataset: Optional[Dataset] = None,
    ) -> dict:
        """Run the full DataPrism pipeline.

        Args:
            model: Pretrained base model (loaded if None).
            tokenizer: Model tokenizer (loaded if None).
            dataset: Training dataset (loaded if None).

        Returns:
            Dict with results including:
                - final_model: Trained PeftModel
                - dataset_sizes: Dict tracking dataset size through phases
                - phase_stats: Per-phase statistics
        """
        results = {"dataset_sizes": {}, "phase_stats": {}}

        # ── Phase 0: Load ──────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("DataPrism Pipeline Starting")
        logger.info("=" * 60)

        if model is None or tokenizer is None:
            from dataprism.models.model_registry import load_model_and_tokenizer
            logger.info("Loading model: %s", self._config.model.name)
            model, tokenizer = load_model_and_tokenizer(
                self._config.model, device=self._config.device,
            )

        if dataset is None:
            from dataprism.data.dataset import load_and_prepare_dataset
            dataset = load_and_prepare_dataset(self._config.data, tokenizer)

        results["dataset_sizes"]["raw"] = len(dataset)
        logger.info("Phase 0 complete: %d samples loaded", len(dataset))

        # ── Phase 1: Offline TracInCP Screening ────────────────────
        if self._config.phase1.enabled:
            phase1 = Phase1Pipeline(self._config)
            model, dataset = phase1.run(model, tokenizer, dataset)

            results["dataset_sizes"]["after_phase1"] = len(dataset)
            results["phase_stats"]["phase1"] = {
                "checkpoints_saved": phase1.checkpoint_manager.num_checkpoints
                if phase1.checkpoint_manager else 0,
                "samples_retained": len(dataset),
            }

            # Pass checkpoint manager to subsequent phases
            checkpoint_manager = phase1.checkpoint_manager
        else:
            logger.info("Phase 1 skipped (disabled in config)")
            checkpoint_manager = None
            results["dataset_sizes"]["after_phase1"] = len(dataset)

        # ── Phase 2: Online Importance Sampling ────────────────────
        if self._config.phase2.enabled:
            phase2 = Phase2Pipeline(self._config)
            model = phase2.run(model, tokenizer, dataset)

            results["phase_stats"]["phase2"] = {
                "variance_tracker": phase2.variance_tracker.get_stats()
                if phase2.variance_tracker else {},
                "batch_stats_count": len(phase2.sampler.get_batch_stats())
                if phase2.sampler else 0,
            }
        else:
            logger.info("Phase 2 skipped (disabled in config)")

        results["dataset_sizes"]["after_phase2"] = len(dataset)

        # ── Phase 3: Multi-Objective Iterative Reselection ─────────
        if self._config.phase3.enabled:
            phase3 = Phase3Pipeline(self._config, checkpoint_manager)
            model = phase3.run(model, tokenizer, dataset)

            results["phase_stats"]["phase3"] = (
                phase3.selector.get_influence_report()
                if phase3.selector else {}
            )
        else:
            logger.info("Phase 3 skipped (disabled in config)")

        # ── Final ──────────────────────────────────────────────────
        results["final_model"] = model
        results["dataset_sizes"]["final"] = len(dataset)

        logger.info("=" * 60)
        logger.info("DataPrism Pipeline Complete")
        logger.info("  Raw data: %d", results["dataset_sizes"]["raw"])
        logger.info("  After Phase 1: %d", results["dataset_sizes"].get("after_phase1", "N/A"))
        logger.info("  Final: %d", results["dataset_sizes"]["final"])
        logger.info("=" * 60)

        return results
