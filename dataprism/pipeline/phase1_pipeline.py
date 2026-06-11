"""
Phase 1 Pipeline: LoRA-Space TracInCP Offline Data Quality Screening.

Orchestrates the complete Phase 1 workflow:
1. Load model → apply LoRA
2. Run initial SFT with checkpoint saving
3. Compute self-influence scores
4. Filter outliers and redundant samples
5. Output a clean, deduplicated dataset
"""

import logging
from typing import Optional, Tuple

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments

from dataprism.config.dataclass import DataPrismConfig, Phase1TracInConfig
from dataprism.influence.checkpoint_manager import CheckpointManager
from dataprism.influence.gradient_collector import GradientCollector
from dataprism.influence.tracin_cp import TracInCP
from dataprism.influence.influence_store import InfluenceStore
from dataprism.selection.tracin_selector import TracInSelector
from dataprism.training.trainer import DataPrismTrainer

logger = logging.getLogger("dataprism.pipeline.phase1")


class Phase1Pipeline:
    """Orchestrates Phase 1: LoRA-space TracInCP offline screening.

    Usage:
        pipeline = Phase1Pipeline(config)
        clean_dataset = pipeline.run(model, tokenizer, raw_dataset)
    """

    def __init__(self, config: DataPrismConfig):
        """Initialize Phase 1 pipeline.

        Args:
            config: Full DataPrism configuration.
        """
        self._config = config
        self._phase_config = config.phase1
        self._checkpoint_manager: Optional[CheckpointManager] = None
        self._tracin: Optional[TracInCP] = None
        self._selector: Optional[TracInSelector] = None
        self._store: Optional[InfluenceStore] = None

    def run(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Dataset,
    ) -> Tuple[PeftModel, Dataset]:
        """Run the complete Phase 1 pipeline.

        Args:
            model: Pretrained base model (without LoRA).
            tokenizer: Model tokenizer.
            dataset: Full raw tokenized training dataset.

        Returns:
            Tuple of (trained_peft_model, filtered_dataset).
        """
        logger.info("=" * 60)
        logger.info("Phase 1: LoRA-Space TracInCP Offline Screening")
        logger.info("=" * 60)

        # Step 1: Apply LoRA
        from dataprism.models.lora_manager import apply_lora, get_lora_parameter_names
        peft_model = apply_lora(model, self._config.lora)
        logger.info("LoRA applied: %d trainable parameters",
                    sum(p.numel() for p in peft_model.parameters() if p.requires_grad))

        # Step 2: Initialize checkpoint manager (scans existing checkpoints)
        self._checkpoint_manager = CheckpointManager(
            checkpoint_dir=self._phase_config.checkpoint_dir,
            max_checkpoints=self._phase_config.max_checkpoints,
        )

        # Step 3: SFT checkpoint generation (skip if already done)
        existing_ckpts = self._checkpoint_manager.list_checkpoints()
        if existing_ckpts and not self._phase_config.force_retrain:
            logger.info("Found %d existing checkpoints, skipping SFT (use force_retrain=true to redo)",
                       len(existing_ckpts))
            # Load the last checkpoint's weights into the model
            last_ckpt = existing_ckpts[-1]
            self._checkpoint_manager.load(last_ckpt, peft_model)
            logger.info("Loaded checkpoint-%d weights", last_ckpt)
        else:
            if self._phase_config.force_retrain and existing_ckpts:
                logger.info("Force retrain: clearing %d existing checkpoints", len(existing_ckpts))
                self._checkpoint_manager.prune_all()

            sft_n = self._phase_config.sft_num_samples or len(dataset)
            sft_dataset = dataset.select(range(min(sft_n, len(dataset))))

            logger.info("SFT training on %d samples (full dataset: %d)", len(sft_dataset), len(dataset))

            # Resolve mixed-precision: bf16 > fp16 > none
            _bf16, _fp16 = False, False
            _dtype = self._config.model.torch_dtype
            if _dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                _bf16 = True
            elif _dtype in ("float16", "bfloat16") and torch.cuda.is_available():
                _fp16 = True
                logger.info("bfloat16 requested but unsupported by GPU — using float16 instead")

            training_args = TrainingArguments(
                output_dir=f"{self._config.output_dir}/phase1_sft",
                num_train_epochs=self._phase_config.num_epochs,
                per_device_train_batch_size=self._config.training.per_device_train_batch_size,
                gradient_accumulation_steps=self._config.training.gradient_accumulation_steps,
                learning_rate=self._config.training.learning_rate,
                warmup_steps=self._config.training.warmup_steps
                if hasattr(self._config.training, "warmup_steps") and self._config.training.warmup_steps
                else int(self._config.training.warmup_ratio
                         * self._phase_config.num_epochs
                         * len(sft_dataset)
                         // (self._config.training.per_device_train_batch_size
                            * self._config.training.gradient_accumulation_steps)),
                logging_steps=self._config.training.logging_steps,
                save_steps=self._config.training.save_steps,
                fp16=_fp16,
                bf16=_bf16,
                remove_unused_columns=False,
                report_to="none",
                run_name=f"{self._config.experiment_name}_phase1",
                seed=self._config.seed,
            )

            trainer = DataPrismTrainer(
                model=peft_model,
                args=training_args,
                train_dataset=sft_dataset,
                tokenizer=tokenizer,
                checkpoint_manager=self._checkpoint_manager,
            )

            # Register time-guard callback if available (SCOW platform)
            try:
                from dataprism.training.time_guard_callback import TimeGuardCallback
                from dataprism.utils.time_guard import TimeGuard
                guard = TimeGuard.from_environment()
                trainer.add_callback(
                    TimeGuardCallback(guard, checkpoint_manager=self._checkpoint_manager)
                )
                logger.info("Time-guard callback registered.")
            except ImportError:
                pass

            trainer.train()

            # Save checkpoints at the specified interval
            for step in range(
                0,
                training_args.num_train_epochs * len(dataset) // training_args.per_device_train_batch_size,
                self._phase_config.checkpoint_every_n_steps,
            ):
                if step > 0:
                    self._checkpoint_manager.save(peft_model, step)

            logger.info("SFT complete: %d checkpoints saved",
                       len(self._checkpoint_manager.list_checkpoints()))

        # Step 4: Initialize TracInCP
        gradient_collector = GradientCollector(peft_model)
        self._tracin = TracInCP(
            model=peft_model,
            checkpoint_manager=self._checkpoint_manager,
            collector=gradient_collector,
        )

        # Step 5: TracInCP dataset (can be larger than SFT subset)
        tracin_n = self._phase_config.tracin_num_samples or len(dataset)
        tracin_dataset = dataset.select(range(min(tracin_n, len(dataset))))
        logger.info("TracInCP evaluating %d samples", len(tracin_dataset))

        # Step 6: Initialize influence store and selector
        self._store = InfluenceStore(
            path=self._phase_config.influence_store_path,
            format="npz",
        )
        self._selector = TracInSelector(
            config=self._phase_config,
            tracin=self._tracin,
            store=self._store,
        )

        # Step 7: Filter dataset
        filtered_dataset = self._selector.select(tracin_dataset, model=peft_model, tokenizer=tokenizer)

        logger.info("Phase 1 pipeline complete — dataset: %d → %d",
                    len(dataset), len(filtered_dataset))

        return peft_model, filtered_dataset

    @property
    def checkpoint_manager(self) -> Optional[CheckpointManager]:
        return self._checkpoint_manager

    @property
    def tracin(self) -> Optional[TracInCP]:
        return self._tracin

    @property
    def selector(self) -> Optional[TracInSelector]:
        return self._selector
