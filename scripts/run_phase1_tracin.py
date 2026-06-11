#!/usr/bin/env python3
"""
Run Phase 1 only: LoRA-space TracInCP offline data quality screening.

Usage:
    python scripts/run_phase1_tracin.py --config default phase1_tracin
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataprism.config.loader import load_config, create_arg_parser, parse_cli_overrides
from dataprism.pipeline.phase1_pipeline import Phase1Pipeline
from dataprism.models import load_model_and_tokenizer
from dataprism.data import load_and_prepare_dataset
from dataprism.utils import seed_everything, setup_logging


def main():
    parser = create_arg_parser()
    parser.add_argument(
        "--data_config", type=str, default="data/alpaca",
        help="Data configuration to use",
    )
    args = parser.parse_args()

    overrides = parse_cli_overrides(args.override)
    if args.device:
        overrides["device"] = args.device
    if args.seed is not None:
        overrides["seed"] = str(args.seed)

    # Force Phase 1 enabled, others disabled
    overrides["phase1.enabled"] = "true"
    overrides["phase2.enabled"] = "false"
    overrides["phase3.enabled"] = "false"

    config = load_config(config_paths=args.config, overrides=overrides)
    setup_logging(log_dir=f"{config.output_dir}/logs", experiment_name="phase1")
    seed_everything(config.seed)

    print("=" * 60)
    print("DataPrism — Phase 1: TracInCP Data Quality Screening")
    print("=" * 60)

    # Load model and full data (pipeline subsamples internally for SFT vs TracIn)
    config.data.num_samples = None  # Always load full dataset
    model, tokenizer = load_model_and_tokenizer(config.model, device=config.device)
    dataset = load_and_prepare_dataset(config.data, tokenizer)

    print(f"Model: {config.model.name}")
    print(f"Data: {config.data.name} ({len(dataset)} samples)")
    print(f"LoRA: r={config.lora.r}, alpha={config.lora.alpha}")

    # Run Phase 1
    phase1 = Phase1Pipeline(config)
    model, filtered_dataset = phase1.run(model, tokenizer, dataset)

    print("\n" + "=" * 60)
    print("Phase 1 Complete")
    print(f"  Input: {len(dataset)} samples")
    print(f"  Output: {len(filtered_dataset)} samples")
    print(f"  Reduction: {(1 - len(filtered_dataset)/len(dataset))*100:.1f}%")
    print(f"  Checkpoints saved: {phase1.checkpoint_manager.num_checkpoints}")


if __name__ == "__main__":
    main()
