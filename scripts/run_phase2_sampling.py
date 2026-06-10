#!/usr/bin/env python3
"""
Run Phase 2 only: Online importance sampling training.

Usage:
    python scripts/run_phase2_sampling.py --config default phase2_importance
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataprism.config.loader import load_config, create_arg_parser, parse_cli_overrides
from dataprism.pipeline.phase2_pipeline import Phase2Pipeline
from dataprism.models import load_model_and_tokenizer, apply_lora
from dataprism.data import load_and_prepare_dataset
from dataprism.utils import seed_everything, setup_logging


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    overrides = parse_cli_overrides(args.override)
    if args.device:
        overrides["device"] = args.device
    if args.seed is not None:
        overrides["seed"] = str(args.seed)

    overrides["phase2.enabled"] = "true"
    overrides["phase1.enabled"] = "false"
    overrides["phase3.enabled"] = "false"

    config = load_config(config_paths=args.config, overrides=overrides)
    setup_logging(log_dir=f"{config.output_dir}/logs", experiment_name="phase2")
    seed_everything(config.seed)

    print("=" * 60)
    print("DataPrism — Phase 2: Online Importance Sampling")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer(config.model, device=config.device)
    peft_model = apply_lora(model, config.lora)
    dataset = load_and_prepare_dataset(config.data, tokenizer)

    print(f"Model: {config.model.name}")
    print(f"Data: {config.data.name} ({len(dataset)} samples)")

    phase2 = Phase2Pipeline(config)
    model = phase2.run(peft_model, tokenizer, dataset)

    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()
