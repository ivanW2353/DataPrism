#!/usr/bin/env python3
"""
Run Phase 3 only: Multi-objective validation-driven iterative reselection.

Usage:
    python scripts/run_phase3_reselect.py --config default phase3_multieval
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataprism.config.loader import load_config, create_arg_parser, parse_cli_overrides
from dataprism.pipeline.phase3_pipeline import Phase3Pipeline
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

    overrides["phase3.enabled"] = "true"
    overrides["phase1.enabled"] = "false"
    overrides["phase2.enabled"] = "false"

    config = load_config(config_paths=args.config, overrides=overrides)
    setup_logging(log_dir=f"{config.output_dir}/logs", experiment_name="phase3")
    seed_everything(config.seed)

    print("=" * 60)
    print("DataPrism — Phase 3: Multi-Objective Iterative Reselection")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer(config.model, device=config.device)
    peft_model = apply_lora(model, config.lora)
    dataset = load_and_prepare_dataset(config.data, tokenizer)

    print(f"Model: {config.model.name}")
    print(f"Data: {config.data.name} ({len(dataset)} samples)")

    phase3 = Phase3Pipeline(config)
    model = phase3.run(peft_model, tokenizer, dataset)

    if phase3.selector:
        report = phase3.selector.get_influence_report()
        print("\nInfluence Report:")
        print(f"  Proponents: {report['n_proponent']}")
        print(f"  Opponents: {report['n_opponent']}")
        print(f"  Neutral: {report['n_neutral']}")


if __name__ == "__main__":
    main()
