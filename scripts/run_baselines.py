#!/usr/bin/env python3
"""
Run all baseline methods for comparison.

Usage:
    python scripts/run_baselines.py --config default baseline/uniform
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataprism.config.loader import load_config, create_arg_parser, parse_cli_overrides
from dataprism.core.registry import get_selector
from dataprism.models import load_model_and_tokenizer
from dataprism.data import load_and_prepare_dataset
from dataprism.utils import seed_everything, setup_logging


BASELINES = [
    ("uniform", 0.2, "Uniform Random (20%)"),
    ("uniform", 0.5, "Uniform Random (50%)"),
    ("full", 1.0, "Full Data (100%)"),
    ("less", 0.2, "LESS (20%)"),
    ("rho_loss", 0.2, "RHO-LOSS (20%)"),
    ("dsir", 0.2, "DSIR (20%)"),
]


def run_baseline(name, fraction, description, model, tokenizer, dataset, config):
    """Run a single baseline method."""
    print(f"\n{'='*60}")
    print(f"Baseline: {description}")
    print(f"{'='*60}")

    if name == "full":
        return dataset  # Full data — no selection

    selector_cls = get_selector(name)
    selector = selector_cls(fraction=fraction, seed=config.seed)
    selected = selector.select(dataset, model=model, tokenizer=tokenizer)

    print(f"  Selected: {len(selected)}/{len(dataset)} ({len(selected)/len(dataset)*100:.1f}%)")
    return selected


def main():
    parser = create_arg_parser()
    parser.add_argument(
        "--baseline", type=str, default=None,
        help="Run a specific baseline (e.g., 'less'). If None, runs all.",
    )
    args = parser.parse_args()

    overrides = parse_cli_overrides(args.override)
    if args.device:
        overrides["device"] = args.device

    config = load_config(config_paths=args.config, overrides=overrides)
    setup_logging(log_dir=f"{config.output_dir}/logs", experiment_name="baselines")
    seed_everything(config.seed)

    # Load model and data
    print("Loading model and data...")
    model, tokenizer = load_model_and_tokenizer(config.model, device=config.device)
    dataset = load_and_prepare_dataset(config.data, tokenizer)
    print(f"Dataset: {len(dataset)} samples")

    # Run baselines
    results = {}
    for name, fraction, description in BASELINES:
        if args.baseline and name != args.baseline:
            continue

        selected = run_baseline(name, fraction, description, model, tokenizer, dataset, config)
        results[description] = len(selected)

    print("\n" + "=" * 60)
    print("Baseline Results Summary")
    print("=" * 60)
    for desc, n in results.items():
        print(f"  {desc}: {n} samples ({n/len(dataset)*100:.1f}%)")


if __name__ == "__main__":
    main()
