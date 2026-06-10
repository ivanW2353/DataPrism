#!/usr/bin/env python3
"""
Main entry point for the DataPrism pipeline.

Usage:
    python scripts/run_pipeline.py --config configs/default.yaml
    python scripts/run_pipeline.py --config default phase1_tracin --override phase1.num_epochs=2
"""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataprism.config.loader import load_config, create_arg_parser, parse_cli_overrides
from dataprism.pipeline import DataPrismPipeline


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    # Parse CLI overrides
    overrides = parse_cli_overrides(args.override)

    # Add device and seed overrides
    if args.device:
        overrides["device"] = args.device
    if args.seed is not None:
        overrides["seed"] = str(args.seed)

    # Load configuration
    print(f"Loading configs: {args.config}")
    config = load_config(config_paths=args.config, overrides=overrides)

    # Run pipeline
    pipeline = DataPrismPipeline(config)
    results = pipeline.run()

    print("\n" + "=" * 60)
    print("DataPrism Pipeline Complete")
    print("=" * 60)
    print(f"Raw data: {results['dataset_sizes'].get('raw', 'N/A')}")
    print(f"After Phase 1: {results['dataset_sizes'].get('after_phase1', 'N/A')}")
    print(f"After Phase 2: {results['dataset_sizes'].get('after_phase2', 'N/A')}")
    print(f"Final: {results['dataset_sizes'].get('final', 'N/A')}")

    return results


if __name__ == "__main__":
    main()
