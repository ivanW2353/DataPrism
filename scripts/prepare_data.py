#!/usr/bin/env python3
"""
Download and preprocess datasets for DataPrism experiments.

Usage:
    python scripts/prepare_data.py --dataset alpaca
    python scripts/prepare_data.py --dataset wizardlm --output_dir ./data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import os

from datasets import load_dataset


DATASETS = {
    "alpaca": {
        "name": "tatsu-lab/alpaca",
        "split": "train",
        "description": "Alpaca-52K instruction-following dataset",
    },
    "wizardlm": {
        "name": "WizardLM/WizardLM_evol_instruct_V2_196k",
        "split": "train",
        "description": "WizardLM evolved instruction dataset",
    },
    "openorca": {
        "name": "Open-Orca/OpenOrca",
        "split": "train",
        "description": "OpenOrca instruction dataset",
    },
    "gsm8k": {
        "name": "gsm8k",
        "split": "test",
        "config": "main",
        "description": "GSM8K math reasoning benchmark",
    },
    "mmlu": {
        "name": "cais/mmlu",
        "split": "test",
        "config": "all",
        "description": "MMLU benchmark",
    },
    "humaneval": {
        "name": "openai_humaneval",
        "split": "test",
        "description": "HumanEval code generation benchmark",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for DataPrism")
    parser.add_argument("--dataset", type=str, choices=list(DATASETS.keys()),
                       help="Dataset to download")
    parser.add_argument("--all", action="store_true",
                       help="Download all datasets")
    parser.add_argument("--output_dir", type=str, default="./data",
                       help="Directory to save datasets")
    parser.add_argument("--list", action="store_true",
                       help="List available datasets")
    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for name, info in DATASETS.items():
            print(f"  {name}: {info['description']}")
        return

    to_download = list(DATASETS.keys()) if args.all else [args.dataset]
    if args.dataset is None and not args.all:
        parser.error("Must specify --dataset or --all")

    os.makedirs(args.output_dir, exist_ok=True)

    for name in to_download:
        info = DATASETS[name]
        print(f"\nDownloading {name}: {info['description']}")

        load_kwargs = {"path": info["name"], "split": info["split"]}
        if "config" in info:
            load_kwargs["name"] = info["config"]

        try:
            dataset = load_dataset(**load_kwargs)
            save_path = os.path.join(args.output_dir, name)
            dataset.save_to_disk(save_path)
            print(f"  Saved {len(dataset)} samples to {save_path}")
        except Exception as e:
            print(f"  Error downloading {name}: {e}")


if __name__ == "__main__":
    main()
