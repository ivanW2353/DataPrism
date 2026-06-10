#!/usr/bin/env python3
"""
Run evaluation benchmarks on a trained model.

Usage:
    python scripts/run_evaluation.py --model_path outputs/phase2_training/final_model --config default
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

from dataprism.config.loader import load_config, parse_cli_overrides
from dataprism.models import load_model_and_tokenizer
from dataprism.evaluation import (
    evaluate_mmlu,
    evaluate_gsm8k,
    evaluate_humaneval,
    evaluate_mt_bench,
    evaluate_alpaca,
)


def main():
    parser = argparse.ArgumentParser(description="DataPrism Evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model or HuggingFace model name")
    parser.add_argument("--config", type=str, nargs="*", default=["default"],
                       help="Config files")
    parser.add_argument("--benchmarks", type=str, nargs="*",
                       default=["mmlu", "gsm8k"],
                       help="Benchmarks to run")
    parser.add_argument("--output", type=str, default="outputs/results/eval_results.json",
                       help="Path to save results")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Limit samples per benchmark")
    args = parser.parse_args()

    config = load_config(config_paths=args.config)
    print(f"Loading model: {args.model_path}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(config.model, device=config.device)

    # If using a fine-tuned checkpoint, load it
    if os.path.exists(args.model_path):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model_path)
        print("Loaded LoRA checkpoint")

    # Run benchmarks
    results = {}
    evaluators = {
        "mmlu": evaluate_mmlu,
        "gsm8k": evaluate_gsm8k,
        "humaneval": evaluate_humaneval,
        "mt_bench": evaluate_mt_bench,
        "alpaca_eval": evaluate_alpaca,
    }

    for benchmark in args.benchmarks:
        if benchmark in evaluators:
            print(f"\nRunning {benchmark}...")
            results[benchmark] = evaluators[benchmark](
                model=model,
                tokenizer=tokenizer,
                max_samples=args.max_samples,
            )
            print(f"  {benchmark}: {results[benchmark]}")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
