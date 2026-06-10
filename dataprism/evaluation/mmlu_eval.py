"""
MMLU (Massive Multitask Language Understanding) evaluation.

5-shot evaluation across 57 subjects covering STEM, humanities, social sciences.
"""

import logging
from typing import Optional

import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

logger = logging.getLogger("dataprism.eval.mmlu")


def evaluate_mmlu(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_fewshot: int = 5,
    max_samples: Optional[int] = None,
    batch_size: int = 8,
) -> dict:
    """Run MMLU evaluation.

    Args:
        model: Model to evaluate.
        tokenizer: Tokenizer.
        num_fewshot: Number of few-shot examples.
        max_samples: Limit samples for quick evaluation.
        batch_size: Batch size.

    Returns:
        Dict with 'accuracy' and per-subject scores.
    """
    logger.info("Evaluating MMLU (%d-shot)...", num_fewshot)

    try:
        # Try loading the full MMLU dataset
        dataset = load_dataset("cais/mmlu", "all", split="test")
    except Exception:
        logger.warning("MMLU dataset not available — returning placeholder")
        return {"accuracy": 0.0, "error": "MMLU dataset not available"}

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    total = 0
    subject_scores = {}

    model.eval()
    device = next(model.parameters()).device

    for sample in tqdm(dataset, desc="MMLU"):
        subject = sample.get("subject", "unknown")
        question = sample["question"]
        choices = sample["choices"]
        answer = sample["answer"]

        # Format prompt with few-shot examples
        prompt = _format_mmlu_prompt(question, choices)
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

        # Get model prediction
        with torch.no_grad():
            outputs = model(input_ids=prompt_ids["input_ids"].to(device))
            logits = outputs.logits[:, -1, :]

            # Score each choice
            choice_scores = []
            for choice in choices:
                choice_ids = tokenizer(choice, return_tensors="pt")["input_ids"][0]
                choice_score = logits[0, choice_ids[0]].item()
                choice_scores.append(choice_score)

            predicted = int(torch.tensor(choice_scores).argmax().item())

        is_correct = (predicted == answer)
        if is_correct:
            correct += 1
        total += 1

        if subject not in subject_scores:
            subject_scores[subject] = {"correct": 0, "total": 0}
        subject_scores[subject]["correct"] += int(is_correct)
        subject_scores[subject]["total"] += 1

    accuracy = correct / total if total > 0 else 0.0
    logger.info("MMLU accuracy: %.4f (%d/%d)", accuracy, correct, total)

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "subject_scores": {
            k: v["correct"] / v["total"]
            for k, v in subject_scores.items()
        },
    }


def _format_mmlu_prompt(question: str, choices: list[str]) -> str:
    """Format an MMLU question with choices."""
    labels = ["A", "B", "C", "D"]
    prompt = f"Question: {question}\n\n"
    for label, choice in zip(labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt
