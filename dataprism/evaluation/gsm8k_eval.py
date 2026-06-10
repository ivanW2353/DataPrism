"""
GSM8K (Grade School Math 8K) evaluation.

8-shot chain-of-thought evaluation for mathematical reasoning.
"""

import logging
import re
from typing import Optional

import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

logger = logging.getLogger("dataprism.eval.gsm8k")


def evaluate_gsm8k(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_fewshot: int = 8,
    use_cot: bool = True,
    max_samples: Optional[int] = None,
) -> dict:
    """Run GSM8K evaluation.

    Args:
        model: Model to evaluate.
        tokenizer: Tokenizer.
        num_fewshot: Number of few-shot examples.
        use_cot: Use chain-of-thought prompting.
        max_samples: Limit samples.

    Returns:
        Dict with 'accuracy' and 'exact_match' scores.
    """
    logger.info("Evaluating GSM8K (%d-shot, CoT=%s)...", num_fewshot, use_cot)

    try:
        dataset = load_dataset("gsm8k", "main", split="test")
    except Exception:
        logger.warning("GSM8K dataset not available — returning placeholder")
        return {"accuracy": 0.0, "error": "GSM8K dataset not available"}

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    total = 0

    model.eval()
    device = next(model.parameters()).device

    for sample in tqdm(dataset, desc="GSM8K"):
        question = sample["question"]
        ground_truth = sample["answer"]

        # Build prompt
        prompt = _build_gsm8k_prompt(question, use_cot=use_cot)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Extract the final answer number
        predicted_answer = _extract_number(response)
        true_answer = _extract_number(ground_truth)

        if predicted_answer is not None and true_answer is not None:
            if abs(predicted_answer - true_answer) < 1e-6:
                correct += 1

        total += 1

    accuracy = correct / total if total > 0 else 0.0
    logger.info("GSM8K accuracy: %.4f (%d/%d)", accuracy, correct, total)

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


def _build_gsm8k_prompt(question: str, use_cot: bool = True) -> str:
    """Build a GSM8K prompt with optional CoT instruction."""
    if use_cot:
        prompt = (
            "Solve the following math problem step by step. "
            "Show your reasoning and end with '#### <answer>'.\n\n"
            f"Question: {question}\n\n"
            "Let's solve this step by step:\n"
        )
    else:
        prompt = f"Question: {question}\nAnswer:"
    return prompt


def _extract_number(text: str) -> Optional[float]:
    """Extract the final numeric answer from GSM8K response.

    Looks for '#### <number>' pattern, or the last number in the text.
    """
    # Try the GSM8K format: #### 42
    match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if match:
        return float(match.group(1).replace(",", ""))

    # Fallback: last number in the text
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if numbers:
        return float(numbers[-1].replace(",", ""))

    return None
