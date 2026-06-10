"""
Dataset loading, tokenization, and schema normalization.

Handles multiple data formats (Alpaca, ShareGPT, ChatML) by normalizing
to a common schema before tokenization.
"""

import logging
from typing import Optional

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

from dataprism.config.dataclass import DataConfig

logger = logging.getLogger("dataprism.data")


# ── Schema Normalization ─────────────────────────────────────────────

def _normalize_alpaca(example: dict) -> dict:
    """Normalize Alpaca-format example to {text, prompt, response}."""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")

    if inp:
        prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
    else:
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"

    return {
        "prompt": prompt,
        "response": output,
        "text": prompt + output,
    }


def _normalize_sharegpt(example: dict) -> dict:
    """Normalize ShareGPT-format example to {text, prompt, response}."""
    conversations = example.get("conversations", [])

    prompt_parts = []
    response = ""

    for turn in conversations:
        role = turn.get("from", turn.get("role", ""))
        value = turn.get("value", turn.get("content", ""))

        if role in ("human", "user"):
            prompt_parts.append(f"### Human:\n{value}\n")
        elif role in ("gpt", "assistant"):
            prompt_parts.append(f"### Assistant:\n{value}\n")
            if not response:
                response = value

    prompt = "".join(prompt_parts)
    return {
        "prompt": prompt,
        "response": response,
        "text": prompt,
    }


def _normalize_chatml(example: dict) -> dict:
    """Normalize ChatML-format example to {text, prompt, response}."""
    messages = example.get("messages", [])

    prompt_parts = []
    response = ""

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            prompt_parts.append(f"<|system|>\n{content}<|end|>\n")
        elif role == "user":
            prompt_parts.append(f"<|user|>\n{content}<|end|>\n")
        elif role == "assistant":
            prompt_parts.append(f"<|assistant|>\n{content}<|end|>\n")
            if not response:
                response = content

    prompt = "".join(prompt_parts)
    return {
        "prompt": prompt,
        "response": response,
        "text": prompt + (f"<|assistant|>\n{response}<|end|>" if response else ""),
    }


NORMALIZERS = {
    "alpaca": _normalize_alpaca,
    "sharegpt": _normalize_sharegpt,
    "chatml": _normalize_chatml,
}


# ── Dataset Loading ──────────────────────────────────────────────────

def load_and_prepare_dataset(
    data_config: DataConfig,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    """Load and preprocess a dataset for training.

    Args:
        data_config: Data configuration.
        tokenizer: Tokenizer for the model.

    Returns:
        Tokenized HuggingFace Dataset ready for training.
    """
    logger.info("Loading dataset: %s (split=%s)", data_config.name, data_config.split)

    # Load raw dataset
    try:
        dataset = load_dataset(data_config.name, split=data_config.split)
    except Exception:
        # Some datasets don't have a validation split
        logger.warning("Split '%s' not found for %s, loading 'train'",
                       data_config.split, data_config.name)
        dataset = load_dataset(data_config.name, split="train")

    # Subsample if requested
    if data_config.num_samples is not None and data_config.num_samples < len(dataset):
        dataset = dataset.shuffle(seed=data_config.shuffle_seed).select(
            range(data_config.num_samples)
        )
        logger.info("Subsampled to %d examples", data_config.num_samples)

    logger.info("Raw dataset: %d examples", len(dataset))

    # Normalize schema
    normalizer = NORMALIZERS.get(data_config.prompt_template)
    if normalizer:
        logger.info("Normalizing schema: %s", data_config.prompt_template)
        dataset = dataset.map(normalizer, desc="Normalizing schema")
    else:
        logger.info("Using raw schema (no normalization)")

    # Tokenize
    logger.info("Tokenizing (max_length=%d)...", data_config.training_max_seq_length
                if hasattr(data_config, 'training_max_seq_length') else 2048)
    max_length = getattr(data_config, 'training_max_seq_length', 2048)

    def tokenize_fn(example):
        if "prompt" in example and "response" in example:
            # Response-only loss: tokenize prompt+response together,
            # then mask prompt tokens in labels
            full_text = example["prompt"] + example["response"]

            tokenized_full = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )

            tokenized_prompt = tokenizer(
                example["prompt"],
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )

            prompt_len = len(tokenized_prompt["input_ids"])
            labels = tokenized_full["input_ids"].copy()

            if data_config.response_only_loss:
                # Mask prompt tokens with -100 (ignored in loss)
                labels[:prompt_len] = [-100] * prompt_len

            return {
                "input_ids": tokenized_full["input_ids"],
                "attention_mask": tokenized_full["attention_mask"],
                "labels": labels,
            }
        else:
            # Full text mode: no masking
            text = example.get("text", example.get("prompt", ""))
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

    dataset = dataset.map(
        tokenize_fn,
        remove_columns=[c for c in dataset.column_names
                       if c not in ("input_ids", "attention_mask", "labels")],
        desc="Tokenizing",
    )

    logger.info("Tokenized dataset: %d examples", len(dataset))
    return dataset


def load_validation_set(
    source: str,
    split: str,
    num_samples: int,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
) -> Dataset:
    """Load and tokenize a validation set from HuggingFace datasets.

    Args:
        source: HuggingFace dataset name.
        split: Dataset split (e.g., 'test', 'validation').
        num_samples: Number of samples to load.
        tokenizer: Tokenizer for the model.
        max_length: Maximum sequence length.

    Returns:
        Tokenized validation Dataset.
    """
    logger.info("Loading validation set: %s (split=%s, n=%d)", source, split, num_samples)

    try:
        dataset = load_dataset(source, split=split)
    except Exception:
        logger.warning("Split '%s' not found for %s, trying 'train'", split, source)
        dataset = load_dataset(source, split="train")

    # Subsample
    if num_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(num_samples))

    # Tokenize
    def tokenize_fn(example):
        # Try to find text content
        text = (
            example.get("text") or
            example.get("question") or
            example.get("instruction") or
            example.get("prompt") or
            ""
        )

        # Handle GSM8K-style (question + answer)
        if "answer" in example and example["answer"]:
            text = f"Question: {example.get('question', '')}\nAnswer: {example['answer']}"

        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    dataset = dataset.map(tokenize_fn, desc="Tokenizing validation set")
    return dataset


def create_toy_dataset(num_samples: int = 100) -> Dataset:
    """Create a tiny synthetic dataset for testing.

    Args:
        num_samples: Number of samples to generate.

    Returns:
        A small Dataset with instruction-response format.
    """
    import random
    random.seed(42)

    templates = [
        ("What is {topic}?", "{topic} is a fascinating subject that involves many aspects."),
        ("Explain the concept of {topic}.", "The concept of {topic} can be understood as follows..."),
        ("How does {topic} work?", "{topic} works through a series of interconnected mechanisms."),
        ("Define {topic} and give an example.", "{topic} is defined as... For example, consider..."),
    ]

    topics = [
        "machine learning", "neural networks", "gradient descent",
        "attention mechanism", "transformer architecture", "loss function",
        "optimization", "regularization", "backpropagation", "data preprocessing",
    ]

    data = []
    for i in range(num_samples):
        template = random.choice(templates)
        topic = random.choice(topics)
        instruction = template[0].format(topic=topic)
        output = template[1].format(topic=topic)

        # Inject some noise for realism
        if i % 20 == 0 and i > 0:
            output = "I'm sorry, I cannot answer that question."  # Mislabeled example
        if i % 15 == 0 and i > 0:
            instruction = instruction + " " + instruction  # Duplicate-ish

        data.append({
            "instruction": instruction,
            "input": "",
            "output": output,
        })

    return Dataset.from_list(data)
