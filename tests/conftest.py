"""
Shared test fixtures for DataPrism.

Uses tiny models (GPT-2) and small synthetic datasets so tests run
fast on CPU without requiring GPU or downloading large models.
"""

import os
import sys
import pytest
import tempfile

import torch
from datasets import Dataset

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def toy_dataset():
    """A tiny synthetic instruction-response dataset."""
    data = [
        {"instruction": f"Question {i}", "input": "", "output": f"Answer {i}"}
        for i in range(20)
    ]
    # Add a noisy sample
    data[5] = {"instruction": "Bad question", "input": "", "output": "I cannot answer"}
    return Dataset.from_list(data)


@pytest.fixture(scope="session")
def tokenized_toy_dataset(toy_dataset, tiny_tokenizer):
    """Tokenize the toy dataset."""
    def tokenize(example):
        text = f"Instruction: {example['instruction']}\n\nResponse: {example['output']}"
        tokenized = tiny_tokenizer(text, truncation=True, max_length=128, padding=False)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    return toy_dataset.map(tokenize)


@pytest.fixture(scope="session")
def tiny_tokenizer():
    """Load GPT-2 tokenizer (fast, small)."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="session")
def tiny_model():
    """Load GPT-2 Small — fast enough for unit tests."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return model


@pytest.fixture(scope="session")
def tiny_lora_model(tiny_model):
    """Apply LoRA to the tiny model."""
    from peft import LoraConfig, get_peft_model, TaskType
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["c_attn"],
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(tiny_model, config)


@pytest.fixture
def temp_dir():
    """Create a temporary directory that cleans up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_config_dict():
    """A minimal valid config dict."""
    return {
        "seed": 42,
        "device": "cpu",
        "project_name": "test",
        "experiment_name": "test_experiment",
        "output_dir": "/tmp/dataprism_test",
        "model": {"name": "gpt2", "torch_dtype": "float32"},
        "lora": {"r": 4, "alpha": 8},
        "training": {"num_epochs": 1, "per_device_train_batch_size": 2},
        "data": {"name": "toy", "prompt_template": "alpaca"},
        "phase1": {"enabled": False},
        "phase2": {"enabled": False},
        "phase3": {"enabled": False},
        "evaluation": {},
    }
