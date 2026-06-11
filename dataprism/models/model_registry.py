"""
Model factory for loading pretrained LLMs with automatic architecture detection.

Supports Llama, Qwen, and other HuggingFace causal LMs.
"""

import logging
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from dataprism.config.dataclass import ModelConfig

logger = logging.getLogger("dataprism.models")


def load_model_and_tokenizer(
    model_config: ModelConfig,
    device: str = "auto",
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a pretrained LLM and its tokenizer.

    Args:
        model_config: Model configuration (name, dtype, etc.).
        device: Device to load model on.

    Returns:
        Tuple of (model, tokenizer).
    """
    logger.info("Loading model: %s", model_config.name)

    # Determine torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(model_config.torch_dtype, torch.bfloat16)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.name,
        trust_remote_code=model_config.trust_remote_code,
    )

    # Set padding token if not present (common for Llama)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token")

    # Load model
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": model_config.trust_remote_code,
    }

    if model_config.use_flash_attention_2:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            logger.warning("Flash Attention 2 not available, falling back to SDPA")

    if device == "cpu":
        model_kwargs["device_map"] = None
    else:
        # cuda or auto: load to GPU
        model_kwargs["device_map"] = model_config.device_map or "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_config.name,
        **model_kwargs,
    )

    # Enable gradient checkpointing to save VRAM during training
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    logger.info("Model loaded: %d parameters", sum(p.numel() for p in model.parameters()))
    return model, tokenizer


def get_target_modules_for_model(model_name: str) -> list[str]:
    """Return recommended LoRA target modules based on model architecture.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        List of module names to target with LoRA.
    """
    model_lower = model_name.lower()

    if "llama" in model_lower:
        # Llama architecture: attention q/k/v/o + MLP gate/up/down
        return ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
    elif "qwen" in model_lower:
        # Qwen2 architecture: same naming as Llama
        return ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
    elif "mistral" in model_lower:
        return ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
    elif "gpt2" in model_lower:
        # GPT-2 uses different naming
        return ["c_attn", "c_proj", "c_fc"]
    else:
        # Default: just attention projections
        logger.warning("Unknown model architecture: %s. Using default LoRA targets.", model_name)
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
