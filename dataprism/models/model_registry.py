"""
Model factory for loading pretrained LLMs with automatic architecture detection.

Supports Llama, Qwen, and other HuggingFace causal LMs.
Handles dtype mapping, device placement, flash-attention, and offline operation.
"""

import logging
import os
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

# Mapping from config string to torch dtype
_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_model_and_tokenizer(
    model_config: ModelConfig,
    device: str = "auto",
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a pretrained LLM and its tokenizer.

    Respects TRANSFORMERS_OFFLINE and HF_HOME environment variables
    for offline operation on compute nodes.

    Args:
        model_config: ModelConfig with name/path, dtype, and options.
        device: Device string — "auto", "cuda", or "cpu".

    Returns:
        Tuple of (model, tokenizer).
    """
    model_name_or_path = model_config.name

    torch_dtype = _DTYPE_MAP.get(model_config.torch_dtype, torch.bfloat16)

    logger.info("Loading model: %s (dtype=%s, flash_attn=%s)",
                model_name_or_path, model_config.torch_dtype,
                model_config.use_flash_attention_2)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token")

    # Build model loading kwargs
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": model_config.trust_remote_code,
        "device_map": model_config.device_map if device == "auto" else None,
    }

    if model_config.use_flash_attention_2:
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            logger.warning(
                "flash_attention_2 requested but flash-attn not installed. "
                "Falling back to sdpa."
            )
            model_kwargs["attn_implementation"] = "sdpa"

    if device == "cpu":
        model_kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )

    # Enable gradient checkpointing to save VRAM during training
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    logger.info("Model loaded: %s (%d parameters)",
                model_config.name,
                sum(p.numel() for p in model.parameters()))

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
        return ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
    elif "qwen" in model_lower:
        return ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
    elif "mistral" in model_lower:
        return ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
    elif "gpt2" in model_lower:
        return ["c_attn", "c_proj", "c_fc"]
    else:
        logger.warning("Unknown model architecture: %s. Using default LoRA targets.", model_name)
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
