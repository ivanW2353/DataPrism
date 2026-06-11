"""
LoRA adapter application and parameter inspection utilities.

Provides thin wrappers around the PEFT library for consistent LoRA
configuration across all DataPrism phases.
"""

import logging
from typing import List

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import PreTrainedModel

from dataprism.config.dataclass import LoRAConfig

logger = logging.getLogger("dataprism.models.lora")


def apply_lora(model: PreTrainedModel, lora_config: LoRAConfig) -> "PeftModel":
    """Apply a LoRA adapter to a pretrained model.

    Merges the standard target_modules with target_modules_extra (if any)
    to support both attention-only and attention+MLP LoRA configurations.

    Args:
        model: Pretrained base model (e.g., LlamaForCausalLM).
        lora_config: LoRA hyperparameters from DataPrismConfig.

    Returns:
        PeftModel with LoRA adapter attached and trainable.
    """
    # Merge target modules
    target_modules = list(lora_config.target_modules)
    if lora_config.target_modules_extra:
        target_modules.extend(lora_config.target_modules_extra)

    # Deduplicate while preserving order
    seen = set()
    unique_modules = []
    for m in target_modules:
        if m not in seen:
            seen.add(m)
            unique_modules.append(m)

    logger.info(
        "Applying LoRA: r=%d alpha=%d dropout=%.2f target_modules=%s",
        lora_config.r,
        lora_config.alpha,
        lora_config.dropout,
        unique_modules,
    )

    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=unique_modules,
        bias=lora_config.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    return peft_model


def get_lora_parameter_names(model: "PeftModel") -> List[str]:
    """Return the names of all LoRA trainable parameters.

    Args:
        model: PeftModel with LoRA adapters applied.

    Returns:
        List of parameter names (strings) that contain 'lora_' and require grad.
    """
    return [name for name, param in model.named_parameters()
            if param.requires_grad and "lora_" in name]


def get_lora_parameter_count(model: "PeftModel") -> int:
    """Return the total number of trainable LoRA parameters.

    Args:
        model: PeftModel with LoRA adapters applied.

    Returns:
        Integer count of LoRA scalar parameters.
    """
    return sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and "lora_" in n
    )
