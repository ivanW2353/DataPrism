"""Model management — LoRA hooks, adapter utilities, and model loading registry."""

from dataprism.models.gradient_hooks import LoRAGradientCapture, compute_per_sample_lora_gradients
from dataprism.models.lora_manager import apply_lora, get_lora_parameter_names
from dataprism.models.model_registry import load_model_and_tokenizer

__all__ = [
    "LoRAGradientCapture",
    "compute_per_sample_lora_gradients",
    "apply_lora",
    "get_lora_parameter_names",
    "load_model_and_tokenizer",
]
