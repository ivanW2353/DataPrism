"""
LoRA gradient capture hooks for TracInCP influence estimation.

Registers backward hooks on all LoRA parameters so gradient vectors can be
flattened into a single vector for dot-product / cosine similarity computations.
"""

import logging
from typing import List

import torch
import torch.nn as nn
from peft import PeftModel

logger = logging.getLogger("dataprism.models.gradient_hooks")


def _get_lora_trainable_params(model: PeftModel) -> List[nn.Parameter]:
    """Return all trainable LoRA parameters from a PeftModel.

    This captures both lora_A and lora_B weights (and biases if present)
    across all target modules.

    Args:
        model: A PeftModel with LoRA adapters applied.

    Returns:
        List of nn.Parameter tensors (LoRA weights only).
    """
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad and "lora_" in name:
            params.append(param)
    return params


class LoRAGradientCapture:
    """Captures and flattens LoRA gradient vectors for influence computations.

    This class provides a lightweight way to extract gradient vectors from
    the LoRA parameter space without storing full computational graphs.

    For LoRA (r=64, α=128) on 7 transformer attention + MLP projections,
    the total trainable dimension is roughly 2M — compared to 8B+ for the
    full model, a ~4000× reduction.

    Attributes:
        total_lora_dim: Total number of scalar LoRA parameters.
    """

    def __init__(self, model: PeftModel):
        """Initialize gradient capture for a PeftModel.

        Args:
            model: PeftModel with LoRA adapters applied.
        """
        self._model = model
        self._lora_params = _get_lora_trainable_params(model)
        self._total_lora_dim = sum(p.numel() for p in self._lora_params)

        if self._total_lora_dim == 0:
            raise RuntimeError(
                "No LoRA parameters found on model. "
                "Ensure LoRA adapters have been applied via apply_lora()."
            )

        logger.debug(
            "LoRAGradientCapture: %d LoRA parameters across %d tensors "
            "(total dim = %d ≈ %.1fM)",
            self._total_lora_dim,
            len(self._lora_params),
            self._total_lora_dim,
            self._total_lora_dim / 1e6,
        )

    @property
    def total_lora_dim(self) -> int:
        """Total number of scalar LoRA parameters."""
        return self._total_lora_dim

    def get_flattened_gradients(self) -> torch.Tensor:
        """Return all LoRA gradients concatenated into a single 1D tensor.

        Returns:
            1D tensor of shape (total_lora_dim,) containing the gradient
            of every LoRA parameter, ordered by parameter iteration order.
            Returns zeros for parameters that received no gradient.
        """
        grad_parts = []
        for param in self._lora_params:
            if param.grad is not None:
                grad_parts.append(param.grad.detach().view(-1))
            else:
                grad_parts.append(torch.zeros(param.numel(), device=param.device))
        return torch.cat(grad_parts)

    def clear_gradients(self) -> None:
        """Zero out all LoRA parameter gradients."""
        for param in self._lora_params:
            param.grad = None


def compute_per_sample_lora_gradients(
    model: PeftModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    capture: LoRAGradientCapture,
) -> List[torch.Tensor]:
    """Compute per-sample LoRA gradients for a batch.

    Uses batch forward + per-sample backward via torch.autograd.grad:
    one forward pass computes logits for the whole batch, then per-sample
    losses are differentiated individually.  For decoder-only transformers
    there is no cross-sample interaction, so each sample's gradient is
    correct and independent.

    Args:
        model: PeftModel with LoRA adapters.
        input_ids: (batch_size, seq_len) token ids.
        attention_mask: (batch_size, seq_len) attention mask.
        labels: (batch_size, seq_len) labels.
        capture: LoRAGradientCapture instance for this model.

    Returns:
        List of 1D tensors, one per sample, each of shape (total_lora_dim,).
    """
    batch_size = input_ids.size(0)
    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "lora_" in n]

    # Single batched forward pass with gradient computation enabled.
    # Activations are retained so per-sample autograd.grad is cheap.
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits  # (batch_size, seq_len, vocab_size)

    # Shift for causal LM loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    per_sample_grads: List[torch.Tensor] = []

    for i in range(batch_size):
        # Per-sample cross-entropy loss
        sample_logits = shift_logits[i:i+1]  # (1, seq_len, vocab_size)
        sample_labels = shift_labels[i:i+1]  # (1, seq_len)

        loss = torch.nn.functional.cross_entropy(
            sample_logits.view(-1, sample_logits.size(-1)),
            sample_labels.view(-1),
            reduction="mean",
        )

        # Compute d(loss_i)/d(params) using batched forward activations
        grads_i = torch.autograd.grad(loss, lora_params, retain_graph=True)

        # Flatten manually (Capture.get_flattened_gradients depends on .grad)
        grad_vec = torch.cat([g.detach().view(-1) for g in grads_i])
        per_sample_grads.append(grad_vec)

    # Clean up computation graph
    model.zero_grad()

    return per_sample_grads
