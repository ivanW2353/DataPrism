"""
Custom data collators for DataPrism.

Supports:
- Standard LM collation with padding
- Importance-weighted batches (Phase 2)
- Weighted sampling metadata (Phase 3)
"""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForLM:
    """Standard causal LM data collator with padding."""

    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch = {}

        # Pad input_ids and attention_mask
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        # Find max length
        max_len = max(len(ids) for ids in input_ids)
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)
        if self.pad_to_multiple_of is not None:
            max_len = ((max_len + self.pad_to_multiple_of - 1)
                       // self.pad_to_multiple_of * self.pad_to_multiple_of)

        # Pad sequences
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for ids, lbls in zip(input_ids, labels):
            pad_len = max_len - len(ids)
            pad_token_id = self.tokenizer.pad_token_id or 0

            padded_input_ids.append(ids + [pad_token_id] * pad_len)
            padded_attention_mask.append([1] * len(ids) + [0] * pad_len)
            padded_labels.append(lbls + [-100] * pad_len)

        batch["input_ids"] = torch.tensor(padded_input_ids, dtype=torch.long)
        batch["attention_mask"] = torch.tensor(padded_attention_mask, dtype=torch.long)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        # Pass through additional fields (importance weights, indices, etc.)
        for key in ["importance_weight", "global_index"]:
            if key in features[0]:
                batch[key] = torch.tensor(
                    [f[key] for f in features], dtype=torch.float32
                    if key == "importance_weight" else torch.long,
                )

        return batch


@dataclass
class ImportanceWeightedCollator(DataCollatorForLM):
    """Collator that passes importance weights to the training loop.

    Used in Phase 2 for importance-sampled batches.
    """

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch = super().__call__(features)

        # Attach importance weights for loss re-weighting
        if "importance_weight" in features[0]:
            batch["importance_weight"] = torch.tensor(
                [f["importance_weight"] for f in features],
                dtype=torch.float32,
            )

        return batch
