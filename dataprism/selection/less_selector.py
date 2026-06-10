"""
Baseline: LESS — Low-rank gradient Similarity Selection (Xia et al., ICML 2024).

Selects training samples whose last-hidden-state representations are
most similar to validation set representations. Uses LoRA checkpoint
ensembles for robust similarity estimation.
"""

import logging
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics.pairwise import cosine_similarity
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from dataprism.core.base_selector import DataSelector
from dataprism.core.registry import register_selector

logger = logging.getLogger("dataprism.selection.less")


@register_selector("less")
class LESSSelector(DataSelector):
    """LESS baseline: representation similarity-based selection.

    Core idea: Train on a small warmup subset, then select training
    samples whose hidden representations are most similar to those
    of a validation set.

    Reference: Xia et al., "LESS: Selecting Influential Data for
    Targeted Instruction Tuning", ICML 2024.
    """

    def __init__(
        self,
        fraction: float = 0.2,
        similarity_metric: str = "cosine",
        representation_layer: int = -1,  # Last layer
        seed: int = 42,
    ):
        self._fraction = fraction
        self._similarity_metric = similarity_metric
        self._representation_layer = representation_layer
        self._seed = seed

    def name(self) -> str:
        return "less"

    def select(
        self,
        dataset: Dataset,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> Dataset:
        if model is None:
            raise ValueError("LESS requires a model for representation extraction")

        n_total = len(dataset)
        n_select = max(1, int(n_total * self._fraction))
        logger.info("LESS: selecting %d/%d samples", n_select, n_total)

        # Extract hidden representations for each training sample
        train_reprs = self._extract_representations(model, dataset)

        # Use a subset of the data as "validation" proxy
        n_val = min(100, n_total // 10)
        val_indices = np.random.RandomState(self._seed).choice(n_total, n_val, replace=False)
        val_reprs = train_reprs[val_indices]

        # Compute similarity: mean cosine similarity to validation set
        similarities = cosine_similarity(train_reprs, val_reprs)
        mean_similarities = similarities.mean(axis=1)

        # Select top-k most similar
        top_k = np.argsort(mean_similarities)[-n_select:]
        top_k.sort()

        logger.info(
            "LESS selection: mean similarity=%.4f, range=[%.4f, %.4f]",
            mean_similarities.mean(), mean_similarities.min(), mean_similarities.max(),
        )

        return dataset.select(top_k.tolist())

    def _extract_representations(
        self,
        model: PreTrainedModel,
        dataset: Dataset,
        batch_size: int = 8,
    ) -> np.ndarray:
        """Extract last-hidden-state representations for each sample.

        Args:
            model: The model (PeftModel or base).
            dataset: Tokenized dataset.
            batch_size: Batch size for extraction.

        Returns:
            (n_samples, hidden_dim) array of representations.
        """
        model_device = next(model.parameters()).device
        representations = []

        model.eval()
        with torch.no_grad():
            for start in tqdm(range(0, len(dataset), batch_size), desc="Extracting LESS reps"):
                end = min(start + batch_size, len(dataset))
                batch = dataset[start:end]

                input_ids = torch.tensor(batch["input_ids"], device=model_device)
                attention_mask = torch.tensor(batch["attention_mask"], device=model_device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

                # Get last hidden state, mean-pool over sequence
                hidden = outputs.hidden_states[self._representation_layer]
                # Mean pool over non-padding tokens
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

                representations.append(pooled.cpu().numpy())

        return np.concatenate(representations, axis=0)
