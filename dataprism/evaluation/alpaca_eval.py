"""
AlpacaEval evaluation — LLM-as-judge comparison against reference model.

Requires GPT-4 or equivalent as judge.
"""

import logging
from typing import Optional

logger = logging.getLogger("dataprism.eval.alpaca")


def evaluate_alpaca(
    model,
    tokenizer,
    max_samples: Optional[int] = None,
) -> dict:
    """Evaluate on AlpacaEval (LLM-judge comparison).

    Args:
        model: Model to evaluate.
        tokenizer: Tokenizer.
        max_samples: Limit samples.

    Returns:
        Dict with 'win_rate' against text-davinci-003 baseline.
    """
    logger.info("Evaluating AlpacaEval...")

    try:
        from datasets import load_dataset
        dataset = load_dataset("tatsu-lab/alpaca_eval", split="eval")
    except Exception:
        logger.warning("AlpacaEval dataset not available — returning placeholder")
        return {"win_rate": 0.0, "error": "AlpacaEval dataset not available"}

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    logger.info(
        "AlpacaEval: %d prompts loaded. Full evaluation requires GPT-4 judge API.",
        len(dataset),
    )

    return {
        "win_rate": 0.0,
        "note": "AlpacaEval requires GPT-4 judge. "
                "Use the 'alpaca_eval' package for full evaluation.",
        "prompts_loaded": len(dataset),
    }
