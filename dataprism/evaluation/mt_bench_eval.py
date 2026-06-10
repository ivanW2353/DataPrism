"""
MT-Bench evaluation — multi-turn conversation quality assessment.

Requires GPT-4 as judge for the standard evaluation protocol.
"""

import logging
from typing import Optional

logger = logging.getLogger("dataprism.eval.mt_bench")


def evaluate_mt_bench(
    model,
    tokenizer,
    num_turns: int = 2,
    max_samples: Optional[int] = None,
) -> dict:
    """Evaluate on MT-Bench (multi-turn chat benchmark).

    Args:
        model: Model to evaluate.
        tokenizer: Tokenizer.
        num_turns: Number of conversation turns.
        max_samples: Limit samples.

    Returns:
        Dict with MT-Bench score.
    """
    logger.info("Evaluating MT-Bench (%d-turn)...", num_turns)

    try:
        from datasets import load_dataset
        dataset = load_dataset("lmsys/mt_bench", split="train")
    except Exception:
        logger.warning("MT-Bench dataset not available — returning placeholder")
        return {"score": 0.0, "error": "MT-Bench dataset not available"}

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # MT-Bench evaluation requires GPT-4 as judge
    logger.info(
        "MT-Bench: %d questions loaded. Full evaluation requires GPT-4 judge API.",
        len(dataset),
    )

    return {
        "score": 0.0,
        "note": "MT-Bench evaluation requires GPT-4 as judge. "
                "Use the 'fastchat' package for full evaluation.",
        "questions_loaded": len(dataset),
    }
