"""
Registry pattern for DataPrism components.

Components (selectors, pipelines, evaluation tasks) are registered via
decorators for dynamic discovery. Use get_*() functions to access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Type

if TYPE_CHECKING:
    from dataprism.core.base_selector import DataSelector


# ── Data Selector Registry ────────────────────────────────────────────

SELECTORS: dict[str, Type[DataSelector]] = {}


def register_selector(name: str) -> Callable:
    """Decorator to register a DataSelector subclass.

    Usage:
        @register_selector("tracin_cp")
        class TracInSelector(DataSelector):
            ...
    """
    def decorator(cls: Type[DataSelector]) -> Type[DataSelector]:
        if name in SELECTORS:
            raise ValueError(f"Selector '{name}' already registered")
        SELECTORS[name] = cls
        return cls
    return decorator


def get_selector(name: str) -> Type[DataSelector]:
    """Get a registered selector class by name.

    Raises LookupError if not found.
    """
    if name not in SELECTORS:
        available = ", ".join(sorted(SELECTORS.keys()))
        raise LookupError(
            f"Selector '{name}' not found. Available: {available}"
        )
    return SELECTORS[name]


def list_selectors() -> list[str]:
    """Return sorted list of registered selector names."""
    return sorted(SELECTORS.keys())


# ── Pipeline Registry ─────────────────────────────────────────────────

PIPELINES: dict[str, Callable] = {}


def register_pipeline(name: str) -> Callable:
    """Decorator to register a pipeline function/class.

    Usage:
        @register_pipeline("phase1")
        class Phase1Pipeline:
            ...
    """
    def decorator(func: Callable) -> Callable:
        PIPELINES[name] = func
        return func
    return decorator


def get_pipeline(name: str) -> Callable:
    """Get a registered pipeline by name."""
    if name not in PIPELINES:
        available = ", ".join(sorted(PIPELINES.keys()))
        raise LookupError(
            f"Pipeline '{name}' not found. Available: {available}"
        )
    return PIPELINES[name]


# ── Evaluation Task Registry ──────────────────────────────────────────

EVALUATORS: dict[str, Callable] = {}


def register_evaluator(name: str) -> Callable:
    """Decorator to register an evaluation function.

    Usage:
        @register_evaluator("mmlu")
        def evaluate_mmlu(model, tokenizer, config):
            ...
    """
    def decorator(func: Callable) -> Callable:
        EVALUATORS[name] = func
        return func
    return decorator
