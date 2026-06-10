"""
Temperature scheduler for importance sampling softmax.

Controls the exploration-exploitation trade-off:
- High tau → near-uniform sampling (exploration)
- Low tau → aggressive focus on high-loss samples (exploitation)
"""

import logging
import math
from typing import Optional

logger = logging.getLogger("dataprism.sampling.temperature")


class TemperatureScheduler:
    """Anneal the softmax temperature tau over training steps.

    Supported strategies:
    - linear: tau goes linearly from initial → min over schedule_length steps
    - cosine: cosine annealing from initial → min
    - constant: tau stays fixed at initial value
    """

    def __init__(
        self,
        strategy: str = "linear",
        initial_tau: float = 1.0,
        tau_min: float = 0.1,
        schedule_length: int = 5000,
    ):
        """Initialize temperature scheduler.

        Args:
            strategy: Annealing strategy ('linear', 'cosine', 'constant').
            initial_tau: Starting temperature.
            tau_min: Minimum temperature.
            schedule_length: Number of steps to anneal over.
        """
        self._strategy = strategy
        self._initial_tau = initial_tau
        self._tau_min = tau_min
        self._schedule_length = schedule_length

        logger.info(
            "TemperatureScheduler: %s, tau: %.2f→%.2f over %d steps",
            strategy, initial_tau, tau_min, schedule_length,
        )

    def get_tau(self, step: int) -> float:
        """Get the temperature for the current training step.

        Args:
            step: Current training step (0-indexed).

        Returns:
            Temperature value.
        """
        if self._strategy == "constant":
            return self._initial_tau

        if step >= self._schedule_length:
            return self._tau_min

        progress = step / max(self._schedule_length, 1)

        if self._strategy == "linear":
            return self._initial_tau - (self._initial_tau - self._tau_min) * progress
        elif self._strategy == "cosine":
            return self._tau_min + 0.5 * (self._initial_tau - self._tau_min) * (
                1.0 + math.cos(math.pi * progress)
            )
        else:
            raise ValueError(f"Unknown strategy: {self._strategy}")

    @property
    def current_tau_range(self) -> tuple[float, float]:
        """Return the (min, max) tau range."""
        return (self._tau_min, self._initial_tau)
