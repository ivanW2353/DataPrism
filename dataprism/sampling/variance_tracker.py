"""
EMA-based variance reduction tracker.

Monitors whether importance sampling is providing sufficient variance
reduction vs uniform sampling. Falls back to uniform when the benefit
drops below a configured threshold.

Reference: Katharopoulos & Fleuret, ICML 2018, Section 3.2
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger("dataprism.sampling.variance")


class VarianceTracker:
    """Tracks variance reduction of importance sampling via EMA.

    When the estimated variance reduction η drops below the threshold,
    signals to revert to uniform sampling (where the method provides
    no benefit).

    η = Var_uniform - Var_importance / Var_uniform

    If η < 0 → importance sampling has higher variance (bad)
    If η ≈ 0 → no benefit
    If η > 0 → importance sampling reduces variance (good)
    """

    def __init__(
        self,
        alpha: float = 0.95,
        threshold: float = 0.05,
    ):
        """Initialize variance tracker.

        Args:
            alpha: EMA smoothing factor for variance estimate.
            threshold: Minimum variance reduction to keep importance sampling active.
        """
        self._alpha = alpha
        self._threshold = threshold

        # EMA of variance reduction
        self._ema_variance_reduction: Optional[float] = None
        self._uniform_mode = False

        # Counters
        self._steps_in_uniform = 0
        self._steps_in_importance = 0

    def update(
        self,
        importance_weighted_variance: float,
        uniform_variance: float,
    ) -> None:
        """Update the variance reduction estimate.

        Args:
            importance_weighted_variance: Estimated variance of importance-sampled gradients.
            uniform_variance: Estimated variance of uniformly-sampled gradients
                              (or a running estimate).
        """
        if uniform_variance == 0:
            return

        eta = 1.0 - importance_weighted_variance / uniform_variance

        # EMA update
        if self._ema_variance_reduction is None:
            self._ema_variance_reduction = eta
        else:
            self._ema_variance_reduction = (
                self._alpha * self._ema_variance_reduction + (1 - self._alpha) * eta
            )

    def update_from_losses(
        self,
        importance_losses: torch.Tensor,
        sampling_probs: torch.Tensor,
        uniform_losses: Optional[torch.Tensor] = None,
    ) -> None:
        """Update from batch losses with importance sampling.

        Simplified estimation: variance of (loss / prob) vs variance of loss.
        This approximates the true gradient variance via loss variance.

        Args:
            importance_losses: Per-sample losses from importance-sampled batch (B,).
            sampling_probs: Probability each sample was selected with (B,).
            uniform_losses: Optional losses from a uniformly-sampled reference batch.
        """
        # Importance-weighted losses
        weighted_losses = importance_losses / (sampling_probs + 1e-8)
        imp_var = weighted_losses.var().item()

        if uniform_losses is not None:
            uni_var = uniform_losses.var().item()
        else:
            # Approximate: use unweighted losses as uniform proxy
            uni_var = importance_losses.var().item()

        self.update(imp_var, uni_var)

    @property
    def should_use_uniform(self) -> bool:
        """Check if we should fall back to uniform sampling.

        Returns True if importance sampling is not providing sufficient benefit.
        """
        if self._ema_variance_reduction is None:
            return False
        return self._ema_variance_reduction < self._threshold

    @property
    def variance_reduction(self) -> Optional[float]:
        """Current EMA estimate of variance reduction η."""
        return self._ema_variance_reduction

    @property
    def uniform_mode_active(self) -> bool:
        """Whether uniform mode is currently active."""
        return self._uniform_mode

    def set_uniform_mode(self, active: bool) -> None:
        """Manually set uniform mode."""
        self._uniform_mode = active
        if active:
            self._steps_in_uniform += 1
        else:
            self._steps_in_importance += 1

    def get_stats(self) -> dict:
        """Get current tracker statistics."""
        return {
            "variance_reduction": self._ema_variance_reduction,
            "threshold": self._threshold,
            "uniform_mode": self._uniform_mode,
            "steps_in_uniform": self._steps_in_uniform,
            "steps_in_importance": self._steps_in_importance,
        }
