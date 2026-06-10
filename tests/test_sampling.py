"""Tests for the Phase 2 importance sampling components."""

import pytest
import torch

from dataprism.sampling.temperature_scheduler import TemperatureScheduler
from dataprism.sampling.variance_tracker import VarianceTracker


class TestTemperatureScheduler:
    def test_constant(self):
        sched = TemperatureScheduler(strategy="constant", initial_tau=1.0)
        assert sched.get_tau(0) == 1.0
        assert sched.get_tau(100) == 1.0
        assert sched.get_tau(10000) == 1.0

    def test_linear(self):
        sched = TemperatureScheduler(
            strategy="linear", initial_tau=1.0, tau_min=0.1, schedule_length=100,
        )
        assert sched.get_tau(0) == 1.0
        assert 0.5 < sched.get_tau(50) < 0.6  # Halfway
        assert sched.get_tau(100) == 0.1
        assert sched.get_tau(200) == 0.1  # Clamped at min

    def test_cosine(self):
        sched = TemperatureScheduler(
            strategy="cosine", initial_tau=1.0, tau_min=0.0, schedule_length=100,
        )
        assert sched.get_tau(0) == 1.0
        assert sched.get_tau(100) == 0.0
        assert sched.get_tau(200) == 0.0


class TestVarianceTracker:
    def test_initial_state(self):
        tracker = VarianceTracker(alpha=0.95, threshold=0.05)
        assert tracker.variance_reduction is None
        assert not tracker.should_use_uniform

    def test_update_and_fallback(self):
        tracker = VarianceTracker(alpha=0.95, threshold=0.1)
        # Simulate poor variance reduction
        for _ in range(20):
            tracker.update(
                importance_weighted_variance=1.0,
                uniform_variance=1.0,  # Same = no reduction
            )
        # After many updates, variance reduction should be near 0
        assert tracker.variance_reduction is not None
        assert tracker.variance_reduction < 0.1

    def test_update_from_losses(self):
        tracker = VarianceTracker(alpha=0.5, threshold=0.1)
        losses = torch.tensor([1.0, 2.0, 1.5, 0.5])
        probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
        tracker.update_from_losses(losses, probs)
        assert tracker.variance_reduction is not None

    def test_set_uniform_mode(self):
        tracker = VarianceTracker()
        assert not tracker.uniform_mode_active
        tracker.set_uniform_mode(True)
        assert tracker.uniform_mode_active
        tracker.set_uniform_mode(False)
        assert not tracker.uniform_mode_active

    def test_stats(self):
        tracker = VarianceTracker(alpha=0.9, threshold=0.05)
        tracker.update(0.5, 1.0)
        stats = tracker.get_stats()
        assert "variance_reduction" in stats
        assert stats["threshold"] == 0.05
