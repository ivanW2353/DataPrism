"""Tests for data selection algorithms."""

import pytest
import numpy as np
from datasets import Dataset

from dataprism.core.registry import get_selector, list_selectors


@pytest.fixture
def sample_dataset():
    """A small dataset for testing selectors."""
    return Dataset.from_list([
        {"text": f"Sample text number {i}"} for i in range(100)
    ])


class TestUniformSelector:
    def test_select_fraction(self, sample_dataset):
        selector_cls = get_selector("uniform")
        selector = selector_cls(fraction=0.2, seed=42)
        result = selector.select(sample_dataset)
        assert len(result) == 20  # 20% of 100

    def test_select_minimum(self, sample_dataset):
        selector_cls = get_selector("uniform")
        selector = selector_cls(fraction=0.001, seed=42)
        result = selector.select(sample_dataset)
        assert len(result) >= 1  # At least 1 sample

    def test_deterministic(self, sample_dataset):
        selector_cls = get_selector("uniform")
        s1 = selector_cls(fraction=0.2, seed=42)
        s2 = selector_cls(fraction=0.2, seed=42)
        r1 = s1.select(sample_dataset)
        r2 = s2.select(sample_dataset)
        assert r1[0]["text"] == r2[0]["text"]  # Same first sample


class TestDSIRSelector:
    def test_select(self, sample_dataset):
        selector_cls = get_selector("dsir")
        selector = selector_cls(fraction=0.2, seed=42)
        result = selector.select(sample_dataset)
        assert len(result) == 20

    def test_select_small(self):
        tiny = Dataset.from_list([{"text": f"Item {i}"} for i in range(5)])
        selector_cls = get_selector("dsir")
        selector = selector_cls(fraction=0.5, seed=42)
        result = selector.select(tiny)
        assert len(result) >= 1


class TestSelectorRegistry:
    def test_all_registered(self):
        """Verify all expected selectors are importable."""
        expected = {"uniform", "less", "rho_loss", "dsir", "tracin_cp",
                    "importance_sampling", "multi_obj_tracin"}
        available = set(list_selectors())
        for name in expected:
            assert name in available, f"Selector '{name}' not registered"

    def test_get_unknown_selector(self):
        with pytest.raises(LookupError):
            get_selector("nonexistent_selector")
