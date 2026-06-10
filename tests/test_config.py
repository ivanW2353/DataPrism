"""Tests for the configuration system."""

import pytest
import tempfile
import os
import yaml

from dataprism.config.loader import (
    deep_merge,
    set_nested,
    parse_cli_overrides,
    _infer_type,
)
from dataprism.config.dataclass import DataPrismConfig


class TestDeepMerge:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 20, "z": 30}}
        result = deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3}

    def test_empty_base(self):
        result = deep_merge({}, {"a": 1})
        assert result == {"a": 1}


class TestSetNested:
    def test_shallow_key(self):
        d = {"a": 1}
        result = set_nested(d, "b", "2")
        assert result["b"] == 2

    def test_nested_key(self):
        d = {"a": {"b": {"c": 1}}}
        result = set_nested(d, "a.b.c", "42")
        assert result["a"]["b"]["c"] == 42

    def test_create_nested(self):
        d = {}
        result = set_nested(d, "x.y.z", "hello")
        assert result["x"]["y"]["z"] == "hello"


class TestInferType:
    def test_bool(self):
        assert _infer_type("true") is True
        assert _infer_type("false") is False

    def test_none(self):
        assert _infer_type("null") is None
        assert _infer_type("none") is None

    def test_int(self):
        assert _infer_type("42") == 42
        assert isinstance(_infer_type("42"), int)

    def test_float(self):
        assert _infer_type("3.14") == 3.14
        assert isinstance(_infer_type("3.14"), float)

    def test_list(self):
        result = _infer_type("a,b,c")
        assert result == ["a", "b", "c"]

    def test_string(self):
        assert _infer_type("hello") == "hello"


class TestParseCLIOverrides:
    def test_single_override(self):
        result = parse_cli_overrides(["phase1.num_epochs=3"])
        assert result == {"phase1.num_epochs": "3"}

    def test_multiple_overrides(self):
        result = parse_cli_overrides(["a.b=1", "c.d=hello"])
        assert result == {"a.b": "1", "c.d": "hello"}

    def test_empty(self):
        assert parse_cli_overrides([]) == {}
        assert parse_cli_overrides(None) == {}

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            parse_cli_overrides(["no_equals_sign"])


class TestDataPrismConfig:
    def test_default_config(self):
        """Test that default config can be instantiated."""
        config = DataPrismConfig()
        assert config.seed == 42
        assert config.model.name == "meta-llama/Meta-Llama-3-8B"
        assert config.lora.r == 64
        assert config.lora.alpha == 128
        assert config.phase1.outlier_percentile == 95.0

    def test_disabled_phases(self):
        config = DataPrismConfig()
        assert config.phase1.enabled is True
        assert config.phase2.enabled is True
        assert config.phase3.enabled is True

    def test_lambda_weights_validation(self):
        """Test that invalid lambda weights raise error."""
        config = DataPrismConfig()
        config.phase3.lambda_weights = {"a": 0.5, "b": 0.3}  # Sum = 0.8
        with pytest.raises(ValueError):
            config._validate()

    def test_tau_strategy_validation(self):
        config = DataPrismConfig()
        config.phase2.tau_annealing = "invalid"
        with pytest.raises(ValueError):
            config._validate()
