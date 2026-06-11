"""
YAML configuration loader with deep merging and CLI override support.

Merge order (later overrides earlier):
1. configs/default.yaml
2. Specific configs (e.g., configs/model/llama3_8b.yaml)
3. CLI overrides (--override key=value, dot-separated for nested keys)
"""

import argparse
import os
from pathlib import Path
from typing import Any, Optional

import yaml

from dataprism.config.dataclass import DataPrismConfig


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def set_nested(config_dict: dict, key_path: str, value: str) -> dict:
    """Set a nested dict value using dot-separated key path, auto-converting types.

    Examples:
        set_nested(d, "phase1.num_epochs", "3")  -> d["phase1"]["num_epochs"] = 3 (int)
        set_nested(d, "training.learning_rate", "1e-4") -> float
    """
    keys = key_path.split(".")
    current = config_dict
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    # Type conversion
    typed_value = _infer_type(value)
    current[keys[-1]] = typed_value
    return config_dict


def _infer_type(value: str) -> Any:
    """Attempt to convert string to appropriate Python type."""
    # Bool
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    # None
    if value.lower() in ("none", "null"):
        return None
    # Int
    try:
        return int(value)
    except ValueError:
        pass
    # Float
    try:
        return float(value)
    except ValueError:
        pass
    # List (comma-separated)
    if "," in value:
        return [_infer_type(v.strip()) for v in value.split(",")]
    return value


def _load_yaml(path: str) -> dict:
    """Load and parse a single YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    return data


def _find_config_path(config_name: str) -> str:
    """Resolve a config name to an absolute path.

    Looks in:
    1. Exact path (if absolute or exists as-is)
    2. configs/ directory relative to project root
    """
    if os.path.isabs(config_name) and os.path.exists(config_name):
        return config_name

    # Search in configs/ directory
    project_root = Path(__file__).parent.parent.parent
    config_dir = project_root / "configs"
    candidate = config_dir / config_name
    if candidate.exists():
        return str(candidate)
    # Try with .yaml extension
    candidate_yaml = config_dir / f"{config_name}.yaml"
    if candidate_yaml.exists():
        return str(candidate_yaml)

    raise FileNotFoundError(
        f"Config '{config_name}' not found at '{config_dir}' or as absolute path"
    )


def load_config(
    config_paths: Optional[list[str]] = None,
    overrides: Optional[dict[str, str]] = None,
) -> DataPrismConfig:
    """Load and merge configuration from YAML files and CLI overrides.

    Args:
        config_paths: List of config file names or paths (merged in order).
                     If None, only loads configs/default.yaml.
        overrides: Dict of dot-separated key -> value string for CLI overrides.

    Returns:
        Validated DataPrismConfig instance.

    Example:
        config = load_config(
            config_paths=["configs/default.yaml", "model/llama3_8b"],
            overrides={"phase1.num_epochs": "3", "training.learning_rate": "1e-4"}
        )
    """
    if config_paths is None:
        config_paths = []

    merged: dict = {}

    for config_name in config_paths:
        path = _find_config_path(config_name)
        data = _load_yaml(path)
        merged = deep_merge(merged, data)

    # Apply CLI overrides
    if overrides:
        for key_path, value in overrides.items():
            merged = set_nested(merged, key_path, value)

    # Build structured config
    # Flatten top-level keys that map to sub-configs
    return _dict_to_config(merged)


def _dict_to_config(raw: dict) -> DataPrismConfig:
    """Convert merged dict to DataPrismConfig, handling nested sub-config keys.

    The top-level dict may have keys like 'model', 'lora', 'phase1', etc.
    that map to sub-config dataclasses.
    """
    from dataprism.config.dataclass import (
        ModelConfig, LoRAConfig, TrainingConfig, DataConfig,
        Phase1TracInConfig, Phase2ImportanceConfig,
        Phase3MultiEvalConfig, EvaluationConfig,
    )

    # Map sub-config keys to their dataclass types
    sub_config_map = {
        "model": (ModelConfig, "model"),
        "lora": (LoRAConfig, "lora"),
        "training": (TrainingConfig, "training"),
        "data": (DataConfig, "data"),
        "phase1": (Phase1TracInConfig, "phase1"),
        "phase2": (Phase2ImportanceConfig, "phase2"),
        "phase3": (Phase3MultiEvalConfig, "phase3"),
        "evaluation": (EvaluationConfig, "evaluation"),
    }

    top_level = {}
    sub_configs = {}

    for key, value in raw.items():
        if key in sub_config_map:
            cls, field_name = sub_config_map[key]
            sub_configs[field_name] = cls(**value) if isinstance(value, dict) else value
        else:
            top_level[key] = value

    return DataPrismConfig(**top_level, **sub_configs)


def parse_cli_overrides(override_args: Optional[list[str]] = None) -> dict[str, str]:
    """Parse --override key=value pairs from command line.

    Args:
        override_args: List of "key=value" strings.

    Returns:
        Dict of key_path -> value string.
    """
    if not override_args:
        return {}

    overrides = {}
    for arg in override_args:
        if "=" not in arg:
            raise ValueError(f"Override must be in 'key=value' format, got '{arg}'")
        key, value = arg.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def create_arg_parser() -> argparse.ArgumentParser:
    """Create the standard CLI argument parser for DataPrism scripts."""
    parser = argparse.ArgumentParser(
        description="DataPrism: Gradient-Driven Data Selection for LLM Fine-Tuning"
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="*",
        default=["default"],
        help="Config file names or paths (merged in order)",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="CLI overrides in 'key=value' format (e.g., 'phase1.num_epochs=3')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda, cpu, auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    return parser
