from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_config_path = config.pop("base_config", None)
    if base_config_path:
        base_path = Path(base_config_path)
        if not base_path.is_absolute():
            base_path = (config_path.parent / base_path).resolve()
        base_config = load_config(str(base_path))
        config = merge_dicts(base_config, config)

    if overrides:
        config = merge_dicts(config, overrides)
    return config


def merge_dicts(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
