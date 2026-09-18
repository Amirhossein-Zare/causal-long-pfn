from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a mapping: {config_path}")
    return config


def pick(config: dict[str, Any], key: str, default: Any = None) -> Any:
    return config[key] if key in config and config[key] is not None else default
