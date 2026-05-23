from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def pick(config: dict[str, Any], key: str, default: Any = None) -> Any:
    return config[key] if key in config and config[key] is not None else default
