"""CPU-only classical baselines evaluated from PFN-ready benchmark files."""

from __future__ import annotations

from typing import Any

from clpfn.evaluation.ready_baselines.models import SPECS, normalize_method
from clpfn.evaluation.ready_baselines.runner import run_all as _run_all


DEFAULT_CONFIGS = {
    "persistence": "configs/eval/persistence.yaml",
    "linear_autoregressive": "configs/eval/linear_autoregressive.yaml",
}


def available_methods() -> tuple[str, ...]:
    return tuple(sorted(SPECS))


def is_method(method: str) -> bool:
    try:
        normalize_method(method)
    except ValueError:
        return False
    return True


def default_config_for(method: str) -> str:
    return DEFAULT_CONFIGS[normalize_method(method)]


def run_all(method: str, **kwargs: Any) -> dict[str, Any]:
    return _run_all(normalize_method(method), **kwargs)


__all__ = [
    "available_methods",
    "default_config_for",
    "is_method",
    "normalize_method",
    "run_all",
]
