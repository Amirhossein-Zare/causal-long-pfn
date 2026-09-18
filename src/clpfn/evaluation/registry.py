from __future__ import annotations

from typing import Any

from clpfn.baselines.registry import available_baselines, default_config_for as baseline_default_config_for
from clpfn.baselines.registry import load_adapter, normalize_method as normalize_baseline_method
from clpfn.baselines.common.runner import run_all as run_baseline_adapter
from clpfn.evaluation import pfn as pfn_method
from clpfn.evaluation import ready_baselines


PFN_METHOD = "pfn"


def available_methods() -> tuple[str, ...]:
    return tuple(sorted(("pfn", *available_baselines(), *ready_baselines.available_methods())))


def is_pfn_method(method: str) -> bool:
    return str(method).strip().lower() == PFN_METHOD


def is_ready_baseline_method(method: str) -> bool:
    return ready_baselines.is_method(method)


def normalize_method(method: str) -> str:
    if is_pfn_method(method):
        return "pfn"
    if is_ready_baseline_method(method):
        return ready_baselines.normalize_method(method)
    return normalize_baseline_method(method)


def default_config_for(method: str) -> str:
    normalized = normalize_method(method)
    if normalized == "pfn":
        return "configs/eval/pfn.yaml"
    if is_ready_baseline_method(normalized):
        return ready_baselines.default_config_for(normalized)
    return baseline_default_config_for(normalized)


def run_evaluation(method: str, **kwargs: Any) -> dict[str, Any]:
    normalized = normalize_method(method)
    if normalized == "pfn":
        return pfn_method.run_all(**kwargs)
    if is_ready_baseline_method(normalized):
        return ready_baselines.run_all(normalized, **kwargs)
    return run_baseline_adapter(load_adapter(normalized), **kwargs)
