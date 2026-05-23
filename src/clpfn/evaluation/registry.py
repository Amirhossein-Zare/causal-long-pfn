from __future__ import annotations

from typing import Any

from clpfn.baselines.registry import available_baselines, default_config_for as baseline_default_config_for
from clpfn.baselines.registry import load_adapter, normalize_method as normalize_baseline_method
from clpfn.baselines.common.runner import run_all as run_baseline_adapter
from clpfn.evaluation import pfn as pfn_method


PFN_METHODS = ("pfn", "causal_long_pfn", "causallongpfn")


def available_methods() -> tuple[str, ...]:
    return tuple(sorted(("pfn", *available_baselines())))


def is_pfn_method(method: str) -> bool:
    key = str(method).lower().replace("-", "_")
    compact = key.replace("_", "")
    return key in PFN_METHODS or compact in PFN_METHODS


def normalize_method(method: str) -> str:
    if is_pfn_method(method):
        return "pfn"
    return normalize_baseline_method(method)


def default_config_for(method: str) -> str:
    normalized = normalize_method(method)
    if normalized == "pfn":
        return "configs/eval/pfn.yaml"
    return baseline_default_config_for(normalized)


def run_evaluation(method: str, **kwargs: Any) -> dict[str, Any]:
    normalized = normalize_method(method)
    if normalized == "pfn":
        return pfn_method.run_all(**kwargs)
    return run_baseline_adapter(load_adapter(normalized), **kwargs)
