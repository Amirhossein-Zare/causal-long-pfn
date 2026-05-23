from __future__ import annotations

from importlib import import_module
from typing import Any

from clpfn.baselines.common.runner import run_all as run_adapter


ADAPTER_MODULES = {
    "crn": "clpfn.baselines.crn.train",
    "ct": "clpfn.baselines.ct.train",
    "gnet": "clpfn.baselines.gnet.train",
    "gtransformer": "clpfn.baselines.gtransformer.train",
    "msm": "clpfn.baselines.msm.train",
    "rmsn": "clpfn.baselines.rmsn.train",
}


DEFAULT_CONFIGS = {
    "crn": "configs/eval/crn.yaml",
    "ct": "configs/eval/ct.yaml",
    "gnet": "configs/eval/gnet.yaml",
    "gtransformer": "configs/eval/gtransformer.yaml",
    "msm": "configs/eval/msm.yaml",
    "rmsn": "configs/eval/rmsn.yaml",
}


def available_baselines() -> tuple[str, ...]:
    return tuple(sorted(ADAPTER_MODULES))


def normalize_method(method: str) -> str:
    key = str(method).lower().replace("-", "").replace("_", "")
    if key not in ADAPTER_MODULES:
        choices = ", ".join(available_baselines())
        raise ValueError(f"Unknown baseline '{method}'. Available baselines: {choices}.")
    return key


def load_adapter(method: str) -> Any:
    module = import_module(ADAPTER_MODULES[normalize_method(method)])
    adapter = module.ADAPTER
    adapter.configure_from_eval_config = module.configure_from_eval_config
    return adapter


def run_all(method: str, **kwargs: Any) -> dict[str, Any]:
    return run_adapter(load_adapter(method), **kwargs)


def default_config_for(method: str) -> str:
    return DEFAULT_CONFIGS[normalize_method(method)]
