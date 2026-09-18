from __future__ import annotations

from typing import Any

import numpy as np
import torch

REPORT_SCHEMA_VERSION = 1


def hardware_report_fields(device: Any = None) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "hardware_device": str(device if device is not None else ("cuda" if cuda_available else "cpu")),
        "cuda_available": cuda_available,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else "",
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda or ""),
    }


def reported_task(task_name: str, horizon: int | float) -> str:
    if task_name == "one_step_exhaustive" and int(horizon) == 1:
        return "one_step_exhaustive"
    if task_name == "sequential_rollout" and 1 <= int(horizon) <= 5:
        return f"sequential_h{int(horizon)}"
    raise ValueError(f"Unsupported task/horizon combination: {task_name!r}, {horizon!r}")


def task_identity_fields(meta: dict[str, Any], task_name: str, tau: int | float) -> dict[str, Any]:
    label = reported_task(task_name, tau)
    return {
        "report_schema_version": int(REPORT_SCHEMA_VERSION),
        "n_support": int(meta["support_size"]),
        "task_index": int(meta["dataset_id"]),
        "repeat_id": int(meta["replicate"]),
        "horizon_label": label,
        "reported_task": label,
        "task_step": label,
        "is_horizon5": bool(label == "sequential_h5"),
    }


def finalize_task_timing(
    rows: list[dict[str, Any]],
    *,
    refit_time_sec: float = 0.0,
) -> list[dict[str, Any]]:
    n_queries = int(len(rows))
    predict_times = np.asarray(
        [float(row["predict_time_sec"]) for row in rows],
        dtype=np.float64,
    )
    predict_sum = float(np.sum(predict_times)) if n_queries else 0.0
    predict_mean = float(np.mean(predict_times)) if n_queries else 0.0
    total = float(refit_time_sec) + predict_sum
    for row in rows:
        row["n_queries_in_task"] = n_queries
        row["refit_time_sec"] = float(refit_time_sec)
        row["total_task_time_sec"] = total
        row["ms_per_query"] = 1000.0 * predict_mean
    return rows


def no_tuning_fields() -> dict[str, Any]:
    return {
        "tuning_mode": "none",
        "initial_search_n": 0,
        "search_candidates_evaluated": 0,
        "search_time_sec": 0.0,
        "selected_candidate": -1,
    }


def pfn_checkpoint_fields(checkpoint_diag: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_basename": str(checkpoint_diag["checkpoint_basename"]),
        "checkpoint_step_count": int(checkpoint_diag["checkpoint_step_count"]),
        "trainable_parameters": int(checkpoint_diag["trainable_parameters"]),
        "pfn_target_refit_time_sec": 0.0,
    }
