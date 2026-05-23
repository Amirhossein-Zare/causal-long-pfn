from __future__ import annotations

from typing import Any

import numpy as np

from clpfn.evaluation.core import benchmark as common
from clpfn.evaluation.core.tasks import task_step_from_task_and_tau


REQUIRED_PREDICTION_COLUMNS = (
    "method",
    "method_family",
    "run_id",
    "domain",
    "dataset_id",
    "global_dataset_id",
    "source_file",
    "row_id",
    "query_id",
    "gamma",
    "support_size",
    "task_name",
    "task_step",
    "tau",
    "t_obs",
    "t_target",
    "pred_norm",
    "target_norm",
    "error_norm",
    "sq_error_norm",
    "abs_error_norm",
)


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Ready benchmark record is missing required key '{key}'.")
    return mapping[key]


def _base_prediction_record(
    *,
    method_name: str,
    method_family: str,
    run_id: str,
    domain: str,
    dataset_id: int,
    global_dataset_id: int,
    source_file: str,
    row_id: int,
    query_id: int,
    gamma: Any,
    support_size: int,
    task_name: str,
    t_obs: int,
    t_target: int,
    pred_norm: float,
    target_norm: float,
    predict_time_sec: float | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    t_obs = int(t_obs)
    t_target = int(t_target)
    tau = int(max(1, t_target - t_obs))
    pred_norm = float(pred_norm)
    target_norm = float(target_norm)
    error_norm = pred_norm - target_norm

    row = {
        "method": str(method_name),
        "method_family": str(method_family),
        "run_id": str(run_id),
        "domain": str(domain).lower(),
        "dataset_id": int(dataset_id),
        "global_dataset_id": int(global_dataset_id),
        "source_file": str(source_file),
        "row_id": int(row_id),
        "query_id": int(query_id),
        "gamma": gamma,
        "support_size": int(support_size),
        "task_name": str(task_name),
        "task_step": task_step_from_task_and_tau(task_name, tau),
        "tau": int(tau),
        "t_obs": int(t_obs),
        "t_target": int(t_target),
        "pred_norm": pred_norm,
        "target_norm": target_norm,
        "error_norm": float(error_norm),
        "sq_error_norm": float(error_norm**2),
        "abs_error_norm": float(abs(error_norm)),
    }
    if predict_time_sec is not None:
        row["predict_time_sec"] = float(predict_time_sec)
    if extra_fields:
        row.update(extra_fields)
    return row


def target_norm_from_raw(query_bundle: dict[str, Any], meta: dict[str, Any], row_id: int, t_target: int) -> float:
    out_std = max(float(meta["out_std"]), 1e-6)
    target_raw = float(query_bundle["y_raw"][int(row_id), int(t_target)])
    target_norm_unclipped = float((target_raw - float(meta["out_mean"])) / out_std)
    return float(np.clip(target_norm_unclipped, -common.TARGET_NORM_CLIP, common.TARGET_NORM_CLIP))


def make_raw_prediction_record(
    *,
    method_name: str,
    method_family: str,
    run_id: str,
    query_bundle: dict[str, Any],
    meta: dict[str, Any],
    task_name: str,
    row_id: int,
    query_id: int,
    current_t: int,
    t_target: int,
    pred_norm: float,
    predict_time_sec: float | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _base_prediction_record(
        method_name=method_name,
        method_family=method_family,
        run_id=run_id,
        domain=str(meta["domain"]),
        dataset_id=int(meta["dataset_id"]),
        global_dataset_id=int(meta["global_dataset_id"]),
        source_file=str(meta["source_file"]),
        row_id=int(row_id),
        query_id=int(query_id),
        gamma=meta.get("gamma", np.nan),
        support_size=int(meta["support_size"]),
        task_name=task_name,
        t_obs=int(current_t),
        t_target=int(t_target),
        pred_norm=float(pred_norm),
        target_norm=target_norm_from_raw(query_bundle, meta, row_id, t_target),
        predict_time_sec=predict_time_sec,
        extra_fields=extra_fields,
    )


def ready_map_meta(ready_map: dict[str, Any]) -> dict[str, Any]:
    ready_file_value = (
        ready_map["_ready_file_basename"]
        if "_ready_file_basename" in ready_map
        else _required(ready_map, "ready_file")
    )
    ready_file_basename = str(ready_file_value)
    source_file = str(_required(ready_map, "source_file"))
    return {
        "domain": str(_required(ready_map, "domain")).lower(),
        "dataset_id": int(_required(ready_map, "dataset_id")),
        "global_dataset_id": int(_required(ready_map, "global_dataset_id")),
        "source_file": source_file,
        "ready_file": ready_file_basename,
        "gamma": ready_map.get("gamma", np.nan),
        "support_size": int(_required(ready_map, "support_size")),
    }


def make_ready_prediction_record(
    *,
    method_name: str,
    method_family: str,
    run_id: str,
    ready_map: dict[str, Any],
    task_name: str,
    row_id: int,
    query_id: int,
    pred_norm: float,
    target_norm: float,
    t_obs: int,
    tau: int,
    t_target: int,
    predict_time_sec: float | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = ready_map_meta(ready_map)
    t_obs = int(t_obs)
    t_target = int(t_target)
    tau = int(max(1, tau))
    if t_target <= t_obs:
        raise ValueError(f"Expected t_target > t_obs, got t_obs={t_obs}, t_target={t_target}.")
    if tau != t_target - t_obs:
        raise ValueError(f"Expected tau={t_target - t_obs} from timestamps, got tau={tau}.")
    fields = {"ready_file": meta["ready_file"], "task": str(task_name)}
    if extra_fields:
        fields.update(extra_fields)
    return _base_prediction_record(
        method_name=method_name,
        method_family=method_family,
        run_id=run_id,
        domain=meta["domain"],
        dataset_id=meta["dataset_id"],
        global_dataset_id=meta["global_dataset_id"],
        source_file=meta["source_file"],
        row_id=int(row_id),
        query_id=int(query_id),
        gamma=meta["gamma"],
        support_size=int(meta["support_size"]),
        task_name=task_name,
        t_obs=t_obs,
        t_target=t_target,
        pred_norm=float(pred_norm),
        target_norm=float(target_norm),
        predict_time_sec=predict_time_sec,
        extra_fields=fields,
    )
