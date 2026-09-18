from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from clpfn.evaluation.core import benchmark as common
from clpfn.evaluation.core import inputs as eval_inputs
from clpfn.evaluation.core import outputs as eval_outputs
from clpfn.evaluation.core import records as eval_records
from clpfn.evaluation.core import reporting
from clpfn.evaluation.core import tasks as eval_tasks
from clpfn.evaluation.ready_baselines.models import (
    FittedReadyBaseline,
    fit_ready_baseline,
    predict_model_norm_paths,
    spec_from_config,
)


LOGGER = logging.getLogger(__name__)
READY_FORMAT_VERSION = "causal_long_pfn_ready_static_normalized"
DEFAULT_OUTPUT_ROOT = Path("outputs/eval")


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_task_name(task_name: str) -> str:
    return "one_step_exhaustive" if "one_step" in str(task_name).lower() else "sequential_rollout"


def _model_to_eval_norm(values: np.ndarray | float, support_context: dict[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    model_mean = float(support_context["out_mean"])
    model_std = max(float(support_context["out_std"]), 1e-6)
    eval_mean = float(support_context["eval_out_mean"])
    eval_std = max(float(support_context["eval_out_std"]), 1e-6)
    return (values * model_std + model_mean - eval_mean) / eval_std


def _model_to_raw(value: float, support_context: dict[str, Any]) -> float:
    return float(value * max(float(support_context["out_std"]), 1e-6) + float(support_context["out_mean"]))


def _empty_checkpoint_fields() -> dict[str, Any]:
    return {
        "checkpoint_basename": "",
        "checkpoint_step_count": None,
        "trainable_parameters": 0,
        "pfn_target_refit_time_sec": None,
    }


def _fit_fields(
    fitted: FittedReadyBaseline,
    *,
    fit_id: str,
    fit_time_sec: float,
) -> dict[str, Any]:
    diag = fitted.fit_diagnostics
    return {
        "fit_id": fit_id,
        "fit_task_scope": "support_context_all_valid_transitions",
        "fit_time_sec": float(fit_time_sec),
        "fit_status": str(diag["fit_status"]),
        "fit_samples": int(diag["fit_samples"]),
        "fit_features": int(diag["fit_features"]),
        "ready_baseline_kind": fitted.spec.kind,
        "ar_order": 1 if fitted.spec.kind == "ar" else None,
    }


def _task_identity(task: dict[str, Any], idx: int) -> dict[str, Any]:
    required = (
        "patient_id",
        "patient_uid",
        "origin_uid",
        "plan_uid",
        "query_uid",
        "planned_action_sequence",
    )
    missing = [key for key in required if key not in task]
    if missing:
        raise KeyError(f"Ready task is missing stable identity fields: {missing}")
    plan = np.asarray(task["planned_action_sequence"][idx], dtype=np.int64).reshape(-1)
    return {
        "patient_id": int(task["patient_id"][idx]),
        "patient_uid": str(task["patient_uid"][idx]),
        "origin_uid": str(task["origin_uid"][idx]),
        "plan_uid": str(task["plan_uid"][idx]),
        "query_uid": str(task["query_uid"][idx]),
        "plan": plan,
        "planned_action_sequence": json.dumps(plan.tolist()),
    }


def _target_at_horizon(
    task: dict[str, Any],
    support_context: dict[str, Any],
    idx: int,
    horizon: int,
    output_task_name: str,
) -> tuple[float, float, float, float]:
    step_idx = int(horizon) - 1
    if output_task_name == "sequential_rollout":
        required = (
            "target_path_raw",
            "target_path_model_norm",
            "target_path_eval_norm_unclipped",
            "target_path_eval_norm_reported",
        )
        missing = [key for key in required if key not in task]
        if missing:
            raise KeyError(f"Sequential ready task is missing target-path fields: {missing}")
        return (
            float(task["target_path_raw"][idx, step_idx]),
            float(task["target_path_model_norm"][idx, step_idx]),
            float(task["target_path_eval_norm_unclipped"][idx, step_idx]),
            float(task["target_path_eval_norm_reported"][idx, step_idx]),
        )
    target_raw = float(task["target_raw"][idx])
    target_model = float(task["target_model_norm"][idx])
    target_eval_unclipped = float(
        (target_raw - float(support_context["eval_out_mean"]))
        / max(float(support_context["eval_out_std"]), 1e-6)
    )
    target_eval_reported = float(task["target_eval_norm"][idx])
    return target_raw, target_model, target_eval_unclipped, target_eval_reported


def evaluate_ready_task(
    *,
    fitted: FittedReadyBaseline,
    ready_map: dict[str, Any],
    task_name: str,
    run_id: str,
    evaluation_config_hash: str,
    fit_id: str,
    fit_time_sec: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task = ready_map["tasks"][task_name]
    support_context = ready_map["support_context"]
    n_eval = eval_tasks.ready_task_n_eval(task)
    if n_eval <= 0:
        return [], []

    started = time.perf_counter()
    model_paths = predict_model_norm_paths(fitted, task)
    predict_elapsed = float(time.perf_counter() - started)
    if model_paths.shape[0] != n_eval:
        raise RuntimeError(f"Prediction row count mismatch: expected {n_eval}, got {model_paths.shape[0]}.")

    output_task_name = _output_task_name(task_name)
    rows_all = eval_tasks.ready_task_row_ids(task)
    t_obs_all = np.asarray(task["t_obs"], dtype=np.int64)
    tau_all = np.asarray(task["tau"], dtype=np.int64)
    hardware_fields = reporting.hardware_report_fields("cpu")
    tuning_fields = reporting.no_tuning_fields()
    checkpoint_fields = _empty_checkpoint_fields()
    fit_fields = _fit_fields(fitted, fit_id=fit_id, fit_time_sec=fit_time_sec)
    ready_hash = str(ready_map["_ready_file_hash"])
    raw_hash = str(ready_map["_raw_file_hash"])
    total_output_rows = int(np.maximum(tau_all, 1).sum())
    predict_time_per_output = predict_elapsed / max(total_output_rows, 1)

    prediction_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []

    for idx in range(n_eval):
        identity = _task_identity(task, idx)
        current_t = int(t_obs_all[idx])
        final_horizon = int(max(1, tau_all[idx]))
        available_horizons = range(1, final_horizon + 1) if output_task_name == "sequential_rollout" else range(1, 2)
        if final_horizon > model_paths.shape[1]:
            raise RuntimeError(
                f"Method {fitted.spec.method_name} produced only {model_paths.shape[1]} horizons "
                f"for a task requiring {final_horizon}."
            )

        current_y_raw = float(task["current_y_raw"][idx])
        current_y_model = float(task["current_y_model_norm"][idx])
        current_y_eval_unclipped = float(task["current_y_eval_norm_unclipped"][idx])

        for horizon in available_horizons:
            step_idx = int(horizon) - 1
            pred_model_unclipped = float(model_paths[idx, step_idx])
            pred_eval_unclipped = float(_model_to_eval_norm(pred_model_unclipped, support_context))
            pred_raw_unclipped = _model_to_raw(pred_model_unclipped, support_context)
            target_raw, target_model, target_eval_unclipped, target_eval_reported = _target_at_horizon(
                task,
                support_context,
                idx,
                horizon,
                output_task_name,
            )
            target_time = current_t + int(horizon)
            action = int(identity["plan"][step_idx]) if step_idx < len(identity["plan"]) else None
            reported_task = reporting.reported_task(output_task_name, horizon)

            shared_fields = {
                "reported_task": reported_task,
                "current_y_raw": current_y_raw,
                "current_y_model_norm": current_y_model,
                "current_y_eval_norm_unclipped": current_y_eval_unclipped,
                "prediction_raw": pred_raw_unclipped,
                "prediction_raw_unclipped": pred_raw_unclipped,
                "prediction_model_norm": pred_model_unclipped,
                "prediction_model_norm_unclipped": pred_model_unclipped,
                "prediction_eval_norm_unclipped": pred_eval_unclipped,
                "prediction_eval_norm": pred_eval_unclipped,
                "target_raw": target_raw,
                "target_model_norm": target_model,
                "target_eval_norm_unclipped": target_eval_unclipped,
                "target_eval_norm": target_eval_reported,
            }

            rollout_mode = (
                "last_value_constant"
                if fitted.spec.kind == "persistence"
                else "recursive_one_step_ar"
            )
            rollout_rows.append(
                {
                    "evaluation_id": run_id,
                    "method": fitted.spec.method_name,
                    "method_family": fitted.spec.family,
                    "dataset_uid": str(ready_map["dataset_uid"]),
                    "patient_id": identity["patient_id"],
                    "patient_uid": identity["patient_uid"],
                    "origin_uid": identity["origin_uid"],
                    "plan_uid": identity["plan_uid"],
                    "query_uid": identity["query_uid"],
                    "task_name": output_task_name,
                    "reported_task": reported_task,
                    "horizon": int(horizon),
                    "observation_time": current_t,
                    "target_time": target_time,
                    "action_at_rollout_step": action,
                    "planned_action_sequence": identity["planned_action_sequence"],
                    **shared_fields,
                    "squared_error": float((pred_eval_unclipped - target_eval_reported) ** 2),
                    "absolute_error": float(abs(pred_eval_unclipped - target_eval_reported)),
                    "rollout_mode": rollout_mode,
                    "schema_version": 1,
                    "fit_id": fit_id,
                    "pfn_checkpoint_id": "",
                    "pfn_checkpoint_hash": "",
                    "pretraining_seed": None,
                    "model_variant": fitted.spec.method_name,
                    "prior_variant": "",
                    "evaluation_seed": int(common.SEED),
                    "raw_dataset_hash": raw_hash,
                    "ready_dataset_hash": ready_hash,
                    "evaluation_config_hash": evaluation_config_hash,
                    "number_rollout_steps": final_horizon,
                    "batch_size": n_eval,
                    "support_size": int(ready_map["support_size"]),
                    "peak_batch_gpu_memory": 0,
                }
            )

            row = eval_records.make_ready_prediction_record(
                method_name=fitted.spec.method_name,
                method_family=fitted.spec.family,
                run_id=run_id,
                ready_map=ready_map,
                task_name=output_task_name,
                row_id=int(rows_all[idx]),
                query_id=idx,
                pred_norm=pred_eval_unclipped,
                target_norm=target_eval_reported,
                t_obs=current_t,
                tau=int(horizon),
                t_target=target_time,
                predict_time_sec=predict_time_per_output,
                extra_fields={
                    "evaluation_id": run_id,
                    "dataset_uid": str(ready_map["dataset_uid"]),
                    "patient_id": identity["patient_id"],
                    "patient_uid": identity["patient_uid"],
                    "origin_uid": identity["origin_uid"],
                    "plan_uid": identity["plan_uid"],
                    "query_uid": identity["query_uid"],
                    "planned_action_sequence": identity["planned_action_sequence"],
                    "action_at_rollout_step": action,
                    "raw_dataset_hash": raw_hash,
                    "ready_dataset_hash": ready_hash,
                    "evaluation_config_hash": evaluation_config_hash,
                    "schema_version": 1,
                    **shared_fields,
                },
            )
            row.update(reporting.task_identity_fields(ready_map, output_task_name, horizon))
            row.update(tuning_fields)
            row.update(checkpoint_fields)
            row.update(hardware_fields)
            row.update(fit_fields)
            prediction_rows.append(row)

    reporting.finalize_task_timing(prediction_rows, refit_time_sec=fit_time_sec)
    return prediction_rows, rollout_rows


def _prepare_ready_map(ready_file: str | Path) -> dict[str, Any]:
    ready_map = eval_inputs.load_pickle(ready_file)
    ready_map["_ready_file_basename"] = os.path.basename(str(ready_file))
    ready_map["_ready_file_path"] = str(ready_file)
    ready_map["_ready_file_hash"] = _file_sha256(ready_file)
    source = Path(str(ready_map["source_file"]))
    if not source.is_file():
        raise FileNotFoundError(f"Ready source_file does not exist: {source}")
    ready_map["_raw_file_hash"] = _file_sha256(source)
    return ready_map


def run_all(
    method: str,
    *,
    ready_dirs: tuple[str, ...] | list[str] | None = None,
    ready_paths: tuple[str, ...] | list[str] | None = None,
    wanted_domains: tuple[str, ...] = common.WANTED_DOMAINS,
    output_dir: str | Path | None = None,
    baseline_config: dict[str, Any] | None = None,
    evaluation_id: str | None = None,
) -> dict[str, Any]:
    spec = spec_from_config(method, baseline_config)
    output_paths = eval_outputs.prepare_output_paths(output_dir or (DEFAULT_OUTPUT_ROOT / spec.method_name))
    ready_files = eval_inputs.find_ready_pickles(
        eval_inputs.ReadyBenchmarkInputs(
            ready_dirs=tuple(str(path) for path in (ready_dirs or ())),
            ready_paths=tuple(str(path) for path in (ready_paths or ())),
        )
    )
    file_hashes = {str(Path(path).resolve()): _file_sha256(path) for path in ready_files}
    config_payload = {
        "method": spec.method_name,
        "spec": {key: getattr(spec, key) for key in spec.__dataclass_fields__},
        "wanted_domains": list(wanted_domains),
        "ready_files": file_hashes,
    }
    evaluation_config_hash = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    run_id = evaluation_id or f"{spec.method_name}_{evaluation_config_hash[:16]}"

    LOGGER.info(
        "Starting %s | method=%s | device=cpu | ready_files=%s | output=%s",
        spec.title,
        spec.method_name,
        len(ready_files),
        output_paths.output_dir,
    )
    LOGGER.info("Evaluation ID: %s", run_id)

    wanted = {str(domain).lower() for domain in wanted_domains}
    skipped: list[dict[str, Any]] = []
    n_files_used = 0
    started_all = time.time()

    for file_idx, ready_file in enumerate(ready_files):
        LOGGER.info("[%s/%s] %s", file_idx + 1, len(ready_files), os.path.basename(ready_file))
        ready_map = _prepare_ready_map(ready_file)
        if ready_map["ready_format_version"] != READY_FORMAT_VERSION:
            skipped.append({"ready_file": str(ready_file), "reason": "wrong_ready_format_version"})
            continue
        domain = str(ready_map["domain"]).lower()
        if domain not in wanted:
            skipped.append({"ready_file": str(ready_file), "reason": f"unwanted_domain_{domain}"})
            continue

        fit_started = time.perf_counter()
        fitted = fit_ready_baseline(spec, ready_map["support_context"])
        fit_time_sec = float(time.perf_counter() - fit_started)
        fit_id = hashlib.sha256(
            json.dumps(
                {
                    "method": spec.method_name,
                    "dataset_uid": str(ready_map["dataset_uid"]),
                    "ready_hash": ready_map["_ready_file_hash"],
                    "spec": {key: getattr(spec, key) for key in spec.__dataclass_fields__},
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:24]
        n_files_used += 1
        LOGGER.info(
            "domain=%s dataset_id=%s support=%s fit_status=%s samples=%s features=%s fit_sec=%.3f",
            domain,
            ready_map["dataset_id"],
            ready_map["support_size"],
            fitted.fit_diagnostics["fit_status"],
            fitted.fit_diagnostics["fit_samples"],
            fitted.fit_diagnostics["fit_features"],
            fit_time_sec,
        )

        for task_name in eval_tasks.ready_task_names(ready_map):
            task = ready_map["tasks"][task_name]
            if eval_tasks.ready_task_n_eval(task) <= 0:
                continue
            output_task_name = _output_task_name(task_name)
            if eval_outputs.partition_is_complete(
                output_paths,
                evaluation_id=run_id,
                method=spec.method_name,
                dataset_uid=str(ready_map["dataset_uid"]),
                task_name=output_task_name,
            ):
                LOGGER.info("task=%s skipped: completed partition exists", task_name)
                continue
            prediction_rows, rollout_rows = evaluate_ready_task(
                fitted=fitted,
                ready_map=ready_map,
                task_name=task_name,
                run_id=run_id,
                evaluation_config_hash=evaluation_config_hash,
                fit_id=fit_id,
                fit_time_sec=fit_time_sec,
            )
            expected = int(np.maximum(np.asarray(task["tau"], dtype=np.int64), 1).sum())
            if output_task_name == "one_step_exhaustive":
                expected = int(eval_tasks.ready_task_n_eval(task))
            if len(prediction_rows) != expected or len(rollout_rows) != expected:
                raise RuntimeError(
                    f"Unexpected output row count for {task_name}: expected {expected}, "
                    f"predictions={len(prediction_rows)}, rollout={len(rollout_rows)}."
                )
            eval_outputs.write_completed_partition(
                output_paths,
                evaluation_id=run_id,
                method=spec.method_name,
                dataset_uid=str(ready_map["dataset_uid"]),
                task_name=output_task_name,
                rollout_steps=pd.DataFrame(rollout_rows),
                prediction_rows=pd.DataFrame(prediction_rows),
            )
            LOGGER.info("task=%s prediction_rows=%s", task_name, len(prediction_rows))

        del ready_map, fitted
        gc.collect()

    prediction_df = eval_outputs.collect_completed_partitions(
        output_paths,
        "prediction_rows.parquet",
        evaluation_id=run_id,
    )
    if prediction_df.empty:
        raise RuntimeError("No evaluation rows were produced. Check ready paths and wanted domains.")
    summaries = eval_outputs.write_prediction_summaries(prediction_df, paths=output_paths)
    elapsed = float(time.time() - started_all)
    summaries.update(
        {
            "method": spec.method_name,
            "method_family": spec.family,
            "run_id": run_id,
            "evaluation_id": run_id,
            "evaluation_config_hash": evaluation_config_hash,
            "output_dir": str(output_paths.output_dir),
            "device": "cpu",
            "seed": int(spec.random_seed),
            "n_ready_files_found": int(len(ready_files)),
            "n_ready_files_used": int(n_files_used),
            "n_skipped": int(len(skipped)),
            "skipped": skipped,
            "baseline_spec": config_payload["spec"],
            "elapsed_min": elapsed / 60.0,
            "metric": "normalized_rmse",
        }
    )
    LOGGER.info(
        "Finished %s | prediction_rows=%s | elapsed_min=%.2f",
        spec.title,
        len(prediction_df),
        elapsed / 60.0,
    )
    return summaries
