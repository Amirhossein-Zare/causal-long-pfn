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
import torch
from scipy.stats import spearmanr

from clpfn.config.defaults import N_SUPPORT_ANCHORS
from clpfn.evaluation.core import inputs as eval_inputs
from clpfn.evaluation.core import outputs as eval_outputs
from clpfn.evaluation.core import reporting
from clpfn.evaluation.core.summaries import print_summary_table
from clpfn.evaluation.pfn import calibration as cal
from clpfn.evaluation.pfn.calibration_summaries import summarize_domain_calibration
from clpfn.evaluation.core import benchmark as common
from clpfn.evaluation.core import records as eval_records
from clpfn.evaluation.core import tasks as eval_tasks
from clpfn.evaluation.pfn.batches import (
    collate_ready_batch,
    resolve_n_support_anchors,
    move_batch_to_device,
)
from clpfn.models.causal_long_pfn import (
    predictive_mean_from_gmm,
)
from clpfn.training.checkpointing import (
    load_causal_long_pfn_checkpoint,
    resolve_checkpoint_path,
)

LOGGER = logging.getLogger(__name__)


READY_FORMAT_VERSION = "causal_long_pfn_ready_static_normalized"

PFN_BATCH_SIZE = 32
WANTED_DOMAINS = common.WANTED_DOMAINS
DEFAULT_OUTPUT_DIR = Path("outputs/eval/causal_long_pfn")
CALIBRATION_SUMMARY_FILENAME = "calibration_summary_domain.csv"
CALIBRATION_ROWS_FILENAME = "calibration_rows.parquet"


def _calibration_rows(prediction_df: pd.DataFrame) -> pd.DataFrame:
    if prediction_df.empty or "calibration_available" not in prediction_df.columns:
        return pd.DataFrame()
    return prediction_df[prediction_df["calibration_available"] == True].copy()


def _write_calibration_summary(
    calibration_rows_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    output_paths: eval_outputs.EvaluationOutputPaths,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    calibration_df = _calibration_rows(calibration_rows_df)
    calibration_summary = summarize_domain_calibration(calibration_df, prediction_df)
    calibration_rows_parquet = output_paths.output_dir / CALIBRATION_ROWS_FILENAME
    calibration_summary_csv = output_paths.output_dir / CALIBRATION_SUMMARY_FILENAME
    calibration_df.to_parquet(calibration_rows_parquet, index=False)
    calibration_summary.to_csv(calibration_summary_csv, index=False)
    print_summary_table(calibration_summary, "One-step calibration by domain")

    return calibration_df, {
        "one_step_calibration_rows": calibration_df,
        "calibration_summary_domain": calibration_summary,
        "calibration_rows_parquet": str(calibration_rows_parquet),
        "calibration_summary_csv": str(calibration_summary_csv),
    }


def _cuda_mem_string() -> str:
    if not torch.cuda.is_available():
        return "CUDA not available"

    device_idx = torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(device_idx) / 1024**3
    reserved = torch.cuda.memory_reserved(device_idx) / 1024**3
    max_alloc = torch.cuda.max_memory_allocated(device_idx) / 1024**3
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
    free = free_bytes / 1024**3
    total = total_bytes / 1024**3

    return (
        f"allocated={alloc:.3f} GiB | reserved={reserved:.3f} GiB | "
        f"max_alloc={max_alloc:.3f} GiB | free={free:.3f}/{total:.3f} GiB"
    )


def _spearman_corr(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3 or np.std(x[mask]) < 1e-12 or np.std(y[mask]) < 1e-12:
        return float("nan")

    return float(spearmanr(x[mask], y[mask]).statistic)


def _metrics_from_arrays(pred_norm, target_norm) -> dict[str, Any]:
    pred_norm = np.asarray(pred_norm, dtype=np.float64)
    target_norm = np.asarray(target_norm, dtype=np.float64)
    mask = np.isfinite(pred_norm) & np.isfinite(target_norm)

    out = {
        "n_test_examples": int(mask.sum()),
        "normalized_rmse": float("nan"),
        "normalized_mae": float("nan"),
        "normalized_nmse": float("nan"),
        "normalized_bias": float("nan"),
        "normalized_srcc": float("nan"),
        "pred_norm_mean": float("nan"),
        "target_norm_mean": float("nan"),
        "pred_norm_std": float("nan"),
        "target_norm_std": float("nan"),
    }

    if mask.any():
        pred = pred_norm[mask]
        target = target_norm[mask]
        error = pred - target
        mse = float(np.mean(error**2))
        var_target = float(np.var(target))
        out.update({
            "normalized_rmse": float(np.sqrt(mse)),
            "normalized_mae": float(np.mean(np.abs(error))),
            "normalized_nmse": float(mse / max(var_target, 1e-8)),
            "normalized_bias": float(abs(float(np.mean(pred)) - float(np.mean(target)))),
            "normalized_srcc": _spearman_corr(pred, target),
            "pred_norm_mean": float(np.mean(pred)),
            "target_norm_mean": float(np.mean(target)),
            "pred_norm_std": float(np.std(pred)),
            "target_norm_std": float(np.std(target)),
        })

    return out


def _model_to_eval_norm(values, support_context: dict[str, Any]):
    values = np.asarray(values, dtype=np.float64)
    model_mean = float(support_context["out_mean"])
    model_std = max(float(support_context["out_std"]), 1e-6)
    eval_mean = float(support_context["eval_out_mean"])
    eval_std = max(float(support_context["eval_out_std"]), 1e-6)
    return (values * model_std + model_mean - eval_mean) / eval_std


def _model_sigma_to_eval_norm(sigma, support_context: dict[str, Any]):
    sigma = np.asarray(sigma, dtype=np.float64)
    model_std = max(float(support_context["out_std"]), 1e-6)
    eval_std = max(float(support_context["eval_out_std"]), 1e-6)
    return sigma * (model_std / eval_std)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_task_name(task_name: str) -> str:
    return "one_step_exhaustive" if "one_step" in task_name else "sequential_rollout"


def find_pfn_checkpoint(checkpoint_path: str | Path | None) -> str:
    if checkpoint_path is None:
        raise ValueError("Provide a checkpoint path.")

    return str(resolve_checkpoint_path(checkpoint_path))


@torch.no_grad()
def evaluate_ready_task(
    model,
    device,
    ready_map,
    ckpt_diag: dict[str, Any],
    task_name: str,
    run_id: str,
    batch_size: int = PFN_BATCH_SIZE,
    report_calibration: bool = True,
    evaluation_id: str | None = None,
    evaluation_config_hash: str | None = None,
    n_support_anchors: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    task = ready_map["tasks"][task_name]
    support_context = ready_map["support_context"]

    n_anchors_used = resolve_n_support_anchors(n_support_anchors)

    n_eval = eval_tasks.ready_task_n_eval(task)

    if n_eval <= 0:
        return _metrics_from_arrays([], []), [], [], [], []

    preds_norm = []
    targets_norm = []
    prediction_rows = []
    rollout_rows = []
    gmm_rows = []

    support_size = int(ready_map["support_size"])
    rows_all = eval_tasks.ready_task_row_ids(task)
    hardware_fields = reporting.hardware_report_fields(device)
    checkpoint_fields = reporting.pfn_checkpoint_fields(ckpt_diag)
    tuning_fields = reporting.no_tuning_fields()

    for start in range(0, n_eval, batch_size):
        end = min(start + batch_size, n_eval)
        batch = collate_ready_batch(ready_map, task_name, start, end, n_support_anchors=n_anchors_used)
        current_batch_size = end - start

        batch = move_batch_to_device(batch, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            log_pi, mu, sigma, trace = model.rollout(
                batch,
                return_trace=True,
                rollout_mode="mean",
                feedback_clip=common.OUTCOME_CLIP_TRAIN,
            )
            pred_norm = predictive_mean_from_gmm(log_pi, mu)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        predict_time_sec = float((time.perf_counter() - started) / max(1, current_batch_size))

        pn_model_unclipped = pred_norm.detach().float().cpu().numpy()
        pn = _model_to_eval_norm(pn_model_unclipped, support_context)
        tn = batch["target_eval_norm"].detach().float().cpu().numpy()

        t_obs_np = batch["t_obs"].detach().cpu().numpy().astype(np.int64)
        t_target_np = batch["t_target"].detach().cpu().numpy().astype(np.int64)
        tau_np = batch["tau"].detach().cpu().numpy().astype(np.int64)

        row_ids = rows_all[start:end]
        trace_log_pi = trace["log_pi"].detach().float().cpu().numpy()
        trace_mu = trace["mu"].detach().float().cpu().numpy()
        trace_sigma = trace["sigma"].detach().float().cpu().numpy()
        trace_mean = trace["mixture_mean"].detach().float().cpu().numpy()
        trace_var = trace["mixture_variance"].detach().float().cpu().numpy()
        trace_feedback = trace["feedback_outcome"].detach().float().cpu().numpy()
        n_trace_steps = int(trace_mean.shape[1])
        output_task_name = _output_task_name(task_name)

        one_step_mask = np.asarray(
            [eval_tasks.is_one_step_task(task_name, tau_np[i]) for i in range(current_batch_size)],
            dtype=bool,
        )

        calibration = None

        if report_calibration and bool(one_step_mask.any()):
            log_pi_np = log_pi.detach().float().cpu().numpy()
            mu_model_np = mu.detach().float().cpu().numpy()
            sigma_model_np = sigma.detach().float().cpu().numpy()
            mu_np = _model_to_eval_norm(mu_model_np, support_context)
            sigma_np = _model_sigma_to_eval_norm(sigma_model_np, support_context)

            calibration = cal.compute_one_step_gmm_calibration_np(
                log_pi_np=log_pi_np,
                mu_np=mu_np,
                sigma_np=sigma_np,
                target_norm_np=tn,
            )

        preds_norm.extend(pn.tolist())
        targets_norm.extend(tn.tolist())

        for batch_idx in range(current_batch_size):
            global_idx = start + batch_idx
            plan = np.asarray(task["planned_action_sequence"][global_idx], dtype=np.int64)
            horizons = range(1, n_trace_steps + 1) if output_task_name == "sequential_rollout" else range(1, 2)
            for horizon in horizons:
                step_idx = horizon - 1
                pred_model_unclipped = float(trace_mean[batch_idx, step_idx])
                pred_eval_unclipped = float(_model_to_eval_norm(np.asarray([pred_model_unclipped]), support_context)[0])
                if horizon == int(tau_np[batch_idx]) and pred_eval_unclipped != float(pn[batch_idx]):
                    raise RuntimeError("Stored PFN rollout endpoint differs from the existing final evaluation prediction.")
                if output_task_name == "sequential_rollout":
                    target_raw = float(task["target_path_raw"][global_idx, step_idx])
                    target_model = float(task["target_path_model_norm"][global_idx, step_idx])
                    target_eval_unclipped = float(task["target_path_eval_norm_unclipped"][global_idx, step_idx])
                    target_eval_reported = float(task["target_path_eval_norm_reported"][global_idx, step_idx])
                else:
                    target_raw = float(task["target_raw"][global_idx])
                    target_model = float(task["target_model_norm"][global_idx])
                    target_eval_unclipped = float((target_raw - support_context["eval_out_mean"]) / max(float(support_context["eval_out_std"]), 1e-6))
                    target_eval_reported = float(task["target_eval_norm"][global_idx])
                target_time = int(t_obs_np[batch_idx] + horizon)
                current_y_raw = float(task["current_y_raw"][global_idx])
                current_y_model = float(task["current_y_model_norm"][global_idx])
                current_y_eval_unclipped = float(task["current_y_eval_norm_unclipped"][global_idx])
                pred_raw_unclipped = float(pred_model_unclipped * float(support_context["out_std"]) + float(support_context["out_mean"]))
                common_fields = {
                    "evaluation_id": evaluation_id or run_id,
                    "dataset_uid": str(ready_map["dataset_uid"]),
                    "patient_id": int(task["patient_id"][global_idx]), "patient_uid": str(task["patient_uid"][global_idx]), "origin_uid": str(task["origin_uid"][global_idx]),
                    "plan_uid": str(task["plan_uid"][global_idx]), "query_uid": str(task["query_uid"][global_idx]),
                    "planned_action_sequence": json.dumps(plan.tolist()),
                    "action_at_rollout_step": int(plan[step_idx]) if step_idx < len(plan) else None,
                    "prediction_raw": pred_raw_unclipped, "prediction_raw_unclipped": pred_raw_unclipped,
                    "prediction_model_norm": pred_model_unclipped, "prediction_model_norm_unclipped": pred_model_unclipped,
                    "prediction_eval_norm_unclipped": pred_eval_unclipped, "prediction_eval_norm": pred_eval_unclipped,
                    "current_y_raw": current_y_raw, "current_y_model_norm": current_y_model,
                    "current_y_eval_norm_unclipped": current_y_eval_unclipped, "target_raw": target_raw,
                    "target_model_norm": target_model, "target_eval_norm_unclipped": target_eval_unclipped,
                    "target_eval_norm": target_eval_reported, "squared_error": float((pred_eval_unclipped - target_eval_reported) ** 2),
                    "absolute_error": float(abs(pred_eval_unclipped - target_eval_reported)),
                    "rollout_mode": "mean",
                    "schema_version": 1, "evaluation_config_hash": evaluation_config_hash or "",
                    "pfn_checkpoint_id": ckpt_diag["checkpoint_basename"], "pfn_checkpoint_hash": ckpt_diag["checkpoint_file_sha256"],
                    "pretraining_seed": ckpt_diag["pretraining_seed"], "model_variant": ckpt_diag["model_variant"], "prior_variant": ckpt_diag["prior_variant"],
                    "evaluation_seed": int(common.SEED), "fit_id": "", "number_rollout_steps": n_trace_steps,
                    "batch_size": current_batch_size, "support_size": support_size, "n_support_anchors": int(n_anchors_used),
                    "n_pfn_layers_used": int(model.n_pfn_layers),
                    "raw_dataset_hash": ready_map["_raw_file_hash"], "ready_dataset_hash": ready_map["_ready_file_hash"],
                    "peak_batch_gpu_memory": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                }
                rollout_rows.append({"method": "causal_long_pfn", "method_family": "PFN", "task_name": output_task_name, "reported_task": reporting.reported_task(output_task_name, horizon), "horizon": horizon, "observation_time": int(t_obs_np[batch_idx]), "target_time": target_time, **common_fields})
                for component_idx in range(trace_mu.shape[-1]):
                    gmm_rows.append({"evaluation_id": evaluation_id or run_id, "dataset_uid": str(ready_map["dataset_uid"]), "query_uid": str(task["query_uid"][global_idx]), "task_name": output_task_name, "horizon": horizon, "component_index": component_idx, "mixture_weight": float(np.exp(trace_log_pi[batch_idx, step_idx, component_idx])), "component_mean": float(trace_mu[batch_idx, step_idx, component_idx]), "component_std": float(trace_sigma[batch_idx, step_idx, component_idx]), "mixture_mean": pred_model_unclipped, "mixture_variance": float(trace_var[batch_idx, step_idx]), "feedback_outcome": float(trace_feedback[batch_idx, step_idx]), "conditional_path": "deterministic_self_fed_mean", "schema_version": 1})
                row = eval_records.make_ready_prediction_record(method_name="causal_long_pfn", method_family="PFN", ready_map={**ready_map, "support_size": support_size}, task_name=output_task_name, run_id=run_id, row_id=row_ids[batch_idx], query_id=global_idx, pred_norm=pred_eval_unclipped, target_norm=target_eval_reported, t_obs=t_obs_np[batch_idx], tau=horizon, t_target=target_time, predict_time_sec=predict_time_sec / max(n_trace_steps, 1), extra_fields=common_fields)
                if report_calibration:
                    row = cal.add_empty_calibration_fields(row)
                    if output_task_name == "one_step_exhaustive" and calibration is not None:
                        row = cal.add_calibration_fields(row, batch_idx, calibration)
                row.update(reporting.task_identity_fields(ready_map, output_task_name, horizon))
                row.update(tuning_fields)
                row.update(checkpoint_fields)
                row.update(hardware_fields)
                prediction_rows.append(row)

        del batch, log_pi, mu, sigma, pred_norm

    metrics = _metrics_from_arrays(preds_norm, targets_norm)
    reporting.finalize_task_timing(prediction_rows, refit_time_sec=0.0)
    calibration_rows = [
        row.copy()
        for row in prediction_rows
        if bool(row.get("calibration_available", False))
    ]
    return metrics, prediction_rows, calibration_rows, rollout_rows, gmm_rows


def _prepare_ready_map(ready_file: str | Path) -> dict[str, Any]:
    ready_map = eval_inputs.load_pickle(ready_file)
    ready_map["_ready_file_basename"] = os.path.basename(str(ready_file))
    ready_map["_ready_file_path"] = str(ready_file)
    ready_map["_ready_file_hash"] = _file_sha256(ready_file)
    source = Path(str(ready_map["source_file"]))
    if not source.is_file():
        raise FileNotFoundError(f"Ready dataset source file not found: {source}")
    ready_map["_raw_file_hash"] = _file_sha256(source)
    return ready_map


@torch.no_grad()
def evaluate_ready_files(
    model,
    device,
    ready_files: list[str],
    ckpt_diag: dict[str, Any],
    run_id: str,
    *,
    output_paths: eval_outputs.EvaluationOutputPaths,
    batch_size: int = PFN_BATCH_SIZE,
    wanted_domains=WANTED_DOMAINS,
    report_calibration: bool = True,
    evaluation_config_hash: str = "",
    n_support_anchors: int | None = None,
) -> dict[str, Any]:
    LOGGER.info(
        "Starting CausalLongPFN evaluation | checkpoint=%s | sha256_prefix=%s | "
        "batch_size=%s | report_calibration=%s | ready_files=%s | output_dir=%s",
        ckpt_diag["checkpoint_path"],
        ckpt_diag["checkpoint_file_sha256_prefix"],
        batch_size,
        bool(report_calibration),
        len(ready_files),
        output_paths.output_dir,
    )
    LOGGER.debug("CUDA at evaluation start: %s", _cuda_mem_string())

    n_anchors_used = resolve_n_support_anchors(n_support_anchors)

    model.eval()

    calibration_rows = []
    skipped: list[dict[str, Any]] = []
    n_files_used = 0
    eval_start_time = time.time()

    for file_idx, ready_file in enumerate(ready_files):
        LOGGER.info("[file %s/%s] %s", file_idx + 1, len(ready_files), os.path.basename(ready_file))

        ready_map = _prepare_ready_map(ready_file)

        if ready_map["ready_format_version"] != READY_FORMAT_VERSION:
            skipped.append({"source_file": os.path.basename(ready_file), "reason": "wrong_ready_format_version"})
            LOGGER.info("Skipped %s: ready_format_version is not %s", os.path.basename(ready_file), READY_FORMAT_VERSION)
            del ready_map
            continue

        domain = str(ready_map["domain"]).lower()

        if domain not in wanted_domains:
            skipped.append({"source_file": os.path.basename(ready_file), "reason": f"unwanted_domain_{domain}"})
            LOGGER.info("Skipped %s: unwanted domain %s", os.path.basename(ready_file), domain)
            del ready_map
            continue

        n_files_used += 1

        support_context = ready_map["support_context"]
        global_dataset_id = int(ready_map["global_dataset_id"])
        dataset_id = int(ready_map["dataset_id"])

        support_count = int(support_context["n_support"])
        d_input = int(support_context["d_input"])

        LOGGER.info(
            "domain=%s | global_dataset_id=%s | dataset_id=%s | support=%s | d_input=%s | tasks=%s",
            domain,
            global_dataset_id,
            dataset_id,
            support_count,
            d_input,
            eval_tasks.ready_task_names(ready_map),
        )

        for task_name in eval_tasks.ready_task_names(ready_map):
            task = ready_map["tasks"][task_name]
            n_eval = eval_tasks.ready_task_n_eval(task)

            if n_eval <= 0:
                LOGGER.info("task=%s skipped: n_eval=%s", task_name, n_eval)
                continue
            output_task_name = _output_task_name(task_name)
            if eval_outputs.partition_is_complete(output_paths, evaluation_id=run_id, method="causal_long_pfn", dataset_uid=str(ready_map["dataset_uid"]), task_name=output_task_name):
                LOGGER.info("task=%s skipped: completed partition exists", task_name)
                continue

            t_obs_arr = np.asarray(task["t_obs"])
            t_target_arr = np.asarray(task["t_target"])
            tau_arr = np.asarray(task["tau"])

            LOGGER.info(
                "task=%s | n_eval=%s | t_obs=%s..%s | t_target=%s..%s | tau=%s",
                task_name,
                n_eval,
                t_obs_arr.min() if t_obs_arr.size else None,
                t_obs_arr.max() if t_obs_arr.size else None,
                t_target_arr.min() if t_target_arr.size else None,
                t_target_arr.max() if t_target_arr.size else None,
                sorted(np.unique(tau_arr).tolist()) if tau_arr.size else [],
            )

            metrics, pred_records, cal_records, rollout_records, gmm_records = evaluate_ready_task(
                model=model,
                device=device,
                ready_map=ready_map,
                ckpt_diag=ckpt_diag,
                task_name=task_name,
                run_id=run_id,
                batch_size=batch_size,
                report_calibration=bool(report_calibration),
                evaluation_id=run_id,
                evaluation_config_hash=evaluation_config_hash,
                n_support_anchors=n_anchors_used,
            )
            calibration_rows.extend(cal_records)
            expected_rows = n_eval * (1 if output_task_name == "one_step_exhaustive" else int(common.PROJECTION_HORIZON))
            if len(pred_records) != expected_rows or len(rollout_records) != expected_rows:
                raise RuntimeError(f"Unexpected PFN rollout row count for {task_name}: expected {expected_rows}, got predictions={len(pred_records)}, rollout={len(rollout_records)}.")
            eval_outputs.write_completed_partition(
                output_paths,
                evaluation_id=run_id,
                method="causal_long_pfn",
                dataset_uid=str(ready_map["dataset_uid"]),
                task_name=output_task_name,
                rollout_steps=pd.DataFrame(rollout_records),
                prediction_rows=pd.DataFrame(pred_records),
                gmm_components=pd.DataFrame(gmm_records),
            )

            LOGGER.info(
                "metrics | normRMSE=%.6f | normMAE=%.6f | pred_norm_mean=%.6f | pred_norm_std=%.6f",
                metrics["normalized_rmse"],
                metrics["normalized_mae"],
                metrics["pred_norm_mean"],
                metrics["pred_norm_std"],
            )

        del ready_map
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        LOGGER.debug("CUDA after file: %s", _cuda_mem_string())

    prediction_df = eval_outputs.collect_completed_partitions(output_paths, "prediction_rows.parquet", evaluation_id=run_id)
    if prediction_df.empty:
        raise RuntimeError("No evaluation rows were produced. Check ready files, domains, and n_eval values.")
    summaries = eval_outputs.write_prediction_summaries(prediction_df, paths=output_paths)
    calibration_df = pd.DataFrame()
    if report_calibration:
        calibration_rows_df = prediction_df
        calibration_df, calibration_outputs = _write_calibration_summary(
            calibration_rows_df,
            prediction_df,
            output_paths,
        )
        summaries.update(calibration_outputs)

    elapsed = time.time() - eval_start_time
    summaries.update({
        "method": "causal_long_pfn",
        "method_family": "PFN",
        "run_id": run_id,
        "output_dir": str(output_paths.output_dir),
        "device": str(device),
        "seed": int(common.SEED),
        "n_ready_files_found": int(len(ready_files)),
        "n_ready_files_used": int(n_files_used),
        "n_skipped": int(len(skipped)),
        "skipped": skipped,
        "checkpoint": ckpt_diag,
        "evaluation_id": run_id,
        "evaluation_config_hash": evaluation_config_hash,
        "report_calibration": bool(report_calibration),
        "metric": "normalized_rmse",
        "metric_definition": (
            "RMSE(pred_eval_norm - target_eval_norm), both on the shared "
            "support-only evaluation normalization"
        ),
        "feedback_clip": float(common.OUTCOME_CLIP_TRAIN),
        "elapsed_min": float(elapsed / 60.0),
    })

    LOGGER.info(
        "Finished CausalLongPFN checkpoint evaluation | ready_files_used=%s | skipped=%s | "
        "prediction_rows=%s | one_step_calibration_rows=%s | "
        "domain_balanced_mean_normalized_RMSE=%.6f | elapsed_min=%.2f",
        n_files_used,
        len(skipped),
        len(prediction_df),
        len(calibration_df),
        summaries["domain_balanced_norm_rmse"],
        elapsed / 60.0,
    )
    LOGGER.info("Saved prediction rows: %s", summaries["prediction_rows_parquet"])
    LOGGER.info("Saved domain/task summary: %s", summaries["domain_task_summary_csv"])
    if report_calibration:
        LOGGER.info("Saved calibration rows: %s", summaries["calibration_rows_parquet"])
        LOGGER.info("Saved calibration summary: %s", summaries["calibration_summary_csv"])

    return summaries


def run_all(
    checkpoint_path,
    ready_dirs=None,
    ready_paths=None,
    batch_size=PFN_BATCH_SIZE,
    wanted_domains=WANTED_DOMAINS,
    output_dir=None,
    report_calibration: bool = True,
    evaluation_id: str | None = None,
) -> dict[str, Any]:
    output_paths = eval_outputs.prepare_output_paths(output_dir or DEFAULT_OUTPUT_DIR)
    common.configure_torch_runtime(seed=common.SEED)

    LOGGER.info(
        "Runtime | torch=%s | cuda_available=%s",
        torch.__version__,
        torch.cuda.is_available(),
    )
    LOGGER.debug("PYTORCH_CUDA_ALLOC_CONF=%s", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))

    if torch.cuda.is_available():
        LOGGER.info("CUDA device: %s", torch.cuda.get_device_name(0))
        LOGGER.debug("CUDA initial: %s", _cuda_mem_string())

    ready_files = eval_inputs.find_ready_pickles(
        eval_inputs.ReadyBenchmarkInputs(
            ready_dirs=tuple(str(path) for path in (ready_dirs or ())),
            ready_paths=tuple(str(path) for path in (ready_paths or ())),
        )
    )

    ckpt_path = find_pfn_checkpoint(checkpoint_path)

    LOGGER.info("Found %s ready datasets.", len(ready_files))
    LOGGER.info("First ready file: %s", ready_files[0])
    LOGGER.info("Using checkpoint: %s", ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    LOGGER.info("Loading checkpoint.")

    model, ckpt_diag = load_causal_long_pfn_checkpoint(
        path=ckpt_path,
        device=device,
    )
    ckpt_diag["checkpoint_file_sha256"] = _file_sha256(ckpt_path)
    checkpoint_n_support_anchors = int(ckpt_diag["prior_config"]["N_SUPPORT_ANCHORS"])
    if not 1 <= checkpoint_n_support_anchors <= int(N_SUPPORT_ANCHORS):
        raise ValueError(
            f"Checkpoint support-anchor count {checkpoint_n_support_anchors} is outside "
            f"[1, {int(N_SUPPORT_ANCHORS)}]; ready files were built with "
            f"{int(N_SUPPORT_ANCHORS)} anchors per support trajectory."
        )
    if checkpoint_n_support_anchors != int(N_SUPPORT_ANCHORS):
        LOGGER.info(
            "Checkpoint uses %s support anchor(s); reading the leading slice of the "
            "%s built anchors.",
            checkpoint_n_support_anchors,
            int(N_SUPPORT_ANCHORS),
        )

    evaluation_config_hash = hashlib.sha256(json.dumps({
        "checkpoint": ckpt_diag["checkpoint_file_sha256"], "batch_size": int(batch_size),
        "wanted_domains": list(wanted_domains), "rollout_mode": "mean",
        "n_support_anchors": checkpoint_n_support_anchors,
        "n_pfn_layers_used": int(model.n_pfn_layers),
        "ready_files": {str(Path(path).resolve()): _file_sha256(path) for path in ready_files},
    }, sort_keys=True).encode("utf-8")).hexdigest()
    run_id = evaluation_id or f"causal_long_pfn_{ckpt_diag['checkpoint_file_sha256'][:16]}_{evaluation_config_hash[:12]}"

    LOGGER.info("Run ID: %s", run_id)
    LOGGER.info(
        "Checkpoint loaded | basename=%s | sha256_prefix=%s",
        ckpt_diag["checkpoint_basename"],
        ckpt_diag["checkpoint_file_sha256_prefix"],
    )

    for key, value in ckpt_diag.items():
        LOGGER.debug("  %s: %s", key, value)

    if torch.cuda.is_available():
        LOGGER.debug("CUDA after model load: %s", _cuda_mem_string())

    return evaluate_ready_files(
        model=model,
        device=device,
        ready_files=ready_files,
        ckpt_diag=ckpt_diag,
        run_id=run_id,
        output_paths=output_paths,
        batch_size=batch_size,
        wanted_domains=tuple(wanted_domains),
        report_calibration=bool(report_calibration),
        evaluation_config_hash=evaluation_config_hash,
        n_support_anchors=checkpoint_n_support_anchors,
    )
