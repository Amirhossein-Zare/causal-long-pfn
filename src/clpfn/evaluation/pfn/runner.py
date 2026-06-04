from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from clpfn.evaluation.core import inputs as eval_inputs
from clpfn.evaluation.core import outputs as eval_outputs
from clpfn.evaluation.core.summaries import print_summary_table
from clpfn.evaluation.pfn import calibration as cal
from clpfn.evaluation.pfn.calibration_summaries import summarize_domain_calibration
from clpfn.evaluation.core import benchmark as common
from clpfn.evaluation.core import records as eval_records
from clpfn.evaluation.core import tasks as eval_tasks
from clpfn.evaluation.pfn.batches import (
    collate_ready_batch,
    move_batch_to_device,
)
from clpfn.models.causal_long_pfn import (
    load_causal_long_pfn_checkpoint,
    predictive_mean_from_gmm,
)

LOGGER = logging.getLogger(__name__)


READY_FORMAT_VERSION = "causal_long_pfn_ready"

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


def find_pfn_checkpoint(checkpoint_path: str | Path | None) -> str:
    if checkpoint_path is None:
        raise ValueError("Provide a checkpoint path.")

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint}")

    return str(checkpoint)


@torch.no_grad()
def evaluate_ready_task(
    model,
    device,
    ready_map,
    task_name: str,
    run_id: str,
    batch_size: int = PFN_BATCH_SIZE,
    report_calibration: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    task = ready_map["tasks"][task_name]
    support_context = ready_map["support_context"]

    n_eval = eval_tasks.ready_task_n_eval(task)

    if n_eval <= 0:
        return _metrics_from_arrays([], []), [], []

    preds_norm = []
    targets_norm = []
    prediction_rows = []
    calibration_rows = []

    support_size = int(ready_map["support_size"])
    rows_all = eval_tasks.ready_task_row_ids(task)

    for start in range(0, n_eval, batch_size):
        end = min(start + batch_size, n_eval)
        batch = collate_ready_batch(ready_map, task_name, start, end)
        current_batch_size = end - start

        batch = move_batch_to_device(batch, device)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            log_pi, mu, sigma = model.rollout(batch)
            pred_norm = predictive_mean_from_gmm(log_pi, mu)

        pn_model_unclipped = pred_norm.detach().float().cpu().numpy()
        pn_unclipped = _model_to_eval_norm(pn_model_unclipped, support_context)
        pn = np.clip(
            pn_unclipped,
            -common.PRED_CLIP_REPORT,
            common.PRED_CLIP_REPORT,
        )
        tn_model = batch["oracle_Y_final"].detach().float().cpu().numpy()
        tn = batch.get("target_eval_norm", batch["oracle_Y_final"]).detach().float().cpu().numpy()

        t_obs_np = batch["t_obs"].detach().cpu().numpy().astype(np.int64)
        t_target_np = batch["t_target"].detach().cpu().numpy().astype(np.int64)
        tau_np = batch["tau"].detach().cpu().numpy().astype(np.int64)

        row_ids = rows_all[start:end]

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
            row = eval_records.make_ready_prediction_record(
                method_name="causal_long_pfn",
                method_family="PFN",
                ready_map={**ready_map, "support_size": support_size},
                task_name=task_name,
                run_id=run_id,
                row_id=row_ids[batch_idx],
                query_id=start + batch_idx,
                pred_norm=pn[batch_idx],
                target_norm=tn[batch_idx],
                t_obs=t_obs_np[batch_idx],
                tau=tau_np[batch_idx],
                t_target=t_target_np[batch_idx],
                extra_fields={
                    "pred_norm_unclipped": float(pn_unclipped[batch_idx]),
                    "pred_model_norm_unclipped": float(pn_model_unclipped[batch_idx]),
                    "target_model_norm": float(tn_model[batch_idx]),
                    "pred_clip_report": float(common.PRED_CLIP_REPORT),
                },
            )

            if report_calibration:
                row = cal.add_empty_calibration_fields(row)

                if one_step_mask[batch_idx] and calibration is not None:
                    row = cal.add_calibration_fields(row, batch_idx, calibration)
                    calibration_rows.append(row.copy())

            prediction_rows.append(row)

        del batch, log_pi, mu, sigma, pred_norm

    metrics = _metrics_from_arrays(preds_norm, targets_norm)
    return metrics, prediction_rows, calibration_rows


def _prepare_ready_map(ready_file: str | Path) -> dict[str, Any]:
    ready_map = eval_inputs.load_pickle(ready_file)
    ready_map["_ready_file_basename"] = os.path.basename(str(ready_file))
    ready_map["_ready_file_path"] = str(ready_file)
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

    model.eval()

    prediction_rows = []
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

        n_ctx = int(support_context["n_support"])
        d_input = int(support_context["d_input"])

        LOGGER.info(
            "domain=%s | global_dataset_id=%s | dataset_id=%s | n_ctx=%s | d_input=%s | tasks=%s",
            domain,
            global_dataset_id,
            dataset_id,
            n_ctx,
            d_input,
            eval_tasks.ready_task_names(ready_map),
        )

        for task_name in eval_tasks.ready_task_names(ready_map):
            task = ready_map["tasks"][task_name]
            n_eval = eval_tasks.ready_task_n_eval(task)

            if n_eval <= 0:
                LOGGER.info("task=%s skipped: n_eval=%s", task_name, n_eval)
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

            metrics, pred_records, cal_records = evaluate_ready_task(
                model=model,
                device=device,
                ready_map=ready_map,
                task_name=task_name,
                run_id=run_id,
                batch_size=batch_size,
                report_calibration=bool(report_calibration),
            )

            prediction_rows.extend(pred_records)
            calibration_rows.extend(cal_records)

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

    if len(prediction_rows) == 0:
        raise RuntimeError("No evaluation rows were produced. Check ready files, domains, and n_eval values.")

    prediction_df = pd.DataFrame(prediction_rows)
    summaries = eval_outputs.write_prediction_summaries(prediction_df, paths=output_paths)
    calibration_df = pd.DataFrame()
    if report_calibration:
        calibration_rows_df = pd.DataFrame(calibration_rows)
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
        "report_calibration": bool(report_calibration),
        "metric": "normalized_rmse",
        "metric_definition": (
            "RMSE(pred_norm_report_clipped_to_[-20,20] - "
            "clip((target_raw - eval_out_mean) / eval_out_std, -10, 10))"
        ),
        "target_norm_clipped_to_match": True,
        "target_norm_clip": float(common.TARGET_NORM_CLIP),
        "pred_norm_clipped_for_report": True,
        "pred_clip_report": float(common.PRED_CLIP_REPORT),
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
        strict=True,
    )

    run_id = f"causal_long_pfn_{ckpt_diag['checkpoint_basename'].replace('.pt', '')}_{int(time.time())}"

    LOGGER.info("Run ID: %s", run_id)
    LOGGER.info(
        "Checkpoint loaded | basename=%s | sha256_prefix=%s",
        ckpt_diag.get("checkpoint_basename", ""),
        ckpt_diag.get("checkpoint_file_sha256_prefix", ""),
    )

    for key, value in ckpt_diag.items():
        if key in {"missing_keys", "unexpected_keys"}:
            LOGGER.debug("  %s: %s", key, len(value))
        else:
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
    )
