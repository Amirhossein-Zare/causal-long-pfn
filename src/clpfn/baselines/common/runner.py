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

from clpfn.baselines.common.api import BaselineAdapter
from clpfn.baselines.common.tuning import select_hparams_for_dataset
from clpfn.evaluation.core import inputs as eval_inputs
from clpfn.evaluation.core import outputs as eval_outputs
from clpfn.evaluation.core import benchmark as common
from clpfn.evaluation.core.records import make_raw_prediction_record
from clpfn.evaluation.core import tasks as eval_tasks


LOGGER = logging.getLogger(__name__)


def _eval_task_rows(
    adapter: BaselineAdapter,
    payload: Any,
    query_bundle: dict[str, Any],
    rows: np.ndarray,
    current_ts: np.ndarray,
    target_ts: np.ndarray,
    task_name: str,
    meta: dict[str, Any],
    train_diag: dict[str, Any] | None = None,
    tune_info: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = np.asarray(rows, dtype=np.int64)
    current_ts = np.asarray(current_ts, dtype=np.int64)
    target_ts = np.asarray(target_ts, dtype=np.int64)
    if len(rows) == 0:
        return []

    predictions = adapter.predict_rows(payload, query_bundle, rows, current_ts, target_ts)
    records = []
    for query_id, (row_id, current_t, t_target, prediction) in enumerate(
        zip(rows, current_ts, target_ts, predictions)
    ):
        row = make_raw_prediction_record(
            method_name=adapter.method_name,
            method_family=adapter.method_family,
            run_id=adapter.run_id,
            query_bundle=query_bundle,
            meta=meta,
            task_name=task_name,
            row_id=int(row_id),
            query_id=query_id,
            current_t=int(current_t),
            t_target=int(t_target),
            pred_norm=float(prediction.pred_norm),
            predict_time_sec=float(prediction.predict_time_sec),
        )
        row.update(adapter.extra_record_fields(train_diag or {}, prediction, tune_info or {}, meta))
        records.append(row)
    return records


def evaluate_one_pickle(
    adapter: BaselineAdapter,
    pfile: str,
    global_dataset_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pm = eval_inputs.load_pickle(pfile)

    support_bundle, meta = common.prepare_dataset_bundle(pm, pfile, global_dataset_id)
    domain = meta["domain"]
    cfg = meta["cfg"]
    task_seed = (
        common.SEED
        + common.stable_domain_seed(domain)
        + int(meta["dataset_id"])
        + common.stable_file_seed(pfile)
    )
    rng = np.random.default_rng(task_seed)

    best_hparams, tune_info = select_hparams_for_dataset(
        adapter,
        support_bundle,
        meta=meta,
        source_file=pfile,
    )
    final_idx = np.arange(int(support_bundle["covariates"].shape[0]), dtype=np.int64)

    artifacts = adapter.train_final(
        support_bundle,
        best_hparams,
        final_idx,
        task_seed + 9999,
    )
    artifacts.train_diag["n_train"] = int(len(final_idx))
    all_records: list[dict[str, Any]] = []
    task_counts: dict[str, int] = {}
    for task_rows in eval_tasks.iter_raw_task_rows(
        pm=pm,
        domain=domain,
        cfg=cfg,
        rng=rng,
        max_rows=None,
    ):
        query_bundle = common.make_query_bundle(task_rows.raw, meta)
        task_records = _eval_task_rows(
            adapter,
            artifacts.payload,
            query_bundle,
            task_rows.rows,
            task_rows.current_ts,
            task_rows.target_ts,
            task_name=task_rows.task_name,
            meta=meta,
            train_diag=artifacts.train_diag,
            tune_info=tune_info,
        )
        all_records.extend(task_records)
        task_counts[task_rows.task_name] = int(len(task_records))

    if adapter.cleanup_payload is not None:
        adapter.cleanup_payload(artifacts.payload)
    del pm, support_bundle, artifacts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_records, {"meta": meta, "task_counts": task_counts}


def run_all(
    adapter: BaselineAdapter,
    *,
    wanted_domains=None,
    raw_inputs=None,
    baseline_config=None,
    initial_random_search=40,
    top_k_reuse=1,
    output_dir=None,
) -> dict[str, Any]:
    configure = getattr(adapter, "configure_from_eval_config", None)
    if configure is None:
        raise ValueError(f"Baseline adapter '{adapter.method_name}' does not expose configure_from_eval_config.")
    configure(baseline_config)

    if output_dir is not None:
        adapter.output_dir = Path(output_dir)
    output_paths = eval_outputs.prepare_output_paths(adapter.output_dir)
    adapter.output_dir = output_paths.output_dir

    adapter.initial_random_search = int(initial_random_search)
    adapter.top_k_reuse = int(top_k_reuse)
    adapter.refresh_run_id()
    wanted_domains = common.WANTED_DOMAINS if wanted_domains is None else tuple(wanted_domains)

    common.configure_torch_runtime()
    LOGGER.info(
        "Starting %s | device=%s | run_id=%s | output=%s | min_t_obs=%s | "
        "initial_random_search=%s | top_k_reuse=%s | tune_group_key=domain + support_size",
        adapter.title,
        adapter.device_label or common.DEVICE,
        adapter.run_id,
        adapter.output_dir,
        common.MIN_T_OBS,
        adapter.initial_random_search,
        adapter.top_k_reuse,
    )

    raw_files = common.find_raw_pickles(eval_inputs.raw_inputs_from_config(raw_inputs))
    raw_files = [
        pfile
        for pfile in raw_files
        if common.dataset_domain(pfile, eval_inputs.load_pickle(pfile)) in wanted_domains
    ]
    LOGGER.info("Found %s raw dataset pickles.", len(raw_files))
    if not raw_files:
        raise FileNotFoundError("No raw dataset pickles found. Mount or generate the new datasets first.")
    LOGGER.info("First raw file: %s", raw_files[0])

    all_records: list[dict[str, Any]] = []
    t0_all = time.time()
    global_dataset_id = 0
    for file_idx, pfile in enumerate(raw_files):
        LOGGER.info("[%s/%s] %s", file_idx + 1, len(raw_files), os.path.basename(pfile))
        t0 = time.time()
        records, info = evaluate_one_pickle(
            adapter,
            pfile,
            global_dataset_id,
        )
        all_records.extend(records)
        task_counts = info.get("task_counts", {})
        row_rmse = common.normalized_rmse_from_sqerr([row["sq_error_norm"] for row in records])
        meta = info.get("meta", {})
        LOGGER.info(
            "domain=%s raw_dataset_id=%s gamma=%s support=%s | tasks=%s | rowRMSE_norm=%.4f | %.1fs",
            meta.get("domain"),
            meta.get("dataset_id"),
            meta.get("gamma"),
            meta.get("support_size"),
            task_counts,
            row_rmse,
            time.time() - t0,
        )
        global_dataset_id += 1

    LOGGER.info(
        "Finished %s | prediction_rows=%s | elapsed_min=%.2f",
        adapter.title,
        len(all_records),
        (time.time() - t0_all) / 60.0,
    )
    if not all_records:
        raise RuntimeError("No predictions were produced. Check input files and domain filters.")

    pred_df = pd.DataFrame(all_records)
    summaries = eval_outputs.write_prediction_summaries(pred_df, paths=output_paths)

    elapsed_min = float((time.time() - t0_all) / 60.0)
    meta = {
        "method": adapter.method_name,
        "method_family": adapter.method_family,
        "run_id": adapter.run_id,
        "output_dir": str(adapter.output_dir),
        "domain_task_summary_csv": summaries["domain_task_summary_csv"],
        "device": str(adapter.device_label or common.DEVICE),
        "seed": int(common.SEED),
        "n_raw_files_found": int(len(raw_files)),
        "n_predictions": int(len(pred_df)),
        "tuning_strategy": adapter.tuning_strategy,
        "initial_random_search": int(adapter.initial_random_search),
        "top_k_reuse": int(adapter.top_k_reuse),
        "tune_group_key": "domain + support_size",
        "tuning_cache_groups": len(adapter.tuning_cache),
        "metric": "normalized_rmse",
        "metric_definition": (
            "RMSE(pred_norm_report_clipped_to_[-20,20] - "
            "clip((target_raw - out_mean) / out_std, -10, 10))"
        ),
        "target_norm_clipped_to_match": True,
        "target_norm_clip": float(common.TARGET_NORM_CLIP),
        "pred_norm_clipped_for_report": True,
        "pred_clip_report": float(common.PRED_CLIP_REPORT),
        "elapsed_min": elapsed_min,
    }
    meta["min_t_obs"] = int(common.MIN_T_OBS)
    meta.update(adapter.extra_meta_fields(meta))
    summaries.update(meta)
    summaries["metadata"] = meta

    LOGGER.info("Saved prediction rows: %s", summaries["prediction_rows_parquet"])
    LOGGER.info("Saved domain/task summary: %s", summaries["domain_task_summary_csv"])
    return summaries
