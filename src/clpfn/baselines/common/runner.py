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

from clpfn.baselines.common.api import BaselineAdapter
from clpfn.baselines.common.persistence import BaselinePersistence, dataset_content_hash
from clpfn.baselines.common.tuning import select_hparams_for_dataset, tune_info_from_selection
from clpfn.evaluation.core import inputs as eval_inputs
from clpfn.evaluation.core import outputs as eval_outputs
from clpfn.evaluation.core import benchmark as common
from clpfn.evaluation.core import reporting
from clpfn.evaluation.core.records import make_raw_prediction_record
from clpfn.evaluation.core import tasks as eval_tasks


LOGGER = logging.getLogger(__name__)


def _output_task_name(task_name: str) -> str:
    return "one_step_exhaustive" if "one_step" in str(task_name).lower() else "sequential_rollout"


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _baseline_tuning_fields(tune_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "tuning_mode": str(tune_info["tuning_mode"]),
        "initial_search_n": int(tune_info["initial_search_n"]),
        "search_candidates_evaluated": int(tune_info["search_candidates_evaluated"]),
        "search_time_sec": float(tune_info["search_time_sec"]),
        "selected_candidate": int(tune_info["selected_candidate"]),
    }


def _baseline_checkpoint_fields() -> dict[str, Any]:
    return {
        "checkpoint_basename": "",
        "checkpoint_step_count": None,
        "trainable_parameters": None,
        "pfn_target_refit_time_sec": None,
    }


def _eval_task_rows(
    adapter: BaselineAdapter,
    payload: Any,
    query_bundle: dict[str, Any],
    rows: np.ndarray,
    current_ts: np.ndarray,
    target_ts: np.ndarray,
    task_name: str,
    meta: dict[str, Any],
    train_diag: dict[str, Any],
    tune_info: dict[str, Any],
    fit_info: dict[str, Any],
    evaluation_id: str = "",
    raw_dataset_hash: str = "",
    final_fit_seed_root: int = 0,
    actual_fit_seed: int = 0,
    query_seed: int = 0,
    manifest_hash: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = np.asarray(rows, dtype=np.int64)
    current_ts = np.asarray(current_ts, dtype=np.int64)
    target_ts = np.asarray(target_ts, dtype=np.int64)
    if len(rows) == 0:
        return [], []

    predictions = adapter.predict_rows(payload, query_bundle, rows, current_ts, target_ts)
    if len(predictions) != len(rows):
        raise RuntimeError(
            f"{adapter.method_name} returned {len(predictions)} predictions for {len(rows)} "
            "evaluation rows; zipping them would silently drop rows."
        )
    records = []
    rollout_records = []
    hardware_fields = reporting.hardware_report_fields(adapter.device_label or common.DEVICE)
    tuning_fields = _baseline_tuning_fields(tune_info)
    checkpoint_fields = _baseline_checkpoint_fields()
    for query_id, (row_id, current_t, t_target, prediction) in enumerate(
        zip(rows, current_ts, target_ts, predictions)
    ):
        output_task_name = _output_task_name(task_name)
        path = np.asarray(prediction.path, dtype=np.float64).reshape(-1) if prediction.path is not None else np.asarray([], dtype=np.float64)
        unclipped_path = np.asarray(prediction.unclipped_path, dtype=np.float64).reshape(-1) if prediction.unclipped_path is not None else np.asarray([], dtype=np.float64)
        if len(path) and not np.isclose(path[-1], float(prediction.pred_norm), rtol=0.0, atol=1e-7):
            raise RuntimeError("Baseline rollout endpoint differs from the existing reported final prediction.")
        requested_horizon = int(t_target - current_t)
        if output_task_name == "sequential_rollout" and len(path) == 0:
            raise RuntimeError(
                f"{adapter.method_name} produced no rollout path for a sequential row "
                f"(current_t={int(current_t)}, target_t={int(t_target)})."
            )
        if len(path) and len(path) != requested_horizon:
            raise RuntimeError(
                f"{adapter.method_name} produced a {len(path)}-step rollout for a "
                f"{requested_horizon}-step request (current_t={int(current_t)}, target_t={int(t_target)})."
            )
        if len(unclipped_path) and len(unclipped_path) != len(path):
            raise RuntimeError(
                f"{adapter.method_name} unclipped rollout has {len(unclipped_path)} steps "
                f"but the reported rollout has {len(path)}."
            )
        if len(path) and not np.isfinite(path).all():
            raise RuntimeError(
                f"{adapter.method_name} produced a non-finite rollout path "
                f"(current_t={int(current_t)}, target_t={int(t_target)})."
            )
        available_horizons = range(1, len(path) + 1) if output_task_name == "sequential_rollout" and len(path) else (requested_horizon,)
        raw = query_bundle
        required_identity = ("patient_id", "patient_uid", "origin_uid", "plan_uid", "query_uid", "planned_action_sequence")
       
        missing_identity = [key for key in required_identity if key not in raw]
        if missing_identity:
            raise KeyError(f"Raw query bundle is missing stable identity fields: {missing_identity}")
        plan = np.asarray(raw["planned_action_sequence"][row_id], dtype=np.int64).reshape(-1)
        for horizon in available_horizons:
            step_idx = int(horizon) - 1
            pred_reported = float(path[step_idx]) if step_idx < len(path) else float(prediction.pred_norm)
            pred_unclipped = float(unclipped_path[step_idx]) if step_idx < len(unclipped_path) else float(prediction.pred_norm_unclipped if prediction.pred_norm_unclipped is not None else pred_reported)
            target_t_step = int(current_t + horizon)
            target_raw = float(raw["y_raw"][row_id, target_t_step])
            target_unclipped = float((target_raw - float(meta["out_mean"])) / max(float(meta["out_std"]), 1e-6))
            current_y_raw = float(raw["y_raw"][row_id, current_t])
            current_y_model = float(raw["y_norm_clip"][row_id, current_t])
            current_y_eval_unclipped = float((current_y_raw - float(meta["out_mean"])) / max(float(meta["out_std"]), 1e-6))
            reported_task = reporting.reported_task(output_task_name, horizon)
            pred_raw_unclipped = float(pred_unclipped * float(meta["out_std"]) + float(meta["out_mean"]))
            shared_fields = {
                "reported_task": reported_task, "current_y_raw": current_y_raw, "current_y_model_norm": current_y_model,
                "current_y_eval_norm_unclipped": current_y_eval_unclipped,
                "prediction_raw_unclipped": pred_raw_unclipped,
                "prediction_model_norm_unclipped": pred_unclipped, "prediction_eval_norm_unclipped": pred_unclipped,
                "prediction_eval_norm": pred_unclipped,
                "target_raw": target_raw, "target_model_norm": target_unclipped,
                "target_eval_norm_unclipped": target_unclipped, "target_eval_norm": target_unclipped,
            }
            rollout_records.append({
                "evaluation_id": evaluation_id, "method": adapter.method_name, "method_family": adapter.method_family,
                "dataset_uid": str(meta["dataset_uid"]), "patient_id": int(raw["patient_id"][row_id]), "patient_uid": str(raw["patient_uid"][row_id]),
                "origin_uid": str(raw["origin_uid"][row_id]), "plan_uid": str(raw["plan_uid"][row_id]),
                "query_uid": str(raw["query_uid"][row_id]), "task_name": output_task_name, "horizon": int(horizon),
                "observation_time": int(current_t), "target_time": target_t_step,
                "action_at_rollout_step": int(plan[step_idx]) if step_idx < len(plan) else None,
                "planned_action_sequence": json.dumps(plan.tolist()), "prediction_raw": pred_raw_unclipped,
                "prediction_model_norm": pred_unclipped, **shared_fields,
                "squared_error": float((pred_unclipped - target_unclipped) ** 2),
                "absolute_error": float(abs(pred_unclipped - target_unclipped)),
                "rollout_mode": "method_existing_rollout" if len(path) else "direct_horizon_prediction",
                "schema_version": 1,
                "fit_id": fit_info["fit_id"],
                "pfn_checkpoint_id": "", "pfn_checkpoint_hash": "", "pretraining_seed": None, "model_variant": "", "prior_variant": "",
                "evaluation_seed": int(query_seed), "final_fit_seed": int(final_fit_seed_root), "actual_fit_seed": int(actual_fit_seed),
                "query_sampling_seed": int(query_seed), "tuning_manifest_hash": manifest_hash,
                "raw_dataset_hash": raw_dataset_hash, "ready_dataset_hash": "",
                "evaluation_config_hash": "", "number_rollout_steps": int(len(path)) if len(path) else 1,
                "batch_size": 1, "support_size": int(meta["support_size"]), "peak_batch_gpu_memory": 0,
            })
            row = make_raw_prediction_record(
            method_name=adapter.method_name,
            method_family=adapter.method_family,
            run_id=adapter.run_id,
            query_bundle=query_bundle,
            meta=meta,
            task_name=output_task_name,
            row_id=int(row_id),
            query_id=query_id,
            current_t=int(current_t),
            t_target=target_t_step,
            pred_norm=pred_unclipped,
            predict_time_sec=float(prediction.predict_time_sec) / max(1, len(available_horizons)),
        )
            row.update(reporting.task_identity_fields(meta, output_task_name, row["tau"]))
            row.update(tuning_fields)
            row.update({"fit_id": fit_info["fit_id"], "fit_task_scope": fit_info["task_scope"]})
            row.update({"evaluation_id": evaluation_id, "dataset_uid": str(meta["dataset_uid"]), "patient_uid": str(raw["patient_uid"][row_id]), "raw_dataset_hash": raw_dataset_hash, "schema_version": 1, "horizon": int(horizon), "evaluation_seed": int(query_seed), "final_fit_seed": int(final_fit_seed_root), "actual_fit_seed": int(actual_fit_seed), "query_sampling_seed": int(query_seed), "tuning_manifest_hash": manifest_hash,})
            row.update(shared_fields)
            row.update(checkpoint_fields)
            row.update(hardware_fields)
            row.update(adapter.extra_record_fields(train_diag, prediction, tune_info, meta))
            records.append(row)
    return reporting.finalize_task_timing(
        records,
        refit_time_sec=float(train_diag["fit_time_sec"]),
    ), rollout_records


def _derived_seed(root_seed: int, *parts: Any) -> int:
    text = "||".join([str(int(root_seed)), *map(str, parts)])
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def evaluate_one_pickle(
    adapter: BaselineAdapter,
    pfile: str,
    global_dataset_id: int,
    persistence: BaselinePersistence,
    *,
    mode: str,
    evaluation_id: str = "",
    final_fit_seed: int = 101,
    query_seed: int = 4242,
    manifest_hash: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pm = eval_inputs.load_pickle(pfile)
    raw_dataset_hash = _file_sha256(pfile)
    support_bundle, meta = common.prepare_dataset_bundle(pm, pfile, global_dataset_id)
    domain = meta["domain"]
    cfg = meta["cfg"]
    dataset_hash = dataset_content_hash(support_bundle)
    final_idx = np.arange(int(support_bundle["covariates"].shape[0]), dtype=np.int64)

    selector = adapter.select_hparams or select_hparams_for_dataset
    if mode == "tune":
        best_hparams, tune_info = selector(
            adapter,
            support_bundle,
            meta=meta,
            source_file=pfile,
            trial_store=persistence,
            dataset_hash=dataset_hash,
            raw_dataset_hash=raw_dataset_hash,
        )
    else:
        saved = persistence.load_selection(
            dataset_uid=str(meta["dataset_uid"]),
            dataset_hash=dataset_hash,
            raw_dataset_hash=raw_dataset_hash,
            strict_hash=True,
        )
        if saved is None:
            raise FileNotFoundError(
                f"No selected hyperparameters for method={adapter.method_name} "
                f"dataset_uid={meta['dataset_uid']}. Run --mode tune first."
            )
        best_hparams = adapter.canonical_hparams(json.loads(saved["selected_hparams_json"]))
        tune_info = tune_info_from_selection(saved)

    if mode == "tune":
        del pm, support_bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return [], {
            "meta": meta,
            "task_counts": {},
            "fit_info": None,
            "partitions": [],
            "tune_info": tune_info,
            "selected_hparams": best_hparams,
        }

    actual_fit_seed = _derived_seed(
        final_fit_seed, adapter.method_name, meta["dataset_uid"], "final_fit"
    )
 
    task_query_seed = _derived_seed(
        query_seed, meta["dataset_uid"], "query_sampling"
    )
    rng = np.random.default_rng(task_query_seed)

    artifacts = adapter.train_final(support_bundle, best_hparams, final_idx, actual_fit_seed)
    train_diag = artifacts.train_diag
    train_diag["n_train"] = int(len(final_idx))
    fit_info = persistence.record_fit(
        meta=meta,
        hparams=adapter.canonical_hparams(best_hparams),
        tune_info=tune_info,
        fit_seed=actual_fit_seed,
        final_fit_seed_root=final_fit_seed,
        query_seed=task_query_seed,
        dataset_hash=dataset_hash,
        train_diag=train_diag,
        run_id=adapter.run_id,
        manifest_hash=manifest_hash,
    )
    all_records: list[dict[str, Any]] = []
    partitions: list[dict[str, Any]] = []
    task_counts: dict[str, int] = {}
    for task_rows in eval_tasks.iter_raw_task_rows(
        pm=pm,
        domain=domain,
        cfg=cfg,
        rng=rng,
        max_rows=None,
    ):
        query_bundle = common.make_query_bundle(task_rows.raw, meta)
        for key in (
            "patient_id", "patient_uid", "origin_uid", "plan_uid",
            "query_uid", "planned_action_sequence",
        ):
            query_bundle[key] = task_rows.raw[key]
        task_records, rollout_records = _eval_task_rows(
            adapter,
            artifacts.payload,
            query_bundle,
            task_rows.rows,
            task_rows.current_ts,
            task_rows.target_ts,
            task_name=task_rows.task_name,
            meta=meta,
            train_diag=train_diag,
            tune_info=tune_info,
            fit_info=fit_info,
            evaluation_id=evaluation_id,
            raw_dataset_hash=raw_dataset_hash,
            final_fit_seed_root=final_fit_seed,
            actual_fit_seed=actual_fit_seed,
            query_seed=task_query_seed,
            manifest_hash=manifest_hash,
        )
        all_records.extend(task_records)
        output_task_name = _output_task_name(task_rows.task_name)
        task_counts[output_task_name] = int(len(task_records))
        partitions.append({
            "task_name": output_task_name,
            "prediction_rows": task_records,
            "rollout_steps": rollout_records,
        })

    del pm, support_bundle, artifacts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_records, {
        "meta": meta,
        "task_counts": task_counts,
        "fit_info": fit_info,
        "partitions": partitions,
        "tune_info": tune_info,
        "selected_hparams": best_hparams,
    }


def run_all(
    adapter: BaselineAdapter,
    *,
    wanted_domains=None,
    raw_inputs=None,
    baseline_config=None,
    initial_random_search=40,
    output_dir=None,
    fit_root="outputs/fits",
    evaluation_id: str | None = None,
    mode: str,
    tuning_state_dir: str | None = None,
    selected_hparams_path: str | None = None,
    final_fit_seed: int = 101,
    query_seed: int = 4242,
    hpo_plan_seed: int = 1701,
    hpo_split_seed: int = 2701,
    hpo_train_seed: int = 3701,
    hpo_validation_seed: int = 4701,
) -> dict[str, Any]:
    mode = str(mode).lower().strip()
    if mode not in {"tune", "evaluate"}:
        raise ValueError("mode must be either tune or evaluate")

    configure = adapter.configure_from_eval_config
    if configure is None:
        raise ValueError(
            f"Baseline adapter '{adapter.method_name}' does not expose configure_from_eval_config."
        )
    configure(baseline_config)

    if output_dir is not None:
        adapter.output_dir = Path(output_dir)
    if mode != "tune":
        output_paths = eval_outputs.prepare_output_paths(adapter.output_dir)
        adapter.output_dir = output_paths.output_dir
    else:
        Path(adapter.output_dir).mkdir(parents=True, exist_ok=True)
        output_paths = None

    adapter.initial_random_search = int(initial_random_search)
    adapter.hpo_plan_seed = int(hpo_plan_seed)
    adapter.hpo_split_seed = int(hpo_split_seed)
    adapter.hpo_train_seed = int(hpo_train_seed)
    adapter.hpo_validation_seed = int(hpo_validation_seed)
    persistence = BaselinePersistence(
        adapter.method_name,
        root=fit_root,
        tuning_state_dir=tuning_state_dir,
        selected_hparams_path=selected_hparams_path,
    )
    wanted_domains = common.WANTED_DOMAINS if wanted_domains is None else tuple(wanted_domains)

    common.configure_torch_runtime()
    LOGGER.info(
        "Starting %s | mode=%s | device=%s | output=%s | "
        "task-specific tuning=%s | final_fit_seed=%s | query_seed=%s",
        adapter.title,
        mode,
        adapter.device_label or common.DEVICE,
        adapter.output_dir,
        adapter.initial_random_search,
        final_fit_seed,
        query_seed,
    )

    raw_files = common.find_raw_pickles(eval_inputs.raw_inputs_from_config(raw_inputs))
    raw_files = [
        pfile for pfile in raw_files
        if common.dataset_domain(pfile, eval_inputs.load_pickle(pfile)) in wanted_domains
    ]
    if not raw_files:
        raise FileNotFoundError("No raw dataset pickles found. Mount or generate the datasets first.")
    LOGGER.info("Found %s raw dataset pickles.", len(raw_files))

    raw_hashes = {str(Path(path).resolve()): _file_sha256(path) for path in raw_files}
    manifest_hash = persistence.manifest_hash() if persistence.selected_hparams_path.exists() else ""
    evaluation_config_hash = hashlib.sha256(json.dumps({
        "method": adapter.method_name,
        "mode": mode,
        "baseline_config": baseline_config,
        "initial_random_search": int(initial_random_search),
        "hpo_seeds": {
            "plan": int(hpo_plan_seed), "split": int(hpo_split_seed),
            "train": int(hpo_train_seed), "validation": int(hpo_validation_seed),
        },
        "final_fit_seed": int(final_fit_seed),
        "query_seed": int(query_seed),
        "manifest_hash": manifest_hash,
        "raw_files": raw_hashes,
    }, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    if mode == "tune":
        evaluation_id = evaluation_id or f"{adapter.method_name}_tune_{evaluation_config_hash[:20]}"
    else:
        evaluation_id = evaluation_id or (
            f"{adapter.method_name}_manifest{manifest_hash[:10]}_"
            f"fitseed{int(final_fit_seed)}_qseed{int(query_seed)}"
        )
    adapter.run_id = evaluation_id
    LOGGER.info("Run ID: %s", evaluation_id)

    expected_dataset_uids = [str(eval_inputs.load_pickle(path)["dataset_uid"]) for path in raw_files]
    if mode == "evaluate":
        if not persistence.selected_hparams_path.exists():
            raise FileNotFoundError(
                f"Selected-hyperparameter manifest not found: {persistence.selected_hparams_path}"
            )
        manifest = pd.read_parquet(persistence.selected_hparams_path)
        required_manifest_columns = {
            "method", "dataset_uid", "selected_hparams_json", "dataset_hash",
            "raw_dataset_hash", "status", "tuning_protocol_hash",
        }
        missing_columns = required_manifest_columns.difference(manifest.columns)
        if missing_columns:
            raise RuntimeError(
                "Selected-hyperparameter manifest is missing required columns: "
                f"{sorted(missing_columns)}"
            )
        method_manifest = manifest.loc[
            manifest["method"].astype(str) == adapter.method_name
        ].copy()
        if method_manifest["dataset_uid"].astype(str).duplicated().any():
            raise RuntimeError(
                f"Evaluation manifest contains duplicate dataset_uid rows for {adapter.method_name}."
            )
        bad_status = method_manifest.loc[method_manifest["status"].astype(str) != "completed"]
        if len(bad_status):
            raise RuntimeError(
                f"Evaluation manifest contains {len(bad_status)} non-completed selection row(s)."
            )
        empty_protocol = method_manifest["tuning_protocol_hash"].fillna("").astype(str).eq("")
        if empty_protocol.any():
            raise RuntimeError(
                "Evaluation manifest contains selections without a tuning protocol hash. "
                "Re-run tuning with the resumable task-specific workflow."
            )
        available = set(method_manifest["dataset_uid"].astype(str))
        missing = sorted(set(expected_dataset_uids) - available)
        if missing:
            raise RuntimeError(
                f"Evaluation manifest is incomplete: {len(missing)} task(s) missing. "
                f"First missing dataset_uid={missing[0]}"
            )

    t0_all = time.time()
    global_dataset_id = 0
    tuned_count = 0
    for file_idx, pfile in enumerate(raw_files):
        LOGGER.info("[%s/%s] %s", file_idx + 1, len(raw_files), os.path.basename(pfile))
        pm_identity = eval_inputs.load_pickle(pfile)
        dataset_uid = str(pm_identity["dataset_uid"])
        task_names = []
        if "test_data" in pm_identity:
            task_names.append("one_step_exhaustive")
        if "test_data_seq" in pm_identity:
            task_names.append("sequential_rollout")
        if mode != "tune" and task_names and all(
            eval_outputs.partition_is_complete(
                output_paths,
                evaluation_id=evaluation_id,
                method=adapter.method_name,
                dataset_uid=dataset_uid,
                task_name=task_name,
            )
            for task_name in task_names
        ):
            LOGGER.info("dataset_uid=%s skipped: completed partitions exist", dataset_uid)
            global_dataset_id += 1
            continue

        t0 = time.time()
        records, info = evaluate_one_pickle(
            adapter,
            pfile,
            global_dataset_id,
            persistence,
            mode=mode,
            evaluation_id=evaluation_id,
            final_fit_seed=final_fit_seed,
            query_seed=query_seed,
            manifest_hash=manifest_hash,
        )
        if mode == "tune":
            tuned_count += 1
            meta = info["meta"]
            LOGGER.info(
                "tuned domain=%s dataset_id=%s gamma=%s support=%s | val=%.4f | %.1fs",
                meta["domain"], meta["dataset_id"], meta["gamma"],
                meta["support_size"],
                float(info["tune_info"]["val_rmse_norm"]),
                time.time() - t0,
            )
            global_dataset_id += 1
            continue

        for partition in info["partitions"]:
            if eval_outputs.partition_is_complete(
                output_paths, evaluation_id=evaluation_id, method=adapter.method_name,
                dataset_uid=str(info["meta"]["dataset_uid"]), task_name=partition["task_name"],
            ):
                continue
            eval_outputs.write_completed_partition(
                output_paths,
                evaluation_id=evaluation_id,
                method=adapter.method_name,
                dataset_uid=str(info["meta"]["dataset_uid"]),
                task_name=partition["task_name"],
                rollout_steps=pd.DataFrame(partition["rollout_steps"]),
                prediction_rows=pd.DataFrame(partition["prediction_rows"]),
            )
        task_counts = info["task_counts"]
        row_rmse = common.normalized_rmse_from_sqerr([row["sq_error_norm"] for row in records])
        meta = info["meta"]
        LOGGER.info(
            "domain=%s dataset_id=%s gamma=%s support=%s | tasks=%s | rowRMSE=%.4f | %.1fs",
            meta["domain"], meta["dataset_id"], meta["gamma"],
            meta["support_size"], task_counts, row_rmse, time.time() - t0,
        )
        global_dataset_id += 1

    if mode == "tune":
        manifest_meta = persistence.consolidate_tuning_state(
            expected_dataset_uids=expected_dataset_uids
        )
        elapsed_min = float((time.time() - t0_all) / 60.0)
        LOGGER.info(
            "Tuning finished/resumed | completed=%s/%s | elapsed_min=%.2f | manifest=%s",
            len(manifest_meta["completed_dataset_uids"]), len(expected_dataset_uids),
            elapsed_min, persistence.selected_hparams_path,
        )
        return {
            "method": adapter.method_name,
            "method_family": adapter.method_family,
            "mode": mode,
            "run_id": evaluation_id,
            "evaluation_id": evaluation_id,
            "selected_hparams_path": str(persistence.selected_hparams_path),
            "tuning_trials_path": str(persistence.tuning_trials_path),
            "completed_tasks": len(manifest_meta["completed_dataset_uids"]),
            "expected_tasks": len(expected_dataset_uids),
            "missing_dataset_uids": manifest_meta["missing_dataset_uids"],
            "elapsed_min": elapsed_min,
            "prediction_rows": [],
        }

    pred_df = eval_outputs.collect_completed_partitions(
        output_paths, "prediction_rows.parquet", evaluation_id=evaluation_id
    )
    if pred_df.empty:
        raise RuntimeError("No predictions were produced. Check manifest, inputs, and partitions.")
    summaries = eval_outputs.write_prediction_summaries(pred_df, paths=output_paths)
    elapsed_min = float((time.time() - t0_all) / 60.0)
    metadata = {
        "method": adapter.method_name,
        "method_family": adapter.method_family,
        "mode": mode,
        "run_id": adapter.run_id,
        "evaluation_id": evaluation_id,
        "evaluation_config_hash": evaluation_config_hash,
        "tuning_manifest_hash": manifest_hash,
        "selected_hparams_path": str(persistence.selected_hparams_path),
        "output_dir": str(adapter.output_dir),
        "domain_task_summary_csv": summaries["domain_task_summary_csv"],
        "device": str(adapter.device_label or common.DEVICE),
        "final_fit_seed": int(final_fit_seed),
        "query_seed": int(query_seed),
        "n_raw_files_found": int(len(raw_files)),
        "n_predictions": int(len(pred_df)),
        "tuning_strategy": adapter.tuning_strategy,
        "initial_random_search": int(adapter.initial_random_search),
        "tune_group_key": "dataset_uid",
        "fit_registry_parquet": str(persistence.fit_registry_path),
        "metric": "normalized_rmse",
        "elapsed_min": elapsed_min,
        "min_history_points": int(common.MIN_HISTORY_POINTS),
        "min_t_obs": int(common.MIN_T_OBS),
    }
    metadata.update(adapter.extra_meta_fields(metadata))
    summaries.update(metadata)
    summaries["metadata"] = metadata
    LOGGER.info("Saved prediction rows: %s", summaries["prediction_rows_parquet"])
    return summaries
