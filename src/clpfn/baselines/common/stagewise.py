from __future__ import annotations

import gc
import json
import time
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from clpfn.baselines.common.api import BaselineAdapter
from clpfn.baselines.common.persistence import BaselinePersistence, canonical_json
from clpfn.baselines.common.tuning import (
    selection_row,
    split_context_indices,
    task_tuning_seeds,
    tune_info_from_selection,
)
from clpfn.evaluation.core import benchmark as common


def component_space(space: dict[str, list[Any]], predicate: Callable[[str], bool]) -> dict[str, list[Any]]:
    return {key: values for key, values in space.items() if predicate(key)}


def cleanup_torch(*_objects: Any) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def stage_trial_row(
    *,
    adapter: BaselineAdapter,
    meta: dict[str, Any],
    stage: str,
    candidate_index: int,
    candidate: dict[str, Any],
    seed: int,
    objective: str,
    score: float,
    fit_time: float,
    seeds: dict[str, int],
    tuning_protocol_hash: str,
    status: str = "completed",
    error: str = "",
) -> dict[str, Any]:
    return {
        "method": adapter.method_name,
        "domain": str(meta["domain"]),
        "dataset_uid": str(meta["dataset_uid"]),
        "trial_id": f"{meta['dataset_uid']}:stagewise:{stage}:{candidate_index}",
        "stage": stage,
        "candidate_index": int(candidate_index),
        "hparams_json": canonical_json(adapter.canonical_hparams(candidate)),
        "seed": int(seed),
        "plan_seed": int(seeds["plan_seed"]),
        "split_seed": int(seeds["split_seed"]),
        "validation_seed": int(seeds["validation_seed"]),
        "tuning_protocol_hash": str(tuning_protocol_hash),
        "validation_objective": objective,
        "validation_rmse": float(score),
        "fit_time": float(fit_time),
        "status": status,
        "error": error,
    }


def prepare_stagewise_context(
    adapter: BaselineAdapter,
    bundle: dict[str, Any],
    meta: dict[str, Any],
):
    n_ctx = int(bundle["covariates"].shape[0])
    seeds = task_tuning_seeds(adapter, meta)
    train_idx, val_idx = split_context_indices(adapter, n_ctx, seed=seeds["split_seed"])
    return seeds, train_idx, val_idx


def cached_stagewise_result(
    adapter: BaselineAdapter,
    *,
    trial_store: BaselinePersistence,
    meta: dict[str, Any],
    dataset_hash: str,
    raw_dataset_hash: str,
    tuning_protocol_hash: str,
):
    del adapter
    row = trial_store.load_selection(
        dataset_uid=str(meta["dataset_uid"]),
        dataset_hash=dataset_hash,
        raw_dataset_hash=raw_dataset_hash,
    )
    if row is None:
        return None
    saved_protocol = str(row["tuning_protocol_hash"])
    if saved_protocol != str(tuning_protocol_hash):
        raise ValueError(
            "Saved stagewise selection was produced by a different search protocol. "
            "Use a new tuning_state_dir or restore the original stage budgets/seeds/search space."
        )
    return json.loads(row["selected_hparams_json"]), tune_info_from_selection(row)


def saved_or_run_trial(
    *,
    trial_store: BaselinePersistence,
    adapter: BaselineAdapter,
    meta: dict[str, Any],
    stage: str,
    candidate_index: int,
    candidate: dict[str, Any],
    seed: int,
    objective: str,
    seeds: dict[str, int],
    tuning_protocol_hash: str,
    run: Callable[[], tuple[float, Any]],
) -> tuple[float, Any, dict[str, Any], bool]:
    candidate = adapter.canonical_hparams(candidate)
    cached = trial_store.load_trial(
        dataset_uid=str(meta["dataset_uid"]),
        stage=stage,
        candidate_index=candidate_index,
        hparams=candidate,
        tuning_protocol_hash=tuning_protocol_hash,
        seed=seed,
    )
    if cached is not None and str(cached["status"]) == "completed":
        return float(cached["validation_rmse"]), None, cached, True

    started = time.time()
    try:
        score, diag = run()
        row = stage_trial_row(
            adapter=adapter,
            meta=meta,
            stage=stage,
            candidate_index=candidate_index,
            candidate=candidate,
            seed=seed,
            objective=objective,
            score=score,
            fit_time=time.time() - started,
            seeds=seeds,
            tuning_protocol_hash=tuning_protocol_hash,
        )
    except Exception as exc:
        row = stage_trial_row(
            adapter=adapter,
            meta=meta,
            stage=stage,
            candidate_index=candidate_index,
            candidate=candidate,
            seed=seed,
            objective=objective,
            score=float("nan"),
            fit_time=time.time() - started,
            seeds=seeds,
            tuning_protocol_hash=tuning_protocol_hash,
            status="failed",
            error=repr(exc),
        )
        trial_store.write_tuning_trial(row)
        raise
    trial_store.write_tuning_trial(row)
    return float(score), diag, row, False


def finalize_stagewise_result(
    adapter: BaselineAdapter,
    *,
    meta: dict[str, Any],
    source_file: str,
    seeds: dict[str, int],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    space_info: dict[str, Any],
    best_hparams: dict[str, Any],
    final_score: float,
    selected_candidate: int,
    search_started: float,
    trial_rows: list[dict[str, Any]],
    stagewise_trials: dict[str, int],
    trial_store: BaselinePersistence,
    dataset_hash: str,
    raw_dataset_hash: str,
    tuning_protocol_hash: str,
):
    elapsed = float(time.time() - search_started)
    persisted_rows = trial_store.iter_trial_rows(dataset_uid=str(meta["dataset_uid"]))
    total_trial_fit_time = float(sum(
        float(row["fit_time"])
        for row in persisted_rows
        if np.isfinite(float(row["fit_time"]))
    ))
    selected_trial_id = None
    for row in reversed(trial_rows):
        if row["status"] == "completed" and int(row["candidate_index"]) == int(selected_candidate):
            selected_trial_id = row["trial_id"]
            break
    if selected_trial_id is None:
        raise RuntimeError("Stagewise tuning could not identify the selected trial.")
    info = {
        "val_rmse_norm": float(final_score),
        "n_train_tune": int(train_idx.size),
        "n_val_tune": int(val_idx.size),
        "selected_candidate": int(selected_candidate),
        "tuning_mode": "task_specific_stagewise_search",
        "tuning_failures": [row["error"] for row in trial_rows if row["status"] == "failed"],
        "initial_search_n": int(sum(stagewise_trials.values())),
        "search_candidates_evaluated": int(len([r for r in trial_rows if r["status"] == "completed"])),
        "search_time_sec": total_trial_fit_time,
        "resume_wall_time_sec": elapsed,
        "space_info": dict(space_info),
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "split_seed": int(seeds["split_seed"]),
        "plan_seed": int(seeds["plan_seed"]),
        "hpo_train_seed": int(seeds["train_seed"]),
        "validation_seed": int(seeds["validation_seed"]),
        "selected_trial_id": selected_trial_id,
        "stagewise_trials": dict(stagewise_trials),
    }
    trial_store.write_selection(selection_row(
        adapter=adapter,
        meta=meta,
        source_file=source_file,
        dataset_hash=dataset_hash,
        raw_dataset_hash=raw_dataset_hash,
        hparams=best_hparams,
        tune_info=info,
        seeds=seeds,
        tuning_protocol_hash=tuning_protocol_hash,
    ))
    return adapter.canonical_hparams(best_hparams), info


def candidates_for_horizon(
    bundle: dict[str, Any],
    val_idx,
    *,
    horizon: int,
    seed: int,
    max_val_origins: int,
):
    rng = np.random.default_rng(int(seed))
    y_norm = bundle["y_norm_clip"]
    lengths = bundle["sequence_lengths"]
    candidates = []
    for i in np.asarray(val_idx, dtype=np.int64):
        max_origin = min(int(lengths[i]) - int(horizon), y_norm.shape[1] - int(horizon) - 1, common.MAX_INPUT_INDEX)
        if max_origin < common.MIN_T_OBS:
            continue
        for origin in range(common.MIN_T_OBS, max_origin + 1):
            candidates.append((int(i), int(origin), int(origin + horizon)))
    if len(candidates) > int(max_val_origins):
        keep = rng.choice(len(candidates), size=int(max_val_origins), replace=False)
        candidates = [candidates[int(k)] for k in keep]
    return candidates


def rmse_for_candidates(bundle, candidates, predict: Callable[[int, int, int], float]) -> float:
    if not candidates:
        return float("nan")
    preds = []
    targets = []
    for row_id, t_obs, t_target in candidates:
        preds.append(float(predict(row_id, t_obs, t_target)))
        targets.append(float(bundle["y_norm_clip"][row_id, t_target]))
    pred = np.asarray(preds, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(target)
    return float(np.sqrt(np.mean((pred[mask] - target[mask]) ** 2))) if mask.any() else float("nan")


@torch.no_grad()
def teacher_forced_decoder_rmse(
    decoder,
    dataset,
    *,
    batch_size: int,
    move_batch_to_device: Callable[[dict[str, Any]], dict[str, Any]],
    outcome_of: Callable[[Any, dict[str, Any]], torch.Tensor],
) -> float:
    """Pooled masked RMSE of a single teacher-forced decoder pass."""
    decoder.eval()
    loader = DataLoader(
        dataset,
        batch_size=max(1, min(int(batch_size), len(dataset))),
        shuffle=False,
    )
    total_sq = 0.0
    total_active = 0.0
    for batch in loader:
        batch = move_batch_to_device(batch)
        outcome_pred = outcome_of(decoder, batch)
        active = batch["active_entries"]
        total_sq += float((active * (outcome_pred - batch["outputs"]) ** 2).sum().detach().cpu())
        total_active += float(active.sum().detach().cpu())
    if total_active <= 0:
        return float("nan")
    return float(np.sqrt(total_sq / total_active))
