from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

import numpy as np

from clpfn.baselines.common.api import BaselineAdapter
from clpfn.baselines.common.persistence import BaselinePersistence, canonical_json, sha256_json


LOGGER = logging.getLogger(__name__)


def _stable_int(*parts: Any) -> int:
    text = "||".join(map(str, parts))
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def all_grid_candidates(
    space: dict[str, list[Any]],
    *,
    default_hparams: dict[str, Any],
    canonical_hparams,
) -> list[dict[str, Any]]:
    import itertools

    keys = list(space.keys())
    candidates = []
    for combo in itertools.product(*(space[key] for key in keys)):
        hparams = dict(default_hparams)
        for key, value in zip(keys, combo):
            hparams[key] = list(value) if isinstance(value, list) else value
        candidates.append(canonical_hparams(hparams))
    return candidates


def sample_random_hparams(
    space: dict[str, list[Any]],
    n: int,
    seed: int,
    *,
    default_hparams: dict[str, Any],
    canonical_hparams,
    transform_sample=None,
    is_valid=None,
    min_attempts: int = 500,
    attempts_per_candidate: int = 100,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    candidates = []
    seen = set()

    for _ in range(max(int(min_attempts), int(n) * int(attempts_per_candidate))):
        sample = {key: values[int(rng.integers(0, len(values)))] for key, values in space.items()}
        hparams = transform_sample(sample) if transform_sample is not None else {**default_hparams, **sample}
        hparams = canonical_hparams(hparams)
        if is_valid is not None and not is_valid(hparams):
            continue
        key = canonical_json(hparams)
        if key not in seen:
            seen.add(key)
            candidates.append(hparams)
        if len(candidates) >= int(n):
            break
    return candidates


def split_context_indices(adapter: BaselineAdapter, n_ctx: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(int(n_ctx))
    n_val = int(round(n_ctx * adapter.tune_val_frac))
    n_val = max(adapter.val_min, n_val)
    n_val = min(adapter.val_max, n_val)
    n_val = min(n_val, max(1, n_ctx - 20))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if train_idx.size < 5 or val_idx.size == 0:
        raise ValueError(
            f"Baseline tuning requires at least five training rows and one validation row; "
            f"got n_ctx={n_ctx}, n_train={train_idx.size}, n_val={val_idx.size}."
        )
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def task_tuning_seeds(adapter: BaselineAdapter, meta: dict[str, Any]) -> dict[str, int]:
    dataset_uid = str(meta["dataset_uid"])
    domain = str(meta["domain"])
    return {
        "plan_seed": int(adapter.hpo_plan_seed + _stable_int(adapter.method_name, domain, "candidate_plan") % 1_000_000_000),
        "split_seed": int(adapter.hpo_split_seed + _stable_int(adapter.method_name, dataset_uid, "support_split") % 1_000_000_000),
        "train_seed": int(adapter.hpo_train_seed + _stable_int(adapter.method_name, dataset_uid, "hpo_train") % 1_000_000_000),
        "validation_seed": int(adapter.hpo_validation_seed + _stable_int(adapter.method_name, dataset_uid, "validation_origins") % 1_000_000_000),
    }


def tune_info_from_selection(row: dict[str, Any]) -> dict[str, Any]:
    stored = row["tune_info_json"]
    if not isinstance(stored, str):
        raise TypeError("tune_info_json must be a JSON string.")
    info = json.loads(stored)
    if not isinstance(info, dict):
        raise TypeError("tune_info_json must decode to an object.")
    return info


def selection_row(
    *,
    adapter: BaselineAdapter,
    meta: dict[str, Any],
    source_file: str,
    dataset_hash: str,
    raw_dataset_hash: str,
    hparams: dict[str, Any],
    tune_info: dict[str, Any],
    seeds: dict[str, int],
    status: str = "completed",
    tuning_protocol_hash: str = "",
) -> dict[str, Any]:
    return {
        "method": adapter.method_name,
        "domain": str(meta["domain"]),
        "dataset_uid": str(meta["dataset_uid"]),
        "dataset_id": int(meta["dataset_id"]),
        "support_size": int(meta["support_size"]),
        "gamma": float(meta["gamma"]),
        "replicate": int(meta["replicate"]),
        "source_file": str(source_file),
        "dataset_hash": str(dataset_hash),
        "raw_dataset_hash": str(raw_dataset_hash),
        "selected_hparams_json": canonical_json(adapter.canonical_hparams(hparams)),
        "validation_score": float(tune_info["val_rmse_norm"]),
        "selected_candidate": int(tune_info["selected_candidate"]),
        "selected_trial_id": tune_info["selected_trial_id"],
        "search_candidates_evaluated": int(tune_info["search_candidates_evaluated"]),
        "search_time_sec": float(tune_info["search_time_sec"]),
        "plan_seed": int(seeds["plan_seed"]),
        "split_seed": int(seeds["split_seed"]),
        "hpo_train_seed": int(seeds["train_seed"]),
        "validation_seed": int(seeds["validation_seed"]),
        "tune_info_json": canonical_json(tune_info),
        "tuning_protocol_hash": str(tuning_protocol_hash),
        "status": str(status),
    }


def select_hparams_for_dataset(
    adapter: BaselineAdapter,
    bundle: dict[str, Any],
    meta: dict[str, Any],
    source_file: str,
    trial_store: BaselinePersistence,
    dataset_hash: str,
    raw_dataset_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    n_ctx = int(bundle["covariates"].shape[0])
    space, space_info = adapter.hyperparameter_space(bundle)
    seeds = task_tuning_seeds(adapter, meta)
    train_idx, val_idx = split_context_indices(adapter, n_ctx, seed=seeds["split_seed"])
    tuning_protocol_hash = sha256_json({
        "method": adapter.method_name,
        "strategy": "task_specific_random_search",
        "initial_random_search": int(adapter.initial_random_search),
        "space": space,
        "space_info": space_info,
        "seeds": seeds,
    })

    saved = trial_store.load_selection(
        dataset_uid=str(meta["dataset_uid"]),
        dataset_hash=dataset_hash,
        raw_dataset_hash=raw_dataset_hash,
    )
    if saved is not None:
        saved_protocol = str(saved["tuning_protocol_hash"])
        if saved_protocol != tuning_protocol_hash:
            raise ValueError(
                "Saved tuning selection was produced by a different search protocol. "
                "Use a new tuning_state_dir or restore the original budgets/seeds/search space."
            )
        return json.loads(saved["selected_hparams_json"]), tune_info_from_selection(saved)

    search_started = time.time()
    candidates = adapter.sample_candidates(space, n=adapter.initial_random_search, seed=seeds["plan_seed"])
    LOGGER.info(
        "task-specific tuning dataset_uid=%s candidates=%s plan_seed=%s split_seed=%s",
        meta["dataset_uid"], len(candidates), seeds["plan_seed"], seeds["split_seed"],
    )
    results: list[dict[str, Any]] = []
    failures: list[str] = []

    for ci, candidate in enumerate(candidates):
        candidate = adapter.canonical_hparams(candidate)
        cached = trial_store.load_trial(
            dataset_uid=str(meta["dataset_uid"]), stage="single",
            candidate_index=ci, hparams=candidate,
            tuning_protocol_hash=tuning_protocol_hash, seed=seeds["train_seed"],
        )
        if cached is not None and str(cached["status"]) == "completed":
            LOGGER.info("resume cand=%02d val_rmse=%.4f", ci, float(cached["validation_rmse"]))
            results.append({
                "candidate_index": ci,
                "val_rmse_norm": float(cached["validation_rmse"]),
                "hparams": candidate,
                "trial_id": cached["trial_id"],
            })
            continue

        started = time.time()
        trial_id = f"{meta['dataset_uid']}:single:{ci}"
        try:
            val_rmse, diag = adapter.evaluate_candidate(
                bundle, candidate, train_idx, val_idx, seeds["train_seed"]
            )
            diag_fields = adapter.tuning_diag_fields(diag) if adapter.tuning_diag_fields is not None else {}
            row = {
                "method": adapter.method_name,
                "domain": str(meta["domain"]),
                "dataset_uid": str(meta["dataset_uid"]),
                "trial_id": trial_id,
                "stage": "single",
                "candidate_index": int(ci),
                "hparams_json": canonical_json(candidate),
                "seed": int(seeds["train_seed"]),
                "plan_seed": int(seeds["plan_seed"]),
                "split_seed": int(seeds["split_seed"]),
                "validation_seed": int(seeds["validation_seed"]),
                "tuning_protocol_hash": tuning_protocol_hash,
                "validation_objective": (
                    "normalized_rmse_support_validation_one_step"
                    if adapter.method_name in {"ct", "gnet", "msm"}
                    else "normalized_rmse_support_validation_pooled_rollout"
                ),
                "validation_rmse": float(val_rmse),
                "fit_time": float(time.time() - started),
                "status": "completed",
                "error": "",
                "diag_json": canonical_json(diag_fields),
            }
            trial_store.write_tuning_trial(row)
            results.append({
                "candidate_index": ci,
                "val_rmse_norm": float(val_rmse),
                "hparams": candidate,
                "trial_id": trial_id,
            })
            label = adapter.tuning_candidate_label(candidate) if adapter.tuning_candidate_label else ""
            LOGGER.info("cand=%02d %s val_rmse=%.4f", ci, label, val_rmse)
        except Exception as exc:
            error = repr(exc)
            failures.append(error)
            row = {
                "method": adapter.method_name,
                "domain": str(meta["domain"]),
                "dataset_uid": str(meta["dataset_uid"]),
                "trial_id": trial_id,
                "stage": "single",
                "candidate_index": int(ci),
                "hparams_json": canonical_json(candidate),
                "seed": int(seeds["train_seed"]),
                "plan_seed": int(seeds["plan_seed"]),
                "split_seed": int(seeds["split_seed"]),
                "validation_seed": int(seeds["validation_seed"]),
                "tuning_protocol_hash": tuning_protocol_hash,
                "validation_objective": "normalized_rmse_support_validation",
                "validation_rmse": float("nan"),
                "fit_time": float(time.time() - started),
                "status": "failed",
                "error": error,
                "diag_json": "{}",
            }
            trial_store.write_tuning_trial(row)
            LOGGER.exception("candidate %s failed", ci)

    finite_results = sorted(
        [row for row in results if np.isfinite(row["val_rmse_norm"])],
        key=lambda row: row["val_rmse_norm"],
    )
    if finite_results:
        best = finite_results[0]
        best_hp = dict(best["hparams"])
        best_rmse = float(best["val_rmse_norm"])
        selected_candidate = int(best["candidate_index"])
        selected_trial_id = best["trial_id"]
    else:
        raise RuntimeError(
            f"All tuning candidates failed or produced non-finite validation scores for "
            f"method={adapter.method_name}, dataset_uid={meta['dataset_uid']}. "
            "Atomic trial records were retained for diagnosis and resume."
        )

    elapsed = float(time.time() - search_started)
    persisted_rows = trial_store.iter_trial_rows(dataset_uid=str(meta["dataset_uid"]))
    total_trial_fit_time = float(sum(
        float(row["fit_time"])
        for row in persisted_rows
        if np.isfinite(float(row["fit_time"]))
    ))
    info = {
        "val_rmse_norm": best_rmse,
        "n_train_tune": int(train_idx.size),
        "n_val_tune": int(val_idx.size),
        "selected_candidate": selected_candidate,
        "tuning_mode": "task_specific_random_search",
        "tuning_failures": failures,
        "initial_search_n": int(adapter.initial_random_search),
        "search_candidates_evaluated": int(len(results)),
        "search_time_sec": total_trial_fit_time,
        "resume_wall_time_sec": elapsed,
        "space_info": space_info,
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "split_seed": int(seeds["split_seed"]),
        "plan_seed": int(seeds["plan_seed"]),
        "hpo_train_seed": int(seeds["train_seed"]),
        "validation_seed": int(seeds["validation_seed"]),
        "selected_trial_id": selected_trial_id,
        "stagewise_trials": {},
    }
    trial_store.write_selection(selection_row(
        adapter=adapter, meta=meta, source_file=source_file,
        dataset_hash=dataset_hash, raw_dataset_hash=raw_dataset_hash,
        hparams=best_hp, tune_info=info, seeds=seeds,
        tuning_protocol_hash=tuning_protocol_hash,
    ))
    return adapter.canonical_hparams(best_hp), info
