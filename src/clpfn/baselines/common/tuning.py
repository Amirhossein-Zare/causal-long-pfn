from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np

from clpfn.evaluation.core import benchmark as common
from clpfn.baselines.common.api import BaselineAdapter


LOGGER = logging.getLogger(__name__)


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
        sample = {
            key: values[int(rng.integers(0, len(values)))]
            for key, values in space.items()
        }
        hparams = (
            transform_sample(sample)
            if transform_sample is not None
            else {**default_hparams, **sample}
        )
        hparams = canonical_hparams(hparams)
        if is_valid is not None and not is_valid(hparams):
            continue
        key = json.dumps(hparams, sort_keys=True)
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
    if train_idx.size < 5:
        train_idx = idx
        val_idx = idx[:0]
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def make_tuning_group_key(adapter: BaselineAdapter, meta: dict[str, Any]) -> tuple[Any, ...]:
    return (
        ("domain", meta["domain"]),
        ("support_size", int(meta["support_size"])),
    )


def default_tune_info(
    adapter: BaselineAdapter,
    n_train: int,
    *,
    mode: str,
    failures: list[str] | None = None,
    space_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "val_rmse_norm": float("nan"),
        "val_loss": float("nan"),
        "n_train_tune": int(n_train),
        "n_val_tune": 0,
        "selected_candidate": -1,
        "tuning_mode": mode,
        "tuning_group_key": "",
        "tuning_failures": list(failures or []),
        "tuning_cache_rank": -1,
        "initial_search_n": 0,
        "reuse_top_k": 0,
        "space_info": dict(space_info or {}),
    }


def select_hparams_for_dataset(
    adapter: BaselineAdapter,
    bundle: dict[str, Any],
    meta: dict[str, Any],
    source_file: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    n_ctx = int(bundle["covariates"].shape[0])
    space, space_info = adapter.hyperparameter_space(bundle)

    split_seed = (
        common.SEED
        + common.stable_domain_seed(meta["domain"])
        + int(meta["dataset_id"])
        + common.stable_file_seed(source_file)
    )
    train_idx, val_idx = split_context_indices(adapter, n_ctx, seed=split_seed)
    if val_idx.size == 0:
        info = default_tune_info(
            adapter,
            int(train_idx.size),
            mode="no_validation_split_default_used",
            failures=["no_validation_split"],
            space_info=space_info,
        )
        return dict(adapter.default_hparams), info

    group_key = make_tuning_group_key(adapter, meta)
    group_key_json = json.dumps(group_key)

    if group_key not in adapter.tuning_cache:
        candidates = adapter.sample_candidates(
            space,
            n=adapter.initial_random_search,
            seed=split_seed + 12345,
        )
        mode = "initial_random_search"
        LOGGER.info("tuning group=%s", group_key_json)
        LOGGER.info(
            "initial random search candidates=%s top_k=%s",
            len(candidates),
            adapter.top_k_reuse,
        )
        for key, value in space_info.items():
            if key.endswith("_grid") or key.startswith("C_") or key == "n_possible_combinations":
                LOGGER.info("%s=%s", key, value)

        results: list[dict[str, Any]] = []
        for ci, candidate in enumerate(candidates):
            val_rmse, diag = adapter.evaluate_candidate(
                bundle,
                candidate,
                train_idx,
                val_idx,
                split_seed + 1000 + ci,
            )
            diag_fields = (
                adapter.tuning_diag_fields(diag)
                if adapter.tuning_diag_fields is not None
                else {}
            )
            results.append(
                {
                    "candidate_index": int(ci),
                    "val_rmse_norm": float(val_rmse),
                    "hparams": adapter.canonical_hparams(candidate),
                    "diag": diag_fields,
                }
            )
            label = adapter.tuning_candidate_label(candidate) if adapter.tuning_candidate_label else ""
            LOGGER.info("cand=%02d %s val_rmse=%.4f", ci, label, val_rmse)

        finite_results = sorted(
            [row for row in results if np.isfinite(row["val_rmse_norm"])],
            key=lambda row: row["val_rmse_norm"],
        )
        if finite_results:
            top_results = finite_results[: adapter.top_k_reuse]
            best_hp = dict(top_results[0]["hparams"])
            best_rmse = float(top_results[0]["val_rmse_norm"])
            selected_candidate = int(top_results[0]["candidate_index"])
        else:
            top_results = []
            best_hp = dict(adapter.default_hparams)
            best_rmse = float("nan")
            selected_candidate = -1

        adapter.tuning_cache[group_key] = {
            "group_key_json": group_key_json,
            "anchor_source_file": os.path.basename(source_file),
            "space_info": space_info,
            "top_results": top_results,
            "all_results": results,
        }
        cache_rank = 0 if top_results else -1
    else:
        cache = adapter.tuning_cache[group_key]
        top_results = cache.get("top_results", [])
        mode = "reuse_group_top_k"
        LOGGER.info("tuning group=%s", group_key_json)
        LOGGER.info(
            "reusing top %s candidates from %s",
            len(top_results),
            cache.get("anchor_source_file", ""),
        )
        candidates = (
            [dict(adapter.default_hparams)]
            if not top_results
            else [dict(row["hparams"]) for row in top_results]
        )
        local_results: list[dict[str, Any]] = []
        for ci, candidate in enumerate(candidates):
            val_rmse, diag = adapter.evaluate_candidate(
                bundle,
                candidate,
                train_idx,
                val_idx,
                split_seed + 5000 + ci,
            )
            diag_fields = (
                adapter.tuning_diag_fields(diag)
                if adapter.tuning_diag_fields is not None
                else {}
            )
            local_results.append(
                {
                    "candidate_index": int(ci),
                    "cache_rank": int(ci),
                    "val_rmse_norm": float(val_rmse),
                    "hparams": adapter.canonical_hparams(candidate),
                    "diag": diag_fields,
                }
            )
            label = adapter.tuning_candidate_label(candidate) if adapter.tuning_candidate_label else ""
            LOGGER.info("topk=%s %s val_rmse=%.4f", ci, label, val_rmse)

        finite_results = sorted(
            [row for row in local_results if np.isfinite(row["val_rmse_norm"])],
            key=lambda row: row["val_rmse_norm"],
        )
        if finite_results:
            best_hp = dict(finite_results[0]["hparams"])
            best_rmse = float(finite_results[0]["val_rmse_norm"])
            selected_candidate = int(finite_results[0]["candidate_index"])
            cache_rank = int(finite_results[0].get("cache_rank", selected_candidate))
        else:
            best_hp = dict(adapter.default_hparams)
            best_rmse = float("nan")
            selected_candidate = -1
            cache_rank = -1

    return best_hp, {
        "val_rmse_norm": best_rmse,
        "val_loss": best_rmse,
        "n_train_tune": int(train_idx.size),
        "n_val_tune": int(val_idx.size),
        "selected_candidate": int(selected_candidate),
        "tuning_mode": mode,
        "tuning_group_key": group_key_json,
        "tuning_failures": [],
        "tuning_cache_rank": int(cache_rank),
        "initial_search_n": int(adapter.initial_random_search),
        "reuse_top_k": int(adapter.top_k_reuse),
        "space_info": space_info,
    }


_default_tune_info = default_tune_info
