from __future__ import annotations

import os
import pickle
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


RAW_SCHEMA_VERSION = "clpfn_benchmark_v2"
CANONICAL_SPLIT_KEYS = (
    "states",
    "outcomes",
    "actions",
    "sequence_lengths",
    "static_features",
    "patient_id",
    "state_observed_mask",
    "outcome_observed_mask",
)
DEFAULT_STATIC_WIDTH = 5


def as_path(path: str | os.PathLike[str] | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def ensure_output_dir(path: str | os.PathLike[str] | Path, *, overwrite: bool = False) -> Path:
    out = as_path(path)
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Benchmark build directory already exists and is non-empty: {out}. Use overwrite=True to replace it.")
        for child in out.iterdir():
            if child.is_file() and (child.suffix in {".p", ".pkl", ".pickle"} or child.name.startswith("benchmark_dataset_manifest_")):
                child.unlink()
    out.mkdir(parents=True, exist_ok=True)
    return out


def take_rows(raw: Mapping[str, Any], idx: Iterable[int]) -> dict[str, Any]:
    idx = np.asarray(idx, dtype=np.int64)
    if "sequence_lengths" not in raw:
        raise KeyError("Raw split is missing sequence_lengths.")
    n_rows = len(np.asarray(raw["sequence_lengths"]))
    if np.any((idx < 0) | (idx >= n_rows)):
        raise IndexError(f"Row selection is outside [0, {n_rows - 1}].")
    out: dict[str, Any] = {}

    for key, value in raw.items():
        if (
            isinstance(value, np.ndarray)
            and value.ndim > 0
            and value.shape[0] == n_rows
        ):
            out[key] = value[idx].copy()
        elif isinstance(value, np.ndarray):
            out[key] = value.copy()
        else:
            out[key] = value

    return out


def concat_raw(raw_list: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not raw_list:
        raise ValueError("concat_raw received an empty list")

    keys = set(raw_list[0])
    for index, raw in enumerate(raw_list[1:], start=1):
        if set(raw) != keys:
            raise KeyError(
                f"Raw split {index} has different keys: "
                f"missing={sorted(keys - set(raw))}, extra={sorted(set(raw) - keys)}"
            )

    out: dict[str, Any] = {}
    for key in keys:
        values = [raw[key] for raw in raw_list]
        if isinstance(values[0], np.ndarray):
            if not all(isinstance(value, np.ndarray) for value in values):
                raise TypeError(f"Raw key {key!r} mixes array and non-array values.")
            out[key] = np.concatenate(values, axis=0)
        else:
            if any(value != values[0] for value in values[1:]):
                raise ValueError(f"Raw key {key!r} differs across concatenated splits.")
            out[key] = values[0]
    return out


def save_pickle(obj: Any, path: str | os.PathLike[str] | Path) -> Path:
    path = as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def sha256_file(path: str | os.PathLike[str] | Path) -> str:
    digest = hashlib.sha256()
    with as_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_uid(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_patient_ids(raw: Mapping[str, Any], n_rows: int) -> np.ndarray:
    for key in ("patient_ids_all_trajectories", "patient_ids", "patient_id"):
        if key in raw:
            value = np.asarray(raw[key]).reshape(-1)
            if value.shape[0] != n_rows:
                raise ValueError(
                    f"Patient identifier key '{key}' has {value.shape[0]} rows, expected {n_rows}."
                )
            return value.astype(np.int64)
    return np.arange(n_rows, dtype=np.int64)


def _add_query_identity(raw: dict[str, Any], *, dataset_uid: str, task_name: str, is_factual_default: bool) -> None:
    n_rows = int(np.asarray(raw["outcomes"]).shape[0])
    outcomes = np.asarray(raw["outcomes"], dtype=np.float32)
    actions = np.asarray(raw["actions"], dtype=np.int64)
    lengths = np.asarray(raw["sequence_lengths"], dtype=np.int64)
    patient_ids = np.asarray(raw["patient_id"], dtype=np.int64)
    if patient_ids.shape != (n_rows,):
        raise ValueError(f"{task_name} patient identifiers have inconsistent shape.")
    current_times = np.asarray(raw["patient_current_t"], dtype=np.int64)
    if current_times.shape[0] != n_rows:
        raise ValueError(f"{task_name} current-time metadata has inconsistent row count.")
    if task_name == "sequence":
        current_times = current_times + 1
    if np.any((current_times < 0) | (current_times >= outcomes.shape[1])):
        raise ValueError(f"{task_name} current-time metadata is outside the outcome array.")
    horizon = int(raw["max_horizon"])
    if horizon < 1:
        raise ValueError(f"{task_name} max_horizon must be positive.")
    plan = np.zeros((n_rows, horizon), dtype=np.int64)
    target_path = np.full((n_rows, horizon), np.nan, dtype=np.float32)
    target_state_path = np.full((n_rows, horizon, raw["states"].shape[-1]), np.nan, dtype=np.float32)
    for row in range(n_rows):
        start = int(current_times[row])
        end = start + horizon
        target_start = start + 1
        target_end = target_start + horizon
        if end > actions.shape[1]:
            raise ValueError(f"{task_name} row {row} has an incomplete action plan.")
        if target_end > outcomes.shape[1] or target_end > raw["states"].shape[1]:
            raise ValueError(f"{task_name} row {row} has an incomplete target path.")
        plan[row] = actions[row, start:end]
        target_path[row] = outcomes[row, target_start:target_end]
        target_state_path[row] = raw["states"][row, target_start:target_end]
    if not np.isfinite(target_path).all() or not np.isfinite(target_state_path).all():
        raise ValueError(f"{task_name} rows must retain finite complete target paths.")
    first_action = plan[:, 0]
    patient_uid = np.asarray([stable_uid(dataset_uid, task_name, int(pid)) for pid in patient_ids], dtype="U64")
    origin_uid = np.asarray([stable_uid(dataset_uid, task_name, int(pid), int(origin)) for pid, origin in zip(patient_ids, current_times)], dtype="U64")
    plan_uid = np.asarray([stable_uid(origin, tuple(int(x) for x in sequence)) for origin, sequence in zip(origin_uid, plan)], dtype="U64")
    query_uid = np.asarray([stable_uid(dataset_uid, task_name, int(row), plan_id) for row, plan_id in enumerate(plan_uid)], dtype="U64")
    if "is_factual" not in raw:
        raw["is_factual"] = np.full(n_rows, is_factual_default, dtype=bool)
    factual = np.asarray(raw["is_factual"], dtype=bool)
    if factual.shape[0] != n_rows:
        raise ValueError(f"{task_name} factual metadata has inconsistent row count.")
    raw.update({
        "dataset_uid": np.full(n_rows, dataset_uid, dtype="U64"),
        "original_row_id": np.arange(n_rows, dtype=np.int64),
        "patient_id": patient_ids,
        "patient_uid": patient_uid,
        "origin_time": current_times,
        "origin_uid": origin_uid,
        "first_action": first_action,
        "planned_action_sequence": plan,
        "plan_uid": plan_uid,
        "query_uid": query_uid,
        "is_factual": factual,
        "is_counterfactual": ~factual,
        "factual_prefix_length": current_times + 1,
        "max_horizon": np.full(n_rows, horizon, dtype=np.int64),
    })
    if task_name == "sequence":
        raw["target_path_raw"] = target_path
        raw["target_state_path_raw"] = target_state_path


def add_benchmark_metadata(pickle_map: dict[str, Any], domain: str) -> dict[str, Any]:
    dataset_uid = stable_uid(domain, pickle_map["dataset_id"], pickle_map["gamma"], pickle_map["rep"], pickle_map["seed"], pickle_map["support_size"])
    out = dict(pickle_map)
    out["dataset_uid"] = dataset_uid
    out["gamma_semantics"] = "split_index" if domain == "mimic" else "confounding_strength"
    out["propensity_source"] = "not_available" if domain == "mimic" else "simulator_policy"
    horizon = int(out["projection_horizon"])
    for split_key, task_name, factual in (
        ("test_data", "one_step", bool(domain == "mimic")),
        ("test_data_seq", "sequence", bool(domain == "mimic")),
    ):
        if split_key not in out:
            continue
        raw = out[split_key]
        raw["max_horizon"] = horizon if task_name == "sequence" else 1
        _add_query_identity(raw, dataset_uid=dataset_uid, task_name=task_name, is_factual_default=factual)
        raw["propensity_source"] = np.full(len(raw["query_uid"]), out["propensity_source"], dtype="U32")
        if domain != "mimic":
            raw["confounding_strength"] = np.full(len(raw["query_uid"]), float(out["gamma"]), dtype=np.float32)
    return out


def write_dataset_manifest(pickle_map: Mapping[str, Any], raw_file: str | os.PathLike[str] | Path, *, generator_config: Mapping[str, Any], generator_source: str | os.PathLike[str] | Path) -> Path:
    raw_file = as_path(raw_file)
    support = pickle_map["support_data"]
    query_splits = [pickle_map[key] for key in ("test_data", "test_data_seq") if key in pickle_map]
    support_ids = np.asarray(support["patient_id"], dtype=np.int64)
    query_ids = np.concatenate(
        [np.asarray(split["patient_id"], dtype=np.int64) for split in query_splits]
    )
    if str(pickle_map["domain"]) == "mimic" and set(support_ids).intersection(query_ids):
        raise AssertionError("MIMIC support and query patients must be disjoint.")
    source_path = as_path(generator_source)
    manifest = {
        "benchmark_build_id": raw_file.parent.parent.name,
        "dataset_uid": str(pickle_map["dataset_uid"]),
        "domain": str(pickle_map["domain"]),
        "gamma": pickle_map["gamma"],
        "gamma_semantics": str(pickle_map["gamma_semantics"]),
        "replicate": int(pickle_map["rep"]),
        "support_size": int(pickle_map["support_size"]),
        "generator_seed": int(pickle_map["seed"]),
        "generator_configuration": dict(generator_config),
        "generator_code_hash": sha256_file(source_path),
        "raw_file": str(raw_file),
        "raw_file_hash": sha256_file(raw_file),
        "support_patient_count": int(len(np.unique(support_ids))),
        "query_patient_count": int(len(np.unique(query_ids))),
        "origin_count": int(len(np.unique(np.concatenate([split["origin_uid"] for split in query_splits])))),
        "plan_count": int(len(np.unique(np.concatenate([split["plan_uid"] for split in query_splits])))),
        "one_step_query_count": int(len(pickle_map["test_data"]["query_uid"])),
        "sequence_query_count": int(len(pickle_map["test_data_seq"]["query_uid"])),
        "maximum_sequence_horizon": int(pickle_map["projection_horizon"]),
        "propensity_source": str(pickle_map["propensity_source"]),
    }
    if str(pickle_map["domain"]) == "mimic":
        manifest["cohort_counts"] = {
            "support_patients": int(len(np.unique(support_ids))),
            "query_patients": int(len(np.unique(query_ids))),
            "support_query_patient_disjoint": True,
        }
    path = raw_file.parent / f"benchmark_dataset_manifest_{pickle_map['dataset_uid']}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return path


def encode_binary_pair_actions(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.int64)
    second = np.asarray(second, dtype=np.int64)
    if first.shape != second.shape:
        raise ValueError("Binary treatment arrays must have identical shapes.")
    if np.any((first < 0) | (first > 1) | (second < 0) | (second > 1)):
        raise ValueError("Binary treatment arrays must contain only zero and one.")
    return (first + 2 * second).astype(np.int64)


def repeat_static_as_state(values: np.ndarray, time_steps: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Expected static values with shape [N] or [N, D], got {arr.shape}.")
    return np.repeat(arr[:, None, :], int(time_steps), axis=1).astype(np.float32)


def fixed_static_features(values: np.ndarray, *, rows: int | None = None, width: int = DEFAULT_STATIC_WIDTH) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Static features must be rank 1 or 2, got shape {arr.shape}.")
    n_rows = int(arr.shape[0] if rows is None else rows)
    if arr.shape[0] != n_rows:
        raise ValueError(f"Static features have {arr.shape[0]} rows, expected {n_rows}.")
    if arr.shape[1] > int(width):
        raise ValueError(f"Static feature width {arr.shape[1]} exceeds canonical width {width}.")
    if not np.isfinite(arr).all():
        raise ValueError("Static features contain non-finite values.")
    out = np.zeros((n_rows, int(width)), dtype=np.float32)
    out[:, : arr.shape[1]] = arr
    return out


def _stack_static_columns(raw: Mapping[str, Any], keys: Iterable[str], rows: int) -> np.ndarray:
    cols = []
    for key in keys:
        if key not in raw:
            raise KeyError(f"Raw split is missing required static key '{key}'.")
        value = np.asarray(raw[key], dtype=np.float32)
        if value.ndim == 1:
            value = value[:, None]
        elif value.ndim > 2:
            raise ValueError(f"Static key '{key}' must be rank one or two, got {value.shape}.")
        if value.shape[0] != rows:
            raise ValueError(f"Static key '{key}' has {value.shape[0]} rows, expected {rows}.")
        cols.append(value)
    if not cols:
        return np.zeros((rows, 0), dtype=np.float32)
    return np.concatenate(cols, axis=1).astype(np.float32)


def standardize_raw_split(
    raw: Mapping[str, Any],
    *,
    outcome_key: str,
    state_key: str | None = "states",
    action_key: str | None = "actions",
    action_pair_keys: tuple[str, str] | None = None,
    static_key: str | None = "static_features",
    static_keys: Iterable[str] = (),
    state_from_static_key: str | None = None,
    static_width: int = DEFAULT_STATIC_WIDTH,
) -> dict[str, Any]:
    if "sequence_lengths" not in raw:
        raise KeyError("Raw split is missing required key 'sequence_lengths'.")
    sequence_lengths = np.asarray(raw["sequence_lengths"], dtype=np.int64)

    if outcome_key not in raw:
        raise KeyError(f"Raw split is missing required outcome key '{outcome_key}'.")
    outcomes = np.asarray(raw[outcome_key], dtype=np.float32)
    if outcomes.ndim == 3 and outcomes.shape[-1] == 1:
        outcomes = outcomes[:, :, 0]
    if outcomes.ndim != 2:
        raise ValueError(f"Outcome '{outcome_key}' must have shape [N, T], got {outcomes.shape}.")

    n_rows, time_steps = outcomes.shape
    if sequence_lengths.shape != (n_rows,):
        raise ValueError(
            f"sequence_lengths must have shape ({n_rows},), got {sequence_lengths.shape}."
        )

    if state_key is not None:
        if state_key not in raw:
            raise KeyError(f"Raw split is missing required state key '{state_key}'.")
        states = np.asarray(raw[state_key], dtype=np.float32)
    elif state_from_static_key is not None:
        if state_from_static_key not in raw:
            raise KeyError(f"Raw split is missing required state source key '{state_from_static_key}'.")
        states = repeat_static_as_state(raw[state_from_static_key], time_steps)
    else:
        raise ValueError("Either state_key or state_from_static_key must be provided.")

    if states.ndim == 2:
        states = states[:, :, None]
    if states.ndim != 3:
        raise ValueError(f"Canonical states must have shape [N, T, D], got {states.shape}.")
    if states.shape[0] != n_rows:
        raise ValueError(f"State rows {states.shape[0]} do not match outcome rows {n_rows}.")
    if states.shape[1] != time_steps:
        raise ValueError(
            f"State time dimension {states.shape[1]} does not match outcomes {time_steps}."
        )
    if not np.isfinite(states).all() or not np.isfinite(outcomes).all():
        raise ValueError("Canonical states and outcomes must be finite after preprocessing.")

    if action_key is not None:
        if action_key not in raw:
            raise KeyError(f"Raw split is missing required action key '{action_key}'.")
        actions = np.asarray(raw[action_key])
        if actions.ndim == 3 and actions.shape[-1] == 2:
            actions = encode_binary_pair_actions(actions[:, :, 0], actions[:, :, 1])
        elif actions.ndim == 3 and actions.shape[-1] == 1:
            actions = actions[:, :, 0]
    elif action_pair_keys is not None:
        first_key, second_key = action_pair_keys
        if first_key not in raw or second_key not in raw:
            raise KeyError(f"Raw split is missing required action pair keys {action_pair_keys}.")
        actions = encode_binary_pair_actions(raw[first_key], raw[second_key])
    else:
        raise ValueError("Either action_key or action_pair_keys must be provided.")

    actions = np.asarray(actions, dtype=np.int64)
    if actions.ndim != 2:
        raise ValueError(f"Canonical actions must have shape [N, T], got {actions.shape}.")
    if actions.shape[0] != n_rows:
        raise ValueError(f"Action rows {actions.shape[0]} do not match outcome rows {n_rows}.")
    if actions.shape[1] != time_steps:
        raise ValueError(
            f"Action time dimension {actions.shape[1]} does not match outcomes {time_steps}."
        )
    if np.any((actions < 0) | (actions >= 4)):
        raise ValueError("Canonical actions must be in [0, 3].")
    if np.any((sequence_lengths < 1) | (sequence_lengths > time_steps)):
        raise ValueError(
            f"sequence_lengths must be within [1, {time_steps}]."
        )
    if static_key is not None and static_key in raw:
        static = fixed_static_features(raw[static_key], rows=n_rows, width=static_width)
    else:
        static = fixed_static_features(_stack_static_columns(raw, static_keys, n_rows), rows=n_rows, width=static_width)

    valid_time = np.arange(time_steps, dtype=np.int64)[None, :] <= sequence_lengths[:, None]
    if "state_observed_mask" in raw:
        state_observed_mask = np.asarray(raw["state_observed_mask"], dtype=bool)
        if state_observed_mask.shape != states.shape:
            raise ValueError(
                f"state_observed_mask has shape {state_observed_mask.shape}, expected {states.shape}."
            )
    else:
        state_observed_mask = np.broadcast_to(
            valid_time[:, :, None],
            states.shape,
        ).copy()
    if "outcome_observed_mask" in raw:
        outcome_observed_mask = np.asarray(raw["outcome_observed_mask"], dtype=bool)
        if outcome_observed_mask.shape != outcomes.shape:
            raise ValueError(
                f"outcome_observed_mask has shape {outcome_observed_mask.shape}, expected {outcomes.shape}."
            )
    else:
        outcome_observed_mask = valid_time.copy()

    out = {
        "states": states.astype(np.float32),
        "outcomes": outcomes.astype(np.float32),
        "actions": actions.astype(np.int64),
        "sequence_lengths": sequence_lengths,
        "static_features": static,
        "patient_id": _split_patient_ids(raw, n_rows),
        "state_observed_mask": state_observed_mask,
        "outcome_observed_mask": outcome_observed_mask,
        "raw_schema_version": RAW_SCHEMA_VERSION,
        "canonical_keys": CANONICAL_SPLIT_KEYS,
    }

    for key in ("patient_current_t", "is_factual"):
        if key in raw:
            out[key] = np.asarray(raw[key]).copy()
    return out


def standardize_pickle_map(
    pickle_map: Mapping[str, Any],
    *,
    domain: str,
    outcome_key: str,
    state_key: str | None = "states",
    action_key: str | None = "actions",
    action_pair_keys: tuple[str, str] | None = None,
    static_key: str | None = "static_features",
    static_keys: Iterable[str] = (),
    state_from_static_key: str | None = None,
    target_state_index: int | None = None,
    static_width: int = DEFAULT_STATIC_WIDTH,
) -> dict[str, Any]:
    out = dict(pickle_map)
    for split_key in ("support_data", "test_data", "test_data_factuals", "test_data_seq"):
        if split_key in out:
            out[split_key] = standardize_raw_split(
                out[split_key],
                outcome_key=outcome_key,
                state_key=state_key,
                action_key=action_key,
                action_pair_keys=action_pair_keys,
                static_key=static_key,
                static_keys=static_keys,
                state_from_static_key=state_from_static_key,
                static_width=static_width,
            )

    out["raw_schema_version"] = RAW_SCHEMA_VERSION
    out["canonical_keys"] = CANONICAL_SPLIT_KEYS
    out["domain"] = str(domain)
    out["state_name"] = "states"
    out["outcome_name"] = "outcomes"
    out["action_name"] = "actions"
    out["static_name"] = "static_features"
    if target_state_index is not None:
        out["target_state_index"] = int(target_state_index)
    return add_benchmark_metadata(out, domain)
