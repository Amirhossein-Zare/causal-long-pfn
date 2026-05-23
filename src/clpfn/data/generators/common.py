"""Shared raw-pickle helpers for benchmark generators.

The public raw-pickle contract is intentionally small at the top level:

    support_data, test_data, test_data_factuals, test_data_seq, scaling_data

Each raw split also exposes one canonical shape used by PFN and baseline
evaluators:

    states, outcomes, actions, sequence_lengths, static_features

Generators may use domain-specific working dictionaries internally, but saved
splits are standardized before they reach PFN or baseline evaluation code.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


RAW_SCHEMA_VERSION = "clpfn_benchmark_v1"
CANONICAL_SPLIT_KEYS = ("states", "outcomes", "actions", "sequence_lengths", "static_features")
DEFAULT_STATIC_WIDTH = 5


def as_path(path: str | os.PathLike[str] | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def ensure_output_dir(path: str | os.PathLike[str] | Path) -> Path:
    """Create an output directory and remove stale generated pickle files."""
    out = as_path(path)
    if out.exists():
        for child in out.iterdir():
            if child.is_file() and child.suffix in {".p", ".pkl", ".pickle"}:
                child.unlink()
    out.mkdir(parents=True, exist_ok=True)
    return out


def take_rows(raw: Mapping[str, Any], idx: Iterable[int]) -> dict[str, Any]:
    """Take first-axis rows for arrays whose first dimension matches a dataset."""
    idx = np.asarray(idx, dtype=np.int64)
    out: dict[str, Any] = {}

    # Infer row count from the first non-scalar ndarray.
    n_rows = None
    for value in raw.values():
        if isinstance(value, np.ndarray) and value.ndim > 0:
            n_rows = value.shape[0]
            break

    for key, value in raw.items():
        if (
            isinstance(value, np.ndarray)
            and value.ndim > 0
            and n_rows is not None
            and value.shape[0] == n_rows
        ):
            out[key] = value[idx].copy()
        elif isinstance(value, np.ndarray):
            out[key] = value.copy()
        else:
            out[key] = value

    return out


def concat_raw(raw_list: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Concatenate a list of raw dataset dictionaries along the row axis."""
    if not raw_list:
        raise ValueError("concat_raw received an empty list")

    keys: set[str] = set()
    for raw in raw_list:
        keys.update(raw.keys())

    out: dict[str, Any] = {}
    for key in keys:
        values = [raw[key] for raw in raw_list if key in raw]
        if not values:
            continue
        if isinstance(values[0], np.ndarray):
            out[key] = np.concatenate(values, axis=0)
        else:
            out[key] = values[0]
    return out


def save_pickle(obj: Any, path: str | os.PathLike[str] | Path) -> Path:
    path = as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def encode_binary_pair_actions(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Encode two binary treatment flags as the shared 4-action convention."""
    return (
        np.asarray(first, dtype=np.int64)
        + 2 * np.asarray(second, dtype=np.int64)
    ).astype(np.int64)


def repeat_static_as_state(values: np.ndarray, time_steps: int) -> np.ndarray:
    """Repeat patient-level values over time to build a time-varying state."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Expected static values with shape [N] or [N, D], got {arr.shape}.")
    return np.repeat(arr[:, None, :], int(time_steps), axis=1).astype(np.float32)


def fixed_static_features(values: np.ndarray, *, rows: int | None = None, width: int = DEFAULT_STATIC_WIDTH) -> np.ndarray:
    """Return a fixed-width patient static matrix."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 0:
        if rows is None:
            raise ValueError("rows is required for scalar static features.")
        arr = np.zeros((int(rows), 0), dtype=np.float32)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    n_rows = int(arr.shape[0] if rows is None else rows)
    out = np.zeros((n_rows, int(width)), dtype=np.float32)
    n_copy_rows = min(n_rows, arr.shape[0])
    n_copy_cols = min(int(width), arr.shape[1])
    if n_copy_rows > 0 and n_copy_cols > 0:
        out[:n_copy_rows, :n_copy_cols] = arr[:n_copy_rows, :n_copy_cols]
    out[~np.isfinite(out)] = 0.0
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
            value = value.reshape(value.shape[0], -1)
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
    """Convert a domain working split to the canonical saved split."""
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
    if static_key is not None and static_key in raw:
        static = fixed_static_features(raw[static_key], rows=n_rows, width=static_width)
    else:
        static = fixed_static_features(_stack_static_columns(raw, static_keys, n_rows), rows=n_rows, width=static_width)

    out = {
        "states": states.astype(np.float32),
        "outcomes": outcomes.astype(np.float32),
        "actions": actions.astype(np.int64),
        "sequence_lengths": sequence_lengths,
        "static_features": static,
        "raw_schema_version": RAW_SCHEMA_VERSION,
        "canonical_keys": CANONICAL_SPLIT_KEYS,
    }

    for key in ("patient_current_t", "patient_ids_all_trajectories", "patient_ids"):
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
    """Add canonical fields to every raw split in a generated pickle map."""
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
    return out
