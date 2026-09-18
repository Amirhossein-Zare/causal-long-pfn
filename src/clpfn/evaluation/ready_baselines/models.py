from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression

from clpfn.config.defaults import HIDDEN_SENTINEL
from clpfn.evaluation.core import benchmark as common


@dataclass(frozen=True)
class ReadyBaselineSpec:
    method_name: str
    title: str
    family: str
    kind: str
    max_horizon: int = common.PROJECTION_HORIZON
    min_train_samples: int = 16
    max_train_samples: int = 200_000
    random_seed: int = common.SEED


SPECS: dict[str, ReadyBaselineSpec] = {
    "persistence": ReadyBaselineSpec(
        method_name="persistence",
        title="Persistence (last observed outcome)",
        family="Ready-file classical baseline",
        kind="persistence",
    ),
    "linear_autoregressive": ReadyBaselineSpec(
        method_name="linear_autoregressive",
        title="Linear Autoregressive",
        family="Ready-file classical baseline",
        kind="ar",
    ),
}

def normalize_method(method: str) -> str:
    key = str(method).strip().lower()
    if key not in SPECS:
        choices = ", ".join(sorted(SPECS))
        raise ValueError(f"Unknown ready-file baseline {method!r}. Available methods: {choices}.")
    return key


def spec_from_config(method: str, config: dict[str, Any] | None = None) -> ReadyBaselineSpec:
    base = SPECS[normalize_method(method)]
    values = dict(config or {})
    allowed = {
        "max_horizon",
        "min_train_samples",
        "max_train_samples",
        "random_seed",
    }
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise KeyError(f"Unsupported ready-baseline config keys for {base.method_name}: {unknown}")
    merged = {field: getattr(base, field) for field in base.__dataclass_fields__}
    merged.update(values)
    spec = ReadyBaselineSpec(**merged)
    if spec.max_horizon < 1:
        raise ValueError("max_horizon must be at least 1.")
    if spec.min_train_samples < 1:
        raise ValueError("min_train_samples must be at least 1.")
    if spec.max_train_samples < 1:
        raise ValueError("max_train_samples must be at least 1.")
    return spec


@dataclass
class FittedReadyBaseline:
    spec: ReadyBaselineSpec
    estimator: Any | None
    fit_diagnostics: dict[str, Any]


def _support_arrays(support_context: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = ("support_x", "support_actions", "support_static", "d_input")
    missing = [key for key in required if key not in support_context]
    if missing:
        raise KeyError(f"Ready support_context is missing required keys: {missing}")
    x = np.asarray(support_context["support_x"], dtype=np.float64)
    actions = np.asarray(support_context["support_actions"], dtype=np.int64)
    static = np.asarray(support_context["support_static"], dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"support_x must be rank 3, got shape {x.shape}.")
    if actions.shape[:2] != x.shape[:2]:
        raise ValueError("support_actions must align with support_x in patient/time dimensions.")
    if static.ndim != 2 or static.shape[0] != x.shape[0]:
        raise ValueError("support_static must be rank 2 and align with support patients.")
    return x, actions, static


def _valid_lengths(x: np.ndarray) -> np.ndarray:
    outcome = x[:, :, -1]
    valid = np.isfinite(outcome) & (outcome != float(HIDDEN_SENTINEL))
    has_valid = valid.any(axis=1)
    last_valid = valid.shape[1] - np.argmax(valid[:, ::-1], axis=1)
    return np.where(has_valid, last_valid, 0).astype(np.int64)


def _outcome_usable_mask(
    mapping: dict[str, Any],
    key: str,
    x: np.ndarray,
) -> np.ndarray:
    outcome = np.asarray(x[..., -1], dtype=np.float64)
    usable = np.isfinite(outcome) & (outcome != float(HIDDEN_SENTINEL))
    value = mapping.get(key)
    if value is not None:
        observed = np.asarray(value, dtype=bool)
        if observed.shape != usable.shape:
            raise ValueError(
                f"{key} must exactly match the outcome array; "
                f"got {observed.shape} for outcomes {usable.shape}."
            )
    return usable


def _subsample_rows(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= max_samples:
        return X, y
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(y), size=max_samples, replace=False))
    return X[idx], y[idx]


def fit_ready_baseline(
    spec: ReadyBaselineSpec,
    support_context: dict[str, Any],
) -> FittedReadyBaseline:
    if spec.kind == "persistence":
        return FittedReadyBaseline(
            spec=spec,
            estimator=None,
            fit_diagnostics={
                "fit_status": "not_required",
                "fit_samples": 0,
                "fit_features": 0,
            },
        )

    support_x, _, _ = _support_arrays(support_context)
    lengths = _valid_lengths(support_x)
    outcome_usable = _outcome_usable_mask(
        support_context,
        "support_outcome_observed_mask",
        support_x,
    )

    X_rows: list[np.ndarray] = []
    y_rows: list[float] = []

    if spec.kind != "ar":
        raise ValueError(f"Unsupported ready baseline kind: {spec.kind!r}")
    for patient_idx, length in enumerate(lengths):
        y = support_x[patient_idx, :length, -1]
        for t in range(int(length) - 1):
            required = np.arange(t, t + 2, dtype=np.int64)
            if not outcome_usable[patient_idx, required].all():
                continue
            X_rows.append(np.asarray([y[t]], dtype=np.float64))
            y_rows.append(float(y[t + 1]))

    if not X_rows:
        raise RuntimeError(
            f"{spec.method_name} found no valid support rows for fitting."
        )

    X_train = np.vstack(X_rows).astype(np.float64, copy=False)
    y_train = np.asarray(y_rows, dtype=np.float64)
    finite = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
    X_train = X_train[finite]
    y_train = y_train[finite]
    X_train, y_train = _subsample_rows(
        X_train,
        y_train,
        max_samples=int(spec.max_train_samples),
        seed=int(spec.random_seed),
    )

    if len(y_train) < int(spec.min_train_samples):
        raise RuntimeError(
            f"{spec.method_name} requires at least {spec.min_train_samples} finite training "
            f"rows, but found {len(y_train)}."
        )

    estimator: Any = LinearRegression(fit_intercept=True, n_jobs=1)
    estimator.fit(X_train, y_train)
    return FittedReadyBaseline(
        spec=spec,
        estimator=estimator,
        fit_diagnostics={
            "fit_status": "fit",
            "fit_samples": int(len(y_train)),
            "fit_features": int(X_train.shape[1]),
        },
    )


def _query_arrays(task: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = (
        "query_x",
        "query_actions",
        "query_static",
        "t_obs",
        "tau",
        "current_y_model_norm",
    )
    missing = [key for key in required if key not in task]
    if missing:
        raise KeyError(f"Ready task is missing required keys: {missing}")
    return (
        np.asarray(task["query_x"], dtype=np.float64),
        np.asarray(task["query_actions"], dtype=np.int64),
        np.asarray(task["query_static"], dtype=np.float64),
        np.asarray(task["t_obs"], dtype=np.int64),
        np.asarray(task["tau"], dtype=np.int64),
    )


def _current_model_unclipped(task: dict[str, Any]) -> np.ndarray:
    out_std = float(task["out_std"])
    if out_std <= 0:
        raise ValueError("Ready task out_std must be positive.")
    return (
        np.asarray(task["current_y_raw"], dtype=np.float64) - float(task["out_mean"])
    ) / out_std


def _persistence_paths(task: dict[str, Any], max_horizon: int) -> np.ndarray:
    current = _current_model_unclipped(task)
    return np.repeat(current[:, None], max_horizon, axis=1)


def _ar_paths(
    fitted: FittedReadyBaseline,
    task: dict[str, Any],
    max_horizon: int,
) -> np.ndarray:
    spec = fitted.spec
    if fitted.estimator is None:
        raise RuntimeError(f"{spec.method_name} has no fitted estimator.")
    query_x, _, _, t_obs, _ = _query_arrays(task)
    n_rows = query_x.shape[0]
    paths = np.empty((n_rows, max_horizon), dtype=np.float64)
    histories = np.zeros((n_rows, 1), dtype=np.float64)
    outcome_usable = _outcome_usable_mask(
        task,
        "query_outcome_observed_mask",
        query_x,
    )
    valid_rows: list[int] = []
    for row_idx, t in enumerate(t_obs):
        if int(t) < 0:
            raise RuntimeError(
                f"{spec.method_name} row {row_idx} lacks the required AR history."
            )
        required = np.asarray([int(t)], dtype=np.int64)
        if not outcome_usable[row_idx, required].all():
            raise RuntimeError(
                f"{spec.method_name} row {row_idx} has non-finite values in its AR history."
            )
        histories[row_idx, 0] = query_x[row_idx, int(t), -1]
        valid_rows.append(row_idx)
    active = np.asarray(valid_rows, dtype=np.int64)
    for step in range(max_horizon):
        pred = np.asarray(fitted.estimator.predict(histories[active]), dtype=np.float64).reshape(-1)
        paths[active, step] = pred
        feedback = np.clip(pred, -common.TARGET_NORM_CLIP, common.TARGET_NORM_CLIP)
        histories[active, 0] = feedback
    return paths


def predict_model_norm_paths(
    fitted: FittedReadyBaseline,
    task: dict[str, Any],
) -> np.ndarray:
    tau = np.asarray(task["tau"], dtype=np.int64)
    max_horizon = int(max(1, tau.max(initial=1)))
    max_horizon = min(max_horizon, int(fitted.spec.max_horizon))
    if fitted.spec.kind == "persistence":
        paths = _persistence_paths(task, max_horizon)
    elif fitted.spec.kind == "ar":
        paths = _ar_paths(fitted, task, max_horizon)
    else:
        raise ValueError(f"Unsupported ready baseline kind: {fitted.spec.kind!r}")
    if not np.isfinite(paths).all():
        raise RuntimeError(f"{fitted.spec.method_name} produced non-finite predictions.")
    return paths
