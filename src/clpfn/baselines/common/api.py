from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from clpfn.evaluation.core import benchmark as common


@dataclass
class Prediction:
    pred_norm: float
    predict_time_sec: float
    path: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainedArtifacts:
    payload: Any
    train_diag: dict[str, Any]


@dataclass
class BaselineAdapter:
    method_name: str
    method_family: str
    title: str
    default_hparams: dict[str, Any]
    tuning_cache: dict[Any, Any]

    hyperparameter_space: Callable[[dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]]
    sample_candidates: Callable[[dict[str, Any], int, int], list[dict[str, Any]]]
    canonical_hparams: Callable[[dict[str, Any]], dict[str, Any]]
    evaluate_candidate: Callable[[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray, int], tuple[float, dict[str, Any]]]
    train_final: Callable[[dict[str, Any], dict[str, Any], np.ndarray, int], TrainedArtifacts]
    predict_rows: Callable[[Any, dict[str, Any], np.ndarray, np.ndarray, np.ndarray], list[Prediction]]
    extra_record_fields: Callable[[dict[str, Any], Prediction, dict[str, Any], dict[str, Any]], dict[str, Any]]
    extra_meta_fields: Callable[[dict[str, Any]], dict[str, Any]]

    cleanup_payload: Callable[[Any], None] | None = None
    tuning_diag_fields: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    tuning_candidate_label: Callable[[dict[str, Any]], str] | None = None
    initial_random_search: int = 40
    top_k_reuse: int = 1
    tune_val_frac: float = 0.20
    val_min: int = 8
    val_max: int = 48
    tuning_strategy: str = "grouped_support_search40_reuse1"
    run_id: str = ""
    output_dir: Path | None = None
    device_label: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            self.refresh_run_id()
        if self.output_dir is None:
            self.output_dir = Path("outputs") / "eval" / f"{self.method_name}"

    def refresh_run_id(self) -> None:
        self.run_id = f"{self.method_name}_seed{common.SEED}_{int(time.time())}"


def canonical_hparams(hparams: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in dict(hparams).items():
        if isinstance(value, tuple):
            clean[key] = list(value)
        elif isinstance(value, list):
            clean[key] = [
                float(x) if isinstance(x, float) else int(x) if isinstance(x, int) else x
                for x in value
            ]
        elif isinstance(value, np.integer):
            clean[key] = int(value)
        elif isinstance(value, np.floating):
            clean[key] = float(value)
        elif isinstance(value, np.bool_):
            clean[key] = bool(value)
        else:
            clean[key] = value
    return clean


def single_model_train_final(train_fn: Callable[..., tuple[Any, dict[str, Any]]]):

    def _train_final(bundle, hparams, context_idx, seed):
        model, train_diag = train_fn(bundle, hparams, context_idx, seed)
        return TrainedArtifacts(payload=model, train_diag=train_diag)

    return _train_final


def paired_model_train_final(train_fn: Callable[..., tuple[Any, Any, dict[str, Any]]]):

    def _train_final(bundle, hparams, context_idx, seed):
        encoder, decoder, train_diag = train_fn(bundle, hparams, context_idx, seed)
        return TrainedArtifacts(payload=(encoder, decoder), train_diag=train_diag)

    return _train_final


def single_rollout_predict_rows(
    predict_fn: Callable[..., tuple[float, np.ndarray]],
    *,
    unpack_pair_payload: bool = False,
):

    def _predict_rows(payload, query_bundle, rows, current_ts, target_ts):
        args = payload if unpack_pair_payload else (payload,)
        out = []
        for row_id, current_t, target_t in zip(rows, current_ts, target_ts):
            started = time.time()
            pred_norm, pred_path = predict_fn(
                *args,
                query_bundle,
                row_id=int(row_id),
                t_obs=int(current_t),
                t_target=int(target_t),
            )
            out.append(
                Prediction(
                    pred_norm=float(pred_norm),
                    path=np.asarray(pred_path),
                    predict_time_sec=float(time.time() - started),
                )
            )
        return out

    return _predict_rows


def train_diag_record_fields(
    *,
    float_keys: tuple[str, ...] = (),
    int_keys: tuple[str, ...] = (),
):
    def _fields(train_diag, _prediction, tune_info, _meta):
        fields = {
            key: float(train_diag.get(key, np.nan))
            for key in float_keys
        }
        fields.update(
            {
                key: int(train_diag.get(key, 0))
                for key in int_keys
            }
        )
        return fields

    return _fields


def baseline_port_metadata(**extra: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "baseline_runtime": "pytorch_training",
        "baseline_tuning_contract": "grouped_support_tuning_domain_plus_support_size",
        "baseline_target_convention": "CLPFN benchmark normalized target y[t_target]",
    }
    meta.update(extra)
    return meta


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def require_baseline_config(config: dict[str, Any] | None, method: str) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError(f"{method} evaluation config must include a baseline section.")
    for key in ("limits", "default_hparams", "search_space"):
        if key not in config:
            raise KeyError(f"{method} baseline config is missing required key '{key}'.")
    return config


def replace_mapping(target: dict[str, Any], values: dict[str, Any] | None) -> dict[str, Any]:
    from copy import deepcopy

    target.clear()
    target.update(deepcopy(values or {}))
    return target
