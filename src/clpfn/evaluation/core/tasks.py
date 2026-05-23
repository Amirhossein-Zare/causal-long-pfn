from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from clpfn.evaluation.core import benchmark as common


@dataclass(frozen=True)
class RawTaskSpec:
    raw_key: str
    task_name: str
    row_builder: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawTaskRows:
    spec: RawTaskSpec
    raw: dict[str, Any]
    rows: np.ndarray
    current_ts: np.ndarray
    target_ts: np.ndarray

    @property
    def task_name(self) -> str:
        return self.spec.task_name

    @property
    def n_eval(self) -> int:
        return int(len(self.rows))


def task_step_from_task_and_tau(task_name: str, tau: int | float) -> str:
    task_name_lower = str(task_name).lower()
    tau_i = int(tau)

    if "one_step" in task_name_lower or "one-step" in task_name_lower or tau_i == 1:
        return "one_step"

    if "seq_h5" in task_name_lower or "horizon_5" in task_name_lower or "h5" in task_name_lower or tau_i == 5:
        return "horizon_5"

    return "other"


def is_one_step_task(task_name: str, tau: int | float) -> bool:
    return task_step_from_task_and_tau(task_name, tau) == "one_step"


def raw_task_specs(horizon: int = common.PROJECTION_HORIZON) -> tuple[RawTaskSpec, ...]:
    return (
        RawTaskSpec(
            raw_key="test_data",
            task_name="one_step_cf_final",
            row_builder=common.get_one_step_eval_rows,
        ),
        RawTaskSpec(
            raw_key="test_data_seq",
            task_name=f"seq_h{int(horizon)}_final",
            row_builder=common.get_seq_horizon_eval_rows,
            kwargs={"horizon": int(horizon)},
        ),
    )


def build_raw_task_rows(
    spec: RawTaskSpec,
    *,
    pm: dict[str, Any],
    domain: str,
    cfg: dict[str, Any],
    rng: np.random.Generator,
    max_rows: int | None = None,
) -> RawTaskRows | None:
    if spec.raw_key not in pm:
        return None
    raw = pm[spec.raw_key]

    rows, current_ts, target_ts = spec.row_builder(
        raw=raw,
        domain=domain,
        cfg=cfg,
        max_rows=max_rows,
        rng=rng,
        **spec.kwargs,
    )
    return RawTaskRows(
        spec=spec,
        raw=raw,
        rows=np.asarray(rows, dtype=np.int64),
        current_ts=np.asarray(current_ts, dtype=np.int64),
        target_ts=np.asarray(target_ts, dtype=np.int64),
    )


def iter_raw_task_rows(
    *,
    pm: dict[str, Any],
    domain: str,
    cfg: dict[str, Any],
    rng: np.random.Generator,
    max_rows: int | None = None,
    horizon: int = common.PROJECTION_HORIZON,
) -> Iterator[RawTaskRows]:
    for spec in raw_task_specs(horizon=horizon):
        task_rows = build_raw_task_rows(
            spec,
            pm=pm,
            domain=domain,
            cfg=cfg,
            rng=rng,
            max_rows=max_rows,
        )
        if task_rows is not None:
            yield task_rows


def ready_task_names(ready_map: dict[str, Any]) -> tuple[str, ...]:
    tasks = ready_map["tasks"]
    return tuple(sorted(str(name) for name in tasks.keys()))


def ready_task_row_ids(task: dict[str, Any]) -> np.ndarray:
    return np.asarray(task["rows"], dtype=np.int64)


def ready_task_n_eval(task: dict[str, Any]) -> int:
    return int(task["n_eval"])
