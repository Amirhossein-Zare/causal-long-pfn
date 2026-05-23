from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clpfn.evaluation.core import benchmark as common


READY_PICKLE_GLOB = "causal_long_pfn_ready_*_dataset_*.p"


@dataclass(frozen=True)
class ReadyBenchmarkInputs:
    ready_dirs: tuple[str, ...] = ()
    ready_paths: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None = None) -> "ReadyBenchmarkInputs":
        values = dict(values or {})
        return cls(
            ready_dirs=tuple(str(path) for path in values.get("ready_dirs", []) or []),
            ready_paths=tuple(str(path) for path in values.get("ready_paths", []) or []),
        )

    def validate(self) -> None:
        if not (self.ready_dirs or self.ready_paths):
            raise ValueError("Provide ready benchmark ready_dirs or ready_paths.")


def find_ready_pickles(inputs: ReadyBenchmarkInputs) -> list[str]:
    inputs.validate()

    roots = [Path(path) for path in inputs.ready_dirs]
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Ready benchmark directory not found: {root}")

    pfiles = [str(Path(path)) for path in inputs.ready_paths]
    for path in pfiles:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Ready benchmark pickle not found: {path}")

    for root in roots:
        pfiles.extend(str(path) for path in root.rglob(READY_PICKLE_GLOB))

    ready_files = sorted(set(pfiles))
    if not ready_files:
        raise FileNotFoundError("No ready benchmark pickles found.")
    return ready_files


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def raw_inputs_from_config(config: dict[str, Any] | None) -> common.RawBenchmarkInputs:
    return common.RawBenchmarkInputs.from_dict(config)
