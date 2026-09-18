from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def dataset_content_hash(bundle: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(bundle):
        value = bundle[key]
        if not isinstance(value, np.ndarray):
            continue
        array = np.asarray(value)
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        if array.dtype.kind in {"O", "U", "S"}:
            digest.update(canonical_json(array.tolist()).encode("utf-8"))
        else:
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _safe_component(value: Any, *, prefix: str) -> str:
    text = str(value)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}={digest}"


class BaselinePersistence:
    def __init__(
        self,
        method: str,
        *,
        root: str | Path = "outputs/fits",
        tuning_state_dir: str | Path | None = None,
        selected_hparams_path: str | Path | None = None,
    ):
        self.method = str(method)
        self.root = Path(root) / self.method
        self.root.mkdir(parents=True, exist_ok=True)
        self.fit_registry_path = self.root / "fit_registry.parquet"

        base_state = Path(tuning_state_dir or (Path("outputs") / "tuning_state"))
        self.tuning_state_root = base_state / f"method={self.method}"
        self.tuning_state_root.mkdir(parents=True, exist_ok=True)
        self.trials_root = self.tuning_state_root / "trials"
        self.selections_root = self.tuning_state_root / "selections"
        self.trials_root.mkdir(parents=True, exist_ok=True)
        self.selections_root.mkdir(parents=True, exist_ok=True)

        self.tuning_trials_path = self.tuning_state_root / "tuning_trials.parquet"
        self.selected_hparams_path = Path(selected_hparams_path) if selected_hparams_path else self.tuning_state_root / "selected_hparams.parquet"

    def _trial_path(
        self,
        *,
        dataset_uid: str,
        stage: str,
        candidate_index: int,
        hparams_json: str,
    ) -> Path:
        ds = _safe_component(dataset_uid, prefix="dataset")
        hp_hash = hashlib.sha256(str(hparams_json).encode("utf-8")).hexdigest()[:16]
        stage_safe = str(stage or "single").replace("/", "_")
        return self.trials_root / ds / f"stage={stage_safe}" / f"candidate={int(candidate_index):05d}_{hp_hash}.json"

    def load_trial(
        self,
        *,
        dataset_uid: str,
        stage: str,
        candidate_index: int,
        hparams: dict[str, Any],
        tuning_protocol_hash: str | None = None,
        seed: int | None = None,
    ) -> dict[str, Any] | None:
        hparams_json = canonical_json(hparams)
        path = self._trial_path(
            dataset_uid=dataset_uid,
            stage=stage,
            candidate_index=candidate_index,
            hparams_json=hparams_json,
        )
        if not path.exists():
            return None
        row = json.loads(path.read_text(encoding="utf-8"))
        if str(row["hparams_json"]) != hparams_json:
            return None
        if tuning_protocol_hash is not None and str(row["tuning_protocol_hash"]) != str(tuning_protocol_hash):
            return None
        if seed is not None and int(row["seed"]) != int(seed):
            return None
        return row

    def write_tuning_trial(self, trial: dict[str, Any]) -> None:
        row = dict(trial)
        row["hparams_json"] = canonical_json(json.loads(row["hparams_json"]))
        path = self._trial_path(
            dataset_uid=str(row["dataset_uid"]),
            stage=str(row["stage"]),
            candidate_index=int(row["candidate_index"]),
            hparams_json=str(row["hparams_json"]),
        )
        _atomic_write_text(path, json.dumps(row, sort_keys=True, indent=2, default=_json_default) + "\n")

    def iter_trial_rows(self, *, dataset_uid: str | None = None) -> list[dict[str, Any]]:
        files = sorted(self.trials_root.rglob("*.json"))
        rows: list[dict[str, Any]] = []
        for path in files:
            row = json.loads(path.read_text(encoding="utf-8"))
            if dataset_uid is not None and str(row["dataset_uid"]) != str(dataset_uid):
                continue
            rows.append(row)
        return rows

    def _selection_path(self, dataset_uid: str) -> Path:
        return self.selections_root / _safe_component(dataset_uid, prefix="dataset") / "selection.json"

    def write_selection(self, row: dict[str, Any]) -> None:
        payload = dict(row)
        payload["method"] = self.method
        if not isinstance(payload["selected_hparams_json"], str):
            raise TypeError("selected_hparams_json must be a JSON string.")
        payload["selected_hparams_json"] = canonical_json(
            json.loads(payload["selected_hparams_json"])
        )
        payload["selected_hparams_hash"] = sha256_json(json.loads(payload["selected_hparams_json"]))
        _atomic_write_text(
            self._selection_path(str(payload["dataset_uid"])),
            json.dumps(payload, sort_keys=True, indent=2, default=_json_default) + "\n",
        )
        self.consolidate_tuning_state()

    def load_selection(
        self,
        *,
        dataset_uid: str,
        dataset_hash: str | None = None,
        raw_dataset_hash: str | None = None,
        strict_hash: bool = True,
    ) -> dict[str, Any] | None:
        if self.selected_hparams_path.exists():
            frame = pd.read_parquet(self.selected_hparams_path)
            subset = frame.loc[
                (frame["method"].astype(str) == self.method)
                & (frame["dataset_uid"].astype(str) == str(dataset_uid))
            ]
            if len(subset):
                row = subset.iloc[-1].to_dict()
                self._validate_selection_hashes(row, dataset_hash, raw_dataset_hash, strict_hash)
                return row

        path = self._selection_path(dataset_uid)
        if not path.exists():
            return None
        row = json.loads(path.read_text(encoding="utf-8"))
        self._validate_selection_hashes(row, dataset_hash, raw_dataset_hash, strict_hash)
        return row

    @staticmethod
    def _validate_selection_hashes(
        row: dict[str, Any],
        dataset_hash: str | None,
        raw_dataset_hash: str | None,
        strict_hash: bool,
    ) -> None:
        mismatches = []
        if dataset_hash:
            if "dataset_hash" not in row or str(row["dataset_hash"]) != str(dataset_hash):
                mismatches.append("support dataset hash")
        if raw_dataset_hash:
            if "raw_dataset_hash" not in row or str(row["raw_dataset_hash"]) != str(raw_dataset_hash):
                mismatches.append("raw dataset hash")
        if mismatches and strict_hash:
            raise ValueError(
                "Saved hyperparameters do not match the current benchmark task: "
                + ", ".join(mismatches)
            )

    def consolidate_tuning_state(self, *, expected_dataset_uids: Iterable[str] | None = None) -> dict[str, Any]:
        trials = self.iter_trial_rows()
        if trials:
            _atomic_write_parquet(self.tuning_trials_path, pd.DataFrame(trials))

        selection_rows: list[dict[str, Any]] = []
        for path in sorted(self.selections_root.rglob("selection.json")):
            selection_rows.append(json.loads(path.read_text(encoding="utf-8")))
        if selection_rows:
            frame = pd.DataFrame(selection_rows)
            sort_cols = [col for col in ("domain", "support_size", "gamma", "dataset_uid") if col in frame]
            if sort_cols:
                frame = frame.sort_values(sort_cols).reset_index(drop=True)
            _atomic_write_parquet(self.selected_hparams_path, frame)

        expected = sorted(set(map(str, expected_dataset_uids or [])))
        completed = sorted({str(row["dataset_uid"]) for row in selection_rows})
        missing = sorted(set(expected) - set(completed))
        return {
            "method": self.method,
            "selected_hparams_path": str(self.selected_hparams_path),
            "tuning_trials_path": str(self.tuning_trials_path),
            "expected_dataset_uids": expected,
            "completed_dataset_uids": completed,
            "missing_dataset_uids": missing,
            "manifest_hash": self.manifest_hash() if self.selected_hparams_path.exists() else "",
        }

    def manifest_hash(self) -> str:
        if not self.selected_hparams_path.exists():
            return ""
        digest = hashlib.sha256()
        with self.selected_hparams_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def record_fit(
        self,
        *,
        meta: dict[str, Any],
        hparams: dict[str, Any],
        tune_info: dict[str, Any],
        fit_seed: int,
        final_fit_seed_root: int,
        query_seed: int,
        dataset_hash: str,
        train_diag: dict[str, Any],
        run_id: str,
        manifest_hash: str,
    ) -> dict[str, Any]:
        task_scope = "one_step_and_sequential_h1_to_h5"
        fit_id = hashlib.sha256(canonical_json({
            "method": self.method,
            "dataset_uid": meta["dataset_uid"],
            "fit_seed": int(fit_seed),
            "final_fit_seed_root": int(final_fit_seed_root),
            "query_seed": int(query_seed),
            "run_id": run_id,
            "dataset_hash": dataset_hash,
            "manifest_hash": manifest_hash,
            "hparams": hparams,
            "task_scope": task_scope,
        }).encode("utf-8")).hexdigest()[:24]
        row = {
            "fit_id": fit_id,
            "method": self.method,
            "dataset_uid": meta["dataset_uid"],
            "dataset_hash": dataset_hash,
            "selected_trial_id": tune_info["selected_trial_id"],
            "selected_hparams_json": canonical_json(hparams),
            "selected_hparams_hash": sha256_json(hparams),
            "manifest_hash": manifest_hash,
            "fit_seed": int(fit_seed),
            "final_fit_seed_root": int(final_fit_seed_root),
            "query_seed": int(query_seed),
            "fit_time": float(train_diag["fit_time_sec"]),
            "normalization_identity": "support_bundle_outcome_normalization",
            "task_scope": task_scope,
            "status": "completed",
        }
        existing = pd.read_parquet(self.fit_registry_path) if self.fit_registry_path.exists() else pd.DataFrame()
        if not existing.empty and "fit_id" in existing and fit_id in set(existing["fit_id"].astype(str)):
            return {"fit_id": fit_id, "dataset_hash": dataset_hash, "task_scope": task_scope}
        _atomic_write_parquet(
            self.fit_registry_path,
            pd.concat([existing, pd.DataFrame([row])], ignore_index=True),
        )
        return {"fit_id": fit_id, "dataset_hash": dataset_hash, "task_scope": task_scope}
