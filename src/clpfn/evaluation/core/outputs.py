from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import shutil

import numpy as np
import pandas as pd

from clpfn.evaluation.core.summaries import print_summary_table, summarize_domain_task_rmse


DOMAIN_TASK_SUMMARY_FILENAME = "domain_task_normalized_rmse.csv"
SEQUENTIAL_HORIZON_SUMMARY_FILENAME = "sequential_horizon_normalized_rmse.csv"
PREDICTION_ROWS_FILENAME = "prediction_rows.parquet"
SUMMARY_INPUT_COLUMNS = [
    "domain",
    "method",
    "run_id",
    "gamma",
    "reported_task",
    "task_name",
    "tau",
    "dataset_id",
    "global_dataset_id",
    "source_file",
    "sq_error_norm",
]


@dataclass(frozen=True)
class EvaluationOutputPaths:
    output_dir: Path
    domain_task_summary_csv: Path
    prediction_rows_parquet: Path
    partitions_dir: Path


def prepare_output_paths(output_dir: str | Path) -> EvaluationOutputPaths:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return EvaluationOutputPaths(
        output_dir=root,
        domain_task_summary_csv=root / DOMAIN_TASK_SUMMARY_FILENAME,
        prediction_rows_parquet=root / PREDICTION_ROWS_FILENAME,
        partitions_dir=root / "partitions",
    )


def partition_dir(paths: EvaluationOutputPaths, *, evaluation_id: str, method: str, dataset_uid: str, task_name: str) -> Path:
    return paths.partitions_dir / f"evaluation_id={evaluation_id}" / f"method={method}" / f"dataset_uid={dataset_uid}" / f"task_name={task_name}"


def partition_is_complete(paths: EvaluationOutputPaths, *, evaluation_id: str, method: str, dataset_uid: str, task_name: str) -> bool:
    root = partition_dir(paths, evaluation_id=evaluation_id, method=method, dataset_uid=dataset_uid, task_name=task_name)
    return (root / "_SUCCESS").exists() and (root / "rollout_steps.parquet").exists() and (root / "prediction_rows.parquet").exists()


def write_completed_partition(
    paths: EvaluationOutputPaths,
    *,
    evaluation_id: str,
    method: str,
    dataset_uid: str,
    task_name: str,
    rollout_steps: pd.DataFrame,
    prediction_rows: pd.DataFrame,
    gmm_components: pd.DataFrame | None = None,
) -> Path:
    if rollout_steps.empty or prediction_rows.empty:
        raise ValueError("Completed evaluation partitions require non-empty rollout and prediction rows.")
    required = {
        "evaluation_id", "method", "dataset_uid", "task_name", "reported_task", "horizon",
        "current_y_raw", "current_y_model_norm", "current_y_eval_norm_unclipped",
        "prediction_raw_unclipped", "prediction_model_norm_unclipped",
        "prediction_eval_norm_unclipped", "prediction_eval_norm",
        "target_eval_norm_unclipped", "target_eval_norm",
    }
    missing = required.difference(rollout_steps.columns)
    if missing:
        raise ValueError(f"Rollout partition is missing required columns: {sorted(missing)}")
    prediction_missing = required.difference(prediction_rows.columns)
    if prediction_missing:
        raise ValueError(f"Prediction partition is missing required columns: {sorted(prediction_missing)}")
    finite = rollout_steps[["prediction_eval_norm", "target_eval_norm"]].to_numpy(dtype=float)
    if not np.isfinite(finite).all():
        raise ValueError("Rollout partition contains non-finite required normalized values.")
    root = partition_dir(paths, evaluation_id=evaluation_id, method=method, dataset_uid=dataset_uid, task_name=task_name)
    tmp = root.with_name(root.name + ".tmp")
    if tmp.exists():
        # A prior interrupted write never carries a success marker and can be rebuilt.
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=False)
    rollout_steps.to_parquet(tmp / "rollout_steps.parquet", index=False)
    prediction_rows.to_parquet(tmp / "prediction_rows.parquet", index=False)
    if gmm_components is not None and not gmm_components.empty:
        gmm_components.to_parquet(tmp / "gmm_components.parquet", index=False)
    (tmp / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        raise FileExistsError(f"Completed partition already exists: {root}")
    tmp.replace(root)
    return root


def collect_completed_partitions(paths: EvaluationOutputPaths, filename: str, *, evaluation_id: str) -> pd.DataFrame:
    root = paths.partitions_dir / f"evaluation_id={evaluation_id}"
    files = sorted(root.rglob(filename)) if root.exists() else []
    files = [path for path in files if (path.parent / "_SUCCESS").exists()]
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True) if files else pd.DataFrame()


def domain_summary_from_task_summary(domain_task_summary: pd.DataFrame) -> pd.DataFrame:
    if domain_task_summary.empty:
        return pd.DataFrame(columns=["method", "domain", "mean_normalized_rmse"])
    return (
        domain_task_summary
        .groupby(["method", "domain"], as_index=False)
        .agg(mean_normalized_rmse=("mean_norm_rmse", "mean"))
        .sort_values(["domain", "method"])
        .reset_index(drop=True)
    )


def balanced_domain_rmse(domain_summary: pd.DataFrame) -> float:
    if domain_summary.empty or "mean_normalized_rmse" not in domain_summary:
        return float("nan")
    values = pd.to_numeric(domain_summary["mean_normalized_rmse"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def write_prediction_summaries(
    prediction_df: pd.DataFrame,
    *,
    paths: EvaluationOutputPaths,
) -> dict[str, Any]:
    prediction_df.to_parquet(paths.prediction_rows_parquet, index=False)
    reported_task = prediction_df["reported_task"].astype(str)
    primary_rows = prediction_df.loc[
        reported_task.isin(["one_step_exhaustive", "sequential_h5"]),
        SUMMARY_INPUT_COLUMNS,
    ].copy()
    domain_task_summary = summarize_domain_task_rmse(primary_rows)
    domain_task_summary.to_csv(paths.domain_task_summary_csv, index=False)
    print_summary_table(domain_task_summary, "Domain/task normalized RMSE")
    del primary_rows
    sequential_rows = prediction_df.loc[
        reported_task.str.startswith("sequential_h"),
        SUMMARY_INPUT_COLUMNS,
    ].copy()
    sequential_summary = summarize_domain_task_rmse(sequential_rows)
    del sequential_rows
    sequential_path = paths.output_dir / SEQUENTIAL_HORIZON_SUMMARY_FILENAME
    sequential_summary.to_csv(sequential_path, index=False)

    domain_summary = domain_summary_from_task_summary(domain_task_summary)
    out: dict[str, Any] = {
        "prediction_rows": prediction_df,
        "domain_task_summary": domain_task_summary,
        "sequential_horizon_summary": sequential_summary,
        "summary_domain": domain_summary,
        "domain_balanced_norm_rmse": balanced_domain_rmse(domain_summary),
        "prediction_rows_parquet": str(paths.prediction_rows_parquet),
        "domain_task_summary_csv": str(paths.domain_task_summary_csv),
        "sequential_horizon_summary_csv": str(sequential_path),
    }

    return out
