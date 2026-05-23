from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from clpfn.evaluation.core.summaries import print_summary_table, summarize_domain_task_rmse


DOMAIN_TASK_SUMMARY_FILENAME = "domain_task_normalized_rmse.csv"
PREDICTION_ROWS_FILENAME = "prediction_rows.parquet"


@dataclass(frozen=True)
class EvaluationOutputPaths:
    output_dir: Path
    domain_task_summary_csv: Path
    prediction_rows_parquet: Path


def prepare_output_paths(output_dir: str | Path) -> EvaluationOutputPaths:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return EvaluationOutputPaths(
        output_dir=root,
        domain_task_summary_csv=root / DOMAIN_TASK_SUMMARY_FILENAME,
        prediction_rows_parquet=root / PREDICTION_ROWS_FILENAME,
    )


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
    domain_task_summary = summarize_domain_task_rmse(prediction_df)
    domain_task_summary.to_csv(paths.domain_task_summary_csv, index=False)
    print_summary_table(domain_task_summary, "Domain/task normalized RMSE")

    domain_summary = domain_summary_from_task_summary(domain_task_summary)
    out: dict[str, Any] = {
        "prediction_rows": prediction_df,
        "domain_task_summary": domain_task_summary,
        "summary_domain": domain_summary,
        "domain_balanced_norm_rmse": balanced_domain_rmse(domain_summary),
        "prediction_rows_parquet": str(paths.prediction_rows_parquet),
        "domain_task_summary_csv": str(paths.domain_task_summary_csv),
    }

    return out
