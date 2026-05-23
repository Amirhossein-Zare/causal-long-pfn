from __future__ import annotations

import logging

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


DOMAIN_TASK_RMSE_COLUMNS = [
    "domain",
    "task_step",
    "method",
    "mean_norm_rmse",
    "std_norm_rmse",
]


def _rmse_from_sq_error(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x))) if len(x) else np.nan


def _normalized_predictions(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "domain",
        "method",
        "run_id",
        "gamma",
        "task_step",
        "task_name",
        "tau",
        "dataset_id",
        "global_dataset_id",
        "source_file",
        "sq_error_norm",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Prediction dataframe is missing required columns: {missing}")

    out = df.copy()
    out["domain"] = out["domain"].astype(str).str.lower().str.strip()
    out["method"] = out["method"].astype(str)
    out["task_step"] = out["task_step"].astype(str)
    out["sq_error_norm"] = pd.to_numeric(out["sq_error_norm"], errors="coerce")

    return out[np.isfinite(out["sq_error_norm"])].copy()


def prediction_unit_rmse(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Return one RMSE row per method/run/domain/task/dataset unit."""
    if pred_df.empty:
        return pd.DataFrame(columns=["domain", "task_step", "method", "norm_rmse"])

    df = _normalized_predictions(pred_df)
    if df.empty:
        return pd.DataFrame(columns=["domain", "task_step", "method", "norm_rmse"])

    unit_cols = [
        "method",
        "run_id",
        "domain",
        "gamma",
        "task_step",
        "task_name",
        "tau",
        "dataset_id",
        "global_dataset_id",
        "source_file",
    ]
    units = (
        df.groupby(unit_cols, dropna=False)
        .agg(norm_rmse=("sq_error_norm", _rmse_from_sq_error))
        .reset_index()
    )
    return units[np.isfinite(units["norm_rmse"])].copy()


def summarize_domain_task_rmse(pred_df: pd.DataFrame) -> pd.DataFrame:
    units = prediction_unit_rmse(pred_df)
    if units.empty:
        return pd.DataFrame(columns=DOMAIN_TASK_RMSE_COLUMNS)

    summary = (
        units.groupby(["domain", "task_step", "method"], dropna=False)
        .agg(
            mean_norm_rmse=("norm_rmse", "mean"),
            std_norm_rmse=("norm_rmse", "std"),
        )
        .reset_index()
    )
    summary["std_norm_rmse"] = summary["std_norm_rmse"].fillna(0.0)
    return summary[DOMAIN_TASK_RMSE_COLUMNS].sort_values(["task_step", "domain", "method"]).reset_index(drop=True)


def print_summary_table(df: pd.DataFrame, title: str) -> None:
    LOGGER.info("%s\n%s", title, "=" * len(title))
    if df.empty:
        LOGGER.info("(empty)")
        return
    LOGGER.info("\n%s", df.to_string(index=False))
