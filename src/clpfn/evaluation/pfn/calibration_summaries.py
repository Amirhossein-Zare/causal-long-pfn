from __future__ import annotations

import numpy as np
import pandas as pd

from clpfn.evaluation.core.summaries import summarize_domain_task_rmse
from clpfn.evaluation.pfn import calibration as cal


CALIBRATION_DOMAIN_COLUMNS = [
    "domain",
    "method",
    "rmse_norm",
    "mean_nll_norm",
    "mean_crps_norm",
    "mean_pred_std_norm",
    "pit_hist_ece",
    "coverage_obs_80",
    "coverage_obs_90",
    "coverage_obs_95",
    "coverage_ece",
    "quantile_ece",
    "mean_interval_width_80",
    "mean_interval_width_90",
    "mean_interval_width_95",
]


def _numeric_series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def _pit_hist_ece(values: pd.Series) -> float:
    pit = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    pit = pit[np.isfinite(pit)]
    pit = pit[(pit >= 0.0) & (pit <= 1.0)]
    if len(pit) == 0:
        return np.nan

    counts, _ = np.histogram(pit, bins=np.linspace(0.0, 1.0, 11))
    observed = counts / max(counts.sum(), 1)
    expected = np.ones_like(observed, dtype=float) / len(observed)
    return float(np.mean(np.abs(observed - expected)))


def _coverage_obs(g: pd.DataFrame, level: float) -> float:
    suffix = cal.central_level_suffix(level)
    coverage_col = f"cal_pred_norm_coverage_{suffix}"
    if coverage_col in g.columns:
        coverage = _numeric_series(g, coverage_col)
        if coverage.notna().any():
            return float(coverage.mean())

    lo_prob, hi_prob = cal.central_interval_probs(level)
    lo_col = f"cal_pred_norm_{cal.q_col_suffix(lo_prob)}"
    hi_col = f"cal_pred_norm_{cal.q_col_suffix(hi_prob)}"
    if lo_col not in g.columns or hi_col not in g.columns:
        return np.nan

    y = _numeric_series(g, "target_norm")
    lo = _numeric_series(g, lo_col)
    hi = _numeric_series(g, hi_col)
    mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    return float(((lo[mask] <= y[mask]) & (y[mask] <= hi[mask])).mean()) if mask.any() else np.nan


def _quantile_ece(g: pd.DataFrame) -> float:
    y = _numeric_series(g, "target_norm")
    errors = []
    for prob in cal.CALIBRATION_QUANTILE_PROBS:
        q_col = f"cal_pred_norm_{cal.q_col_suffix(prob)}"
        if q_col not in g.columns:
            continue
        q = _numeric_series(g, q_col)
        mask = np.isfinite(y) & np.isfinite(q)
        if mask.any():
            observed = float((y[mask] <= q[mask]).mean())
            errors.append(abs(observed - float(prob)))
    return float(np.mean(errors)) if errors else np.nan


def _normalized_calibration_rows(calibration_df: pd.DataFrame) -> pd.DataFrame:
    df = calibration_df.copy()
    if df.empty:
        return df

    if "calibration_available" in df.columns:
        df = df[_bool_series(df["calibration_available"])].copy()

    if "task_step" in df.columns:
        task_step = df["task_step"].astype(str)
        df = df[(task_step == "one_step") | (_numeric_series(df, "tau") == 1)].copy()
    elif "tau" in df.columns:
        df = df[_numeric_series(df, "tau") == 1].copy()

    if "domain" not in df.columns:
        raise KeyError("Calibration dataframe is missing required column 'domain'.")
    df["domain"] = df["domain"].astype(str).str.lower().str.strip()

    if "method" not in df.columns:
        raise KeyError("Calibration dataframe is missing required column 'method'.")
    df["method"] = df["method"].astype(str)

    return df


def _one_step_rmse_by_domain_method(prediction_df: pd.DataFrame) -> pd.DataFrame:
    point_summary = summarize_domain_task_rmse(prediction_df)
    one_step = point_summary[point_summary["task_step"].astype(str) == "one_step"].copy()
    return one_step[["domain", "method", "mean_norm_rmse"]].rename(
        columns={"mean_norm_rmse": "rmse_norm"}
    )


def summarize_domain_calibration(calibration_df: pd.DataFrame, prediction_df: pd.DataFrame) -> pd.DataFrame:
    df = _normalized_calibration_rows(calibration_df)
    if df.empty or "cal_pred_norm_gmm_nll" not in df.columns:
        return pd.DataFrame(columns=CALIBRATION_DOMAIN_COLUMNS)

    df = df[_numeric_series(df, "cal_pred_norm_gmm_nll").notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=CALIBRATION_DOMAIN_COLUMNS)

    rmse_by_domain_method = _one_step_rmse_by_domain_method(prediction_df)
    rows = []
    for (domain, method), g in df.groupby(["domain", "method"], dropna=False):
        coverage = {level: _coverage_obs(g, level) for level in cal.CENTRAL_COVERAGE_LEVELS}
        coverage_errors = [
            abs(observed - float(level))
            for level, observed in coverage.items()
            if np.isfinite(observed)
        ]
        row = {
            "domain": domain,
            "method": method,
            "mean_nll_norm": float(_numeric_series(g, "cal_pred_norm_gmm_nll").mean()),
            "mean_crps_norm": float(_numeric_series(g, "cal_pred_norm_gmm_crps").mean()),
            "mean_pred_std_norm": float(_numeric_series(g, "cal_pred_norm_gmm_std").mean()),
            "pit_hist_ece": _pit_hist_ece(_numeric_series(g, "cal_pred_norm_gmm_pit")),
            "coverage_ece": float(np.mean(coverage_errors)) if coverage_errors else np.nan,
            "quantile_ece": _quantile_ece(g),
        }
        for level, observed in coverage.items():
            suffix = cal.central_level_suffix(level)
            row[f"coverage_obs_{suffix}"] = observed
            row[f"mean_interval_width_{suffix}"] = float(
                _numeric_series(g, f"cal_pred_norm_interval_width_{suffix}").mean()
            )
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.merge(rmse_by_domain_method, on=["domain", "method"], how="left")
    if _numeric_series(out, "rmse_norm").isna().any():
        missing = out[_numeric_series(out, "rmse_norm").isna()][["domain", "method"]].drop_duplicates()
        raise ValueError(
            "Missing one-step point-prediction RMSE for calibration summary rows: "
            f"{missing.to_dict(orient='records')}"
        )
    return out[CALIBRATION_DOMAIN_COLUMNS].sort_values(["domain", "method"]).reset_index(drop=True)
