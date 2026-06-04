import math

import numpy as np
from scipy.special import ndtr

CALIBRATION_QUANTILE_PROBS = [0.025, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.975]
CENTRAL_COVERAGE_LEVELS = [0.80, 0.90, 0.95]
MIXTURE_QUANTILE_BISECTION_ITERS = 45
MIXTURE_QUANTILE_BRACKET_SIGMAS = 12.0


def prob_key(prob, ndigits=6):
    return round(float(prob), ndigits)


CALIBRATION_QUANTILE_PROBS = sorted({prob_key(prob) for prob in CALIBRATION_QUANTILE_PROBS})


def q_col_suffix(prob):
    prob = prob_key(prob)
    return f"q{int(round(prob * 1000)):03d}"


def central_level_suffix(level):
    return f"{int(round(float(level) * 100))}"


def central_interval_probs(level):
    level = float(level)
    alpha = 1.0 - level
    return prob_key(alpha / 2.0), prob_key(1.0 - alpha / 2.0)


def normal_cdf_np(x):
    x = np.asarray(x, dtype=np.float64)
    return ndtr(x)


def logsumexp_np(a, axis=-1):
    a = np.asarray(a, dtype=np.float64)
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)), axis=axis)


def normal_logpdf_np(x, mu, sigma):
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)
    z = (np.asarray(x, dtype=np.float64) - np.asarray(mu, dtype=np.float64)) / sigma
    return -0.5 * z * z - np.log(sigma) - 0.5 * math.log(2.0 * math.pi)


def normal_abs_moment_np(x, sigma):
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)
    z = np.asarray(x, dtype=np.float64) / sigma
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return 2.0 * sigma * pdf + np.asarray(x, dtype=np.float64) * (2.0 * normal_cdf_np(z) - 1.0)


def mixture_crps_np(pi, mu, sigma, target):
    pi = np.asarray(pi, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)
    target = np.asarray(target, dtype=np.float64).reshape(-1)

    target_term = np.sum(pi * normal_abs_moment_np(target[:, None] - mu, sigma), axis=1)
    pair_sigma = np.sqrt(sigma[:, :, None] ** 2 + sigma[:, None, :] ** 2)
    pair_term = np.sum(
        pi[:, :, None]
        * pi[:, None, :]
        * normal_abs_moment_np(mu[:, :, None] - mu[:, None, :], pair_sigma),
        axis=(1, 2),
    )
    return np.maximum(target_term - 0.5 * pair_term, 0.0)


def mixture_quantiles_np(pi, mu, sigma, probs):
    pi = np.asarray(pi, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)
    probs = np.asarray(sorted({prob_key(prob) for prob in probs}), dtype=np.float64)

    lo = np.min(mu - MIXTURE_QUANTILE_BRACKET_SIGMAS * sigma, axis=1)
    hi = np.max(mu + MIXTURE_QUANTILE_BRACKET_SIGMAS * sigma, axis=1)

    def cdf_at(x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            z = (x[:, None] - mu) / sigma
            return np.sum(pi * normal_cdf_np(z), axis=1)
        z = (x[:, :, None] - mu[:, None, :]) / sigma[:, None, :]
        return np.sum(pi[:, None, :] * normal_cdf_np(z), axis=2)

    min_prob = float(probs.min())
    max_prob = float(probs.max())

    for _ in range(20):
        bad = cdf_at(lo) > min_prob
        if not bool(np.any(bad)):
            break
        lo[bad] -= np.maximum(hi[bad] - lo[bad], 1.0)

    for _ in range(20):
        bad = cdf_at(hi) < max_prob
        if not bool(np.any(bad)):
            break
        hi[bad] += np.maximum(hi[bad] - lo[bad], 1.0)

    left = np.repeat(lo[:, None], len(probs), axis=1)
    right = np.repeat(hi[:, None], len(probs), axis=1)
    prob_grid = probs[None, :]

    for _ in range(MIXTURE_QUANTILE_BISECTION_ITERS):
        mid = 0.5 * (left + right)
        cdf = cdf_at(mid)
        left = np.where(cdf < prob_grid, mid, left)
        right = np.where(cdf < prob_grid, right, mid)

    return probs, 0.5 * (left + right)


def compute_one_step_gmm_calibration_np(log_pi_np, mu_np, sigma_np, target_norm_np):
    log_pi_np = np.asarray(log_pi_np, dtype=np.float64)
    mu_np = np.asarray(mu_np, dtype=np.float64)
    sigma_np = np.maximum(np.asarray(sigma_np, dtype=np.float64), 1e-8)
    target_norm_np = np.asarray(target_norm_np, dtype=np.float64).reshape(-1)

    log_pi_np = log_pi_np - logsumexp_np(log_pi_np, axis=1)[:, None]
    pi_np = np.exp(log_pi_np)

    mean_norm = np.sum(pi_np * mu_np, axis=1)
    second_norm = np.sum(pi_np * (sigma_np ** 2 + mu_np ** 2), axis=1)
    var_norm = np.maximum(second_norm - mean_norm ** 2, 1e-12)
    std_norm = np.sqrt(var_norm)

    component_logpdf = normal_logpdf_np(target_norm_np[:, None], mu_np, sigma_np)
    log_prob_norm = logsumexp_np(log_pi_np + component_logpdf, axis=1)
    nll_norm = -log_prob_norm
    crps_norm = mixture_crps_np(pi_np, mu_np, sigma_np, target_norm_np)

    pit = np.sum(pi_np * normal_cdf_np((target_norm_np[:, None] - mu_np) / sigma_np), axis=1)
    pit = np.clip(pit, 0.0, 1.0)

    z_norm = (target_norm_np - mean_norm) / np.maximum(std_norm, 1e-8)

    probs, q_arr = mixture_quantiles_np(
        pi=pi_np,
        mu=mu_np,
        sigma=sigma_np,
        probs=CALIBRATION_QUANTILE_PROBS,
    )
    q_norm_rows = [
        {prob_key(prob): float(q_arr[row_idx, prob_idx]) for prob_idx, prob in enumerate(probs)}
        for row_idx in range(log_pi_np.shape[0])
    ]

    return {
        "mean_norm": mean_norm,
        "var_norm": var_norm,
        "std_norm": std_norm,
        "nll_norm": nll_norm,
        "crps_norm": crps_norm,
        "pit": pit,
        "z_norm": z_norm,
        "quantiles_norm": q_norm_rows,
    }


def add_empty_calibration_fields(row):
    row["calibration_available"] = False
    row["calibration_scope"] = ""
    row["calibration_not_reported_reason"] = "not_saved_for_non_one_step_task"

    base_nan_cols = [
        "cal_pred_norm_gmm_mean",
        "cal_pred_norm_gmm_var",
        "cal_pred_norm_gmm_std",
        "cal_pred_norm_gmm_nll",
        "cal_pred_norm_gmm_crps",
        "cal_pred_norm_gmm_pit",
        "cal_pred_norm_gmm_z",
    ]

    for col in base_nan_cols:
        row[col] = np.nan

    for prob in CALIBRATION_QUANTILE_PROBS:
        suffix = q_col_suffix(prob)
        row[f"cal_pred_norm_{suffix}"] = np.nan

    for level in CENTRAL_COVERAGE_LEVELS:
        suffix = central_level_suffix(level)
        row[f"cal_pred_norm_coverage_{suffix}"] = np.nan
        row[f"cal_pred_norm_interval_width_{suffix}"] = np.nan

    return row


def add_calibration_fields(row, batch_idx, calibration):
    row["calibration_available"] = True
    row["calibration_scope"] = "one_step_final_gmm"
    row["calibration_not_reported_reason"] = ""

    row["cal_pred_norm_gmm_mean"] = float(calibration["mean_norm"][batch_idx])
    row["cal_pred_norm_gmm_var"] = float(calibration["var_norm"][batch_idx])
    row["cal_pred_norm_gmm_std"] = float(calibration["std_norm"][batch_idx])
    row["cal_pred_norm_gmm_nll"] = float(calibration["nll_norm"][batch_idx])
    row["cal_pred_norm_gmm_crps"] = float(calibration["crps_norm"][batch_idx])
    row["cal_pred_norm_gmm_pit"] = float(calibration["pit"][batch_idx])
    row["cal_pred_norm_gmm_z"] = float(calibration["z_norm"][batch_idx])

    q_norm = calibration["quantiles_norm"][batch_idx]

    for prob in CALIBRATION_QUANTILE_PROBS:
        pkey = prob_key(prob)
        suffix = q_col_suffix(pkey)
        row[f"cal_pred_norm_{suffix}"] = float(q_norm[pkey])

    target_norm = float(row["target_norm"])

    for level in CENTRAL_COVERAGE_LEVELS:
        lo_prob, hi_prob = central_interval_probs(level)
        suffix = central_level_suffix(level)
        lo_norm, hi_norm = q_norm[lo_prob], q_norm[hi_prob]

        row[f"cal_pred_norm_coverage_{suffix}"] = float(lo_norm <= target_norm <= hi_norm)
        row[f"cal_pred_norm_interval_width_{suffix}"] = float(hi_norm - lo_norm)

    return row
