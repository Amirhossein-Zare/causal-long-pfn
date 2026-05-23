import math

import numpy as np
from scipy.special import ndtr

CALIBRATION_QUANTILE_PROBS = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
CENTRAL_COVERAGE_LEVELS = [0.50, 0.80, 0.90, 0.95]
MIXTURE_QUANTILE_BISECTION_ITERS = 45
MIXTURE_QUANTILE_BRACKET_SIGMAS = 12.0

SUPPORT_CALIBRATION_N_FOLDS = 5
SUPPORT_CALIBRATION_MIN_CTX = 20
SUPPORT_CALIBRATION_MAX_PAIRS_PER_FILE = 2048
SUPPORT_CALIBRATION_SEED = 20260515
SUPPORT_ALPHA_GRID = np.exp(np.linspace(np.log(0.25), np.log(5.0), 121))


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


def mixture_cdf_scalar(x, pi, mu, sigma):
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)

    return float(np.sum(pi * normal_cdf_np((float(x) - mu) / sigma)))


def mixture_quantiles_row(pi, mu, sigma, probs):
    pi = np.asarray(pi, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)

    probs = sorted({prob_key(prob) for prob in probs})

    lo = float(np.min(mu - MIXTURE_QUANTILE_BRACKET_SIGMAS * sigma))
    hi = float(np.max(mu + MIXTURE_QUANTILE_BRACKET_SIGMAS * sigma))

    min_prob = float(min(probs))
    max_prob = float(max(probs))

    for _ in range(20):
        if mixture_cdf_scalar(lo, pi, mu, sigma) <= min_prob:
            break

        lo -= max(hi - lo, 1.0)

    for _ in range(20):
        if mixture_cdf_scalar(hi, pi, mu, sigma) >= max_prob:
            break

        hi += max(hi - lo, 1.0)

    out = {}

    for prob in probs:
        prob = prob_key(prob)
        left = lo
        right = hi

        for _ in range(MIXTURE_QUANTILE_BISECTION_ITERS):
            mid = 0.5 * (left + right)

            if mixture_cdf_scalar(mid, pi, mu, sigma) < prob:
                left = mid
            else:
                right = mid

        out[prob] = float(0.5 * (left + right))

    return out


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

    pit = np.sum(pi_np * normal_cdf_np((target_norm_np[:, None] - mu_np) / sigma_np), axis=1)
    pit = np.clip(pit, 0.0, 1.0)

    z_norm = (target_norm_np - mean_norm) / np.maximum(std_norm, 1e-8)

    out = {
        "mean_norm": mean_norm,
        "var_norm": var_norm,
        "std_norm": std_norm,
        "nll_norm": nll_norm,
        "pit": pit,
        "z_norm": z_norm,
        "quantiles_norm": None,
    }

    q_norm_rows = []

    for row_idx in range(log_pi_np.shape[0]):
        q_norm = mixture_quantiles_row(
            pi=pi_np[row_idx],
            mu=mu_np[row_idx],
            sigma=sigma_np[row_idx],
            probs=CALIBRATION_QUANTILE_PROBS,
        )

        q_norm_rows.append(q_norm)

    out["quantiles_norm"] = q_norm_rows

    return out


def gmm_nll_with_sigma_alpha_np(y, log_pi, mu, sigma, alpha):
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    log_pi = np.asarray(log_pi, dtype=np.float64)
    log_pi = log_pi - logsumexp_np(log_pi, axis=1)[:, None]

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64) * float(alpha), 1e-8)

    component_logpdf = normal_logpdf_np(y[:, None], mu, sigma)

    return -logsumexp_np(log_pi + component_logpdf, axis=1)


def fit_sigma_alpha_from_support_predictions(y, log_pi, mu, sigma):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    log_pi = np.asarray(log_pi, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    finite = (
        np.isfinite(y)
        & np.isfinite(log_pi).all(axis=1)
        & np.isfinite(mu).all(axis=1)
        & np.isfinite(sigma).all(axis=1)
    )

    y = y[finite]
    log_pi = log_pi[finite]
    mu = mu[finite]
    sigma = sigma[finite]

    if len(y) < 10:
        return {
            "alpha": 1.0,
            "n": int(len(y)),
            "nll_uncal": np.nan,
            "nll_cal": np.nan,
            "status": "too_few_support_calibration_rows",
        }

    scores = []

    for alpha in SUPPORT_ALPHA_GRID:
        nll = gmm_nll_with_sigma_alpha_np(y, log_pi, mu, sigma, alpha=alpha)
        scores.append(float(np.mean(nll)))

    scores = np.asarray(scores, dtype=np.float64)
    best_idx = int(np.nanargmin(scores))
    alpha = float(SUPPORT_ALPHA_GRID[best_idx])

    return {
        "alpha": alpha,
        "n": int(len(y)),
        "nll_uncal": float(np.mean(gmm_nll_with_sigma_alpha_np(y, log_pi, mu, sigma, alpha=1.0))),
        "nll_cal": float(scores[best_idx]),
        "status": "ok",
    }


def add_empty_calibration_fields(row):
    row["calibration_available"] = False
    row["calibration_scope"] = ""
    row["calibration_not_reported_reason"] = "not_saved_for_non_one_step_task"

    row["support_sigma_calibration_available"] = False
    row["support_sigma_calibration_alpha"] = np.nan
    row["support_sigma_calibration_source"] = ""
    row["support_sigma_calibration_status"] = ""
    row["support_sigma_calibration_n"] = 0
    row["support_sigma_calibration_n_ctx"] = np.nan
    row["support_sigma_calibration_n_folds"] = np.nan
    row["support_sigma_calibration_max_pairs_per_file"] = np.nan
    row["support_sigma_calibration_nll_uncal"] = np.nan
    row["support_sigma_calibration_nll_cal"] = np.nan

    base_nan_cols = [
        "cal_pred_norm_gmm_mean",
        "cal_pred_norm_gmm_var",
        "cal_pred_norm_gmm_std",
        "cal_pred_norm_gmm_nll",
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


def add_support_scaled_calibration_fields(row, batch_idx, cal_scaled, alpha, meta):
    row["calibration_available"] = True
    row["calibration_scope"] = "one_step_final_gmm_support_sigma_scaled"
    row["calibration_not_reported_reason"] = ""

    row["support_sigma_calibration_available"] = bool(meta["status"] == "ok")
    row["support_sigma_calibration_alpha"] = float(alpha)
    row["support_sigma_calibration_source"] = str(meta["source"])
    row["support_sigma_calibration_status"] = str(meta["status"])
    row["support_sigma_calibration_n"] = int(meta["n"])
    row["support_sigma_calibration_n_ctx"] = int(meta["n_ctx"])
    row["support_sigma_calibration_n_folds"] = int(meta["n_folds"])
    row["support_sigma_calibration_max_pairs_per_file"] = int(meta["max_pairs_per_file"])
    row["support_sigma_calibration_nll_uncal"] = float(meta["nll_uncal"])
    row["support_sigma_calibration_nll_cal"] = float(meta["nll_cal"])

    row["cal_pred_norm_gmm_mean"] = float(cal_scaled["mean_norm"][batch_idx])
    row["cal_pred_norm_gmm_var"] = float(cal_scaled["var_norm"][batch_idx])
    row["cal_pred_norm_gmm_std"] = float(cal_scaled["std_norm"][batch_idx])
    row["cal_pred_norm_gmm_nll"] = float(cal_scaled["nll_norm"][batch_idx])
    row["cal_pred_norm_gmm_pit"] = float(cal_scaled["pit"][batch_idx])
    row["cal_pred_norm_gmm_z"] = float(cal_scaled["z_norm"][batch_idx])

    if cal_scaled["quantiles_norm"] is not None:
        q_norm = cal_scaled["quantiles_norm"][batch_idx]

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
