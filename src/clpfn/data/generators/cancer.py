"""Cancer tumor-growth benchmark generator.

This module follows the tumor-growth simulation family used by longitudinal
counterfactual baselines such as RMSN, CRN, Causal Transformer, and
G-Transformer. It exports the canonical CLPFN benchmark raw-pickle schema while keeping
the domain-specific tumor dynamics and treatment-policy confounding explicit.

The generator preserves:

* CT-style tumor dynamics and treatment-policy confounding.
* log1p(clipped volume) export for PFN-facing data.
* The canonical raw-pickle keys expected by PFN and baseline evaluators.
* One-step counterfactual rows and 5-step random-trajectory rows.

Use from Python:

    from clpfn.data.generators.cancer import CancerGeneratorConfig, generate
    generate(CancerGeneratorConfig(output_dir="outputs/data/cancer"))

or from CLI:

    clpfn-generate-all --config configs/data/all_benchmarks.yaml --only cancer
"""

from __future__ import annotations

import dataclasses
import gc
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from .common import concat_raw, ensure_output_dir, save_pickle, standardize_pickle_map, take_rows

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class CancerGeneratorConfig:
    output_dir: str = "outputs/data/cancer"

    gammas: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    support_sizes: tuple[int, ...] = (40, 80, 160, 320, 500)
    reps_per_cell: int = 2

    test_base_patients: int = 1
    seq_length: int = 60
    window_size: int = 15
    projection_horizon: int = 5
    n_seq_random_trajectories: int | None = None
    min_t_obs: int = 10

    base_seed: int = 1000

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None = None, **overrides: Any) -> "CancerGeneratorConfig":
        values = dict(values or {})
        values.update({k: v for k, v in overrides.items() if v is not None})
        valid = {field.name for field in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in valid})

    def __post_init__(self) -> None:
        self.gammas = tuple(int(x) for x in self.gammas)
        self.support_sizes = tuple(int(x) for x in self.support_sizes)
        if self.n_seq_random_trajectories is None:
            self.n_seq_random_trajectories = int(self.projection_horizon) * 2


# -----------------------------------------------------------------------------
# CT simulation constants
# -----------------------------------------------------------------------------

def calc_volume(diameter: np.ndarray | float) -> np.ndarray | float:
    return 4.0 / 3.0 * np.pi * (diameter / 2.0) ** 3.0


def calc_diameter(volume: np.ndarray | float) -> np.ndarray | float:
    return ((volume / (4.0 / 3.0 * np.pi)) ** (1.0 / 3.0)) * 2.0


TUMOUR_CELL_DENSITY = 5.8 * 10.0**8.0
TUMOUR_DEATH_THRESHOLD = calc_volume(13)

TUMOUR_SIZE_DISTRIBUTIONS = {
    "I": (1.72, 4.70, 0.3, 5.0),
    "II": (1.96, 1.63, 0.3, 13.0),
    "IIIA": (1.91, 9.40, 0.3, 13.0),
    "IIIB": (2.76, 6.87, 0.3, 13.0),
    "IV": (3.86, 8.82, 0.3, 13.0),
}

CANCER_STAGE_OBSERVATIONS = {
    "I": 1432,
    "II": 128,
    "IIIA": 1306,
    "IIIB": 7248,
    "IV": 12840,
}


def get_standard_params(num_patients: int) -> dict[str, np.ndarray]:
    possible_patient_types = [1, 2, 3]
    patient_types = np.random.choice(possible_patient_types, num_patients)

    chemo_mean_adjustments = np.array([0.0 if i < 3 else 0.1 for i in patient_types])
    radio_mean_adjustments = np.array([0.0 if i > 1 else 0.1 for i in patient_types])

    total_obs = sum(CANCER_STAGE_OBSERVATIONS.values())
    cancer_stage_proportions = {k: CANCER_STAGE_OBSERVATIONS[k] / total_obs for k in CANCER_STAGE_OBSERVATIONS}

    possible_stages = sorted(TUMOUR_SIZE_DISTRIBUTIONS.keys())
    initial_stages = np.random.choice(
        possible_stages,
        num_patients,
        p=[cancer_stage_proportions[k] for k in possible_stages],
    )

    output_initial_diam: list[float] = []
    patient_sim_stages: list[str] = []

    for stage in possible_stages:
        count = np.sum((initial_stages == stage) * 1)
        mu, sigma, lower_bound, upper_bound = TUMOUR_SIZE_DISTRIBUTIONS[stage]
        lower_bound = (np.log(lower_bound) - mu) / sigma
        upper_bound = (np.log(upper_bound) - mu) / sigma
        norm_rvs = truncnorm.rvs(lower_bound, upper_bound, size=count)
        initial_diam_by_stage = np.exp((norm_rvs * sigma) + mu)
        output_initial_diam += list(initial_diam_by_stage)
        patient_sim_stages += [stage for _ in range(count)]

    K = calc_volume(30)
    alpha_beta_ratio = 10
    alpha_rho_corr = 0.87
    parameter_lower_bound = 0.0
    parameter_upper_bound = np.inf

    rho_params = (7 * 10**-5, 7.23 * 10**-3)
    alpha_params = (0.0398, 0.168)
    beta_c_params = (0.028, 0.0007)

    alpha_rho_cov = np.array(
        [
            [alpha_params[1] ** 2, alpha_rho_corr * alpha_params[1] * rho_params[1]],
            [alpha_rho_corr * alpha_params[1] * rho_params[1], rho_params[1] ** 2],
        ]
    )
    alpha_rho_mean = np.array([alpha_params[0], rho_params[0]])

    simulated_params = []
    while len(simulated_params) < num_patients:
        param_holder = np.random.multivariate_normal(alpha_rho_mean, alpha_rho_cov, size=num_patients)
        for i in range(param_holder.shape[0]):
            if param_holder[i, 0] > parameter_lower_bound and param_holder[i, 1] > parameter_lower_bound:
                simulated_params.append(param_holder[i, :])
    simulated_params = np.array(simulated_params)[:num_patients, :]

    alpha_adjustments = alpha_params[0] * radio_mean_adjustments
    alpha = simulated_params[:, 0] + alpha_adjustments
    rho = simulated_params[:, 1]
    beta = alpha / alpha_beta_ratio

    beta_c_adjustments = beta_c_params[0] * chemo_mean_adjustments
    beta_c = (
        beta_c_params[0]
        + beta_c_params[1]
        * truncnorm.rvs(
            (parameter_lower_bound - beta_c_params[0]) / beta_c_params[1],
            (parameter_upper_bound - beta_c_params[0]) / beta_c_params[1],
            size=num_patients,
        )
        + beta_c_adjustments
    )

    output_holder = {
        "patient_types": patient_types,
        "initial_stages": np.array(patient_sim_stages),
        "initial_volumes": calc_volume(np.array(output_initial_diam)),
        "alpha": alpha,
        "rho": rho,
        "beta": beta,
        "beta_c": beta_c,
        "K": np.array([K for _ in range(num_patients)]),
    }

    idx = np.arange(num_patients)
    np.random.shuffle(idx)
    return {k: output_holder[k][idx] for k in output_holder}


def get_confounding_params(num_patients: int, gamma: int | float, window_size: int) -> dict[str, np.ndarray]:
    params = get_standard_params(num_patients)
    patient_types = params["patient_types"]
    d_max = calc_diameter(TUMOUR_DEATH_THRESHOLD)

    params["chemo_sigmoid_intercepts"] = np.array([d_max / 2.0 for _ in patient_types])
    params["radio_sigmoid_intercepts"] = np.array([d_max / 2.0 for _ in patient_types])
    params["chemo_sigmoid_betas"] = np.array([float(gamma) / d_max for _ in patient_types])
    params["radio_sigmoid_betas"] = np.array([float(gamma) / d_max for _ in patient_types])
    params["window_size"] = int(window_size)
    return params


# -----------------------------------------------------------------------------
# Raw helpers and scaling
# -----------------------------------------------------------------------------

def filter_factual_min_tobs(raw: dict[str, Any], min_tobs: int) -> tuple[dict[str, Any], np.ndarray]:
    keep = np.where(np.asarray(raw["sequence_lengths"], dtype=np.int64) >= int(min_tobs))[0]
    return take_rows(raw, keep), keep


def log_export(raw: dict[str, Any]) -> dict[str, Any]:
    # Intentional PFN-facing difference from CT:
    # store log1p(clipped tumor volume), not raw tumor volume.
    raw = dict(raw)
    raw["cancer_volume"] = np.log1p(np.clip(raw["cancer_volume"], 0.0, TUMOUR_DEATH_THRESHOLD))
    return raw


def get_scaling_params(sim: dict[str, Any]) -> tuple[pd.Series, pd.Series]:
    real_idx = ["cancer_volume", "chemo_dosage", "radio_dosage"]
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    seq_lengths = sim["sequence_lengths"]

    for key in real_idx:
        if key not in sim:
            continue
        active_values = []
        for i in range(seq_lengths.shape[0]):
            end = int(seq_lengths[i])
            if end > 0:
                active_values += list(sim[key][i, :end])
        means[key] = float(np.mean(active_values)) if len(active_values) else 0.0
        stds[key] = float(np.std(active_values)) if len(active_values) else 1.0

    means["patient_types"] = float(np.mean(sim["patient_types"])) if len(sim["patient_types"]) else 0.0
    stds["patient_types"] = float(np.std(sim["patient_types"])) if len(sim["patient_types"]) else 1.0
    return pd.Series(means), pd.Series(stds)


# -----------------------------------------------------------------------------
# CT factual simulation
# -----------------------------------------------------------------------------

def simulate_factual(simulation_params: dict[str, Any], num_time_steps: int) -> dict[str, np.ndarray]:
    total_num_radio_treatments = 1
    total_num_chemo_treatments = 1

    radio_amt = np.array([2.0 for _ in range(total_num_radio_treatments)])
    chemo_amt = [5.0 for _ in range(total_num_chemo_treatments)]
    chemo_days = [(i + 1) * 7 for i in range(total_num_chemo_treatments)]
    chemo_idx = np.argsort(chemo_days)
    chemo_amt = np.array(chemo_amt)[chemo_idx]
    drug_half_life = 1

    initial_volumes = simulation_params["initial_volumes"]
    alphas = simulation_params["alpha"]
    rhos = simulation_params["rho"]
    betas = simulation_params["beta"]
    beta_cs = simulation_params["beta_c"]
    Ks = simulation_params["K"]
    patient_types = simulation_params["patient_types"]
    window_size = simulation_params["window_size"]

    chemo_sigmoid_intercepts = simulation_params["chemo_sigmoid_intercepts"]
    radio_sigmoid_intercepts = simulation_params["radio_sigmoid_intercepts"]
    chemo_sigmoid_betas = simulation_params["chemo_sigmoid_betas"]
    radio_sigmoid_betas = simulation_params["radio_sigmoid_betas"]

    num_patients = initial_volumes.shape[0]
    cancer_volume = np.zeros((num_patients, num_time_steps))
    chemo_dosage = np.zeros((num_patients, num_time_steps))
    radio_dosage = np.zeros((num_patients, num_time_steps))
    chemo_application_point = np.zeros((num_patients, num_time_steps))
    radio_application_point = np.zeros((num_patients, num_time_steps))
    sequence_lengths = np.zeros(num_patients)
    death_flags = np.zeros((num_patients, num_time_steps))
    recovery_flags = np.zeros((num_patients, num_time_steps))
    chemo_probabilities = np.zeros((num_patients, num_time_steps))
    radio_probabilities = np.zeros((num_patients, num_time_steps))

    noise_terms = 0.01 * np.random.randn(num_patients, num_time_steps)
    recovery_rvs = np.random.rand(num_patients, num_time_steps)
    chemo_application_rvs = np.random.rand(num_patients, num_time_steps)
    radio_application_rvs = np.random.rand(num_patients, num_time_steps)

    for i in range(num_patients):
        noise = noise_terms[i]
        cancer_volume[i, 0] = initial_volumes[i]
        alpha = alphas[i]
        beta = betas[i]
        beta_c = beta_cs[i]
        rho = rhos[i]
        K = Ks[i]
        b_death = False
        b_recover = False

        for t in range(0, num_time_steps - 1):
            current_chemo_dose = 0.0
            previous_chemo_dose = 0.0 if t == 0 else chemo_dosage[i, t - 1]
            cancer_volume_used = cancer_volume[i, max(t - window_size, 0) : t + 1]
            cancer_diameter_used = np.array([calc_diameter(vol) for vol in cancer_volume_used]).mean()

            radio_prob = 1.0 / (1.0 + np.exp(-radio_sigmoid_betas[i] * (cancer_diameter_used - radio_sigmoid_intercepts[i])))
            chemo_prob = 1.0 / (1.0 + np.exp(-chemo_sigmoid_betas[i] * (cancer_diameter_used - chemo_sigmoid_intercepts[i])))
            chemo_probabilities[i, t] = chemo_prob
            radio_probabilities[i, t] = radio_prob

            if radio_application_rvs[i, t] < radio_prob:
                radio_application_point[i, t] = 1
                radio_dosage[i, t] = radio_amt[0]
            if chemo_application_rvs[i, t] < chemo_prob:
                chemo_application_point[i, t] = 1
                current_chemo_dose = chemo_amt[0]

            chemo_dosage[i, t] = previous_chemo_dose * np.exp(-np.log(2) / drug_half_life) + current_chemo_dose
            cancer_volume[i, t + 1] = cancer_volume[i, t] * (
                1
                + rho * np.log(K / cancer_volume[i, t])
                - beta_c * chemo_dosage[i, t]
                - (alpha * radio_dosage[i, t] + beta * radio_dosage[i, t] ** 2)
                + noise[t]
            )

            if cancer_volume[i, t + 1] > TUMOUR_DEATH_THRESHOLD:
                cancer_volume[i, t + 1] = TUMOUR_DEATH_THRESHOLD
                b_death = True
                break
            if recovery_rvs[i, t + 1] < np.exp(-cancer_volume[i, t + 1] * TUMOUR_CELL_DENSITY):
                cancer_volume[i, t + 1] = 0.0
                b_recover = True
                break

        end_t = int(t + 1)
        sequence_lengths[i] = end_t
        if end_t < num_time_steps:
            death_flags[i, end_t] = 1 if b_death else 0
            recovery_flags[i, end_t] = 1 if b_recover else 0

    return {
        "cancer_volume": cancer_volume,
        "chemo_dosage": chemo_dosage,
        "radio_dosage": radio_dosage,
        "chemo_application": chemo_application_point,
        "radio_application": radio_application_point,
        "chemo_probabilities": chemo_probabilities,
        "radio_probabilities": radio_probabilities,
        "sequence_lengths": sequence_lengths,
        "death_flags": death_flags,
        "recovery_flags": recovery_flags,
        "patient_types": patient_types,
    }


def make_support_data(support_size: int, gamma: int, window_size: int, seq_length: int, min_tobs: int) -> dict[str, Any]:
    chunks = []
    n_kept = 0
    attempts = 0
    while n_kept < support_size:
        attempts += 1
        batch_size = max(support_size * 3, support_size + 50)
        params = get_confounding_params(num_patients=batch_size, gamma=gamma, window_size=window_size)
        raw = simulate_factual(params, seq_length)
        raw_kept, keep_idx = filter_factual_min_tobs(raw, min_tobs=min_tobs)
        if len(keep_idx) > 0:
            chunks.append(raw_kept)
            n_kept += len(keep_idx)
        if attempts > 100 and n_kept < support_size:
            raise RuntimeError(
                f"Could not generate enough support trajectories with sequence_lengths >= {min_tobs}. "
                f"Kept {n_kept}/{support_size} after {attempts} attempts."
            )
    support = concat_raw(chunks)
    return take_rows(support, np.arange(support_size))


# -----------------------------------------------------------------------------
# CT one-step counterfactual test rows
# -----------------------------------------------------------------------------

def simulate_counterfactual_1_step(simulation_params: dict[str, Any], num_time_steps: int, min_tobs: int) -> dict[str, np.ndarray]:
    total_num_radio_treatments = 1
    total_num_chemo_treatments = 1
    radio_amt = np.array([2.0 for _ in range(total_num_radio_treatments)])
    chemo_amt = [5.0 for _ in range(total_num_chemo_treatments)]
    chemo_days = [(i + 1) * 7 for i in range(total_num_chemo_treatments)]
    chemo_idx = np.argsort(chemo_days)
    chemo_amt = np.array(chemo_amt)[chemo_idx]
    drug_half_life = 1

    initial_volumes = simulation_params["initial_volumes"]
    alphas = simulation_params["alpha"]
    rhos = simulation_params["rho"]
    betas = simulation_params["beta"]
    beta_cs = simulation_params["beta_c"]
    Ks = simulation_params["K"]
    patient_types = simulation_params["patient_types"]
    window_size = simulation_params["window_size"]
    chemo_sigmoid_intercepts = simulation_params["chemo_sigmoid_intercepts"]
    radio_sigmoid_intercepts = simulation_params["radio_sigmoid_intercepts"]
    chemo_sigmoid_betas = simulation_params["chemo_sigmoid_betas"]
    radio_sigmoid_betas = simulation_params["radio_sigmoid_betas"]

    num_patients = initial_volumes.shape[0]
    num_test_points = num_patients * num_time_steps * 4
    cancer_volume = np.zeros((num_test_points, num_time_steps))
    chemo_application_point = np.zeros((num_test_points, num_time_steps))
    radio_application_point = np.zeros((num_test_points, num_time_steps))
    sequence_lengths = np.zeros(num_test_points)
    patient_types_all = np.zeros(num_test_points)
    test_idx = 0

    for i in range(num_patients):
        noise = 0.01 * np.random.randn(num_time_steps)
        recovery_rvs = np.random.rand(num_time_steps)
        factual_cancer_volume = np.zeros(num_time_steps)
        factual_chemo_dosage = np.zeros(num_time_steps)
        factual_radio_dosage = np.zeros(num_time_steps)
        factual_chemo_application = np.zeros(num_time_steps)
        factual_radio_application = np.zeros(num_time_steps)
        chemo_application_rvs = np.random.rand(num_time_steps)
        radio_application_rvs = np.random.rand(num_time_steps)
        factual_cancer_volume[0] = initial_volumes[i]
        alpha = alphas[i]
        beta = betas[i]
        beta_c = beta_cs[i]
        rho = rhos[i]
        K = Ks[i]

        for t in range(0, num_time_steps - 1):
            current_chemo_dose = 0.0
            previous_chemo_dose = 0.0 if t == 0 else factual_chemo_dosage[t - 1]

            cancer_volume_used = cancer_volume[i, max(t - window_size, 0) : t + 1]
            cancer_diameter_used = np.array([calc_diameter(vol) for vol in cancer_volume_used]).mean()
            radio_prob = 1.0 / (1.0 + np.exp(-radio_sigmoid_betas[i] * (cancer_diameter_used - radio_sigmoid_intercepts[i])))
            chemo_prob = 1.0 / (1.0 + np.exp(-chemo_sigmoid_betas[i] * (cancer_diameter_used - chemo_sigmoid_intercepts[i])))

            if radio_application_rvs[t] < radio_prob:
                factual_radio_application[t] = 1
                factual_radio_dosage[t] = radio_amt[0]
            if chemo_application_rvs[t] < chemo_prob:
                factual_chemo_application[t] = 1
                current_chemo_dose = chemo_amt[0]

            factual_chemo_dosage[t] = previous_chemo_dose * np.exp(-np.log(2) / drug_half_life) + current_chemo_dose
            factual_cancer_volume[t + 1] = factual_cancer_volume[t] * (
                1
                + rho * np.log(K / factual_cancer_volume[t])
                - beta_c * factual_chemo_dosage[t]
                - (alpha * factual_radio_dosage[t] + beta * factual_radio_dosage[t] ** 2)
                + noise[t + 1]
            )
            factual_cancer_volume[t + 1] = np.clip(factual_cancer_volume[t + 1], 0.0, TUMOUR_DEATH_THRESHOLD)

            if (t + 1) >= int(min_tobs):
                cancer_volume[test_idx] = factual_cancer_volume
                chemo_application_point[test_idx] = factual_chemo_application
                radio_application_point[test_idx] = factual_radio_application
                patient_types_all[test_idx] = patient_types[i]
                sequence_lengths[test_idx] = int(t) + 1
                test_idx += 1

                for treatment_option in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                    if factual_chemo_application[t] == treatment_option[0] and factual_radio_application[t] == treatment_option[1]:
                        continue

                    current_chemo_dose = 0.0
                    counterfactual_radio_dosage = 0.0
                    counterfactual_chemo_application = 0
                    counterfactual_radio_application = 0
                    if treatment_option[0] == 1:
                        counterfactual_chemo_application = 1
                        current_chemo_dose = chemo_amt[0]
                    if treatment_option[1] == 1:
                        counterfactual_radio_application = 1
                        counterfactual_radio_dosage = radio_amt[0]

                    counterfactual_chemo_dosage = previous_chemo_dose * np.exp(-np.log(2) / drug_half_life) + current_chemo_dose
                    counterfactual_next_volume = factual_cancer_volume[t] * (
                        1
                        + rho * np.log(K / factual_cancer_volume[t])
                        - beta_c * counterfactual_chemo_dosage
                        - (alpha * counterfactual_radio_dosage + beta * counterfactual_radio_dosage**2)
                        + noise[t + 1]
                    )

                    cancer_volume[test_idx][: t + 2] = np.append(factual_cancer_volume[: t + 1], [counterfactual_next_volume])
                    chemo_application_point[test_idx][: t + 1] = np.append(factual_chemo_application[:t], [counterfactual_chemo_application])
                    radio_application_point[test_idx][: t + 1] = np.append(factual_radio_application[:t], [counterfactual_radio_application])
                    patient_types_all[test_idx] = patient_types[i]
                    sequence_lengths[test_idx] = int(t) + 1
                    test_idx += 1

            if factual_cancer_volume[t + 1] >= TUMOUR_DEATH_THRESHOLD or recovery_rvs[t] <= np.exp(-factual_cancer_volume[t + 1] * TUMOUR_CELL_DENSITY):
                break

    return {
        "cancer_volume": cancer_volume[:test_idx],
        "chemo_application": chemo_application_point[:test_idx],
        "radio_application": radio_application_point[:test_idx],
        "sequence_lengths": sequence_lengths[:test_idx],
        "patient_types": patient_types_all[:test_idx],
    }


# -----------------------------------------------------------------------------
# CT 5-step random-trajectory counterfactual rows
# -----------------------------------------------------------------------------

def simulate_counterfactuals_random_trajectories(
    simulation_params: dict[str, Any],
    num_time_steps: int,
    projection_horizon: int,
    min_tobs: int,
    n_random_trajectories: int,
) -> dict[str, np.ndarray]:
    total_num_radio_treatments = 1
    total_num_chemo_treatments = 1
    radio_amt = np.array([2.0 for _ in range(total_num_radio_treatments)])
    chemo_amt = [5.0 for _ in range(total_num_chemo_treatments)]
    chemo_days = [(i + 1) * 7 for i in range(total_num_chemo_treatments)]
    chemo_idx = np.argsort(chemo_days)
    chemo_amt = np.array(chemo_amt)[chemo_idx]
    drug_half_life = 1

    initial_volumes = simulation_params["initial_volumes"]
    alphas = simulation_params["alpha"]
    rhos = simulation_params["rho"]
    betas = simulation_params["beta"]
    beta_cs = simulation_params["beta_c"]
    Ks = simulation_params["K"]
    patient_types = simulation_params["patient_types"]
    window_size = simulation_params["window_size"]
    chemo_sigmoid_intercepts = simulation_params["chemo_sigmoid_intercepts"]
    radio_sigmoid_intercepts = simulation_params["radio_sigmoid_intercepts"]
    chemo_sigmoid_betas = simulation_params["chemo_sigmoid_betas"]
    radio_sigmoid_betas = simulation_params["radio_sigmoid_betas"]

    num_patients = initial_volumes.shape[0]
    num_test_points = n_random_trajectories * num_patients * num_time_steps
    total_T = num_time_steps + projection_horizon
    cancer_volume = np.zeros((num_test_points, total_T))
    chemo_application_point = np.zeros((num_test_points, total_T))
    radio_application_point = np.zeros((num_test_points, total_T))
    sequence_lengths = np.zeros(num_test_points)
    patient_types_all = np.zeros(num_test_points)
    patient_ids_all = np.zeros(num_test_points)
    patient_current_t = np.zeros(num_test_points)
    test_idx = 0

    for i in range(num_patients):
        noise = 0.01 * np.random.randn(num_time_steps + projection_horizon + 1)
        recovery_rvs = np.random.rand(num_time_steps)
        factual_cancer_volume = np.zeros(num_time_steps)
        factual_chemo_dosage = np.zeros(num_time_steps)
        factual_radio_dosage = np.zeros(num_time_steps)
        factual_chemo_application = np.zeros(num_time_steps)
        factual_radio_application = np.zeros(num_time_steps)
        chemo_application_rvs = np.random.rand(num_time_steps)
        radio_application_rvs = np.random.rand(num_time_steps)
        factual_cancer_volume[0] = initial_volumes[i]
        alpha = alphas[i]
        beta = betas[i]
        beta_c = beta_cs[i]
        rho = rhos[i]
        K = Ks[i]

        for t in range(0, num_time_steps - 1):
            current_chemo_dose = 0.0
            previous_chemo_dose = 0.0 if t == 0 else factual_chemo_dosage[t - 1]
            cancer_volume_used = cancer_volume[i, max(t - window_size, 0) : t + 1]
            cancer_diameter_used = np.array([calc_diameter(vol) for vol in cancer_volume_used]).mean()
            radio_prob = 1.0 / (1.0 + np.exp(-radio_sigmoid_betas[i] * (cancer_diameter_used - radio_sigmoid_intercepts[i])))
            chemo_prob = 1.0 / (1.0 + np.exp(-chemo_sigmoid_betas[i] * (cancer_diameter_used - chemo_sigmoid_intercepts[i])))

            if radio_application_rvs[t] < radio_prob:
                factual_radio_application[t] = 1
                factual_radio_dosage[t] = radio_amt[0]
            if chemo_application_rvs[t] < chemo_prob:
                factual_chemo_application[t] = 1
                current_chemo_dose = chemo_amt[0]

            factual_chemo_dosage[t] = previous_chemo_dose * np.exp(-np.log(2) / drug_half_life) + current_chemo_dose
            factual_cancer_volume[t + 1] = factual_cancer_volume[t] * (
                1
                + rho * np.log(K / factual_cancer_volume[t])
                - beta_c * factual_chemo_dosage[t]
                - (alpha * factual_radio_dosage[t] + beta * factual_radio_dosage[t] ** 2)
                + noise[t + 1]
            )
            factual_cancer_volume[t + 1] = np.clip(factual_cancer_volume[t + 1], 0.0, TUMOUR_DEATH_THRESHOLD)

            # Downstream sequence formatter: current_t = patient_current_t + 1, t_obs = current_t + 1.
            if (t + 2) >= int(min_tobs):
                treatment_options = np.random.randint(0, 2, size=(n_random_trajectories, projection_horizon, 2))
                for treatment_option in treatment_options:
                    counterfactual_cancer_volume = np.zeros(t + 1 + projection_horizon + 1)
                    counterfactual_chemo_application = np.zeros(t + 1 + projection_horizon)
                    counterfactual_radio_application = np.zeros(t + 1 + projection_horizon)
                    counterfactual_chemo_dosage = np.zeros(t + 1 + projection_horizon)
                    counterfactual_radio_dosage = np.zeros(t + 1 + projection_horizon)
                    counterfactual_cancer_volume[: t + 2] = factual_cancer_volume[: t + 2]
                    counterfactual_chemo_application[: t + 1] = factual_chemo_application[: t + 1]
                    counterfactual_radio_application[: t + 1] = factual_radio_application[: t + 1]
                    counterfactual_chemo_dosage[: t + 1] = factual_chemo_dosage[: t + 1]
                    counterfactual_radio_dosage[: t + 1] = factual_radio_dosage[: t + 1]

                    for projection_time in range(projection_horizon):
                        current_t = t + 1 + projection_time
                        previous_chemo_dose = counterfactual_chemo_dosage[current_t - 1]
                        current_chemo_dose = 0.0
                        counterfactual_radio_dosage[current_t] = 0.0
                        if treatment_option[projection_time][0] == 1:
                            counterfactual_chemo_application[current_t] = 1
                            current_chemo_dose = chemo_amt[0]
                        if treatment_option[projection_time][1] == 1:
                            counterfactual_radio_application[current_t] = 1
                            counterfactual_radio_dosage[current_t] = radio_amt[0]

                        counterfactual_chemo_dosage[current_t] = previous_chemo_dose * np.exp(-np.log(2) / drug_half_life) + current_chemo_dose
                        counterfactual_cancer_volume[current_t + 1] = counterfactual_cancer_volume[current_t] * (
                            1
                            + rho * np.log(K / (counterfactual_cancer_volume[current_t] + 1e-7) + 1e-7)
                            - beta_c * counterfactual_chemo_dosage[current_t]
                            - (alpha * counterfactual_radio_dosage[current_t] + beta * counterfactual_radio_dosage[current_t] ** 2)
                            + noise[current_t + 1]
                        )

                    if np.isnan(counterfactual_cancer_volume).any():
                        continue

                    end = t + 1 + projection_horizon + 1
                    cancer_volume[test_idx][:end] = counterfactual_cancer_volume
                    chemo_application_point[test_idx][: end - 1] = counterfactual_chemo_application
                    radio_application_point[test_idx][: end - 1] = counterfactual_radio_application
                    patient_types_all[test_idx] = patient_types[i]
                    patient_ids_all[test_idx] = i
                    patient_current_t[test_idx] = t
                    sequence_lengths[test_idx] = int(t) + projection_horizon + 1
                    test_idx += 1

            if factual_cancer_volume[t + 1] >= TUMOUR_DEATH_THRESHOLD or recovery_rvs[t] <= np.exp(-factual_cancer_volume[t + 1] * TUMOUR_CELL_DENSITY):
                break

    return {
        "cancer_volume": cancer_volume[:test_idx],
        "chemo_application": chemo_application_point[:test_idx],
        "radio_application": radio_application_point[:test_idx],
        "sequence_lengths": sequence_lengths[:test_idx],
        "patient_types": patient_types_all[:test_idx],
        "patient_ids_all_trajectories": patient_ids_all[:test_idx],
        "patient_current_t": patient_current_t[:test_idx],
    }


# -----------------------------------------------------------------------------
# Resampling wrappers and dataset assembly
# -----------------------------------------------------------------------------

def make_valid_factual_test_data(gamma: int, window_size: int, seq_length: int, min_tobs: int, test_base_patients: int, max_attempts: int = 200) -> tuple[dict[str, Any], int]:
    for attempt in range(max_attempts):
        params = get_confounding_params(num_patients=test_base_patients, gamma=gamma, window_size=window_size)
        raw = simulate_factual(params, seq_length)
        raw_kept, keep_idx = filter_factual_min_tobs(raw, min_tobs=min_tobs)
        if len(keep_idx) > 0:
            return raw_kept, attempt + 1
    raise RuntimeError(f"Could not generate non-empty factual test data for gamma={gamma}, min_tobs={min_tobs}.")


def make_valid_one_step_test_data(gamma: int, window_size: int, seq_length: int, min_tobs: int, test_base_patients: int, max_attempts: int = 200) -> tuple[dict[str, Any], int]:
    for attempt in range(max_attempts):
        params = get_confounding_params(num_patients=test_base_patients, gamma=gamma, window_size=window_size)
        raw = simulate_counterfactual_1_step(params, seq_length, min_tobs=min_tobs)
        if raw["cancer_volume"].shape[0] > 0:
            return raw, attempt + 1
    raise RuntimeError(f"Could not generate non-empty one-step test_data for gamma={gamma}, min_tobs={min_tobs}.")


def make_valid_seq_test_data(gamma: int, cfg: CancerGeneratorConfig, max_attempts: int = 200) -> tuple[dict[str, Any], int]:
    for attempt in range(max_attempts):
        params = get_confounding_params(num_patients=cfg.test_base_patients, gamma=gamma, window_size=cfg.window_size)
        raw = simulate_counterfactuals_random_trajectories(
            params,
            num_time_steps=cfg.seq_length,
            projection_horizon=cfg.projection_horizon,
            min_tobs=cfg.min_t_obs,
            n_random_trajectories=int(cfg.n_seq_random_trajectories),
        )
        if raw["cancer_volume"].shape[0] > 0:
            return raw, attempt + 1
    raise RuntimeError(f"Could not generate non-empty seq test_data_seq for gamma={gamma}, min_tobs={cfg.min_t_obs}.")


def make_dataset(dataset_id: int, gamma: int, support_size: int, rep: int, seed: int, cfg: CancerGeneratorConfig) -> dict[str, Any]:
    np.random.seed(seed)
    LOGGER.info(
        "[dataset %03d] domain=cancer, gamma=%s, support=%s, rep=%s, "
        "test_base_patients=%s, min_tobs=%s, seed=%s",
        dataset_id,
        gamma,
        support_size,
        rep,
        cfg.test_base_patients,
        cfg.min_t_obs,
        seed,
    )

    support_data = make_support_data(
        support_size=support_size,
        gamma=gamma,
        window_size=cfg.window_size,
        seq_length=cfg.seq_length,
        min_tobs=cfg.min_t_obs,
    )

    test_data_factuals, factual_attempts = make_valid_factual_test_data(
        gamma=gamma,
        window_size=cfg.window_size,
        seq_length=cfg.seq_length,
        min_tobs=cfg.min_t_obs,
        test_base_patients=cfg.test_base_patients,
    )

    test_data_counterfactuals, one_step_attempts = make_valid_one_step_test_data(
        gamma=gamma,
        window_size=cfg.window_size,
        seq_length=cfg.seq_length,
        min_tobs=cfg.min_t_obs,
        test_base_patients=cfg.test_base_patients,
    )

    test_data_seq, seq_attempts = make_valid_seq_test_data(gamma=gamma, cfg=cfg)

    support_data = log_export(support_data)
    test_data_factuals = log_export(test_data_factuals)
    test_data_counterfactuals = log_export(test_data_counterfactuals)
    test_data_seq = log_export(test_data_seq)

    scaling_data = get_scaling_params(support_data)

    pickle_map = {
        "dataset_id": int(dataset_id),
        "seed": int(seed),
        "rep": int(rep),
        "domain": "cancer",
        "gamma": int(gamma),
        "chemo_coeff": float(gamma),
        "radio_coeff": float(gamma),
        "seq_length": int(cfg.seq_length),
        "num_time_steps": int(cfg.seq_length),
        "window_size": int(cfg.window_size),
        "min_t_obs": int(cfg.min_t_obs),
        "support_size": int(support_size),
        "training_size": int(support_size),
        "validation_size": 0,
        "test_size_base_patients": int(cfg.test_base_patients),
        "projection_horizon": int(cfg.projection_horizon),
        "cf_seq_mode": "random_trajectories",
        "n_seq_random_trajectories": int(cfg.n_seq_random_trajectories),
        "cancer_volume_transform": "log1p_clipped_volume",
        "target_space": "normalized_log1p_volume",
        "test_factual_resample_attempts": int(factual_attempts),
        "test_one_step_resample_attempts": int(one_step_attempts),
        "test_seq_resample_attempts": int(seq_attempts),
        "support_data": support_data,
        "test_data": test_data_counterfactuals,
        "test_data_factuals": test_data_factuals,
        "test_data_seq": test_data_seq,
        "scaling_data": scaling_data,
    }
    return standardize_pickle_map(
        pickle_map,
        domain="cancer",
        outcome_key="cancer_volume",
        state_key=None,
        state_from_static_key="patient_types",
        action_key=None,
        action_pair_keys=("chemo_application", "radio_application"),
        static_key=None,
        static_keys=("patient_types",),
        target_state_index=None,
    )


def generate(config: CancerGeneratorConfig | None = None, **overrides: Any) -> pd.DataFrame:
    cfg = config or CancerGeneratorConfig()
    if overrides:
        cfg = CancerGeneratorConfig.from_dict(dataclasses.asdict(cfg), **overrides)

    output_dir = ensure_output_dir(cfg.output_dir)

    summary_rows: list[dict[str, Any]] = []
    dataset_id = 0

    for gamma in cfg.gammas:
        for support_size in cfg.support_sizes:
            for rep in range(cfg.reps_per_cell):
                seed = int(cfg.base_seed) + dataset_id
                pickle_map = make_dataset(dataset_id=dataset_id, gamma=gamma, support_size=support_size, rep=rep, seed=seed, cfg=cfg)

                file_name = (
                    f"cancer_dataset_{dataset_id:03d}"
                    f"_gamma_{gamma}"
                    f"_support_{support_size}"
                    f"_rep_{rep}"
                    f"_testbase_{cfg.test_base_patients}"
                    f"_mintobs_{cfg.min_t_obs}"
                    f"_seed_{seed}.p"
                )
                file_path = output_dir / file_name
                save_pickle(pickle_map, file_path)

                one_step_rows = pickle_map["test_data"]["outcomes"].shape[0]
                seq_rows = pickle_map["test_data_seq"]["outcomes"].shape[0]
                if one_step_rows == 0 or seq_rows == 0:
                    raise RuntimeError(
                        f"Generated empty test task in dataset_id={dataset_id}: one_step_rows={one_step_rows}, seq_rows={seq_rows}"
                    )

                summary_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "file_name": file_name,
                        "file_path": str(file_path),
                        "seed": seed,
                        "domain": "cancer",
                        "gamma": gamma,
                        "chemo_coeff": float(gamma),
                        "radio_coeff": float(gamma),
                        "support_size": support_size,
                        "training_size": support_size,
                        "validation_size": 0,
                        "rep": rep,
                        "test_size_base_patients": cfg.test_base_patients,
                        "test_data_rows_one_step_counterfactual": one_step_rows,
                        "test_data_seq_rows_multi_step_counterfactual": seq_rows,
                        "test_factual_resample_attempts": pickle_map["test_factual_resample_attempts"],
                        "test_one_step_resample_attempts": pickle_map["test_one_step_resample_attempts"],
                        "test_seq_resample_attempts": pickle_map["test_seq_resample_attempts"],
                        "min_t_obs": cfg.min_t_obs,
                        "seq_length": cfg.seq_length,
                        "window_size": cfg.window_size,
                        "projection_horizon": cfg.projection_horizon,
                        "cf_seq_mode": "random_trajectories",
                        "n_seq_random_trajectories": cfg.n_seq_random_trajectories,
                        "support_min_sequence_lengths": float(np.min(pickle_map["support_data"]["sequence_lengths"])),
                        "one_step_min_sequence_lengths": float(np.min(pickle_map["test_data"]["sequence_lengths"])),
                        "seq_min_downstream_t_obs": float(np.min(pickle_map["test_data_seq"]["patient_current_t"] + 2)),
                        "cancer_volume_transform": "log1p_clipped_volume",
                        "target_space": "normalized_log1p_volume",
                    }
                )

                del pickle_map
                gc.collect()
                dataset_id += 1

    summary = pd.DataFrame(summary_rows)

    LOGGER.info("Generated cancer datasets: %s", len(summary))
    LOGGER.info("Dataset directory: %s", output_dir)
    LOGGER.info("Any empty one-step task: %s", bool((summary["test_data_rows_one_step_counterfactual"] == 0).any()))
    LOGGER.info("Any empty sequence task: %s", bool((summary["test_data_seq_rows_multi_step_counterfactual"] == 0).any()))
    LOGGER.info("Min one-step rows: %s", int(summary["test_data_rows_one_step_counterfactual"].min()))
    LOGGER.info("Min sequence rows: %s", int(summary["test_data_seq_rows_multi_step_counterfactual"].min()))
    LOGGER.info("Min support sequence length: %.3f", float(summary["support_min_sequence_lengths"].min()))
    LOGGER.info("Min one-step sequence length: %.3f", float(summary["one_step_min_sequence_lengths"].min()))
    LOGGER.info("Min sequence downstream t_obs: %.3f", float(summary["seq_min_downstream_t_obs"].min()))

    return summary


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    generate()
