"""Adams/WhyNot-style HIV benchmark generator for CausalLongPFN evaluation.

This generator preserves:

* 6-compartment Adams/WhyNot HIV ODE dynamics.
* 4-action therapy mapping.
* Gamma-controlled behavioral-policy confounding.
* log10(1 + free_virus) target space.

"""

from __future__ import annotations

import copy
import dataclasses
import gc
import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import odeint

from .common import concat_raw, ensure_output_dir, save_pickle, standardize_pickle_map, take_rows

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class HIVGeneratorConfig:
    output_dir: str = "outputs/data/hiv"

    gammas: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    support_sizes: tuple[int, ...] = (40, 80, 160, 320, 500)
    reps_per_cell: int = 2

    test_base_patients: int = 1
    seq_length: int = 60
    projection_horizon: int = 5
    n_seq_random_trajectories: int | None = None
    min_t_obs: int = 5

    base_seed: int = 3000

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None = None, **overrides: Any) -> "HIVGeneratorConfig":
        values = dict(values or {})
        values.update({k: v for k, v in overrides.items() if v is not None})
        valid = {field.name for field in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in valid})

    def __post_init__(self) -> None:
        self.gammas = tuple(int(x) for x in self.gammas)
        self.support_sizes = tuple(int(x) for x in self.support_sizes)
        if self.n_seq_random_trajectories is None:
            self.n_seq_random_trajectories = int(self.projection_horizon) * 2


D_STATE = 6
N_ACTIONS = 4
BIN_DAYS = 4.0 / 24.0
ODE_DELTA_T = 0.05
TARGET_FEATURE_INDEX = 4
TARGET_NAME = "log10(1 + free_virus)"


@dataclasses.dataclass
class HIVODEConfig:
    lambda_1: float = 10000.0
    lambda_2: float = 31.98
    d_1: float = 0.01
    d_2: float = 0.01
    k_1: float = 8e-7
    k_2: float = 1e-4
    f: float = 0.34
    delta: float = 0.7
    m_1: float = 1e-5
    m_2: float = 1e-5
    N_T: float = 100.0
    c: float = 13.0
    rho_1: float = 1.0
    rho_2: float = 1.0
    lambda_E: float = 1.0
    b_E: float = 0.3
    K_B: float = 100.0
    d_E: float = 0.25
    K_D: float = 500.0
    delta_E: float = 0.1

    epsilon_1: float = 0.0  # RTI
    epsilon_2: float = 0.0  # PI

    start_time: float = 0.0
    end_time: float = BIN_DAYS
    delta_t: float = ODE_DELTA_T
    rtol: float = 1e-6
    atol: float = 1e-6

    def update(self, **kwargs: Any) -> "HIVODEConfig":
        params = dataclasses.asdict(self)
        params.update(kwargs)
        return HIVODEConfig(**params)


BASE_CFG = HIVODEConfig()


@dataclasses.dataclass
class PatientProfile:
    config: HIVODEConfig
    eff_rti_scale: float
    eff_pi_scale: float
    virus_threshold: float
    immune_threshold: float
    policy_aggr: float


# Action mapping from WhyNot HIV environment.
ACTION_TO_EPS = {
    0: (0.0, 0.0),  # no therapy
    1: (0.0, 0.3),  # PI only
    2: (0.7, 0.0),  # RTI only
    3: (0.7, 0.3),  # RTI + PI
}


def _ln_mult(rng: np.random.Generator, sigma: float) -> float:
    return float(np.exp(rng.normal(0.0, sigma)))


def state_to_features(raw_state: np.ndarray) -> np.ndarray:
    x = np.asarray(raw_state, dtype=np.float32)
    return np.log10(1.0 + x).astype(np.float32)


def get_scaling_params(sim: dict[str, Any]) -> tuple[pd.Series, pd.Series]:
    means: dict[str, Any] = {}
    stds: dict[str, Any] = {}
    seq_lengths = np.asarray(sim["sequence_lengths"], dtype=np.int64)

    if "states" in sim:
        vals = []
        for i in range(seq_lengths.shape[0]):
            end = int(seq_lengths[i])
            if end > 0:
                vals.append(sim["states"][i, :end, :])
        if len(vals):
            vals = np.concatenate(vals, axis=0)
            means["states"] = vals.mean(axis=0)
            stds["states"] = np.maximum(vals.std(axis=0), 1e-6)
        else:
            means["states"] = np.zeros(D_STATE, dtype=np.float32)
            stds["states"] = np.ones(D_STATE, dtype=np.float32)

    if "hiv_outcome" in sim:
        vals = []
        for i in range(seq_lengths.shape[0]):
            end = int(seq_lengths[i])
            if end > 0:
                vals.append(sim["hiv_outcome"][i, :end])
        if len(vals):
            vals = np.concatenate(vals)
            means["hiv_outcome"] = float(np.mean(vals))
            stds["hiv_outcome"] = float(max(np.std(vals), 1e-6))
        else:
            means["hiv_outcome"] = 0.0
            stds["hiv_outcome"] = 1.0

    for key in ["eff_rti_scale", "eff_pi_scale", "virus_threshold", "immune_threshold", "policy_aggr"]:
        if key in sim:
            means[key] = float(np.mean(sim[key])) if len(sim[key]) else 0.0
            stds[key] = float(max(np.std(sim[key]), 1e-6)) if len(sim[key]) else 1.0

    return pd.Series(means), pd.Series(stds)


# -----------------------------------------------------------------------------
# HIV ODE dynamics
# -----------------------------------------------------------------------------

def hiv_dynamics(state: np.ndarray, time: float, config: HIVODEConfig) -> list[float]:
    (
        uninfected_T1,
        infected_T1,
        uninfected_T2,
        infected_T2,
        free_virus,
        immune_response,
    ) = state

    delta_uninfected_T1 = config.lambda_1 - config.d_1 * uninfected_T1 - (1 - config.epsilon_1) * config.k_1 * free_virus * uninfected_T1
    delta_infected_T1 = (1 - config.epsilon_1) * config.k_1 * free_virus * uninfected_T1 - config.delta * infected_T1 - config.m_1 * immune_response * infected_T1
    delta_uninfected_T2 = config.lambda_2 - config.d_2 * uninfected_T2 - (1 - config.f * config.epsilon_1) * config.k_2 * free_virus * uninfected_T2
    delta_infected_T2 = (1 - config.f * config.epsilon_1) * config.k_2 * free_virus * uninfected_T2 - config.delta * infected_T2 - config.m_2 * immune_response * infected_T2
    delta_virus = (
        (1 - config.epsilon_2) * config.N_T * config.delta * (infected_T1 + infected_T2)
        - config.c * free_virus
        - free_virus
        * (
            (1.0 - config.epsilon_1) * config.rho_1 * config.k_1 * uninfected_T1
            + (1.0 - config.f * config.epsilon_1) * config.rho_2 * config.k_2 * uninfected_T2
        )
    )

    infected_total = infected_T1 + infected_T2
    delta_immune_response = (
        config.lambda_E
        + ((config.b_E * infected_total) / (infected_total + config.K_B)) * immune_response
        - ((config.d_E * infected_total) / (infected_total + config.K_D)) * immune_response
        - config.delta_E * immune_response
    )

    return [
        delta_uninfected_T1,
        delta_infected_T1,
        delta_uninfected_T2,
        delta_infected_T2,
        delta_virus,
        delta_immune_response,
    ]


def integrate_one_bin(raw_state: np.ndarray, profile: PatientProfile, action: int) -> np.ndarray:
    eps1_base, eps2_base = ACTION_TO_EPS[int(action)]
    eps1 = min(0.99, eps1_base * profile.eff_rti_scale)
    eps2 = min(0.99, eps2_base * profile.eff_pi_scale)
    cfg = profile.config.update(epsilon_1=eps1, epsilon_2=eps2)

    t_eval = np.arange(cfg.start_time, cfg.end_time + cfg.delta_t, cfg.delta_t)
    if t_eval[-1] < cfg.end_time:
        t_eval = np.append(t_eval, cfg.end_time)

    sol = odeint(hiv_dynamics, y0=raw_state.astype(np.float64), t=t_eval, args=(cfg,), rtol=cfg.rtol, atol=cfg.atol)
    out = sol[-1].astype(np.float64)

    out = np.maximum(out, 0.0)
    out[0] = min(out[0], 5e6)
    out[1] = min(out[1], 5e6)
    out[2] = min(out[2], 5e5)
    out[3] = min(out[3], 5e5)
    out[4] = min(out[4], 1e8)
    out[5] = min(out[5], 1e6)
    return out


# -----------------------------------------------------------------------------
# Heterogeneity and behavior policy
# -----------------------------------------------------------------------------

def sample_patient_profile(rng: np.random.Generator) -> PatientProfile:
    cfg = HIVODEConfig(
        lambda_1=BASE_CFG.lambda_1 * _ln_mult(rng, 0.18),
        lambda_2=BASE_CFG.lambda_2 * _ln_mult(rng, 0.18),
        d_1=BASE_CFG.d_1 * _ln_mult(rng, 0.10),
        d_2=BASE_CFG.d_2 * _ln_mult(rng, 0.10),
        k_1=BASE_CFG.k_1 * _ln_mult(rng, 0.30),
        k_2=BASE_CFG.k_2 * _ln_mult(rng, 0.30),
        f=float(np.clip(BASE_CFG.f * _ln_mult(rng, 0.10), 0.15, 0.75)),
        delta=BASE_CFG.delta * _ln_mult(rng, 0.18),
        m_1=BASE_CFG.m_1 * _ln_mult(rng, 0.20),
        m_2=BASE_CFG.m_2 * _ln_mult(rng, 0.20),
        N_T=BASE_CFG.N_T * _ln_mult(rng, 0.30),
        c=BASE_CFG.c * _ln_mult(rng, 0.20),
        rho_1=BASE_CFG.rho_1 * _ln_mult(rng, 0.12),
        rho_2=BASE_CFG.rho_2 * _ln_mult(rng, 0.12),
        lambda_E=BASE_CFG.lambda_E * _ln_mult(rng, 0.25),
        b_E=BASE_CFG.b_E * _ln_mult(rng, 0.20),
        K_B=BASE_CFG.K_B * _ln_mult(rng, 0.20),
        d_E=BASE_CFG.d_E * _ln_mult(rng, 0.20),
        K_D=BASE_CFG.K_D * _ln_mult(rng, 0.18),
        delta_E=BASE_CFG.delta_E * _ln_mult(rng, 0.18),
    )

    eff_rti_scale = float(np.clip(rng.lognormal(mean=np.log(0.95), sigma=0.18), 0.55, 1.20))
    eff_pi_scale = float(np.clip(rng.lognormal(mean=np.log(0.95), sigma=0.18), 0.55, 1.20))
    virus_threshold = float(10 ** rng.uniform(1.0, 4.5))
    immune_threshold = float(10 ** rng.uniform(0.8, 1.4))
    policy_aggr = float(rng.uniform(0.75, 1.35))

    return PatientProfile(
        config=cfg,
        eff_rti_scale=eff_rti_scale,
        eff_pi_scale=eff_pi_scale,
        virus_threshold=virus_threshold,
        immune_threshold=immune_threshold,
        policy_aggr=policy_aggr,
    )


def sample_initial_state(rng: np.random.Generator, profile: PatientProfile) -> np.ndarray:
    severity = float(rng.beta(1.8, 1.8))
    uninfected_T1 = float(np.clip(1e6 * (1.15 - 0.70 * severity) * _ln_mult(rng, 0.22), 2e5, 2e6))
    infected_T1 = float(np.clip(10 ** rng.normal(-1.0 + 4.0 * severity, 0.35), 1e-4, 5e5))
    uninfected_T2 = float(np.clip(3198.0 * (1.10 - 0.55 * severity) * _ln_mult(rng, 0.20), 500.0, 8e4))
    infected_T2 = float(np.clip(10 ** rng.normal(-1.0 + 2.8 * severity, 0.35), 1e-4, 5e4))
    free_virus = float(np.clip(10 ** rng.normal(0.3 + 5.0 * severity, 0.45), 1.0, 1e7))
    immune_response = float(np.clip(10 ** rng.normal(1.1 - 0.15 * severity, 0.20), 2.0, 1e4))
    return np.array([uninfected_T1, infected_T1, uninfected_T2, infected_T2, free_virus, immune_response], dtype=np.float64)


def sequential_policy(raw_state: np.ndarray, prev_action: int, gamma: int, profile: PatientProfile, rng: np.random.Generator) -> int:
    _, _, _, _, free_virus, immune_response = raw_state
    virus_term = math.log10(max(free_virus, 1e-8) + 1.0)
    immune_term = math.log10(max(immune_response, 1e-8) + 1.0)
    v_thr = math.log10(profile.virus_threshold + 1.0)
    i_thr = math.log10(profile.immune_threshold + 1.0)
    v_excess = virus_term - v_thr
    i_excess = immune_term - i_thr

    score = profile.policy_aggr * (1.00 * v_excess + 0.35 * i_excess + 0.50 * float((v_excess > 0.0) and (i_excess > 0.0)))
    logits = np.array([1.15, 0.10, 0.40, -0.20], dtype=np.float64)
    sev = float(gamma) * score
    logits[0] -= 1.20 * sev
    logits[1] += 0.35 * sev
    logits[2] += 0.75 * sev
    logits[3] += 1.05 * sev
    logits[int(prev_action)] += 1.25

    if free_virus < 5.0:
        logits[0] += 0.45
        logits[3] -= 0.20

    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return int(rng.choice(N_ACTIONS, p=probs))


# -----------------------------------------------------------------------------
# Patient trajectory simulation
# -----------------------------------------------------------------------------

def simulate_factual_patient(rng: np.random.Generator, gamma: int, seq_length: int) -> dict[str, Any]:
    profile = sample_patient_profile(rng)
    raw_state = sample_initial_state(rng, profile)

    raw_states = np.zeros((seq_length, D_STATE), dtype=np.float32)
    states = np.zeros((seq_length, D_STATE), dtype=np.float32)
    hiv_outcome = np.zeros(seq_length, dtype=np.float32)
    actions = np.zeros(seq_length, dtype=np.int64)
    snapshots: list[np.ndarray] = []
    prev_action = 0

    for t in range(seq_length):
        raw_states[t] = raw_state.astype(np.float32)
        feat = state_to_features(raw_state)
        states[t] = feat
        hiv_outcome[t] = float(feat[TARGET_FEATURE_INDEX])
        snapshots.append(raw_state.copy())
        action = sequential_policy(raw_state=raw_state, prev_action=prev_action, gamma=gamma, profile=profile, rng=rng)
        actions[t] = action
        raw_state = integrate_one_bin(raw_state, profile, action)
        prev_action = action

    profile_summary = {
        "eff_rti_scale": profile.eff_rti_scale,
        "eff_pi_scale": profile.eff_pi_scale,
        "virus_threshold": profile.virus_threshold,
        "immune_threshold": profile.immune_threshold,
        "policy_aggr": profile.policy_aggr,
    }
    return {
        "profile": profile,
        "profile_summary": profile_summary,
        "raw_states": raw_states,
        "states": states,
        "hiv_outcome": hiv_outcome,
        "actions": actions,
        "snapshots": snapshots,
    }


def simulate_factual_dataset(rng: np.random.Generator, num_patients: int, gamma: int, seq_length: int) -> dict[str, np.ndarray]:
    raw_states = np.zeros((num_patients, seq_length, D_STATE), dtype=np.float32)
    states = np.zeros((num_patients, seq_length, D_STATE), dtype=np.float32)
    hiv_outcome = np.zeros((num_patients, seq_length), dtype=np.float32)
    actions = np.zeros((num_patients, seq_length), dtype=np.int64)
    sequence_lengths = np.full(num_patients, seq_length, dtype=np.int64)

    eff_rti_scale = np.zeros(num_patients, dtype=np.float32)
    eff_pi_scale = np.zeros(num_patients, dtype=np.float32)
    virus_threshold = np.zeros(num_patients, dtype=np.float32)
    immune_threshold = np.zeros(num_patients, dtype=np.float32)
    policy_aggr = np.zeros(num_patients, dtype=np.float32)

    for i in range(num_patients):
        sim = simulate_factual_patient(rng, gamma, seq_length)
        ps = sim["profile_summary"]
        raw_states[i] = sim["raw_states"]
        states[i] = sim["states"]
        hiv_outcome[i] = sim["hiv_outcome"]
        actions[i] = sim["actions"]
        eff_rti_scale[i] = ps["eff_rti_scale"]
        eff_pi_scale[i] = ps["eff_pi_scale"]
        virus_threshold[i] = ps["virus_threshold"]
        immune_threshold[i] = ps["immune_threshold"]
        policy_aggr[i] = ps["policy_aggr"]

    return {
        "raw_states": raw_states,
        "states": states,
        "hiv_outcome": hiv_outcome,
        "actions": actions,
        "sequence_lengths": sequence_lengths,
        "eff_rti_scale": eff_rti_scale,
        "eff_pi_scale": eff_pi_scale,
        "virus_threshold": virus_threshold,
        "immune_threshold": immune_threshold,
        "policy_aggr": policy_aggr,
    }


# -----------------------------------------------------------------------------
# Counterfactual test rows
# -----------------------------------------------------------------------------

def simulate_one_step_counterfactual_rows(rng: np.random.Generator, num_patients: int, gamma: int, seq_length: int, min_tobs: int) -> dict[str, np.ndarray]:
    max_rows = num_patients * seq_length * N_ACTIONS
    raw_states = np.zeros((max_rows, seq_length, D_STATE), dtype=np.float32)
    states = np.zeros((max_rows, seq_length, D_STATE), dtype=np.float32)
    hiv_outcome = np.zeros((max_rows, seq_length), dtype=np.float32)
    actions = np.zeros((max_rows, seq_length), dtype=np.int64)
    sequence_lengths = np.zeros(max_rows, dtype=np.int64)
    patient_ids = np.zeros(max_rows, dtype=np.int64)
    patient_current_t = np.zeros(max_rows, dtype=np.int64)
    eff_rti_scale = np.zeros(max_rows, dtype=np.float32)
    eff_pi_scale = np.zeros(max_rows, dtype=np.float32)
    virus_threshold = np.zeros(max_rows, dtype=np.float32)
    immune_threshold = np.zeros(max_rows, dtype=np.float32)
    policy_aggr = np.zeros(max_rows, dtype=np.float32)
    row = 0

    for pid in range(num_patients):
        sim = simulate_factual_patient(rng, gamma, seq_length)
        profile = sim["profile"]
        ps = sim["profile_summary"]
        factual_raw = sim["raw_states"]
        factual_states = sim["states"]
        factual_outcome = sim["hiv_outcome"]
        factual_actions = sim["actions"]
        snapshots = sim["snapshots"]

        for t in range(min_tobs - 1, seq_length - 1):
            for action in range(N_ACTIONS):
                cf_next_raw = integrate_one_bin(raw_state=snapshots[t].copy(), profile=profile, action=action)
                cf_next_feat = state_to_features(cf_next_raw)
                cf_next_outcome = float(cf_next_feat[TARGET_FEATURE_INDEX])

                raw_states[row, : t + 1] = factual_raw[: t + 1]
                states[row, : t + 1] = factual_states[: t + 1]
                hiv_outcome[row, : t + 1] = factual_outcome[: t + 1]
                actions[row, :t] = factual_actions[:t]

                raw_states[row, t + 1] = cf_next_raw.astype(np.float32)
                states[row, t + 1] = cf_next_feat
                hiv_outcome[row, t + 1] = cf_next_outcome
                actions[row, t] = action

                sequence_lengths[row] = t + 1
                patient_ids[row] = pid
                patient_current_t[row] = t
                eff_rti_scale[row] = ps["eff_rti_scale"]
                eff_pi_scale[row] = ps["eff_pi_scale"]
                virus_threshold[row] = ps["virus_threshold"]
                immune_threshold[row] = ps["immune_threshold"]
                policy_aggr[row] = ps["policy_aggr"]
                row += 1

    return {
        "raw_states": raw_states[:row],
        "states": states[:row],
        "hiv_outcome": hiv_outcome[:row],
        "actions": actions[:row],
        "sequence_lengths": sequence_lengths[:row],
        "patient_ids_all_trajectories": patient_ids[:row],
        "patient_current_t": patient_current_t[:row],
        "eff_rti_scale": eff_rti_scale[:row],
        "eff_pi_scale": eff_pi_scale[:row],
        "virus_threshold": virus_threshold[:row],
        "immune_threshold": immune_threshold[:row],
        "policy_aggr": policy_aggr[:row],
    }


def simulate_sequence_counterfactual_rows(
    rng: np.random.Generator,
    num_patients: int,
    gamma: int,
    seq_length: int,
    projection_horizon: int,
    min_tobs: int,
    n_random_trajectories: int,
) -> dict[str, np.ndarray]:
    max_rows = num_patients * seq_length * n_random_trajectories
    total_T = seq_length + projection_horizon
    raw_states = np.zeros((max_rows, total_T, D_STATE), dtype=np.float32)
    states = np.zeros((max_rows, total_T, D_STATE), dtype=np.float32)
    hiv_outcome = np.zeros((max_rows, total_T), dtype=np.float32)
    actions = np.zeros((max_rows, total_T), dtype=np.int64)
    sequence_lengths = np.zeros(max_rows, dtype=np.int64)
    patient_ids = np.zeros(max_rows, dtype=np.int64)
    patient_current_t = np.zeros(max_rows, dtype=np.int64)
    eff_rti_scale = np.zeros(max_rows, dtype=np.float32)
    eff_pi_scale = np.zeros(max_rows, dtype=np.float32)
    virus_threshold = np.zeros(max_rows, dtype=np.float32)
    immune_threshold = np.zeros(max_rows, dtype=np.float32)
    policy_aggr = np.zeros(max_rows, dtype=np.float32)
    row = 0

    for pid in range(num_patients):
        sim = simulate_factual_patient(rng, gamma, seq_length)
        profile = sim["profile"]
        ps = sim["profile_summary"]
        factual_raw = sim["raw_states"]
        factual_states = sim["states"]
        factual_outcome = sim["hiv_outcome"]
        factual_actions = sim["actions"]
        start_t = max(0, min_tobs - 2)

        for t in range(start_t, seq_length - 1):
            treatment_options = rng.integers(0, N_ACTIONS, size=(n_random_trajectories, projection_horizon), endpoint=False)
            for plan in treatment_options:
                hist_len = t + 2
                row_raw = np.zeros((total_T, D_STATE), dtype=np.float32)
                row_states = np.zeros((total_T, D_STATE), dtype=np.float32)
                row_outcome = np.zeros(total_T, dtype=np.float32)
                row_actions = np.zeros(total_T, dtype=np.int64)
                row_raw[:hist_len] = factual_raw[:hist_len]
                row_states[:hist_len] = factual_states[:hist_len]
                row_outcome[:hist_len] = factual_outcome[:hist_len]
                row_actions[: t + 1] = factual_actions[: t + 1]

                cf_raw = factual_raw[t + 1].astype(np.float64).copy()
                for h in range(projection_horizon):
                    current_time = t + 1 + h
                    action = int(plan[h])
                    row_actions[current_time] = action
                    cf_raw = integrate_one_bin(raw_state=cf_raw, profile=profile, action=action)
                    cf_feat = state_to_features(cf_raw)
                    row_raw[current_time + 1] = cf_raw.astype(np.float32)
                    row_states[current_time + 1] = cf_feat
                    row_outcome[current_time + 1] = float(cf_feat[TARGET_FEATURE_INDEX])

                end = t + 1 + projection_horizon + 1
                raw_states[row, :end] = row_raw[:end]
                states[row, :end] = row_states[:end]
                hiv_outcome[row, :end] = row_outcome[:end]
                actions[row, : end - 1] = row_actions[: end - 1]
                patient_ids[row] = pid
                patient_current_t[row] = t
                sequence_lengths[row] = int(t) + projection_horizon + 1
                eff_rti_scale[row] = ps["eff_rti_scale"]
                eff_pi_scale[row] = ps["eff_pi_scale"]
                virus_threshold[row] = ps["virus_threshold"]
                immune_threshold[row] = ps["immune_threshold"]
                policy_aggr[row] = ps["policy_aggr"]
                row += 1

    return {
        "raw_states": raw_states[:row],
        "states": states[:row],
        "hiv_outcome": hiv_outcome[:row],
        "actions": actions[:row],
        "sequence_lengths": sequence_lengths[:row],
        "patient_ids_all_trajectories": patient_ids[:row],
        "patient_current_t": patient_current_t[:row],
        "eff_rti_scale": eff_rti_scale[:row],
        "eff_pi_scale": eff_pi_scale[:row],
        "virus_threshold": virus_threshold[:row],
        "immune_threshold": immune_threshold[:row],
        "policy_aggr": policy_aggr[:row],
    }


# -----------------------------------------------------------------------------
# Valid-data wrappers and dataset assembly
# -----------------------------------------------------------------------------

def make_support_data(rng: np.random.Generator, support_size: int, gamma: int, seq_length: int, min_tobs: int) -> dict[str, Any]:
    raw = simulate_factual_dataset(rng=rng, num_patients=support_size, gamma=gamma, seq_length=seq_length)
    keep = np.where(raw["sequence_lengths"] >= min_tobs)[0]
    raw = take_rows(raw, keep)

    if raw["states"].shape[0] < support_size:
        chunks = [raw]
        n_kept = raw["states"].shape[0]
        while n_kept < support_size:
            extra = simulate_factual_dataset(rng=rng, num_patients=max(support_size, 50), gamma=gamma, seq_length=seq_length)
            keep = np.where(extra["sequence_lengths"] >= min_tobs)[0]
            extra = take_rows(extra, keep)
            chunks.append(extra)
            n_kept += extra["states"].shape[0]
        raw = concat_raw(chunks)

    return take_rows(raw, np.arange(support_size))


def make_valid_one_step_test_data(rng: np.random.Generator, gamma: int, cfg: HIVGeneratorConfig, max_attempts: int = 100) -> tuple[dict[str, Any], int]:
    for attempt in range(max_attempts):
        raw = simulate_one_step_counterfactual_rows(
            rng=rng,
            num_patients=cfg.test_base_patients,
            gamma=gamma,
            seq_length=cfg.seq_length,
            min_tobs=cfg.min_t_obs,
        )
        if raw["states"].shape[0] > 0:
            return raw, attempt + 1
    raise RuntimeError(f"Could not generate non-empty one-step HIV test data for gamma={gamma}.")


def make_valid_seq_test_data(rng: np.random.Generator, gamma: int, cfg: HIVGeneratorConfig, max_attempts: int = 100) -> tuple[dict[str, Any], int]:
    for attempt in range(max_attempts):
        raw = simulate_sequence_counterfactual_rows(
            rng=rng,
            num_patients=cfg.test_base_patients,
            gamma=gamma,
            seq_length=cfg.seq_length,
            projection_horizon=cfg.projection_horizon,
            min_tobs=cfg.min_t_obs,
            n_random_trajectories=int(cfg.n_seq_random_trajectories),
        )
        if raw["states"].shape[0] > 0:
            return raw, attempt + 1
    raise RuntimeError(f"Could not generate non-empty sequence HIV test data for gamma={gamma}.")


def make_dataset(dataset_id: int, gamma: int, support_size: int, rep: int, seed: int, cfg: HIVGeneratorConfig) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    LOGGER.info(
        "[dataset %03d] domain=hiv, gamma=%s, support=%s, rep=%s, "
        "test_base_patients=%s, min_tobs=%s, seed=%s",
        dataset_id,
        gamma,
        support_size,
        rep,
        cfg.test_base_patients,
        cfg.min_t_obs,
        seed,
    )

    support_data = make_support_data(rng=rng, support_size=support_size, gamma=gamma, seq_length=cfg.seq_length, min_tobs=cfg.min_t_obs)
    test_data_counterfactuals, one_step_attempts = make_valid_one_step_test_data(rng=rng, gamma=gamma, cfg=cfg)
    test_data_seq, seq_attempts = make_valid_seq_test_data(rng=rng, gamma=gamma, cfg=cfg)
    test_data_factuals = simulate_factual_dataset(rng=rng, num_patients=cfg.test_base_patients, gamma=gamma, seq_length=cfg.seq_length)
    scaling_data = get_scaling_params(support_data)

    pickle_map = {
        "dataset_id": int(dataset_id),
        "seed": int(seed),
        "rep": int(rep),
        "domain": "hiv",
        "gamma": int(gamma),
        "seq_length": int(cfg.seq_length),
        "num_time_steps": int(cfg.seq_length),
        "min_t_obs": int(cfg.min_t_obs),
        "support_size": int(support_size),
        "training_size": int(support_size),
        "validation_size": 0,
        "test_size_base_patients": int(cfg.test_base_patients),
        "projection_horizon": int(cfg.projection_horizon),
        "cf_seq_mode": "random_trajectories",
        "n_seq_random_trajectories": int(cfg.n_seq_random_trajectories),
        "state_name": "states",
        "outcome_name": "outcomes",
        "action_name": "actions",
        "static_name": "static_features",
        "target_feature_index": int(TARGET_FEATURE_INDEX),
        "target_feature_name": TARGET_NAME,
        "target_space": "log10_1p_free_virus",
        "action_mapping": {0: "no_therapy", 1: "PI_only", 2: "RTI_only", 3: "RTI_plus_PI"},
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
        domain="hiv",
        outcome_key="hiv_outcome",
        state_key="states",
        action_key="actions",
        static_key=None,
        static_keys=("eff_rti_scale", "eff_pi_scale", "virus_threshold", "immune_threshold", "policy_aggr"),
        target_state_index=TARGET_FEATURE_INDEX,
    )


def generate(config: HIVGeneratorConfig | None = None, **overrides: Any) -> pd.DataFrame:
    cfg = config or HIVGeneratorConfig()
    if overrides:
        cfg = HIVGeneratorConfig.from_dict(dataclasses.asdict(cfg), **overrides)

    output_dir = ensure_output_dir(cfg.output_dir)

    summary_rows: list[dict[str, Any]] = []
    dataset_id = 0

    for gamma in cfg.gammas:
        for support_size in cfg.support_sizes:
            for rep in range(cfg.reps_per_cell):
                seed = int(cfg.base_seed) + dataset_id
                pickle_map = make_dataset(dataset_id=dataset_id, gamma=gamma, support_size=support_size, rep=rep, seed=seed, cfg=cfg)

                file_name = (
                    f"hiv_pfn_dataset_{dataset_id:03d}"
                    f"_gamma_{gamma}"
                    f"_support_{support_size}"
                    f"_rep_{rep}"
                    f"_testbase_{cfg.test_base_patients}"
                    f"_mintobs_{cfg.min_t_obs}"
                    f"_seed_{seed}.p"
                )
                file_path = output_dir / file_name
                save_pickle(pickle_map, file_path)

                one_step_rows = pickle_map["test_data"]["states"].shape[0]
                seq_rows = pickle_map["test_data_seq"]["states"].shape[0]
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
                        "domain": "hiv",
                        "gamma": gamma,
                        "support_size": support_size,
                        "training_size": support_size,
                        "validation_size": 0,
                        "rep": rep,
                        "test_size_base_patients": cfg.test_base_patients,
                        "test_data_rows_one_step_counterfactual": one_step_rows,
                        "test_data_seq_rows_multi_step_counterfactual": seq_rows,
                        "test_one_step_resample_attempts": pickle_map["test_one_step_resample_attempts"],
                        "test_seq_resample_attempts": pickle_map["test_seq_resample_attempts"],
                        "min_t_obs": cfg.min_t_obs,
                        "seq_length": cfg.seq_length,
                        "projection_horizon": cfg.projection_horizon,
                        "cf_seq_mode": "random_trajectories",
                        "n_seq_random_trajectories": cfg.n_seq_random_trajectories,
                        "outcome_name": "outcomes",
                        "target_feature_index": TARGET_FEATURE_INDEX,
                        "target_feature_name": TARGET_NAME,
                        "state_dim": D_STATE,
                        "n_actions": N_ACTIONS,
                        "target_space": "log10_1p_free_virus",
                        "support_min_sequence_lengths": float(np.min(pickle_map["support_data"]["sequence_lengths"])),
                        "one_step_min_sequence_lengths": float(np.min(pickle_map["test_data"]["sequence_lengths"])),
                        "seq_min_downstream_t_obs": float(np.min(pickle_map["test_data_seq"]["patient_current_t"] + 2)),
                    }
                )

                del pickle_map
                gc.collect()
                dataset_id += 1

    summary = pd.DataFrame(summary_rows)
    LOGGER.info("Generated HIV datasets: %s", len(summary))
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
