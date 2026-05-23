"""Semi-QSP-lite Warfarin benchmark generator for CausalLongPFN evaluation.

This generator preserves PK/PD dynamics, gamma-controlled behavioral policy,
one-step counterfactual rows, and 5-step random-trajectory rows.
"""

from __future__ import annotations

import dataclasses
import gc
import copy
import math
import logging
from typing import Any
import numpy as np
import pandas as pd

from .common import concat_raw, ensure_output_dir, save_pickle, standardize_pickle_map, take_rows

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class WarfarinGeneratorConfig:
    output_dir: str = "outputs/data/warfarin"
    gammas: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    support_sizes: tuple[int, ...] = (40, 80, 160, 320, 500)
    reps_per_cell: int = 2
    test_base_patients: int = 1
    seq_length: int = 60
    projection_horizon: int = 5
    n_seq_random_trajectories: int | None = None
    min_t_obs: int = 10
    base_seed: int = 2000

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None = None, **overrides: Any) -> "WarfarinGeneratorConfig":
        values = dict(values or {})
        values.update({k: v for k, v in overrides.items() if v is not None})
        valid = {field.name for field in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in valid})

    def __post_init__(self) -> None:
        self.gammas = tuple(int(x) for x in self.gammas)
        self.support_sizes = tuple(int(x) for x in self.support_sizes)
        if self.n_seq_random_trajectories is None:
            self.n_seq_random_trajectories = int(self.projection_horizon) * 2


# ============================================================
# Config
# ============================================================

OUTPUT_DIR = "outputs/data/warfarin"

GAMMAS = [1,2,3,4,5,6,7,8,9,10]
SUPPORT_SIZES = [40, 80, 160, 320, 500]
REPS_PER_CELL = 2

TEST_BASE_PATIENTS = 1

SEQ_LENGTH = 60
PROJECTION_HORIZON = 5
N_SEQ_RANDOM_TRAJECTORIES = PROJECTION_HORIZON * 2

MIN_T_OBS = 5

BASE_SEED = 2000
D_STATE = 10
N_ACTIONS = 4

BIN_HOURS = 4.0
DT_HOURS = 1.0
STEPS_PER_BIN = int(BIN_HOURS / DT_HOURS)

DOSE_MG_DAY = np.array([0.0, 2.0, 5.0, 10.0], dtype=np.float32)
DOSE_MG_BIN = DOSE_MG_DAY * (BIN_HOURS / 24.0)

INR_LOWER = 2.0
INR_UPPER = 3.0
INR_TARGET = 2.5

# ============================================================
# Utilities
# ============================================================

def softmax(x):
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def nearest_dose_class(mg_day):
    return int(np.argmin(np.abs(DOSE_MG_DAY - mg_day)))


def ar1_path(rng, T, mean, rho, sigma, lo, hi):
    x = np.empty(T, dtype=np.float32)
    x[0] = float(np.clip(rng.normal(mean, sigma), lo, hi))

    for t in range(1, T):
        eps = rng.normal(0.0, sigma)
        x[t] = float(np.clip(mean + rho * (x[t - 1] - mean) + eps, lo, hi))

    return x


def get_scaling_params(sim):
    real_idx = ["states", "inr"]

    means = {}
    stds = {}
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

    if "inr" in sim:
        vals = []
        for i in range(seq_lengths.shape[0]):
            end = int(seq_lengths[i])
            if end > 0:
                vals.append(sim["inr"][i, :end])

        if len(vals):
            vals = np.concatenate(vals)
            means["inr"] = float(np.mean(vals))
            stds["inr"] = float(max(np.std(vals), 1e-6))
        else:
            means["inr"] = 0.0
            stds["inr"] = 1.0

    for key in ["cyp_proxy", "vkorc1_proxy", "age_norm", "maint_need_mg_day"]:
        if key in sim:
            means[key] = float(np.mean(sim[key])) if len(sim[key]) else 0.0
            stds[key] = float(max(np.std(sim[key]), 1e-6)) if len(sim[key]) else 1.0

    return pd.Series(means), pd.Series(stds)

# ============================================================
# Warfarin patient simulator
# ============================================================

class WarfarinLitePatient:
    """
    Semi-mechanistic warfarin model.

    PK:
      gut -> plasma -> effect-site

    PD:
      effect-site warfarin reduces effective vitamin K recycling/availability,
      which slows synthesis of delayed coagulation factors.

    INR:
      nonlinear readout from deficits in factors II / VII / X.
    """

    def __init__(self, rng: np.random.Generator, gamma: int):
        self.rng = rng
        self.gamma = int(gamma)

        self.cyp_class = int(rng.choice([0, 1, 2], p=[0.65, 0.25, 0.10]))
        self.cyp_proxy = {0: 0.0, 1: 0.5, 2: 1.0}[self.cyp_class]

        self.vkorc1 = int(rng.choice([0, 1, 2], p=[0.365, 0.485, 0.150]))
        self.vkorc1_proxy = {0: 0.0, 1: 0.5, 2: 1.0}[self.vkorc1]

        self.age = float(np.clip(rng.normal(62.0, 12.0), 35.0, 90.0))
        self.age_norm = float(np.clip((self.age - 60.0) / 20.0, -1.5, 1.5))

        cl_base = 0.23
        cl_mult_cyp = {0: 1.00, 1: 0.65, 2: 0.35}[self.cyp_class]
        cl_mult_age = math.exp(-0.18 * self.age_norm)
        self.cl = float(np.clip(
            cl_base * cl_mult_cyp * cl_mult_age * rng.lognormal(0.0, 0.20),
            0.03,
            0.40,
        ))

        self.vd = float(np.clip(10.0 * rng.lognormal(0.0, 0.20), 5.5, 18.0))
        self.ka = float(np.clip(1.40 * rng.lognormal(0.0, 0.20), 0.5, 3.0))
        self.ke = self.cl / self.vd
        self.ke0 = float(np.clip(0.035 * rng.lognormal(0.0, 0.25), 0.01, 0.08))

        ec50_base = 0.95
        ec50_mult_vkorc = {0: 1.00, 1: 0.78, 2: 0.58}[self.vkorc1]
        ec50_mult_age = math.exp(-0.10 * self.age_norm)
        self.ec50 = float(np.clip(
            ec50_base * ec50_mult_vkorc * ec50_mult_age * rng.lognormal(0.0, 0.22),
            0.12,
            2.5,
        ))

        self.emax = float(np.clip(rng.normal(0.92, 0.03), 0.82, 0.99))
        self.hill = float(np.clip(rng.normal(1.15, 0.15), 0.8, 1.6))

        self.hl_ii = float(np.clip(rng.normal(60.0, 7.0), 40.0, 80.0))
        self.hl_vii = float(np.clip(rng.normal(8.0, 1.5), 5.0, 12.0))
        self.hl_x = float(np.clip(rng.normal(36.0, 6.0), 24.0, 50.0))
        self.hl_pc = float(np.clip(rng.normal(9.0, 2.0), 5.0, 14.0))

        self.kout_ii = math.log(2.0) / self.hl_ii
        self.kout_vii = math.log(2.0) / self.hl_vii
        self.kout_x = math.log(2.0) / self.hl_x
        self.kout_pc = math.log(2.0) / self.hl_pc

        self.sens_ii = float(np.clip(rng.normal(0.85, 0.08), 0.65, 1.05))
        self.sens_vii = float(np.clip(rng.normal(1.15, 0.10), 0.90, 1.35))
        self.sens_x = float(np.clip(rng.normal(1.00, 0.08), 0.80, 1.20))
        self.sens_pc = float(np.clip(rng.normal(1.25, 0.12), 1.00, 1.55))

        self.vk_mean = float(np.clip(rng.normal(1.0, 0.12), 0.75, 1.35))
        self.k_vk_revert = float(np.clip(rng.normal(0.10, 0.02), 0.05, 0.16))
        self.k_diet_revert = float(np.clip(rng.normal(0.18, 0.03), 0.10, 0.25))

        self.base_inr = float(np.clip(rng.normal(1.02, 0.05), 0.90, 1.20))
        self.isi = float(np.clip(rng.normal(1.10, 0.10), 0.90, 1.35))

        self.clinic_bias = float(np.clip(rng.normal(0.0, 0.35), -0.8, 0.8))

        maint = 5.0
        maint *= math.exp(-0.85 * self.cyp_proxy)
        maint *= math.exp(-0.55 * self.vkorc1_proxy)
        maint *= math.exp(-0.18 * self.age_norm)
        maint *= math.exp(0.12 * (self.vk_mean - 1.0))
        maint *= rng.lognormal(0.0, 0.15)

        self.maint_need_mg_day = float(np.clip(maint, 0.4, 11.5))

    def sample_exogenous_paths(self, T_bins):
        diet_path = ar1_path(
            self.rng,
            T_bins,
            mean=self.vk_mean,
            rho=0.92,
            sigma=0.05,
            lo=0.55,
            hi=1.60,
        )

        adh_path = ar1_path(
            self.rng,
            T_bins,
            mean=0.96,
            rho=0.85,
            sigma=0.06,
            lo=0.0,
            hi=1.10,
        )

        for t in range(T_bins):
            u = self.rng.random()

            if u < 0.03:
                adh_path[t] = 0.0
            elif u < 0.08:
                adh_path[t] *= float(self.rng.uniform(0.3, 0.8))

        return {
            "diet_path": diet_path.astype(np.float32),
            "adherence": adh_path.astype(np.float32),
        }

    def initial_state(self):
        return {
            "gut": 0.0,
            "conc": 0.0,
            "eff": 0.0,
            "fii": 1.0,
            "fvii": 1.0,
            "fx": 1.0,
            "pc": 1.0,
            "vk": self.vk_mean,
            "diet": self.vk_mean,
            "inr": self.base_inr,
            "dose_hist": np.zeros(42, dtype=np.float32),
            "adh_hist": np.ones(18, dtype=np.float32),
            "last_inr": self.base_inr,
        }

    def visible_state(self, st):
        dose_load_7d = float(np.clip(np.sum(st["dose_hist"]) / 35.0, 0.0, 3.0))

        return np.array([
            st["conc"],
            st["eff"],
            st["fii"],
            st["fvii"],
            st["vk"],
            st["inr"],
            dose_load_7d,
            self.cyp_proxy,
            self.vkorc1_proxy,
            self.age_norm,
        ], dtype=np.float32)

    def compute_inr(self, fii, fvii, fx):
        deficit = (
            1.30 * max(0.0, 1.0 - fvii)
            + 0.95 * max(0.0, 1.0 - fx)
            + 0.80 * max(0.0, 1.0 - fii)
        )

        pt_ratio = 1.0 + 1.6 * deficit + 0.70 * deficit * deficit
        inr = self.base_inr * (pt_ratio ** self.isi)

        return float(np.clip(inr, 0.8, 8.0))

    def behavioral_action(self, st):
        inr = float(st["inr"])
        trend = float(inr - st["last_inr"])
        eff = float(st["eff"])
        dose_load = float(np.sum(st["dose_hist"]) / 35.0)
        adh_load = float(np.mean(st["adh_hist"]))

        maint_class = nearest_dose_class(self.maint_need_mg_day)

        base = np.array([-0.8, -0.2, 0.0, -0.2], dtype=np.float32)
        idx = np.arange(N_ACTIONS)
        base += -0.55 * (idx - maint_class) ** 2

        err = inr - INR_TARGET
        g = float(self.gamma)

        if inr > 4.0:
            base += np.array([4.5, 1.0, -2.5, -5.5], dtype=np.float32)
        elif inr > INR_UPPER:
            base += np.array([1.2, 0.8, -0.7, -3.0], dtype=np.float32)
            base += g * np.array([0.7 * err, 0.35 * err, -0.45 * err, -0.9 * err], dtype=np.float32)
        elif inr < INR_LOWER:
            low = INR_LOWER - inr
            base += np.array([-1.4, 0.2, 0.9, 0.4], dtype=np.float32)
            base += g * np.array([-0.3 * low, 0.15 * low, 0.55 * low, 0.90 * low], dtype=np.float32)
        else:
            base += np.array([-0.3, 0.1, 0.2, -0.1], dtype=np.float32)

        if trend > 0.20:
            base += np.array([0.5, 0.3, -0.2, -0.6], dtype=np.float32)
        elif trend < -0.20:
            base += np.array([-0.2, 0.1, 0.3, 0.4], dtype=np.float32)

        if dose_load > 1.5:
            base[3] -= 1.2
            base[2] -= 0.4

        if eff > 1.5:
            base[3] -= 0.5
            base[2] -= 0.2
            base[0] += 0.2

        if adh_load < 0.7:
            base[3] -= 0.25
            base[2] -= 0.10
            base[1] += 0.05

        base[0] += 0.15 * max(0.0, self.age_norm)
        base[3] -= 0.20 * max(0.0, self.age_norm)
        base += self.clinic_bias * np.array([0.1, 0.1, -0.05, -0.15], dtype=np.float32)

        probs = softmax(base)
        return int(self.rng.choice(N_ACTIONS, p=probs))

    def step_bin(self, st, action, t_bin, exog):
        delivered_mg = float(DOSE_MG_BIN[action]) * float(exog["adherence"][t_bin])

        st["dose_hist"] = np.roll(st["dose_hist"], -1)
        st["dose_hist"][-1] = delivered_mg

        st["adh_hist"] = np.roll(st["adh_hist"], -1)
        st["adh_hist"][-1] = float(exog["adherence"][t_bin])

        st["gut"] += delivered_mg

        for _ in range(STEPS_PER_BIN):
            diet_target = float(exog["diet_path"][t_bin])

            st["diet"] += self.k_diet_revert * (diet_target - st["diet"]) * DT_HOURS
            st["diet"] = float(np.clip(st["diet"], 0.45, 1.80))

            d_gut = -self.ka * st["gut"] * DT_HOURS
            st["gut"] += d_gut
            st["gut"] = max(st["gut"], 0.0)

            input_rate = self.ka * max(st["gut"], 0.0) / self.vd
            d_conc = (input_rate - self.ke * st["conc"]) * DT_HOURS
            st["conc"] += d_conc
            st["conc"] = float(np.clip(st["conc"], 0.0, 10.0))

            st["eff"] += self.ke0 * (st["conc"] - st["eff"]) * DT_HOURS
            st["eff"] = float(np.clip(st["eff"], 0.0, 10.0))

            ce_h = st["eff"] ** self.hill
            ec_h = self.ec50 ** self.hill
            inhib = self.emax * ce_h / (ec_h + ce_h + 1e-8)

            vk_target = 0.75 * st["diet"] + 0.25 * self.vk_mean
            st["vk"] += self.k_vk_revert * (vk_target - st["vk"]) * DT_HOURS
            st["vk"] = float(np.clip(st["vk"], 0.45, 1.80))

            syn_ii = np.clip(st["vk"] * (1.0 - self.sens_ii * inhib), 0.02, 1.40)
            syn_vii = np.clip(st["vk"] * (1.0 - self.sens_vii * inhib), 0.02, 1.40)
            syn_x = np.clip(st["vk"] * (1.0 - self.sens_x * inhib), 0.02, 1.40)
            syn_pc = np.clip(st["vk"] * (1.0 - self.sens_pc * inhib), 0.02, 1.40)

            st["fii"] += self.kout_ii * (syn_ii - st["fii"]) * DT_HOURS
            st["fvii"] += self.kout_vii * (syn_vii - st["fvii"]) * DT_HOURS
            st["fx"] += self.kout_x * (syn_x - st["fx"]) * DT_HOURS
            st["pc"] += self.kout_pc * (syn_pc - st["pc"]) * DT_HOURS

            st["fii"] = float(np.clip(st["fii"], 0.02, 1.50))
            st["fvii"] = float(np.clip(st["fvii"], 0.02, 1.50))
            st["fx"] = float(np.clip(st["fx"], 0.02, 1.50))
            st["pc"] = float(np.clip(st["pc"], 0.02, 1.50))

        st["inr"] = self.compute_inr(st["fii"], st["fvii"], st["fx"])

# ============================================================
# Simulation routines
# ============================================================

def simulate_factual_patient(patient, seq_length):
    exog = patient.sample_exogenous_paths(seq_length)
    st = patient.initial_state()

    states = np.zeros((seq_length, D_STATE), dtype=np.float32)
    inr = np.zeros(seq_length, dtype=np.float32)
    actions = np.zeros(seq_length, dtype=np.int64)
    snapshots = []

    for t in range(seq_length):
        states[t] = patient.visible_state(st)
        inr[t] = float(st["inr"])
        snapshots.append(copy.deepcopy(st))

        action = patient.behavioral_action(st)
        actions[t] = action

        st["last_inr"] = float(st["inr"])
        patient.step_bin(st, action, t, exog)

    return states, inr, actions, snapshots, exog


def simulate_factual_dataset(num_patients, gamma, seq_length):
    states = np.zeros((num_patients, seq_length, D_STATE), dtype=np.float32)
    inr = np.zeros((num_patients, seq_length), dtype=np.float32)
    actions = np.zeros((num_patients, seq_length), dtype=np.int64)
    sequence_lengths = np.full(num_patients, seq_length, dtype=np.int64)

    cyp_proxy = np.zeros(num_patients, dtype=np.float32)
    vkorc1_proxy = np.zeros(num_patients, dtype=np.float32)
    age_norm = np.zeros(num_patients, dtype=np.float32)
    maint_need_mg_day = np.zeros(num_patients, dtype=np.float32)

    for i in range(num_patients):
        pt = WarfarinLitePatient(np.random.default_rng(np.random.randint(0, 2**31 - 1)), gamma)
        s, y, a, _, _ = simulate_factual_patient(pt, seq_length)

        states[i] = s
        inr[i] = y
        actions[i] = a

        cyp_proxy[i] = pt.cyp_proxy
        vkorc1_proxy[i] = pt.vkorc1_proxy
        age_norm[i] = pt.age_norm
        maint_need_mg_day[i] = pt.maint_need_mg_day

    return {
        "states": states,
        "inr": inr,
        "actions": actions,
        "sequence_lengths": sequence_lengths,
        "cyp_proxy": cyp_proxy,
        "vkorc1_proxy": vkorc1_proxy,
        "age_norm": age_norm,
        "maint_need_mg_day": maint_need_mg_day,
    }


def simulate_one_step_counterfactual_rows(num_patients, gamma, seq_length, min_tobs):
    max_rows = num_patients * seq_length * N_ACTIONS

    states = np.zeros((max_rows, seq_length, D_STATE), dtype=np.float32)
    inr = np.zeros((max_rows, seq_length), dtype=np.float32)
    actions = np.zeros((max_rows, seq_length), dtype=np.int64)
    sequence_lengths = np.zeros(max_rows, dtype=np.int64)
    patient_ids = np.zeros(max_rows, dtype=np.int64)
    patient_current_t = np.zeros(max_rows, dtype=np.int64)

    cyp_proxy = np.zeros(max_rows, dtype=np.float32)
    vkorc1_proxy = np.zeros(max_rows, dtype=np.float32)
    age_norm = np.zeros(max_rows, dtype=np.float32)
    maint_need_mg_day = np.zeros(max_rows, dtype=np.float32)

    row = 0

    for pid in range(num_patients):
        pt = WarfarinLitePatient(np.random.default_rng(np.random.randint(0, 2**31 - 1)), gamma)
        factual_states, factual_inr, factual_actions, snapshots, exog = simulate_factual_patient(pt, seq_length)

        for t in range(min_tobs - 1, seq_length - 1):
            for action in range(N_ACTIONS):
                st_cf = copy.deepcopy(snapshots[t])
                st_cf["last_inr"] = float(st_cf["inr"])
                pt.step_bin(st_cf, action, t, exog)

                next_state = pt.visible_state(st_cf)
                next_inr = float(st_cf["inr"])

                states[row, :t + 1] = factual_states[:t + 1]
                inr[row, :t + 1] = factual_inr[:t + 1]
                actions[row, :t] = factual_actions[:t]

                actions[row, t] = action
                states[row, t + 1] = next_state
                inr[row, t + 1] = next_inr

                sequence_lengths[row] = t + 1
                patient_ids[row] = pid
                patient_current_t[row] = t

                cyp_proxy[row] = pt.cyp_proxy
                vkorc1_proxy[row] = pt.vkorc1_proxy
                age_norm[row] = pt.age_norm
                maint_need_mg_day[row] = pt.maint_need_mg_day

                row += 1

    return {
        "states": states[:row],
        "inr": inr[:row],
        "actions": actions[:row],
        "sequence_lengths": sequence_lengths[:row],
        "patient_ids_all_trajectories": patient_ids[:row],
        "patient_current_t": patient_current_t[:row],
        "cyp_proxy": cyp_proxy[:row],
        "vkorc1_proxy": vkorc1_proxy[:row],
        "age_norm": age_norm[:row],
        "maint_need_mg_day": maint_need_mg_day[:row],
    }


def simulate_sequence_counterfactual_rows(
    num_patients,
    gamma,
    seq_length,
    projection_horizon,
    min_tobs,
    n_random_trajectories,
):
    max_rows = num_patients * seq_length * n_random_trajectories

    total_T = seq_length + projection_horizon

    states = np.zeros((max_rows, total_T, D_STATE), dtype=np.float32)
    inr = np.zeros((max_rows, total_T), dtype=np.float32)
    actions = np.zeros((max_rows, total_T), dtype=np.int64)
    sequence_lengths = np.zeros(max_rows, dtype=np.int64)
    patient_ids = np.zeros(max_rows, dtype=np.int64)
    patient_current_t = np.zeros(max_rows, dtype=np.int64)

    cyp_proxy = np.zeros(max_rows, dtype=np.float32)
    vkorc1_proxy = np.zeros(max_rows, dtype=np.float32)
    age_norm = np.zeros(max_rows, dtype=np.float32)
    maint_need_mg_day = np.zeros(max_rows, dtype=np.float32)

    row = 0

    for pid in range(num_patients):
        pt = WarfarinLitePatient(np.random.default_rng(np.random.randint(0, 2**31 - 1)), gamma)
        exog = pt.sample_exogenous_paths(total_T)

        st = pt.initial_state()

        factual_states = np.zeros((seq_length, D_STATE), dtype=np.float32)
        factual_inr = np.zeros(seq_length, dtype=np.float32)
        factual_actions = np.zeros(seq_length, dtype=np.int64)
        snapshots = []

        for t in range(seq_length):
            factual_states[t] = pt.visible_state(st)
            factual_inr[t] = float(st["inr"])
            snapshots.append(copy.deepcopy(st))

            a = pt.behavioral_action(st)
            factual_actions[t] = a

            st["last_inr"] = float(st["inr"])
            pt.step_bin(st, a, t, exog)

        for t in range(max(0, min_tobs - 2), seq_length - projection_horizon - 1):
            treatment_options = np.random.randint(
                0,
                N_ACTIONS,
                size=(n_random_trajectories, projection_horizon),
            )

            for plan in treatment_options:
                st_cf = copy.deepcopy(snapshots[t + 1])

                row_states = np.zeros((total_T, D_STATE), dtype=np.float32)
                row_inr = np.zeros(total_T, dtype=np.float32)
                row_actions = np.zeros(total_T, dtype=np.int64)

                hist_len = t + 2
                row_states[:hist_len] = factual_states[:hist_len]
                row_inr[:hist_len] = factual_inr[:hist_len]
                row_actions[:t + 1] = factual_actions[:t + 1]

                for h in range(projection_horizon):
                    current_time = t + 1 + h
                    action = int(plan[h])

                    row_actions[current_time] = action

                    st_cf["last_inr"] = float(st_cf["inr"])
                    pt.step_bin(st_cf, action, current_time, exog)

                    row_states[current_time + 1] = pt.visible_state(st_cf)
                    row_inr[current_time + 1] = float(st_cf["inr"])

                end = t + 1 + projection_horizon + 1

                states[row, :end] = row_states[:end]
                inr[row, :end] = row_inr[:end]
                actions[row, :end - 1] = row_actions[:end - 1]

                patient_ids[row] = pid
                patient_current_t[row] = t

                sequence_lengths[row] = int(t) + projection_horizon + 1

                cyp_proxy[row] = pt.cyp_proxy
                vkorc1_proxy[row] = pt.vkorc1_proxy
                age_norm[row] = pt.age_norm
                maint_need_mg_day[row] = pt.maint_need_mg_day

                row += 1

    return {
        "states": states[:row],
        "inr": inr[:row],
        "actions": actions[:row],
        "sequence_lengths": sequence_lengths[:row],
        "patient_ids_all_trajectories": patient_ids[:row],
        "patient_current_t": patient_current_t[:row],
        "cyp_proxy": cyp_proxy[:row],
        "vkorc1_proxy": vkorc1_proxy[:row],
        "age_norm": age_norm[:row],
        "maint_need_mg_day": maint_need_mg_day[:row],
    }

# ============================================================
# Valid-data wrappers
# ============================================================

def make_support_data(support_size, gamma):
    raw = simulate_factual_dataset(
        num_patients=support_size,
        gamma=gamma,
        seq_length=SEQ_LENGTH,
    )

    keep = np.where(raw["sequence_lengths"] >= MIN_T_OBS)[0]
    raw = take_rows(raw, keep)

    if raw["states"].shape[0] < support_size:
        chunks = [raw]
        n_kept = raw["states"].shape[0]

        while n_kept < support_size:
            extra = simulate_factual_dataset(
                num_patients=max(support_size, 50),
                gamma=gamma,
                seq_length=SEQ_LENGTH,
            )
            keep = np.where(extra["sequence_lengths"] >= MIN_T_OBS)[0]
            extra = take_rows(extra, keep)
            chunks.append(extra)
            n_kept += extra["states"].shape[0]

        raw = concat_raw(chunks)

    return take_rows(raw, np.arange(support_size))


def make_valid_one_step_test_data(gamma, max_attempts=100):
    for attempt in range(max_attempts):
        raw = simulate_one_step_counterfactual_rows(
            num_patients=TEST_BASE_PATIENTS,
            gamma=gamma,
            seq_length=SEQ_LENGTH,
            min_tobs=MIN_T_OBS,
        )

        if raw["states"].shape[0] > 0:
            return raw, attempt + 1

    raise RuntimeError(f"Could not generate non-empty one-step warfarin test data for gamma={gamma}.")


def make_valid_seq_test_data(gamma, max_attempts=100):
    for attempt in range(max_attempts):
        raw = simulate_sequence_counterfactual_rows(
            num_patients=TEST_BASE_PATIENTS,
            gamma=gamma,
            seq_length=SEQ_LENGTH,
            projection_horizon=PROJECTION_HORIZON,
            min_tobs=MIN_T_OBS,
            n_random_trajectories=N_SEQ_RANDOM_TRAJECTORIES,
        )

        if raw["states"].shape[0] > 0:
            return raw, attempt + 1

    raise RuntimeError(f"Could not generate non-empty sequence warfarin test data for gamma={gamma}.")

# ============================================================
# Dataset assembly
# ============================================================

def make_dataset(dataset_id, gamma, support_size, rep, seed):
    np.random.seed(seed)

    LOGGER.info(
        "[dataset %03d] domain=warfarin, gamma=%s, support=%s, rep=%s, "
        "test_base_patients=%s, min_tobs=%s, seed=%s",
        dataset_id,
        gamma,
        support_size,
        rep,
        TEST_BASE_PATIENTS,
        MIN_T_OBS,
        seed,
    )

    support_data = make_support_data(
        support_size=support_size,
        gamma=gamma,
    )

    test_data_counterfactuals, one_step_attempts = make_valid_one_step_test_data(
        gamma=gamma,
    )

    test_data_seq, seq_attempts = make_valid_seq_test_data(
        gamma=gamma,
    )

    test_data_factuals = simulate_factual_dataset(
        num_patients=TEST_BASE_PATIENTS,
        gamma=gamma,
        seq_length=SEQ_LENGTH,
    )

    scaling_data = get_scaling_params(support_data)

    pickle_map = {
        "dataset_id": int(dataset_id),
        "seed": int(seed),
        "rep": int(rep),
        "domain": "warfarin",
        "gamma": int(gamma),

        "seq_length": int(SEQ_LENGTH),
        "num_time_steps": int(SEQ_LENGTH),
        "min_t_obs": int(MIN_T_OBS),

        "support_size": int(support_size),
        "training_size": int(support_size),
        "validation_size": 0,
        "test_size_base_patients": int(TEST_BASE_PATIENTS),

        "projection_horizon": int(PROJECTION_HORIZON),
        "cf_seq_mode": "random_trajectories",
        "n_seq_random_trajectories": int(N_SEQ_RANDOM_TRAJECTORIES),

        "state_name": "states",
        "outcome_name": "outcomes",
        "action_name": "actions",
        "static_name": "static_features",
        "target_space": "normalized_inr",

        "dose_mg_day": DOSE_MG_DAY.copy(),
        "dose_mg_bin": DOSE_MG_BIN.copy(),
        "bin_hours": float(BIN_HOURS),

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
        domain="warfarin",
        outcome_key="inr",
        state_key="states",
        action_key="actions",
        static_key=None,
        static_keys=("cyp_proxy", "vkorc1_proxy", "age_norm", "maint_need_mg_day"),
        target_state_index=5,
    )



def generate(config: WarfarinGeneratorConfig | None = None, **overrides: Any) -> pd.DataFrame:
    config = config or WarfarinGeneratorConfig()
    if overrides:
        config = WarfarinGeneratorConfig.from_dict(dataclasses.asdict(config), **overrides)

    global OUTPUT_DIR, GAMMAS, SUPPORT_SIZES, REPS_PER_CELL
    global TEST_BASE_PATIENTS, SEQ_LENGTH, PROJECTION_HORIZON, N_SEQ_RANDOM_TRAJECTORIES
    global MIN_T_OBS, BASE_SEED

    OUTPUT_DIR = str(config.output_dir)
    GAMMAS = list(config.gammas)
    SUPPORT_SIZES = list(config.support_sizes)
    REPS_PER_CELL = int(config.reps_per_cell)
    TEST_BASE_PATIENTS = int(config.test_base_patients)
    SEQ_LENGTH = int(config.seq_length)
    PROJECTION_HORIZON = int(config.projection_horizon)
    N_SEQ_RANDOM_TRAJECTORIES = int(config.n_seq_random_trajectories)
    MIN_T_OBS = int(config.min_t_obs)
    BASE_SEED = int(config.base_seed)

    output_dir = ensure_output_dir(OUTPUT_DIR)
    # ============================================================
    # Generate datasets
    # ============================================================

    summary_rows = []
    dataset_id = 0

    for gamma in GAMMAS:
        for support_size in SUPPORT_SIZES:
            for rep in range(REPS_PER_CELL):
                seed = BASE_SEED + dataset_id

                pickle_map = make_dataset(
                    dataset_id=dataset_id,
                    gamma=gamma,
                    support_size=support_size,
                    rep=rep,
                    seed=seed,
                )

                file_name = (
                    f"warfarin_pfn_dataset_{dataset_id:03d}"
                    f"_gamma_{gamma}"
                    f"_support_{support_size}"
                    f"_rep_{rep}"
                    f"_testbase_{TEST_BASE_PATIENTS}"
                    f"_mintobs_{MIN_T_OBS}"
                    f"_seed_{seed}.p"
                )

                file_path = output_dir / file_name
                save_pickle(pickle_map, file_path)

                one_step_rows = pickle_map["test_data"]["states"].shape[0]
                seq_rows = pickle_map["test_data_seq"]["states"].shape[0]

                if one_step_rows == 0 or seq_rows == 0:
                    raise RuntimeError(
                        f"Generated empty test task in dataset_id={dataset_id}: "
                        f"one_step_rows={one_step_rows}, seq_rows={seq_rows}"
                    )

                summary_rows.append({
                    "dataset_id": dataset_id,
                    "file_name": file_name,
                    "file_path": file_path,
                    "seed": seed,
                    "domain": "warfarin",

                    "gamma": gamma,
                    "support_size": support_size,
                    "training_size": support_size,
                    "validation_size": 0,
                    "rep": rep,

                    "test_size_base_patients": TEST_BASE_PATIENTS,
                    "test_data_rows_one_step_counterfactual": one_step_rows,
                    "test_data_seq_rows_multi_step_counterfactual": seq_rows,

                    "test_one_step_resample_attempts": pickle_map["test_one_step_resample_attempts"],
                    "test_seq_resample_attempts": pickle_map["test_seq_resample_attempts"],

                    "min_t_obs": MIN_T_OBS,
                    "seq_length": SEQ_LENGTH,
                    "projection_horizon": PROJECTION_HORIZON,
                    "cf_seq_mode": "random_trajectories",
                    "n_seq_random_trajectories": N_SEQ_RANDOM_TRAJECTORIES,

                    "outcome_name": "outcomes",
                    "state_dim": D_STATE,
                    "n_actions": N_ACTIONS,
                    "target_space": "normalized_inr",

                    "support_min_sequence_lengths": float(np.min(pickle_map["support_data"]["sequence_lengths"])),
                    "one_step_min_sequence_lengths": float(np.min(pickle_map["test_data"]["sequence_lengths"])),
                    "seq_min_downstream_t_obs": float(np.min(pickle_map["test_data_seq"]["patient_current_t"] + 2)),
                })

                del pickle_map
                gc.collect()

                dataset_id += 1

    summary = pd.DataFrame(summary_rows)

    LOGGER.info("Generated warfarin datasets: %s", len(summary))
    LOGGER.info("Dataset directory: %s", OUTPUT_DIR)
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
