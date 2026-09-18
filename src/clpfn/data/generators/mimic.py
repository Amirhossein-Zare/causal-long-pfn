"""Factual MIMIC-III rolling-origin benchmark generator."""

from __future__ import annotations

import dataclasses
import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import ensure_output_dir, save_pickle, standardize_pickle_map, write_dataset_manifest

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class MIMICGeneratorConfig:
    input_root: str = "data/raw"
    merged_dataset_slug: str = "mimic-iii-extract-session2-merged"
    output_dir: str = "outputs/benchmarks/mimic"
    overwrite: bool = False
    gammas: tuple[int, ...] = (1, 3, 5, 7, 9)
    support_sizes: tuple[int, ...] = (40, 80, 160, 320, 500)
    reps_per_cell: int = 1
    test_base_patients: int = 1
    seq_length: int = 60
    projection_horizon: int = 5
    total_seq_length: int | None = None
    n_seq_random_trajectories: int = 1
    min_t_obs: int = 10
    base_seed: int = 4000

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None = None) -> "MIMICGeneratorConfig":
        values = dict(values or {})
        valid = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(values) - valid)
        if unknown:
            raise KeyError(f"Unknown MIMIC generator configuration keys: {unknown}")
        return cls(**values)

    def __post_init__(self) -> None:
        self.gammas = tuple(int(x) for x in self.gammas)
        self.support_sizes = tuple(int(x) for x in self.support_sizes)
        if self.total_seq_length is None:
            self.total_seq_length = int(self.seq_length) + int(self.projection_horizon)
        if not self.gammas or any(gamma < 0 for gamma in self.gammas):
            raise ValueError("gammas must contain non-negative values.")
        if not self.support_sizes or any(size < 1 for size in self.support_sizes):
            raise ValueError("support_sizes must contain positive values.")
        if self.reps_per_cell < 1 or self.test_base_patients < 1:
            raise ValueError("reps_per_cell and test_base_patients must be positive.")
        if self.seq_length < 2 or self.projection_horizon < 1:
            raise ValueError("seq_length must exceed one and projection_horizon must be positive.")
        if not 1 <= self.min_t_obs < self.seq_length:
            raise ValueError("min_t_obs must be within the generated sequence.")
        if self.projection_horizon >= self.seq_length - self.min_t_obs:
            raise ValueError("The sequence must contain a complete projection after min_t_obs.")
        if self.total_seq_length != self.seq_length + self.projection_horizon:
            raise ValueError("total_seq_length must equal seq_length plus projection_horizon.")
        if self.n_seq_random_trajectories != 1:
            raise ValueError("MIMIC has exactly one observed trajectory per query.")


# Default generator grid

INPUT_ROOT = "data/raw"
MERGED_DATASET_SLUG = "mimic-iii-extract-session2-merged"

OUTPUT_DIR = "outputs/benchmarks/mimic"

GAMMAS = [1, 3, 5, 7, 9]
SUPPORT_SIZES = [40, 80, 160, 320, 500]
REPS_PER_CELL = 1

TEST_BASE_PATIENTS = 1

SEQ_LENGTH = 60
PROJECTION_HORIZON = 5
TOTAL_SEQ_LENGTH = SEQ_LENGTH + PROJECTION_HORIZON
N_SEQ_RANDOM_TRAJECTORIES = 1  # factual MIMIC has one observed trajectory, not random CF plans

MIN_T_OBS = 10

BASE_SEED = 4000
D_STATE = 10
N_ACTIONS = 4

TARGET_COL = "diastolic blood pressure"
TARGET_IDX = 0

BASE_STATE_COLS = [
    "diastolic blood pressure",
    "mean blood pressure",
    "oxygen saturation",
    "heart rate",
    "respiratory rate",
    "glascow coma scale total",
    "glucose",
    "creatinine",
    "bicarbonate",
    "sodium",
]

TREATMENT_LIST = ["vaso", "vent"]
STATIC_LIST = ["gender", "ethnicity", "age"]
D_STATIC_MAX = 5

# MIMIC loading helpers

def find_merged_h5() -> Path:
    hits = sorted(
        path
        for path in Path(INPUT_ROOT).rglob("all_hourly_data.h5")
        if MERGED_DATASET_SLUG in str(path)
    )
    if not hits:
        raise FileNotFoundError(
            f"Could not find all_hourly_data.h5 for mounted dataset slug "
            f"'{MERGED_DATASET_SLUG}' under {INPUT_ROOT}."
        )
    if len(hits) > 1:
        raise RuntimeError(
            f"Expected one MIMIC H5 file for slug '{MERGED_DATASET_SLUG}', found: {hits}"
        )
    return hits[0]


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    return out


def get_group_levels(index_names):
    if "hours_in" in index_names:
        return [n for n in index_names if n != "hours_in"]
    return list(index_names[:-1])


def get_hour_level(index_names):
    return "hours_in" if "hours_in" in index_names else index_names[-1]


def grouped_ffill(df: pd.DataFrame, group_levels, limit=None) -> pd.DataFrame:
    return df.groupby(level=group_levels, sort=False).ffill(limit=limit)


def process_static_features_ct(static_features: pd.DataFrame, drop_first: bool = False) -> pd.DataFrame:
    processed = []

    for feature in static_features.columns:
        s = static_features[feature]

        if pd.api.types.is_numeric_dtype(s):
            processed.append(pd.to_numeric(s, errors="coerce").rename(feature))
        else:
            oh = pd.get_dummies(
                s.astype("string"),
                prefix=feature,
                drop_first=drop_first,
            ).astype(float)
            processed.append(oh)

    return pd.concat(processed, axis=1)


def choose_static_columns(static_ct: pd.DataFrame, max_cols: int = D_STATIC_MAX):
    cols = list(static_ct.columns)
    chosen = []

    for pref in ["age", "gender", "ethnicity"]:
        for c in cols:
            cs = str(c)

            if c in chosen:
                continue

            if cs == pref or cs.startswith(pref + "_"):
                chosen.append(c)

                if len(chosen) >= max_cols:
                    return chosen

    for c in cols:
        if c not in chosen:
            chosen.append(c)

            if len(chosen) >= max_cols:
                break

    return chosen


def exclude_support_ids(eligible, support_patient_ids, label):
    eligible = np.setdiff1d(
        np.asarray(eligible, dtype=np.int64),
        np.asarray(support_patient_ids, dtype=np.int64),
        assume_unique=False,
    )
    if len(eligible) < TEST_BASE_PATIENTS:
        raise RuntimeError(
            f"Not enough disjoint MIMIC {label} stays after excluding support stays: "
            f"need {TEST_BASE_PATIENTS}, have {len(eligible)}."
        )
    return eligible


def get_scaling_params(sim):
    means = {}
    stds = {}
    seq_lengths = np.asarray(sim["sequence_lengths"], dtype=np.int64)

    if "states" in sim:
        vals = []

        for i in range(seq_lengths.shape[0]):
            end = min(int(seq_lengths[i]), sim["states"].shape[1])
            if end > 0:
                vals.append(sim["states"][i, :end, :])

        if len(vals):
            vals = np.concatenate(vals, axis=0)
            means["states"] = vals.mean(axis=0)
            stds["states"] = np.maximum(vals.std(axis=0), 1e-6)
        else:
            raise ValueError("MIMIC support data has no state observations.")

    if "mimic_outcome" in sim:
        vals = []

        for i in range(seq_lengths.shape[0]):
            end = min(int(seq_lengths[i]), sim["mimic_outcome"].shape[1])
            if end > 0:
                vals.append(sim["mimic_outcome"][i, :end])

        if len(vals):
            vals = np.concatenate(vals)
            means["mimic_outcome"] = float(np.mean(vals))
            stds["mimic_outcome"] = float(max(np.std(vals), 1e-6))
        else:
            raise ValueError("MIMIC support data has no outcome observations.")

    if "static_features" in sim:
        sf = np.asarray(sim["static_features"], dtype=np.float32)
        if sf.shape[0] > 0:
            means["static_features"] = sf.mean(axis=0)
            stds["static_features"] = np.maximum(sf.std(axis=0), 1e-6)
        else:
            raise ValueError("MIMIC support data has no static observations.")

    return pd.Series(means), pd.Series(stds)


# Build fixed MIMIC arrays once

def load_mimic_arrays():
    input_h5 = find_merged_h5()

    LOGGER.info("MIMIC input HDF5: %s", input_h5)

    with pd.HDFStore(str(input_h5), mode="r") as store:
        LOGGER.info("Available HDF5 keys: %s", list(store.keys()))

        interventions = flatten_columns(store["/interventions"])
        patients = flatten_columns(store["/patients"])
        vitals_labs_mean = flatten_columns(store["/vitals_labs_mean"])

    missing_treatments = [c for c in TREATMENT_LIST if c not in interventions.columns]
    missing_static = [c for c in STATIC_LIST if c not in patients.columns]
    missing_state = [c for c in BASE_STATE_COLS if c not in vitals_labs_mean.columns]

    if missing_treatments:
        raise KeyError(f"Missing treatment columns: {missing_treatments}")
    if missing_static:
        raise KeyError(f"Missing static columns: {missing_static}")
    if missing_state:
        raise KeyError(f"Missing state columns: {missing_state}")

    index_names = list(vitals_labs_mean.index.names)
    group_levels = get_group_levels(index_names)
    hour_level = get_hour_level(index_names)

    LOGGER.info("MIMIC index names: %s", index_names)
    LOGGER.info("MIMIC group levels: %s", group_levels)
    LOGGER.info("MIMIC hour level: %s", hour_level)

    time_series_raw = vitals_labs_mean[BASE_STATE_COLS].sort_index()
    observed_mask = time_series_raw.notna()
    time_series_dense = grouped_ffill(time_series_raw, group_levels=group_levels)

    treatments = interventions[TREATMENT_LIST].sort_index()
    action4_df = (
        treatments["vaso"].fillna(0).astype(np.int8)
        + 2 * treatments["vent"].fillna(0).astype(np.int8)
    ).rename("action4").to_frame()

    static_raw = patients[STATIC_LIST].copy()
    static_ct_full = process_static_features_ct(static_raw, drop_first=False)
    static_cols_used = choose_static_columns(static_ct_full, D_STATIC_MAX)
    static_ct = static_ct_full[static_cols_used].copy()

    sequence_lengths = (
        time_series_raw
        .groupby(level=group_levels, sort=False)
        .size()
        .rename("sequence_length")
        .to_frame()
    )

    stay_index = sequence_lengths.index
    n_stays = len(stay_index)

    states_total = np.full((n_stays, TOTAL_SEQ_LENGTH, D_STATE), np.nan, dtype=np.float32)
    outcome_total = np.full((n_stays, TOTAL_SEQ_LENGTH), np.nan, dtype=np.float32)
    actions_total = np.zeros((n_stays, TOTAL_SEQ_LENGTH), dtype=np.int64)
    sequence_len_arr = np.zeros(n_stays, dtype=np.int64)
    static_arr = np.full((n_stays, D_STATIC_MAX), np.nan, dtype=np.float32)
    state_observed_mask_total = np.zeros((n_stays, TOTAL_SEQ_LENGTH, D_STATE), dtype=bool)
    outcome_observed_mask_total = np.zeros((n_stays, TOTAL_SEQ_LENGTH), dtype=bool)

    state_pos = time_series_dense.groupby(level=group_levels, sort=False).indices
    observed_pos = observed_mask.groupby(level=group_levels, sort=False).indices
    action_pos = action4_df.groupby(level=group_levels, sort=False).indices

    static_aligned = static_ct.reindex(stay_index).replace([np.inf, -np.inf], np.nan)

    static_vals = static_aligned.to_numpy(dtype=np.float32)
    static_arr[:, :min(static_vals.shape[1], D_STATIC_MAX)] = static_vals[:, :D_STATIC_MAX]

    LOGGER.info("Building fixed MIMIC arrays.")
    for i, key in enumerate(stay_index):
        max_hour_seen = -1

        if key in state_pos:
            g = time_series_dense.iloc[state_pos[key]]
            hrs_all = g.index.get_level_values(hour_level).to_numpy()

            if len(hrs_all) > 0:
                max_hour_seen = max(max_hour_seen, int(np.nanmax(hrs_all)))

            vals = g.to_numpy(dtype=np.float32)
            keep = (hrs_all >= 0) & (hrs_all < TOTAL_SEQ_LENGTH)

            hrs = hrs_all[keep].astype(int)
            vals = vals[keep]

            states_total[i, hrs, :] = vals
            outcome_total[i, hrs] = vals[:, TARGET_IDX]
            direct = observed_mask.iloc[observed_pos[key]].to_numpy(dtype=bool)[keep]
            state_observed_mask_total[i, hrs, :] = direct
            outcome_observed_mask_total[i, hrs] = direct[:, TARGET_IDX]

        if key in action_pos:
            g = action4_df.iloc[action_pos[key]]
            hrs_all = g.index.get_level_values(hour_level).to_numpy()
            keep = (hrs_all >= 0) & (hrs_all < TOTAL_SEQ_LENGTH)

            hrs = hrs_all[keep].astype(int)
            actions_total[i, hrs] = g.iloc[keep, 0].to_numpy(dtype=np.int64)

        if max_hour_seen >= 0:
            sequence_len_arr[i] = min(max_hour_seen + 1, TOTAL_SEQ_LENGTH)
        else:
            sequence_len_arr[i] = min(int(sequence_lengths.iloc[i, 0]), TOTAL_SEQ_LENGTH)

        if (i + 1) % 5000 == 0:
            LOGGER.info("Processed %s/%s MIMIC stays.", i + 1, n_stays)

    actions_total = np.clip(actions_total, 0, N_ACTIONS - 1).astype(np.int64)

    support_eligible = np.where(sequence_len_arr >= SEQ_LENGTH)[0]

    seq_eligible = np.where(sequence_len_arr >= TOTAL_SEQ_LENGTH)[0]

    LOGGER.info("MIMIC states_total shape: %s", states_total.shape)
    LOGGER.info("MIMIC outcome_total shape: %s", outcome_total.shape)
    LOGGER.info("MIMIC actions_total shape: %s", actions_total.shape)
    LOGGER.info("MIMIC static_arr shape: %s", static_arr.shape)
    LOGGER.info("MIMIC support-eligible stays: %s", len(support_eligible))
    LOGGER.info("MIMIC sequence-test-eligible stays: %s", len(seq_eligible))
    LOGGER.info("MIMIC static columns used: %s", [str(c) for c in static_cols_used])

    if len(support_eligible) == 0:
        raise RuntimeError("No MIMIC stays have enough length for support trajectories.")
    if len(seq_eligible) == 0:
        raise RuntimeError("No MIMIC stays have enough length for 5-step sequence test.")

    return {
        "input_h5": str(input_h5),
        "states_total": states_total,
        "outcome_total": outcome_total,
        "state_observed_mask_total": state_observed_mask_total,
        "outcome_observed_mask_total": outcome_observed_mask_total,
        "actions_total": actions_total,
        "static_arr": static_arr,
        "sequence_lengths": sequence_len_arr,
        "support_eligible": support_eligible.astype(np.int64),
        "seq_eligible": seq_eligible.astype(np.int64),
        "static_cols_used": [str(c) for c in static_cols_used],
        "static_continuous_columns": [str(c) for c in static_cols_used if str(c) in STATIC_LIST and pd.api.types.is_numeric_dtype(static_ct[c])],
    }


MIMIC = None

# Raw construction functions

def fit_mimic_preprocessing(support_patient_ids, dataset_uid):
    ids = np.asarray(support_patient_ids, dtype=np.int64)
    states = MIMIC["states_total"][ids, :SEQ_LENGTH, :]
    mask = MIMIC["state_observed_mask_total"][ids, :SEQ_LENGTH, :]
    lengths = np.minimum(MIMIC["sequence_lengths"][ids], SEQ_LENGTH)
    valid = np.arange(states.shape[1])[None, :] < lengths[:, None]
    medians = np.zeros(D_STATE, dtype=np.float32)
    for j, feature in enumerate(BASE_STATE_COLS):
        values = states[:, :, j][mask[:, :, j] & valid & np.isfinite(states[:, :, j])]
        if values.size == 0:
            raise ValueError(f"MIMIC support-only imputation has no observed value: dataset_uid={dataset_uid}, feature={feature}, support_patient_ids={ids.tolist()}")
        medians[j] = np.float32(np.median(values))
    static = MIMIC["static_arr"][ids].copy()
    continuous = [i for i, name in enumerate(MIMIC["static_cols_used"]) if name in MIMIC["static_continuous_columns"]]
    means, stds = [], []
    for col in continuous:
        values = static[:, col][np.isfinite(static[:, col])]
        if values.size == 0:
            raise ValueError(f"MIMIC support-only static normalization has no finite value: dataset_uid={dataset_uid}, feature={MIMIC['static_cols_used'][col]}, support_patient_ids={ids.tolist()}")
        means.append(float(np.mean(values)))
        stds.append(float(max(np.std(values), 1e-6)))
    return {"state_imputation_medians": medians, "static_continuous_columns": continuous, "static_continuous_means": np.asarray(means, dtype=np.float32), "static_continuous_stds": np.asarray(stds, dtype=np.float32), "support_patient_ids": ids, "imputation_policy": "within_stay_forward_fill_then_support_observed_median"}


def apply_mimic_preprocessing(raw, preprocessing):
    out = dict(raw)
    states = np.asarray(out["states"], dtype=np.float32).copy()
    bad = ~np.isfinite(states)
    states[bad] = np.broadcast_to(preprocessing["state_imputation_medians"], states.shape)[bad]
    out["states"] = states
    out["mimic_outcome"] = states[:, :, TARGET_IDX].copy()
    static = np.asarray(out["static_features"], dtype=np.float32).copy()
    for idx, mean, std in zip(preprocessing["static_continuous_columns"], preprocessing["static_continuous_means"], preprocessing["static_continuous_stds"]):
        static[:, idx] = (np.where(np.isfinite(static[:, idx]), static[:, idx], mean) - mean) / std
    if not np.isfinite(static).all():
        raise ValueError("MIMIC categorical static features contain missing values.")
    out["static_features"] = static
    return out

def make_factual_raw_from_patient_ids(patient_ids, length=SEQ_LENGTH):
    patient_ids = np.asarray(patient_ids, dtype=np.int64)

    states = MIMIC["states_total"][patient_ids, :length, :].astype(np.float32)
    outcomes = MIMIC["outcome_total"][patient_ids, :length].astype(np.float32)
    actions = MIMIC["actions_total"][patient_ids, :length].astype(np.int64)
    static_features = MIMIC["static_arr"][patient_ids].astype(np.float32)
    state_observed_mask = MIMIC["state_observed_mask_total"][patient_ids, :length, :].astype(bool)
    outcome_observed_mask = MIMIC["outcome_observed_mask_total"][patient_ids, :length].astype(bool)

    sequence_lengths = np.minimum(
        MIMIC["sequence_lengths"][patient_ids],
        length,
    ).astype(np.int64)

    raw = {
        "states": states,
        "mimic_outcome": outcomes,
        "actions": actions,
        "sequence_lengths": sequence_lengths,
        "static_features": static_features,
        "patient_ids": patient_ids.astype(np.int64),
        "state_observed_mask": state_observed_mask,
        "outcome_observed_mask": outcome_observed_mask,
    }
    return raw


def make_support_data(rng, support_size):
    eligible = MIMIC["support_eligible"]

    if len(eligible) < support_size:
        raise RuntimeError(
            f"Not enough support-eligible MIMIC stays: need {support_size}, have {len(eligible)}."
        )

    chosen = rng.choice(eligible, size=support_size, replace=False)
    return make_factual_raw_from_patient_ids(chosen, length=SEQ_LENGTH)


def simulate_one_step_factual_rows(patient_ids, preprocessing, min_tobs=MIN_T_OBS):
    patient_ids = np.asarray(patient_ids, dtype=np.int64)
    max_rows = len(patient_ids) * SEQ_LENGTH

    states = np.zeros((max_rows, SEQ_LENGTH, D_STATE), dtype=np.float32)
    outcomes = np.zeros((max_rows, SEQ_LENGTH), dtype=np.float32)
    state_observed_mask = np.zeros((max_rows, SEQ_LENGTH, D_STATE), dtype=bool)
    outcome_observed_mask = np.zeros((max_rows, SEQ_LENGTH), dtype=bool)
    actions = np.zeros((max_rows, SEQ_LENGTH), dtype=np.int64)
    sequence_lengths = np.zeros(max_rows, dtype=np.int64)

    static_features = np.zeros((max_rows, D_STATIC_MAX), dtype=np.float32)
    patient_ids_all = np.zeros(max_rows, dtype=np.int64)
    patient_current_t = np.zeros(max_rows, dtype=np.int64)

    row = 0

    for pid in patient_ids:
        seq_len = int(min(MIMIC["sequence_lengths"][pid], SEQ_LENGTH))

        if seq_len < min_tobs + 1:
            continue

        src_states = MIMIC["states_total"][pid, :SEQ_LENGTH, :].copy()
        bad = ~np.isfinite(src_states)
        imputed = np.broadcast_to(preprocessing["state_imputation_medians"], src_states.shape)
        src_states[bad] = imputed[bad]
        src_outcomes = src_states[:, TARGET_IDX]
        src_actions = MIMIC["actions_total"][pid, :SEQ_LENGTH]
        src_static = MIMIC["static_arr"][pid]

        for t in range(min_tobs - 1, min(seq_len - 1, SEQ_LENGTH - 1)):
            target_t = t + 1

            if target_t >= SEQ_LENGTH:
                continue

            if not np.isfinite(src_outcomes[target_t]):
                continue

            states[row] = src_states
            outcomes[row] = src_outcomes
            state_observed_mask[row] = MIMIC["state_observed_mask_total"][pid, :SEQ_LENGTH]
            outcome_observed_mask[row] = MIMIC["outcome_observed_mask_total"][pid, :SEQ_LENGTH]
            actions[row] = src_actions

            sequence_lengths[row] = int(t) + 1
            patient_ids_all[row] = int(pid)
            patient_current_t[row] = int(t)
            static_features[row] = src_static

            row += 1

    raw = {
        "states": states[:row],
        "mimic_outcome": outcomes[:row],
        "actions": actions[:row],
        "sequence_lengths": sequence_lengths[:row],
        "static_features": static_features[:row],
        "patient_ids_all_trajectories": patient_ids_all[:row],
        "patient_current_t": patient_current_t[:row],
        "state_observed_mask": state_observed_mask[:row],
        "outcome_observed_mask": outcome_observed_mask[:row],
    }
    return apply_mimic_preprocessing(raw, preprocessing)


def simulate_sequence_factual_rows(
    patient_ids,
    preprocessing,
    projection_horizon=PROJECTION_HORIZON,
    min_tobs=MIN_T_OBS,
):
    patient_ids = np.asarray(patient_ids, dtype=np.int64)
    max_rows = len(patient_ids) * TOTAL_SEQ_LENGTH

    states = np.zeros((max_rows, TOTAL_SEQ_LENGTH, D_STATE), dtype=np.float32)
    outcomes = np.zeros((max_rows, TOTAL_SEQ_LENGTH), dtype=np.float32)
    state_observed_mask = np.zeros((max_rows, TOTAL_SEQ_LENGTH, D_STATE), dtype=bool)
    outcome_observed_mask = np.zeros((max_rows, TOTAL_SEQ_LENGTH), dtype=bool)
    actions = np.zeros((max_rows, TOTAL_SEQ_LENGTH), dtype=np.int64)
    sequence_lengths = np.zeros(max_rows, dtype=np.int64)

    static_features = np.zeros((max_rows, D_STATIC_MAX), dtype=np.float32)
    patient_ids_all = np.zeros(max_rows, dtype=np.int64)
    patient_current_t = np.zeros(max_rows, dtype=np.int64)

    row = 0

    for pid in patient_ids:
        seq_len = int(min(MIMIC["sequence_lengths"][pid], TOTAL_SEQ_LENGTH))

        if seq_len < min_tobs + projection_horizon:
            continue

        src_states = MIMIC["states_total"][pid, :TOTAL_SEQ_LENGTH, :].copy()
        bad = ~np.isfinite(src_states)
        imputed = np.broadcast_to(preprocessing["state_imputation_medians"], src_states.shape)
        src_states[bad] = imputed[bad]
        src_outcomes = src_states[:, TARGET_IDX]
        src_actions = MIMIC["actions_total"][pid, :TOTAL_SEQ_LENGTH]
        src_static = MIMIC["static_arr"][pid]

        start_t = max(0, min_tobs - 2)

        for t in range(start_t, seq_len - projection_horizon - 1):
            target_t = t + 1 + projection_horizon

            if target_t >= TOTAL_SEQ_LENGTH:
                continue

            if not np.isfinite(src_outcomes[target_t]):
                continue

            states[row] = src_states
            outcomes[row] = src_outcomes
            state_observed_mask[row] = MIMIC["state_observed_mask_total"][pid, :TOTAL_SEQ_LENGTH]
            outcome_observed_mask[row] = MIMIC["outcome_observed_mask_total"][pid, :TOTAL_SEQ_LENGTH]
            actions[row] = src_actions

            patient_ids_all[row] = int(pid)
            patient_current_t[row] = int(t)

            sequence_lengths[row] = int(t) + projection_horizon + 1
            static_features[row] = src_static

            row += 1

    raw = {
        "states": states[:row],
        "mimic_outcome": outcomes[:row],
        "actions": actions[:row],
        "sequence_lengths": sequence_lengths[:row],
        "static_features": static_features[:row],
        "patient_ids_all_trajectories": patient_ids_all[:row],
        "patient_current_t": patient_current_t[:row],
        "state_observed_mask": state_observed_mask[:row],
        "outcome_observed_mask": outcome_observed_mask[:row],
    }
    return apply_mimic_preprocessing(raw, preprocessing)


# Valid-data wrappers

def make_factual_test_data(rng, support_patient_ids, preprocessing):
    eligible = exclude_support_ids(MIMIC["support_eligible"], support_patient_ids, "factual test")
    ids = rng.choice(eligible, size=TEST_BASE_PATIENTS, replace=False)
    return apply_mimic_preprocessing(
        make_factual_raw_from_patient_ids(ids, length=SEQ_LENGTH),
        preprocessing,
    )


def make_one_step_test_data(rng, support_patient_ids, preprocessing):
    eligible = exclude_support_ids(MIMIC["support_eligible"], support_patient_ids, "one-step test")
    ids = rng.choice(eligible, size=TEST_BASE_PATIENTS, replace=False)
    raw = simulate_one_step_factual_rows(
        patient_ids=ids,
        min_tobs=MIN_T_OBS,
        preprocessing=preprocessing,
    )
    if raw["states"].shape[0] == 0:
        raise RuntimeError("MIMIC one-step query construction produced no rows.")
    return raw


def make_sequence_test_data(rng, support_patient_ids, preprocessing):
    eligible = exclude_support_ids(MIMIC["seq_eligible"], support_patient_ids, "sequence test")
    ids = rng.choice(eligible, size=TEST_BASE_PATIENTS, replace=False)
    raw = simulate_sequence_factual_rows(
        patient_ids=ids,
        projection_horizon=PROJECTION_HORIZON,
        min_tobs=MIN_T_OBS,
        preprocessing=preprocessing,
    )
    if raw["states"].shape[0] == 0:
        raise RuntimeError("MIMIC sequence query construction produced no rows.")
    return raw


# Dataset assembly

def make_dataset(dataset_id, gamma, support_size, rep, seed):
    rng = np.random.default_rng(seed)

    LOGGER.info(
        "[dataset %03d] domain=mimic, gamma=%s, support=%s, rep=%s, "
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
        rng=rng,
        support_size=support_size,
    )
    support_patient_ids = support_data["patient_ids"]
    preprocessing = fit_mimic_preprocessing(support_patient_ids, dataset_uid=f"mimic_dataset_{dataset_id}")
    support_data = apply_mimic_preprocessing(support_data, preprocessing)

    test_data_factuals = make_factual_test_data(
        rng=rng,
        support_patient_ids=support_patient_ids,
        preprocessing=preprocessing,
    )
    test_data = make_one_step_test_data(
        rng=rng,
        support_patient_ids=support_patient_ids,
        preprocessing=preprocessing,
    )
    test_data_seq = make_sequence_test_data(
        rng=rng,
        support_patient_ids=support_patient_ids,
        preprocessing=preprocessing,
    )

    scaling_data = get_scaling_params(support_data)

    pickle_map = {
        "dataset_id": int(dataset_id),
        "seed": int(seed),
        "rep": int(rep),
        "domain": "mimic",

        "gamma": int(gamma),
        "gamma_semantics": "split_index",

        "seq_length": int(SEQ_LENGTH),
        "num_time_steps": int(SEQ_LENGTH),
        "total_seq_length": int(TOTAL_SEQ_LENGTH),
        "min_t_obs": int(MIN_T_OBS),

        "support_size": int(support_size),
        "training_size": int(support_size),
        "validation_size": 0,
        "test_size_base_patients": int(TEST_BASE_PATIENTS),

        "projection_horizon": int(PROJECTION_HORIZON),
        "cf_seq_mode": "factual_rolling_origin",
        "n_seq_random_trajectories": int(N_SEQ_RANDOM_TRAJECTORIES),

        "state_name": "states",
        "outcome_name": "outcomes",
        "action_name": "actions",
        "static_name": "static_features",

        "target_feature_index": int(TARGET_IDX),
        "target_feature_name": TARGET_COL,
        "target_space": "raw_diastolic_blood_pressure",
        "is_counterfactual": False,

        "state_feature_names": list(BASE_STATE_COLS),
        "treatment_columns": list(TREATMENT_LIST),
        "action_mapping": {
            0: "none",
            1: "vaso",
            2: "vent",
            3: "vaso_plus_vent",
        },
        "static_features_source": list(STATIC_LIST),
        "static_processed_columns_used": list(MIMIC["static_cols_used"]),
        "mimic_preprocessing": {"imputation_policy": preprocessing["imputation_policy"], "uses_backward_fill": False, "imputation_statistics_scope": "support_only", "imputation_source": "directly_observed_values", "static_normalization_scope": "support_only", "state_imputation_medians": preprocessing["state_imputation_medians"].tolist(), "static_continuous_columns": [MIMIC["static_cols_used"][i] for i in preprocessing["static_continuous_columns"]], "static_continuous_means": preprocessing["static_continuous_means"].tolist(), "static_continuous_stds": preprocessing["static_continuous_stds"].tolist(), "support_patient_ids": preprocessing["support_patient_ids"].tolist()},

        "support_data": support_data,

        "test_data": test_data,
        "test_data_factuals": test_data_factuals,
        "test_data_seq": test_data_seq,

        "scaling_data": scaling_data,
    }

    return standardize_pickle_map(
        pickle_map,
        domain="mimic",
        outcome_key="mimic_outcome",
        state_key="states",
        action_key="actions",
        static_key="static_features",
        target_state_index=TARGET_IDX,
    )


def generate(config: MIMICGeneratorConfig | None = None) -> pd.DataFrame:
    config = MIMICGeneratorConfig() if config is None else config

    global INPUT_ROOT, MERGED_DATASET_SLUG, OUTPUT_DIR
    global GAMMAS, SUPPORT_SIZES, REPS_PER_CELL, TEST_BASE_PATIENTS, SEQ_LENGTH
    global PROJECTION_HORIZON, TOTAL_SEQ_LENGTH, N_SEQ_RANDOM_TRAJECTORIES
    global MIN_T_OBS, BASE_SEED, MIMIC

    INPUT_ROOT = str(config.input_root)
    MERGED_DATASET_SLUG = str(config.merged_dataset_slug)
    OUTPUT_DIR = str(config.output_dir)
    GAMMAS = list(config.gammas)
    SUPPORT_SIZES = list(config.support_sizes)
    REPS_PER_CELL = int(config.reps_per_cell)
    TEST_BASE_PATIENTS = int(config.test_base_patients)
    SEQ_LENGTH = int(config.seq_length)
    PROJECTION_HORIZON = int(config.projection_horizon)
    TOTAL_SEQ_LENGTH = int(config.total_seq_length)
    N_SEQ_RANDOM_TRAJECTORIES = int(config.n_seq_random_trajectories)
    MIN_T_OBS = int(config.min_t_obs)
    BASE_SEED = int(config.base_seed)

    output_dir = ensure_output_dir(OUTPUT_DIR, overwrite=bool(config.overwrite))
    MIMIC = load_mimic_arrays()
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
                    f"mimic_pfn_dataset_{dataset_id:03d}"
                    f"_gamma_{gamma}"
                    f"_support_{support_size}"
                    f"_rep_{rep}"
                    f"_testbase_{TEST_BASE_PATIENTS}"
                    f"_mintobs_{MIN_T_OBS}"
                    f"_seed_{seed}.p"
                )

                file_path = output_dir / file_name
                save_pickle(pickle_map, file_path)
                write_dataset_manifest(pickle_map, file_path, generator_config=dataclasses.asdict(config), generator_source=__file__)

                one_step_rows = pickle_map["test_data"]["states"].shape[0]
                seq_rows = pickle_map["test_data_seq"]["states"].shape[0]

                if one_step_rows == 0 or seq_rows == 0:
                    raise RuntimeError(
                        f"Generated empty MIMIC test task in dataset_id={dataset_id}: "
                        f"one_step_rows={one_step_rows}, seq_rows={seq_rows}"
                    )

                summary_rows.append({
                    "dataset_id": dataset_id,
                    "file_name": file_name,
                    "file_path": file_path,
                    "seed": seed,
                    "domain": "mimic",

                    "gamma": gamma,
                        "gamma_semantics": "split_index",

                    "support_size": support_size,
                    "training_size": support_size,
                    "validation_size": 0,
                    "rep": rep,

                    "test_size_base_patients": TEST_BASE_PATIENTS,
                    "test_data_rows_one_step_factual": one_step_rows,
                    "test_data_seq_rows_multi_step_factual": seq_rows,


                    "min_t_obs": MIN_T_OBS,
                    "seq_length": SEQ_LENGTH,
                    "total_seq_length": TOTAL_SEQ_LENGTH,
                    "projection_horizon": PROJECTION_HORIZON,
                    "cf_seq_mode": "factual_rolling_origin",
                    "n_seq_random_trajectories": N_SEQ_RANDOM_TRAJECTORIES,

                    "outcome_name": "outcomes",
                    "target_feature_index": TARGET_IDX,
                    "target_feature_name": TARGET_COL,
                    "state_dim": D_STATE,
                    "n_actions": N_ACTIONS,
                    "target_space": "raw_diastolic_blood_pressure",
                    "is_counterfactual": False,

                    "support_min_sequence_lengths": float(np.min(pickle_map["support_data"]["sequence_lengths"])),
                    "one_step_min_sequence_lengths": float(np.min(pickle_map["test_data"]["sequence_lengths"])),
                    "seq_min_downstream_t_obs": float(np.min(pickle_map["test_data_seq"]["patient_current_t"] + 2)),
                })

                del pickle_map
                gc.collect()

                dataset_id += 1

    summary = pd.DataFrame(summary_rows)

    LOGGER.info("Generated MIMIC datasets: %s", len(summary))
    LOGGER.info("Dataset directory: %s", OUTPUT_DIR)
    LOGGER.info("Any empty one-step task: %s", bool((summary["test_data_rows_one_step_factual"] == 0).any()))
    LOGGER.info("Any empty sequence task: %s", bool((summary["test_data_seq_rows_multi_step_factual"] == 0).any()))
    LOGGER.info("Min one-step rows: %s", int(summary["test_data_rows_one_step_factual"].min()))
    LOGGER.info("Min sequence rows: %s", int(summary["test_data_seq_rows_multi_step_factual"].min()))
    LOGGER.info("Min support sequence length: %.3f", float(summary["support_min_sequence_lengths"].min()))
    LOGGER.info("Min one-step sequence length: %.3f", float(summary["one_step_min_sequence_lengths"].min()))
    LOGGER.info("Min sequence downstream t_obs: %.3f", float(summary["seq_min_downstream_t_obs"].min()))

    return summary


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    generate()
