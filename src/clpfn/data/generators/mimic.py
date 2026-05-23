"""Factual MIMIC-III rolling-origin benchmark generator.

MIMIC-III does not expose outcomes under unobserved interventions, so this
generator exports factual prediction rows under the observed vasopressor and
ventilation actions. The output uses the same canonical benchmark raw-pickle schema as
the branchable simulated domains, allowing PFN and baseline evaluators to share
one input contract.

The HDF5 file is loaded inside ``generate`` rather than at import time so that
the package can be imported without credentialed MIMIC assets on disk.
"""

from __future__ import annotations

import dataclasses
import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import concat_raw, ensure_output_dir, save_pickle, standardize_pickle_map, take_rows

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class MIMICGeneratorConfig:
    input_root: str = "data/raw"
    merged_dataset_slug: str = "mimic-iii-extract-session2-merged"
    output_dir: str = "outputs/data/mimic"
    gammas: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    support_sizes: tuple[int, ...] = (40, 80, 160, 320, 500)
    reps_per_cell: int = 2
    test_base_patients: int = 1
    seq_length: int = 60
    projection_horizon: int = 5
    total_seq_length: int | None = None
    n_seq_random_trajectories: int = 1
    min_t_obs: int = 5
    base_seed: int = 4000

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None = None, **overrides: Any) -> "MIMICGeneratorConfig":
        values = dict(values or {})
        values.update({k: v for k, v in overrides.items() if v is not None})
        valid = {field.name for field in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in valid})

    def __post_init__(self) -> None:
        self.gammas = tuple(int(x) for x in self.gammas)
        self.support_sizes = tuple(int(x) for x in self.support_sizes)
        if self.total_seq_length is None:
            self.total_seq_length = int(self.seq_length) + int(self.projection_horizon)


# ============================================================
# Default generator grid
# ============================================================

INPUT_ROOT = "data/raw"
MERGED_DATASET_SLUG = "mimic-iii-extract-session2-merged"

OUTPUT_DIR = "outputs/data/mimic"

# Gamma is retained as a benchmark stratification field for compatibility with
# the simulated domains. It does not alter observed MIMIC trajectories.
GAMMAS = [1,2,3,4,5,6,7,8,9,10]
SUPPORT_SIZES = [40, 80, 160, 320, 500]
REPS_PER_CELL = 2

TEST_BASE_PATIENTS = 1

SEQ_LENGTH = 60
PROJECTION_HORIZON = 5
TOTAL_SEQ_LENGTH = SEQ_LENGTH + PROJECTION_HORIZON
N_SEQ_RANDOM_TRAJECTORIES = 1  # factual MIMIC has one observed trajectory, not random CF plans

MIN_T_OBS = 5

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

# ============================================================
# MIMIC loading helpers
# ============================================================

def find_merged_h5() -> Path:
    candidates = list(Path(INPUT_ROOT).rglob("all_hourly_data.h5"))
    hits = [p for p in candidates if MERGED_DATASET_SLUG in str(p)]

    if not hits:
        # Fallback: use the only all_hourly_data.h5 if there is exactly one.
        if len(candidates) == 1:
            return candidates[0]

        raise FileNotFoundError(
            f"Could not find all_hourly_data.h5 for mounted dataset slug "
            f"'{MERGED_DATASET_SLUG}'. Found candidates: {candidates[:5]}"
        )

    if len(hits) > 1:
        LOGGER.info("Multiple matching MIMIC H5 files found; using first:")
        for p in hits:
            LOGGER.info("  %s", p)

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


def grouped_bfill(df: pd.DataFrame, group_levels, limit=None) -> pd.DataFrame:
    return df.groupby(level=group_levels, sort=False).bfill(limit=limit)


def process_static_features_ct(static_features: pd.DataFrame, drop_first: bool = False) -> pd.DataFrame:
    processed = []

    for feature in static_features.columns:
        s = static_features[feature]

        if pd.api.types.is_numeric_dtype(s):
            mean = float(np.nanmean(s))
            std = float(np.nanstd(s))

            if std == 0.0 or not np.isfinite(std):
                std = 1.0

            processed.append(((s - mean) / std).rename(feature))
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


def filter_min_tobs(raw, min_tobs=MIN_T_OBS):
    keep = np.where(np.asarray(raw["sequence_lengths"], dtype=np.int64) >= int(min_tobs))[0]
    return take_rows(raw, keep), keep


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
            means["states"] = np.zeros(D_STATE, dtype=np.float32)
            stds["states"] = np.ones(D_STATE, dtype=np.float32)

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
            means["mimic_outcome"] = 0.0
            stds["mimic_outcome"] = 1.0

    if "static_features" in sim:
        sf = np.asarray(sim["static_features"], dtype=np.float32)
        if sf.shape[0] > 0:
            means["static_features"] = sf.mean(axis=0)
            stds["static_features"] = np.maximum(sf.std(axis=0), 1e-6)
        else:
            means["static_features"] = np.zeros(D_STATIC_MAX, dtype=np.float32)
            stds["static_features"] = np.ones(D_STATIC_MAX, dtype=np.float32)

    return pd.Series(means), pd.Series(stds)


# ============================================================
# Build fixed MIMIC arrays once
# ============================================================

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
    time_series_dense = grouped_ffill(time_series_raw, group_levels=group_levels)
    time_series_dense = grouped_bfill(time_series_dense, group_levels=group_levels)

    global_medians = time_series_dense.median(axis=0, skipna=True)
    global_medians = global_medians.replace([np.inf, -np.inf], np.nan).fillna(0.0)

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
    static_arr = np.zeros((n_stays, D_STATIC_MAX), dtype=np.float32)

    state_pos = time_series_dense.groupby(level=group_levels, sort=False).indices
    action_pos = action4_df.groupby(level=group_levels, sort=False).indices

    static_aligned = (
        static_ct
        .reindex(stay_index)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

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

    med = global_medians[BASE_STATE_COLS].to_numpy(dtype=np.float32)

    for j in range(D_STATE):
        bad = ~np.isfinite(states_total[:, :, j])
        states_total[:, :, j][bad] = med[j]

    outcome_total = states_total[:, :, TARGET_IDX].astype(np.float32)

    actions_total = np.clip(actions_total, 0, N_ACTIONS - 1).astype(np.int64)
    static_arr[~np.isfinite(static_arr)] = 0.0

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
        "actions_total": actions_total,
        "static_arr": static_arr,
        "sequence_lengths": sequence_len_arr,
        "support_eligible": support_eligible.astype(np.int64),
        "seq_eligible": seq_eligible.astype(np.int64),
        "static_cols_used": [str(c) for c in static_cols_used],
    }


MIMIC = None  

# ============================================================
# Raw construction functions
# ============================================================

def make_factual_raw_from_patient_ids(patient_ids, length=SEQ_LENGTH):
    patient_ids = np.asarray(patient_ids, dtype=np.int64)

    states = MIMIC["states_total"][patient_ids, :length, :].astype(np.float32)
    outcomes = MIMIC["outcome_total"][patient_ids, :length].astype(np.float32)
    actions = MIMIC["actions_total"][patient_ids, :length].astype(np.int64)
    static_features = MIMIC["static_arr"][patient_ids].astype(np.float32)

    sequence_lengths = np.minimum(
        MIMIC["sequence_lengths"][patient_ids],
        length,
    ).astype(np.int64)

    return {
        "states": states,
        "mimic_outcome": outcomes,
        "actions": actions,
        "sequence_lengths": sequence_lengths,
        "static_features": static_features,
        "patient_ids": patient_ids.astype(np.int64),
    }


def make_support_data(rng, support_size):
    eligible = MIMIC["support_eligible"]

    if len(eligible) < support_size:
        raise RuntimeError(
            f"Not enough support-eligible MIMIC stays: need {support_size}, have {len(eligible)}."
        )

    chosen = rng.choice(eligible, size=support_size, replace=False)
    raw = make_factual_raw_from_patient_ids(chosen, length=SEQ_LENGTH)
    raw, keep = filter_min_tobs(raw, min_tobs=MIN_T_OBS)

    if raw["states"].shape[0] < support_size:
        remaining = np.setdiff1d(eligible, chosen)
        chunks = [raw]
        n_kept = raw["states"].shape[0]

        while n_kept < support_size:
            extra_n = min(max(support_size, 50), len(remaining))
            if extra_n <= 0:
                raise RuntimeError("Could not refill MIMIC support after min_tobs filtering.")

            extra_ids = rng.choice(remaining, size=extra_n, replace=False)
            remaining = np.setdiff1d(remaining, extra_ids)

            extra = make_factual_raw_from_patient_ids(extra_ids, length=SEQ_LENGTH)
            extra, _ = filter_min_tobs(extra, min_tobs=MIN_T_OBS)

            chunks.append(extra)
            n_kept += extra["states"].shape[0]

        raw = concat_raw(chunks)

    return take_rows(raw, np.arange(support_size))


def simulate_one_step_factual_rows(patient_ids, min_tobs=MIN_T_OBS):
    patient_ids = np.asarray(patient_ids, dtype=np.int64)
    max_rows = len(patient_ids) * SEQ_LENGTH

    states = np.zeros((max_rows, SEQ_LENGTH, D_STATE), dtype=np.float32)
    outcomes = np.zeros((max_rows, SEQ_LENGTH), dtype=np.float32)
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

        src_states = MIMIC["states_total"][pid, :SEQ_LENGTH, :]
        src_outcomes = MIMIC["outcome_total"][pid, :SEQ_LENGTH]
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
            actions[row] = src_actions

            sequence_lengths[row] = int(t) + 1
            patient_ids_all[row] = int(pid)
            patient_current_t[row] = int(t)
            static_features[row] = src_static

            row += 1

    return {
        "states": states[:row],
        "mimic_outcome": outcomes[:row],
        "actions": actions[:row],
        "sequence_lengths": sequence_lengths[:row],
        "static_features": static_features[:row],
        "patient_ids_all_trajectories": patient_ids_all[:row],
        "patient_current_t": patient_current_t[:row],
    }


def simulate_sequence_factual_rows(
    patient_ids,
    projection_horizon=PROJECTION_HORIZON,
    min_tobs=MIN_T_OBS,
):
    patient_ids = np.asarray(patient_ids, dtype=np.int64)
    max_rows = len(patient_ids) * TOTAL_SEQ_LENGTH

    states = np.zeros((max_rows, TOTAL_SEQ_LENGTH, D_STATE), dtype=np.float32)
    outcomes = np.zeros((max_rows, TOTAL_SEQ_LENGTH), dtype=np.float32)
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

        src_states = MIMIC["states_total"][pid, :TOTAL_SEQ_LENGTH, :]
        src_outcomes = MIMIC["outcome_total"][pid, :TOTAL_SEQ_LENGTH]
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
            actions[row] = src_actions

            patient_ids_all[row] = int(pid)
            patient_current_t[row] = int(t)

            sequence_lengths[row] = int(t) + projection_horizon + 1
            static_features[row] = src_static

            row += 1

    return {
        "states": states[:row],
        "mimic_outcome": outcomes[:row],
        "actions": actions[:row],
        "sequence_lengths": sequence_lengths[:row],
        "static_features": static_features[:row],
        "patient_ids_all_trajectories": patient_ids_all[:row],
        "patient_current_t": patient_current_t[:row],
    }


# ============================================================
# Valid-data wrappers
# ============================================================

def make_valid_factual_test_data(rng, max_attempts=500):
    eligible = MIMIC["support_eligible"]

    for attempt in range(max_attempts):
        ids = rng.choice(eligible, size=TEST_BASE_PATIENTS, replace=False)
        raw = make_factual_raw_from_patient_ids(ids, length=SEQ_LENGTH)
        raw, keep = filter_min_tobs(raw, min_tobs=MIN_T_OBS)

        if raw["states"].shape[0] > 0:
            return raw, attempt + 1

    raise RuntimeError("Could not generate non-empty MIMIC factual test data.")


def make_valid_one_step_test_data(rng, max_attempts=500):
    eligible = MIMIC["support_eligible"]

    for attempt in range(max_attempts):
        ids = rng.choice(eligible, size=TEST_BASE_PATIENTS, replace=False)
        raw = simulate_one_step_factual_rows(
            patient_ids=ids,
            min_tobs=MIN_T_OBS,
        )

        if raw["states"].shape[0] > 0:
            return raw, attempt + 1

    raise RuntimeError("Could not generate non-empty MIMIC one-step test_data.")


def make_valid_seq_test_data(rng, max_attempts=500):
    eligible = MIMIC["seq_eligible"]

    for attempt in range(max_attempts):
        ids = rng.choice(eligible, size=TEST_BASE_PATIENTS, replace=False)
        raw = simulate_sequence_factual_rows(
            patient_ids=ids,
            projection_horizon=PROJECTION_HORIZON,
            min_tobs=MIN_T_OBS,
        )

        if raw["states"].shape[0] > 0:
            return raw, attempt + 1

    raise RuntimeError("Could not generate non-empty MIMIC sequence test_data_seq.")


# ============================================================
# Dataset assembly
# ============================================================

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

    test_data_factuals, factual_attempts = make_valid_factual_test_data(rng=rng)
    test_data, one_step_attempts = make_valid_one_step_test_data(rng=rng)
    test_data_seq, seq_attempts = make_valid_seq_test_data(rng=rng)

    scaling_data = get_scaling_params(support_data)

    pickle_map = {
        "dataset_id": int(dataset_id),
        "seed": int(seed),
        "rep": int(rep),
        "domain": "mimic",

        # Kept as a grouping stratum.
        # MIMIC is real factual data; gamma does not change the data-generating process.
        "gamma": int(gamma),
        "gamma_semantics": "pseudo_stratum_only_factual_mimic",

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

        "test_factual_resample_attempts": int(factual_attempts),
        "test_one_step_resample_attempts": int(one_step_attempts),
        "test_seq_resample_attempts": int(seq_attempts),

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


def generate(config: MIMICGeneratorConfig | None = None, **overrides: Any) -> pd.DataFrame:
    config = config or MIMICGeneratorConfig()
    if overrides:
        config = MIMICGeneratorConfig.from_dict(dataclasses.asdict(config), **overrides)

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

    output_dir = ensure_output_dir(OUTPUT_DIR)
    MIMIC = load_mimic_arrays()
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
                    "gamma_semantics": "pseudo_stratum_only_factual_mimic",

                    "support_size": support_size,
                    "training_size": support_size,
                    "validation_size": 0,
                    "rep": rep,

                    "test_size_base_patients": TEST_BASE_PATIENTS,
                    "test_data_rows_one_step_factual": one_step_rows,
                    "test_data_seq_rows_multi_step_factual": seq_rows,

                    "test_factual_resample_attempts": pickle_map["test_factual_resample_attempts"],
                    "test_one_step_resample_attempts": pickle_map["test_one_step_resample_attempts"],
                    "test_seq_resample_attempts": pickle_map["test_seq_resample_attempts"],

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
