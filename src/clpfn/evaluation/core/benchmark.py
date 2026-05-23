from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

SEED = 42

T_OBS_MAX = 60
TAU_MAX = 5
MAX_SEQ_LEN = T_OBS_MAX + TAU_MAX
MAX_INPUT_INDEX = MAX_SEQ_LEN - 1
MAX_TARGET_INDEX = MAX_SEQ_LEN
MIN_T_OBS = 25
PROJECTION_HORIZON = 5

N_ACTIONS = 4
D_STATIC_MAX = 5

TARGET_NORM_CLIP = 10.0
PRED_CLIP_REPORT = 20.0
STATE_CLIP_TRAIN = 5.0
OUTCOME_CLIP_TRAIN = 10.0
WANTED_DOMAINS = ("cancer", "hiv", "warfarin", "mimic")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RAW_PICKLE_GLOBS = (
    "cancer_dataset_*.p",
    "warfarin_pfn_dataset_*.p",
    "hiv_pfn_dataset_*.p",
    "mimic_pfn_dataset_*.p",
)


@dataclass(frozen=True)
class RawBenchmarkInputs:
    pickle_dirs: tuple[str, ...] = ()
    pickle_paths: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None = None) -> "RawBenchmarkInputs":
        values = dict(values or {})
        return cls(
            pickle_dirs=tuple(str(path) for path in values.get("pickle_dirs", []) or []),
            pickle_paths=tuple(str(path) for path in values.get("pickle_paths", []) or []),
        )

    def validate(self) -> None:
        if not (self.pickle_dirs or self.pickle_paths):
            raise ValueError("Provide raw benchmark pickle_dirs or pickle_paths.")


DOMAIN_CONFIGS = {
    "cancer": {
        "target_state_index": None,
    },
    "warfarin": {
        "target_state_index": 5,
    },
    "hiv": {
        "target_state_index": 4,
    },
    "mimic": {
        "target_state_index": 0,
    },
}

def seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def configure_torch_runtime(seed=SEED):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    seed_everything(seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def stable_domain_seed(domain):
    return sum((i + 1) * ord(c) for i, c in enumerate(str(domain)))


def stable_file_seed(path):
    base = os.path.basename(str(path))
    return sum((i + 1) * ord(c) for i, c in enumerate(base))


def finite_or_zero(x):
    x = np.asarray(x, dtype=np.float32)
    x[~np.isfinite(x)] = 0.0
    return x


def fixed_2d_float(arr, rows, cols):
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 0:
        arr = np.zeros((rows, cols), dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1) if rows == 1 else arr.reshape(rows, -1)

    out = np.zeros((rows, cols), dtype=np.float32)
    n_rows = min(rows, arr.shape[0])
    n_cols = min(cols, arr.shape[1])
    out[:n_rows, :n_cols] = arr[:n_rows, :n_cols]
    out[~np.isfinite(out)] = 0.0

    return out


def move_tensor_batch_to_device(batch, device=None, *, float_tensors=False, long_keys=()):
    device = DEVICE if device is None else device
    long_keys = set(long_keys)
    out = {}

    for key, value in batch.items():
        if not torch.is_tensor(value):
            out[key] = value
            continue

        value = value.to(device, non_blocking=True)
        if key in long_keys:
            value = value.long()
        elif float_tensors:
            value = value.float()
        out[key] = value

    return out


def normalized_rmse_from_sqerr(sqerr):
    sqerr = np.asarray(sqerr, dtype=np.float64)
    return float(np.sqrt(np.mean(sqerr))) if sqerr.size else float("nan")


def action_onehot_2d(a_2d, n_actions=N_ACTIONS):
    a_2d = np.asarray(a_2d, dtype=np.int64)
    out = np.zeros((a_2d.shape[0], a_2d.shape[1], n_actions), dtype=np.float32)
    idx = np.clip(a_2d, 0, n_actions - 1)

    for k in range(n_actions):
        out[:, :, k] = (idx == k).astype(np.float32)

    return out


def onehot_action(a, n_actions=N_ACTIONS):
    out = np.zeros(n_actions, dtype=np.float32)
    out[int(np.clip(a, 0, n_actions - 1))] = 1.0
    return out


def find_raw_pickles(inputs: RawBenchmarkInputs) -> list[str]:
    inputs.validate()

    roots = [Path(path) for path in inputs.pickle_dirs]
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Raw benchmark pickle directory not found: {root}")

    pfiles = [str(Path(path)) for path in inputs.pickle_paths]
    for path in pfiles:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Raw benchmark pickle not found: {path}")

    for root in roots:
        for pattern in RAW_PICKLE_GLOBS:
            pfiles.extend(str(path) for path in root.rglob(pattern))

    return sorted(set(pfiles))


def dataset_domain(pfile, pm):
    if "domain" not in pm:
        raise KeyError(f"Raw benchmark pickle is missing required key 'domain': {pfile}")

    domain = str(pm["domain"]).lower()
    if domain not in DOMAIN_CONFIGS:
        raise ValueError(f"Unsupported raw benchmark domain '{domain}' in {pfile}")

    return domain


def domain_config(domain):
    return dict(DOMAIN_CONFIGS[domain])


def get_support_raw(pm):
    if "support_data" not in pm:
        raise KeyError("Raw benchmark pickle is missing required key 'support_data'.")
    return pm["support_data"]


def get_state_array(raw, domain, cfg):
    key = "states"
    if key not in raw:
        raise KeyError(f"Raw {domain} data is missing canonical state key '{key}'.")

    arr = np.asarray(raw[key], dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Raw {domain} canonical states must have shape [N, T, D], got {arr.shape}.")
    return arr


def get_outcome_array(raw, domain, cfg):
    key = "outcomes"
    if key not in raw:
        raise KeyError(f"Raw {domain} data is missing canonical outcome key '{key}'.")

    arr = np.asarray(raw[key], dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"Raw {domain} canonical outcomes must have shape [N, T], got {arr.shape}.")

    return arr


def get_actions(raw, domain):
    key = "actions"
    if key not in raw:
        raise KeyError(f"Raw {domain} data is missing canonical action key '{key}'.")

    A = np.asarray(raw[key])
    if A.ndim == 3 and A.shape[-1] == 1:
        A = A[:, :, 0]
    if A.ndim != 2:
        raise ValueError(f"Raw {domain} canonical actions must have shape [N, T], got {A.shape}.")

    return A.astype(np.int64)


def get_static_array(raw, n_rows):
    key = "static_features"
    if key not in raw:
        raise KeyError("Raw data is missing canonical static key 'static_features'.")

    S = np.asarray(raw[key], dtype=np.float32)
    if S.ndim == 1:
        S = S.reshape(-1, 1)

    if S.shape[0] != n_rows:
        raise ValueError(f"Raw static_features has {S.shape[0]} rows, expected {n_rows}.")

    return fixed_2d_float(S, rows=n_rows, cols=D_STATIC_MAX)


def compute_support_stats(raw_support, domain, cfg, chosen):
    L = np.asarray(raw_support["sequence_lengths"], dtype=np.int64)
    S = get_state_array(raw_support, domain, cfg)
    Y = get_outcome_array(raw_support, domain, cfg)

    target_idx = cfg["target_state_index"]
    if target_idx is not None and 0 <= target_idx < S.shape[-1]:
        cov_idx = [i for i in range(S.shape[-1]) if i != target_idx]
        C = S[:, :, cov_idx]
    else:
        C = S

    cov_vals = []
    out_vals = []

    for i in chosen:
        end_x = min(int(L[i]), S.shape[1], MAX_SEQ_LEN)
        end_y = min(int(L[i]) + 1, Y.shape[1], MAX_TARGET_INDEX + 1)

        if end_x > 0:
            cov_vals.append(C[i, :end_x, :])
        if end_y > 0:
            out_vals.append(Y[i, :end_y])

    if cov_vals:
        cov_vals = np.concatenate(cov_vals, axis=0)
        good = np.isfinite(cov_vals).all(axis=1)
        cov_vals = cov_vals[good]

        if cov_vals.shape[0] > 0:
            state_mean = cov_vals.mean(axis=0).astype(np.float32)
            state_std = np.maximum(cov_vals.std(axis=0), 0.1).astype(np.float32)
        else:
            state_mean = np.zeros(C.shape[-1], dtype=np.float32)
            state_std = np.ones(C.shape[-1], dtype=np.float32)
    else:
        state_mean = np.zeros(C.shape[-1], dtype=np.float32)
        state_std = np.ones(C.shape[-1], dtype=np.float32)

    if out_vals:
        out_vals = np.concatenate(out_vals)
        out_vals = out_vals[np.isfinite(out_vals)]

        if len(out_vals) > 0:
            out_mean = float(np.mean(out_vals))
            out_std = float(max(np.std(out_vals), 1e-6))
        else:
            out_mean, out_std = 0.0, 1.0
    else:
        out_mean, out_std = 0.0, 1.0

    return state_mean, state_std, out_mean, out_std


def build_benchmark_arrays_for_raw(raw, domain, cfg, state_mean, state_std, out_mean, out_std):
    out_std = max(float(out_std), 1e-6)

    L = np.asarray(raw["sequence_lengths"], dtype=np.int64)
    A = get_actions(raw, domain)
    Y = get_outcome_array(raw, domain, cfg).astype(np.float32)
    n = Y.shape[0]
    static = get_static_array(raw, n)
    S = get_state_array(raw, domain, cfg)
    target_idx = cfg["target_state_index"]

    if target_idx is not None and 0 <= target_idx < S.shape[-1]:
        cov_idx = [i for i in range(S.shape[-1]) if i != target_idx]
        C_raw = S[:, :, cov_idx]
    else:
        C_raw = S

    dc = C_raw.shape[-1]
    sm = np.asarray(state_mean, dtype=np.float32).reshape(1, 1, dc)
    ss = np.asarray(state_std, dtype=np.float32).reshape(1, 1, dc)

    C = ((C_raw - sm) / np.maximum(ss, 0.1)).astype(np.float32)
    C = finite_or_zero(np.clip(C, -STATE_CLIP_TRAIN, STATE_CLIP_TRAIN))

    y_norm_unclipped = ((Y - float(out_mean)) / out_std).astype(np.float32)
    y_norm_clip = np.clip(
        finite_or_zero(y_norm_unclipped),
        -OUTCOME_CLIP_TRAIN,
        OUTCOME_CLIP_TRAIN,
    ).astype(np.float32)

    A = np.clip(A, 0, N_ACTIONS - 1).astype(np.int64)

    return {
        "covariates": C.astype(np.float32),
        "y_raw": Y.astype(np.float32),
        "y_norm_unclipped": y_norm_unclipped.astype(np.float32),
        "y_norm_clip": y_norm_clip.astype(np.float32),
        "actions": A.astype(np.int64),
        "static": static.astype(np.float32),
        "sequence_lengths": L.astype(np.int64),
    }


def get_one_step_eval_rows(raw, domain, cfg, max_rows, rng):
    Y = get_outcome_array(raw, domain, cfg)
    L = np.asarray(raw["sequence_lengths"], dtype=np.int64)

    current_t = L - 1
    target_t = L

    valid = np.where(
        (current_t >= MIN_T_OBS)
        & (current_t <= MAX_INPUT_INDEX)
        & (target_t > current_t)
        & (target_t < Y.shape[1])
        & (target_t <= MAX_TARGET_INDEX)
    )[0]

    if len(valid) > 0:
        valid = valid[np.isfinite(Y[valid, target_t[valid]])]

    if max_rows is not None and len(valid) > max_rows:
        valid = rng.choice(valid, size=max_rows, replace=False)

    valid = np.sort(valid)

    return valid.astype(np.int64), current_t[valid].astype(np.int64), target_t[valid].astype(np.int64)


def get_seq_horizon_eval_rows(raw, domain, cfg, horizon, max_rows, rng):
    Y = get_outcome_array(raw, domain, cfg)
    if "patient_current_t" not in raw:
        raise KeyError("Sequence test raw data is missing required key 'patient_current_t'.")
    current_t = np.asarray(raw["patient_current_t"], dtype=np.int64) + 1
    target_t = current_t + int(horizon)

    valid = np.where(
        (current_t >= MIN_T_OBS)
        & (current_t <= MAX_INPUT_INDEX)
        & (target_t > current_t)
        & (target_t < Y.shape[1])
        & (target_t <= MAX_TARGET_INDEX)
    )[0]

    if len(valid) > 0:
        valid = valid[np.isfinite(Y[valid, target_t[valid]])]

    if max_rows is not None and len(valid) > max_rows:
        valid = rng.choice(valid, size=max_rows, replace=False)

    valid = np.sort(valid)

    return valid.astype(np.int64), current_t[valid].astype(np.int64), target_t[valid].astype(np.int64)


def prepare_dataset_bundle(pm, pfile, global_dataset_id):
    domain = dataset_domain(pfile, pm)
    cfg = domain_config(domain)
    support_raw = get_support_raw(pm)

    n_support_total = int(np.asarray(support_raw["sequence_lengths"]).shape[0])
    chosen = np.arange(n_support_total, dtype=np.int64)

    state_mean, state_std, out_mean, out_std = compute_support_stats(
        support_raw,
        domain,
        cfg,
        chosen,
    )

    support_bundle = build_benchmark_arrays_for_raw(
        support_raw,
        domain,
        cfg,
        state_mean=state_mean,
        state_std=state_std,
        out_mean=out_mean,
        out_std=out_std,
    )

    dataset_id = int(pm["dataset_id"])
    actual_n_ctx = int(support_bundle["covariates"].shape[0])
    support_size = int(pm["support_size"])
    meta = {
        "domain": domain,
        "domain_key": domain,
        "cfg": cfg,
        "dataset_id": dataset_id,
        "global_dataset_id": int(global_dataset_id),
        "source_file": os.path.basename(pfile),
        "source_path": str(pfile),
        "gamma": pm["gamma"],
        "support_size": int(support_size),
        "n_ctx": int(actual_n_ctx),
        "replicate": int(pm["rep"]),
        "out_mean": float(out_mean),
        "out_std": float(max(out_std, 1e-6)),
        "state_mean": state_mean,
        "state_std": state_std,
        "outcome_name": "outcomes",
        "target_space": str(pm["target_space"]),
        "max_seq_len": int(MAX_SEQ_LEN),
    }

    return support_bundle, meta


def make_query_bundle(raw_query, meta):
    return build_benchmark_arrays_for_raw(
        raw_query,
        meta["domain"],
        meta["cfg"],
        state_mean=meta["state_mean"],
        state_std=meta["state_std"],
        out_mean=meta["out_mean"],
        out_std=meta["out_std"],
    )
