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
MIN_HISTORY_POINTS = 10
MIN_T_OBS = MIN_HISTORY_POINTS - 1
PROJECTION_HORIZON = 5

N_ACTIONS = 4
D_STATIC_MAX = 5

TARGET_NORM_CLIP = 10.0
PRED_CLIP_REPORT = 20.0
STATE_CLIP_TRAIN = 5.0
STATIC_CLIP_TRAIN = 3.0
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
        unknown = sorted(set(values) - {"pickle_dirs", "pickle_paths"})
        if unknown:
            raise KeyError(f"Unknown raw benchmark input keys: {unknown}")
        return cls(
            pickle_dirs=tuple(str(path) for path in values["pickle_dirs"]) if "pickle_dirs" in values else (),
            pickle_paths=tuple(str(path) for path in values["pickle_paths"]) if "pickle_paths" in values else (),
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


def onehot_action(a, n_actions=N_ACTIONS):
    action = int(a)
    if not 0 <= action < n_actions:
        raise ValueError(f"Action {action} is outside [0, {n_actions - 1}].")
    out = np.zeros(n_actions, dtype=np.float32)
    out[action] = 1.0
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
    if arr.ndim != 2:
        raise ValueError(f"Raw {domain} canonical outcomes must have shape [N, T], got {arr.shape}.")

    return arr


def get_actions(raw, domain):
    key = "actions"
    if key not in raw:
        raise KeyError(f"Raw {domain} data is missing canonical action key '{key}'.")

    A = np.asarray(raw[key])
    if A.ndim != 2:
        raise ValueError(f"Raw {domain} canonical actions must have shape [N, T], got {A.shape}.")

    return A.astype(np.int64)


def get_static_array(raw, n_rows):
    key = "static_features"
    if key not in raw:
        raise KeyError("Raw data is missing canonical static key 'static_features'.")

    S = np.asarray(raw[key], dtype=np.float32)
    if S.shape != (n_rows, D_STATIC_MAX):
        raise ValueError(
            f"Raw static_features must have shape ({n_rows}, {D_STATIC_MAX}), got {S.shape}."
        )
    if not np.isfinite(S).all():
        raise ValueError("Raw static_features contains non-finite values.")
    return S


def compute_static_stats(static_support):
    """Per-column standardization statistics for static features."""
    static_support = np.asarray(static_support, dtype=np.float32)
    if static_support.ndim != 2:
        raise ValueError(f"Static support features must be 2-D, got shape={static_support.shape}.")
    if not np.isfinite(static_support).all():
        raise ValueError("Static support features contain non-finite values.")

    d_static = int(static_support.shape[-1])
    static_mean = np.zeros(d_static, dtype=np.float32)
    static_std = np.ones(d_static, dtype=np.float32)

    for j in range(d_static):
        column = static_support[:, j]
        if np.isin(column, (0.0, 1.0)).all():
            continue
        static_mean[j] = np.float32(column.mean())
        static_std[j] = np.float32(max(float(column.std()), 0.1))

    return static_mean, static_std


def normalize_static_features(static, static_mean, static_std):
    static = np.asarray(static, dtype=np.float32)
    d_static = int(static.shape[-1])
    mean = np.asarray(static_mean, dtype=np.float32).reshape(1, d_static)
    std = np.asarray(static_std, dtype=np.float32).reshape(1, d_static)

    out = ((static - mean) / np.maximum(std, 0.1)).astype(np.float32)
    out = np.clip(out, -STATIC_CLIP_TRAIN, STATIC_CLIP_TRAIN).astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError("Normalized static features contain non-finite values.")
    return out


def compute_support_stats(raw_support, domain, cfg, chosen):
    L = np.asarray(raw_support["sequence_lengths"], dtype=np.int64)
    S = get_state_array(raw_support, domain, cfg)
    Y = get_outcome_array(raw_support, domain, cfg)

    target_idx = cfg["target_state_index"]
    if target_idx is not None:
        if not 0 <= target_idx < S.shape[-1]:
            raise ValueError(
                f"Target state index {target_idx} is outside state width {S.shape[-1]}."
            )
        cov_idx = [i for i in range(S.shape[-1]) if i != target_idx]
        C = S[:, :, cov_idx]
    else:
        C = S

    cov_vals = []
    out_vals = []

    for i in chosen:
        end_x = int(L[i])
        end_y = min(end_x + 1, Y.shape[1])
        if not 1 <= end_x <= min(S.shape[1], MAX_SEQ_LEN):
            raise ValueError(f"Support sequence length {end_x} is outside the canonical range.")
        cov_vals.append(C[i, :end_x, :])
        out_vals.append(Y[i, :end_y])

    if not cov_vals:
        raise ValueError("Support statistics require at least one selected row.")
    cov_vals = np.concatenate(cov_vals, axis=0)
    out_vals = np.concatenate(out_vals)
    if not np.isfinite(cov_vals).all() or not np.isfinite(out_vals).all():
        raise ValueError("Support statistics require finite states and outcomes.")

    state_mean = cov_vals.mean(axis=0).astype(np.float32)
    state_std = np.maximum(cov_vals.std(axis=0), 0.1).astype(np.float32)
    out_mean = float(np.mean(out_vals))
    out_std = float(max(np.std(out_vals), 1e-6))

    static_all = get_static_array(raw_support, int(Y.shape[0]))
    static_mean, static_std = compute_static_stats(static_all[np.asarray(chosen, dtype=np.int64)])

    return state_mean, state_std, static_mean, static_std, out_mean, out_std


def build_benchmark_arrays_for_raw(
    raw, domain, cfg, state_mean, state_std, static_mean, static_std, out_mean, out_std
):
    out_std = float(out_std)
    if out_std <= 0:
        raise ValueError("Outcome standard deviation must be positive.")

    L = np.asarray(raw["sequence_lengths"], dtype=np.int64)
    A = get_actions(raw, domain)
    Y = get_outcome_array(raw, domain, cfg).astype(np.float32)
    n = Y.shape[0]
    static = normalize_static_features(get_static_array(raw, n), static_mean, static_std)
    S = get_state_array(raw, domain, cfg)
    target_idx = cfg["target_state_index"]

    if target_idx is not None:
        if not 0 <= target_idx < S.shape[-1]:
            raise ValueError(
                f"Target state index {target_idx} is outside state width {S.shape[-1]}."
            )
        cov_idx = [i for i in range(S.shape[-1]) if i != target_idx]
        C_raw = S[:, :, cov_idx]
    else:
        C_raw = S
    if S.shape[:2] != Y.shape or A.shape != Y.shape or L.shape != (n,):
        raise ValueError("Canonical state, outcome, action, and length shapes are inconsistent.")

    dc = C_raw.shape[-1]
    sm = np.asarray(state_mean, dtype=np.float32).reshape(1, 1, dc)
    ss = np.asarray(state_std, dtype=np.float32).reshape(1, 1, dc)

    C = ((C_raw - sm) / np.maximum(ss, 0.1)).astype(np.float32)
    C = np.clip(C, -STATE_CLIP_TRAIN, STATE_CLIP_TRAIN)

    y_norm = ((Y - float(out_mean)) / out_std).astype(np.float32)
    y_norm_clip = np.clip(
        y_norm,
        -OUTCOME_CLIP_TRAIN,
        OUTCOME_CLIP_TRAIN,
    ).astype(np.float32)
    if not np.isfinite(C).all() or not np.isfinite(y_norm).all():
        raise ValueError("Canonical benchmark arrays contain non-finite normalized values.")
    if np.any((A < 0) | (A >= N_ACTIONS)):
        raise ValueError(f"Canonical actions must be within [0, {N_ACTIONS - 1}].")

    return {
        "covariates": C.astype(np.float32),
        "y_raw": Y.astype(np.float32),
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

    state_mean, state_std, static_mean, static_std, out_mean, out_std = compute_support_stats(
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
        static_mean=static_mean,
        static_std=static_std,
        out_mean=out_mean,
        out_std=out_std,
    )
    support_bundle["domain"] = domain

    dataset_id = int(pm["dataset_id"])
    actual_support_size = int(support_bundle["covariates"].shape[0])
    support_size = int(pm["support_size"])
    if actual_support_size != support_size:
        raise ValueError(
            f"Raw benchmark support_size={support_size} differs from its {actual_support_size} support rows."
        )
    meta = {
        "dataset_uid": str(pm["dataset_uid"]),
        "domain": domain,
        "domain_key": domain,
        "cfg": cfg,
        "dataset_id": dataset_id,
        "global_dataset_id": int(global_dataset_id),
        "source_file": os.path.basename(pfile),
        "source_path": str(pfile),
        "gamma": pm["gamma"],
        "support_size": int(support_size),
        "replicate": int(pm["rep"]),
        "out_mean": float(out_mean),
        "out_std": float(out_std),
        "state_mean": state_mean,
        "state_std": state_std,
        "static_mean": static_mean,
        "static_std": static_std,
        "outcome_name": "outcomes",
        "target_space": str(pm["target_space"]),
        "max_seq_len": int(MAX_SEQ_LEN),
    }

    return support_bundle, meta


def make_query_bundle(raw_query, meta):
    bundle = build_benchmark_arrays_for_raw(
        raw_query,
        meta["domain"],
        meta["cfg"],
        state_mean=meta["state_mean"],
        state_std=meta["state_std"],
        static_mean=meta["static_mean"],
        static_std=meta["static_std"],
        out_mean=meta["out_mean"],
        out_std=meta["out_std"],
    )
    bundle["domain"] = meta["domain"]
    return bundle
