from __future__ import annotations

import gc
import logging
import os
import pickle
import hashlib
import json
from pathlib import Path

import numpy as np

from clpfn.evaluation.core import benchmark as common
from clpfn.evaluation.core import tasks as eval_tasks
from clpfn.config.defaults import (
    D_INPUT_MAX,
    D_STATIC_MAX,
    HIDDEN_SENTINEL,
    N_SUPPORT_ANCHORS,
)


LOGGER = logging.getLogger(__name__)

READY_FORMAT_VERSION = "causal_long_pfn_ready_static_normalized"

DEFAULT_OUTPUT_DIR = Path("outputs/pfn_ready")

PFN_MAX_CONTEXT = 250
PFN_MAX_TEST_ROWS_PER_TASK = None
RANDOM_SEED = 2026

TARGET_SENTINEL = HIDDEN_SENTINEL


def prepare_output_dir(output_dir=DEFAULT_OUTPUT_DIR, *, overwrite=False) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Ready build directory already exists and is non-empty: {output_dir}. Use overwrite=True to replace it.")
        for path in output_dir.glob("causal_long_pfn_ready_*.p"):
            path.unlink()
        for path in output_dir.glob("*manifest*.json"):
            path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def support_anchor_candidates(Y_i, max_anchor):
    max_anchor = int(max_anchor)
    if not common.MIN_T_OBS + 1 <= max_anchor <= common.MAX_TARGET_INDEX:
        raise ValueError(
            f"Support anchor limit {max_anchor} is outside "
            f"[{common.MIN_T_OBS + 1}, {common.MAX_TARGET_INDEX}]."
        )
    lo = common.MIN_T_OBS + 1
    candidates = np.arange(lo, max_anchor + 1, dtype=np.int64)
    return candidates[np.isfinite(Y_i[candidates])]


def build_features_for_raw(raw, domain, cfg, state_mean, state_std, out_mean, out_std):
    out_std = float(out_std)
    if out_std <= 0:
        raise ValueError("Outcome standard deviation must be positive.")
    states = common.get_state_array(raw, domain, cfg)
    outcomes = common.get_outcome_array(raw, domain, cfg)

    n_rows, raw_T, d_state = states.shape
    target_idx = cfg["target_state_index"]

    if target_idx is not None and 0 <= target_idx < d_state:
        cov_idx = [i for i in range(d_state) if i != target_idx]
        covariates = states[:, :, cov_idx]
    else:
        covariates = states

    d_cov = covariates.shape[-1]

    state_mean = np.asarray(state_mean, dtype=np.float32).reshape(1, 1, d_cov)
    state_std = np.asarray(state_std, dtype=np.float32).reshape(1, 1, d_cov)

    covariates_norm = ((covariates - state_mean) / np.maximum(state_std, 0.1)).astype(np.float32)
    np.clip(covariates_norm, -3.0, 3.0, out=covariates_norm)
    if not np.isfinite(covariates_norm).all():
        raise ValueError(f"{domain} covariates contain non-finite normalized values.")

    y_norm = ((outcomes - out_mean) / out_std).astype(np.float32)
    y_norm = np.clip(y_norm, -common.OUTCOME_CLIP_TRAIN, common.OUTCOME_CLIP_TRAIN)
    if not np.isfinite(y_norm).all():
        raise ValueError(f"{domain} outcomes contain non-finite normalized values.")

    x = np.concatenate([covariates_norm, y_norm[:, :, None]], axis=-1).astype(np.float32)
    d_input = int(x.shape[-1])

    if d_input > D_INPUT_MAX:
        raise ValueError(
            f"{domain} formatted d_input={d_input} exceeds D_INPUT_MAX={D_INPUT_MAX}. "
            "Reduce the active feature set before building PFN-ready files."
        )

    return x, d_input


def make_support_context(
    raw_support, domain, cfg, rng, support_selection_seed, max_context=PFN_MAX_CONTEXT,
):
    lengths = np.asarray(raw_support["sequence_lengths"], dtype=np.int64)
    actions = common.get_actions(raw_support, domain)
    outcomes = common.get_outcome_array(raw_support, domain, cfg)

    n_total = int(lengths.shape[0])
    if n_total == 0:
        raise ValueError("Support data has zero rows.")

    eligible = []

    for row_idx in range(n_total):
        max_anchor = min(int(lengths[row_idx]) - 1, outcomes.shape[1] - 1, common.MAX_TARGET_INDEX)

        if max_anchor < 1:
            continue

        if max_anchor < common.MIN_T_OBS + 1:
            continue
        candidates = support_anchor_candidates(outcomes[row_idx], max_anchor)

        if candidates.size > 0 and max_anchor >= common.MIN_T_OBS + 1:
            eligible.append(row_idx)

    eligible = np.asarray(eligible, dtype=np.int64)

    if len(eligible) == 0:
        raise ValueError(
            f"Support data has no row with a finite anchor at or after t={common.MIN_T_OBS + 1}."
        )

    if len(eligible) > max_context:
        chosen = rng.choice(eligible, size=max_context, replace=False)
    else:
        chosen = eligible.copy()

    chosen = np.sort(chosen)
    support_patient_ids = np.asarray(raw_support["patient_id"])[chosen].copy()

    model_normalizer_rows = chosen
    state_mean, state_std, static_mean, static_std, out_mean, out_std = common.compute_support_stats(
        raw_support,
        domain,
        cfg,
        model_normalizer_rows,
    )
    eval_normalizer_rows = np.arange(n_total, dtype=np.int64)
    *_, eval_out_mean, eval_out_std = common.compute_support_stats(
        raw_support,
        domain,
        cfg,
        eval_normalizer_rows,
    )

    x_all, d_input = build_features_for_raw(
        raw=raw_support,
        domain=domain,
        cfg=cfg,
        state_mean=state_mean,
        state_std=state_std,
        out_mean=out_mean,
        out_std=out_std,
    )

    n_support = int(len(chosen))
    raw_T = int(x_all.shape[1])

    support_x = np.full(
        (n_support, common.MAX_SEQ_LEN, d_input),
        TARGET_SENTINEL,
        dtype=np.float32,
    )
    support_actions = np.zeros((n_support, common.MAX_SEQ_LEN), dtype=np.int64)
    support_anchor_y = np.zeros((n_support, N_SUPPORT_ANCHORS), dtype=np.float32)
    support_anchor_time = np.ones((n_support, N_SUPPORT_ANCHORS), dtype=np.int64)

    static_all = common.normalize_static_features(
        common.get_static_array(raw_support, n_total), static_mean, static_std
    )
    support_static = np.zeros((n_support, D_STATIC_MAX), dtype=np.float32)

    for support_idx, row_idx in enumerate(chosen):
        seq_len = int(lengths[row_idx])
        if not 1 <= seq_len <= raw_T:
            raise ValueError(f"Support sequence length {seq_len} is outside [1, {raw_T}].")
        valid_len = min(seq_len, common.MAX_SEQ_LEN)
        support_x[support_idx, :valid_len, :] = x_all[row_idx, :valid_len, :]

        action_len = min(valid_len, actions.shape[1], common.MAX_SEQ_LEN)
        support_actions[support_idx, :action_len] = actions[row_idx, :action_len]
        support_static[support_idx] = static_all[row_idx]

        max_anchor = min(seq_len - 1, raw_T - 1, outcomes.shape[1] - 1, common.MAX_TARGET_INDEX)
        candidates = support_anchor_candidates(outcomes[row_idx], max_anchor)
        if candidates.size == 0:
            raise ValueError(f"Support row {row_idx} has no finite canonical anchor.")

        max_a = int(candidates.max())
        min_a = int(candidates.min())
        mid_a = int(max(min_a, min(max_a, (min_a + max_a) // 2)))

        anchors = [max_a, mid_a, min_a]
        while len(anchors) < N_SUPPORT_ANCHORS:
            anchors.append(int(rng.choice(candidates)))

        for anchor_idx, anchor_time in enumerate(anchors[:N_SUPPORT_ANCHORS]):
            anchor_time = int(anchor_time)
            if anchor_time not in candidates:
                raise ValueError(f"Support anchor {anchor_time} is not a valid candidate.")

            support_anchor_time[support_idx, anchor_idx] = anchor_time
            support_anchor_y[support_idx, anchor_idx] = np.float32(
                np.clip(
                    (float(outcomes[row_idx, anchor_time]) - out_mean) / float(out_std),
                    -common.TARGET_NORM_CLIP,
                    common.TARGET_NORM_CLIP,
                )
            )

    state_observed = np.asarray(raw_support["state_observed_mask"], dtype=bool)[chosen]
    outcome_observed = np.asarray(raw_support["outcome_observed_mask"], dtype=bool)[chosen]
    support_state_observed_mask = np.zeros(
        (n_support, common.MAX_SEQ_LEN, state_observed.shape[-1]),
        dtype=bool,
    )
    support_outcome_observed_mask = np.zeros(
        (n_support, common.MAX_SEQ_LEN),
        dtype=bool,
    )
    mask_width = min(state_observed.shape[1], common.MAX_SEQ_LEN)
    support_state_observed_mask[:, :mask_width] = state_observed[:, :mask_width]
    support_outcome_observed_mask[:, :mask_width] = outcome_observed[:, :mask_width]

    return {
        "support_x": support_x,
        "support_actions": support_actions,
        "support_anchor_y": support_anchor_y,
        "support_anchor_time": support_anchor_time,
        "support_raw_row_ids": chosen.copy(),
        "support_patient_ids": support_patient_ids,
        "support_selection_seed": int(support_selection_seed),
        "support_static": support_static,
        "support_state_observed_mask": support_state_observed_mask,
        "support_outcome_observed_mask": support_outcome_observed_mask,

        "n_support": int(n_support),
        "d_input": int(d_input),

        "out_mean": np.float32(out_mean),
        "out_std": np.float32(out_std),
        "eval_out_mean": np.float32(eval_out_mean),
        "eval_out_std": np.float32(eval_out_std),

        "state_mean": state_mean,
        "state_std": state_std,
        "static_mean": static_mean,
        "static_std": static_std,

        "support_rows_total": int(n_total),
        "support_rows_eligible": int(len(eligible)),
        "support_rows_used": int(n_support),
        "normalization_rows_used": int(len(model_normalizer_rows)),
        "normalization_scope": "pfn_context",
        "static_normalization_scope": "pfn_context",
        "eval_normalization_rows_used": int(len(eval_normalizer_rows)),
        "eval_normalization_scope": "full_support",
    }


def strip_private_support_stats(support_context):
    out = dict(support_context)
    del out["state_mean"]
    del out["state_std"]
    del out["static_mean"]
    del out["static_std"]
    return out


def make_query_task_ready(raw_query, rows, current_times, target_times, domain, cfg, support_context):
    d_input = int(support_context["d_input"])

    actions = common.get_actions(raw_query, domain)
    outcomes = common.get_outcome_array(raw_query, domain, cfg)

    out_mean = float(support_context["out_mean"])
    out_std = float(support_context["out_std"])
    eval_out_mean = float(support_context["eval_out_mean"])
    eval_out_std = float(support_context["eval_out_std"])
    if out_std <= 0 or eval_out_std <= 0:
        raise ValueError("Ready outcome standard deviations must be positive.")

    x_all, d_input = build_features_for_raw(
        raw=raw_query,
        domain=domain,
        cfg=cfg,
        state_mean=support_context["state_mean"],
        state_std=support_context["state_std"],
        out_mean=out_mean,
        out_std=out_std,
    )

    raw_T = int(outcomes.shape[1])
    n_rows = int(len(rows))

    query_x = np.full((n_rows, common.MAX_SEQ_LEN, d_input), TARGET_SENTINEL, dtype=np.float32)
    query_actions = np.zeros((n_rows, common.MAX_SEQ_LEN), dtype=np.int64)
    query_static = np.zeros((n_rows, D_STATIC_MAX), dtype=np.float32)

    static_all = common.normalize_static_features(
        common.get_static_array(raw_query, outcomes.shape[0]),
        support_context["static_mean"],
        support_context["static_std"],
    )

    target_raw = np.zeros(n_rows, dtype=np.float32)
    target_model_norm = np.zeros(n_rows, dtype=np.float32)
    target_eval_norm = np.zeros(n_rows, dtype=np.float32)
    current_y_raw = np.zeros(n_rows, dtype=np.float32)
    current_y_model_norm = np.zeros(n_rows, dtype=np.float32)
    current_y_eval_norm_unclipped = np.zeros(n_rows, dtype=np.float32)
    current_y_eval_norm_reported = np.zeros(n_rows, dtype=np.float32)
    current_y_observed = np.zeros(n_rows, dtype=bool)
    target_observed = np.zeros(n_rows, dtype=bool)
    target_path_raw = None
    if "target_path_raw" in raw_query:
        target_path_raw = np.asarray(raw_query["target_path_raw"], dtype=np.float32)[rows].copy()
        target_path_model_norm = np.clip((target_path_raw - out_mean) / out_std, -common.TARGET_NORM_CLIP, common.TARGET_NORM_CLIP).astype(np.float32)
        target_path_eval_norm_unclipped = ((target_path_raw - eval_out_mean) / eval_out_std).astype(np.float32)
        target_path_eval_norm_reported = np.clip(target_path_eval_norm_unclipped, -common.TARGET_NORM_CLIP, common.TARGET_NORM_CLIP).astype(np.float32)

    t_obs = np.zeros(n_rows, dtype=np.int64)
    t_target = np.zeros(n_rows, dtype=np.int64)
    current_time_out = np.zeros(n_rows, dtype=np.int64)
    tau = np.zeros(n_rows, dtype=np.int64)

    for out_idx, row_id in enumerate(rows):
        row_id = int(row_id)

        current_time = int(current_times[out_idx])
        target_time = int(target_times[out_idx])

        if not 0 <= current_time <= min(raw_T - 1, common.MAX_INPUT_INDEX):
            raise ValueError(f"Current time {current_time} is outside the canonical query range.")
        if not 1 <= target_time <= min(raw_T - 1, common.MAX_TARGET_INDEX):
            raise ValueError(f"Target time {target_time} is outside the canonical query range.")
        if target_time <= current_time:
            raise ValueError(
                f"Target time {target_time} must be after current time {current_time}."
            )

        visible_len = min(current_time + 1, raw_T, common.MAX_SEQ_LEN)
        query_x[out_idx, :visible_len, :] = x_all[row_id, :visible_len, :]

        action_len = min(actions.shape[1], common.MAX_SEQ_LEN)
        query_actions[out_idx, :action_len] = actions[row_id, :action_len]
        query_static[out_idx] = static_all[row_id]

        y_value = float(outcomes[row_id, target_time])
        y_model_norm = float(
            np.clip(
                (y_value - out_mean) / out_std,
                -common.TARGET_NORM_CLIP,
                common.TARGET_NORM_CLIP,
            )
        )
        y_eval_norm = float(
            np.clip(
                (y_value - eval_out_mean) / eval_out_std,
                -common.TARGET_NORM_CLIP,
                common.TARGET_NORM_CLIP,
            )
        )

        target_raw[out_idx] = np.float32(y_value)
        target_model_norm[out_idx] = np.float32(y_model_norm)
        target_eval_norm[out_idx] = np.float32(y_eval_norm)
        current_value = float(outcomes[row_id, current_time])
        current_model_unclipped = (current_value - out_mean) / out_std
        current_eval_unclipped = (current_value - eval_out_mean) / eval_out_std
        current_y_raw[out_idx] = np.float32(current_value)
        current_y_model_norm[out_idx] = np.float32(np.clip(current_model_unclipped, -common.TARGET_NORM_CLIP, common.TARGET_NORM_CLIP))
        current_y_eval_norm_unclipped[out_idx] = np.float32(current_eval_unclipped)
        current_y_eval_norm_reported[out_idx] = np.float32(np.clip(current_eval_unclipped, -common.TARGET_NORM_CLIP, common.TARGET_NORM_CLIP))
        current_y_observed[out_idx] = bool(np.asarray(raw_query["outcome_observed_mask"])[row_id, current_time])
        target_observed[out_idx] = bool(np.asarray(raw_query["outcome_observed_mask"])[row_id, target_time])

        current_time_out[out_idx] = current_time
        t_obs[out_idx] = current_time
        t_target[out_idx] = target_time
        tau[out_idx] = target_time - current_time

    target_observed_path = None
    if target_path_raw is not None:
        source_mask = np.asarray(raw_query["outcome_observed_mask"], dtype=bool)
        target_observed_path = np.zeros(target_path_raw.shape, dtype=bool)
        for out_idx, row_id in enumerate(rows):
            start = int(t_obs[out_idx]) + 1
            stop = start + target_observed_path.shape[1]
            target_observed_path[out_idx] = source_mask[int(row_id), start:stop]

    raw_state_observed = np.asarray(raw_query["state_observed_mask"], dtype=bool)[rows]
    raw_outcome_observed = np.asarray(raw_query["outcome_observed_mask"], dtype=bool)[rows]
    query_state_observed_mask = np.zeros(
        (n_rows, common.MAX_SEQ_LEN, raw_state_observed.shape[-1]),
        dtype=bool,
    )
    query_outcome_observed_mask = np.zeros(
        (n_rows, common.MAX_SEQ_LEN),
        dtype=bool,
    )
    mask_width = min(raw_state_observed.shape[1], common.MAX_SEQ_LEN)
    query_state_observed_mask[:, :mask_width] = raw_state_observed[:, :mask_width]
    query_outcome_observed_mask[:, :mask_width] = raw_outcome_observed[:, :mask_width]

    out = {
        "rows": rows.astype(np.int64),
        "current_time": current_time_out,
        "t_obs": t_obs,
        "t_target": t_target,
        "tau": tau,

        "query_x": query_x,
        "query_actions": query_actions,
        "query_static": query_static,

        "target_raw": target_raw,
        "target_model_norm": target_model_norm,
        "target_eval_norm": target_eval_norm,
        "current_y_raw": current_y_raw,
        "current_y_model_norm": current_y_model_norm,
        "current_y_eval_norm_unclipped": current_y_eval_norm_unclipped,
        "current_y_eval_norm_reported": current_y_eval_norm_reported,
        "current_y_observed": current_y_observed,
        "target_observed": target_observed,
        "query_state_observed_mask": query_state_observed_mask,
        "query_outcome_observed_mask": query_outcome_observed_mask,

        "out_mean": np.float32(out_mean),
        "out_std": np.float32(out_std),
        "eval_out_mean": np.float32(eval_out_mean),
        "eval_out_std": np.float32(eval_out_std),
        "n_eval": int(n_rows),
    }
    identity_keys = ("dataset_uid", "original_row_id", "patient_id", "patient_uid", "origin_time", "origin_uid", "first_action", "plan_uid", "query_uid", "is_factual", "is_counterfactual", "planned_action_sequence", "factual_prefix_length", "max_horizon")
    missing_identity = [key for key in identity_keys if key not in raw_query]
    if missing_identity:
        raise KeyError(f"Canonical benchmark query is missing identity fields: {missing_identity}")
    for key in identity_keys:
        out[key] = np.asarray(raw_query[key])[rows].copy()
    if target_path_raw is not None:
        endpoint = target_path_raw[np.arange(n_rows), tau - 1]
        if not np.allclose(endpoint, target_raw, rtol=0.0, atol=0.0, equal_nan=False):
            raise ValueError("Sequence target path endpoint differs from the existing horizon target.")
        out.update({
            "target_path_raw": target_path_raw,
            "target_path_model_norm": target_path_model_norm,
            "target_path_eval_norm_unclipped": target_path_eval_norm_unclipped,
            "target_path_eval_norm_reported": target_path_eval_norm_reported,
            "target_state_path_raw": np.asarray(raw_query["target_state_path_raw"], dtype=np.float32)[rows].copy(),
            "target_observed_path": target_observed_path,
            "target_path_normalizers": {
                "model": {"identity": "support_context_outcome_clipped", "mean": out_mean, "std": out_std, "clip": common.TARGET_NORM_CLIP},
                "evaluation": {"identity": "full_support_outcome_reported", "mean": eval_out_mean, "std": eval_out_std, "clip": common.TARGET_NORM_CLIP},
            },
        })
    return out


def build_ready_map_for_pickle(
    pfile,
    global_dataset_id,
    pfn_max_context,
    seed,
    wanted_domains,
    max_test_rows_per_task=PFN_MAX_TEST_ROWS_PER_TASK,
):
    with open(pfile, "rb") as f:
        pm = pickle.load(f)

    domain = common.dataset_domain(pfile, pm)

    if domain not in wanted_domains:
        return None, {"skipped": True, "reason": f"unwanted_domain_{domain}"}

    cfg = common.domain_config(domain)
    support_raw = common.get_support_raw(pm)

    dataset_id = int(pm["dataset_id"])
    support_size = int(pm["support_size"])
    if dataset_id < 0:
        raise ValueError("dataset_id must be non-negative.")

    support_selection_seed = int(seed) + 100000 * int(global_dataset_id) + dataset_id
    rng = np.random.default_rng(support_selection_seed)

    support_context = make_support_context(
        raw_support=support_raw,
        domain=domain,
        cfg=cfg,
        rng=rng,
        max_context=pfn_max_context,
        support_selection_seed=support_selection_seed,
    )

    tasks = {}
    for task_rows in eval_tasks.iter_raw_task_rows(
        pm=pm,
        domain=domain,
        cfg=cfg,
        rng=rng,
        max_rows=max_test_rows_per_task,
    ):
        tasks[task_rows.task_name] = make_query_task_ready(
            raw_query=task_rows.raw,
            rows=task_rows.rows,
            current_times=task_rows.current_ts,
            target_times=task_rows.target_ts,
            domain=domain,
            cfg=cfg,
            support_context=support_context,
        )

    if "one_step_cf_final" not in tasks:
        raise KeyError(f"Raw benchmark {pfile} has no one-step evaluation task.")

    gamma = pm["gamma"]

    ready_map = {
        "ready_format_version": READY_FORMAT_VERSION,
        "global_dataset_id": int(global_dataset_id),
        "dataset_id": int(dataset_id),
        "dataset_uid": str(pm["dataset_uid"]),
        "dataset_file": os.path.basename(pfile),
        "source_file": str(pfile),

        "domain": domain,
        "gamma": gamma,

        "support_size": int(support_size),
        "training_size": int(pm["training_size"]),
        "validation_size": int(pm["validation_size"]),
        "replicate": int(pm["rep"]),

        "max_seq_len": int(common.MAX_SEQ_LEN),
        "max_input_index": int(common.MAX_INPUT_INDEX),
        "max_target_index": int(common.MAX_TARGET_INDEX),
        "projection_horizon": int(common.PROJECTION_HORIZON),
        "min_history_points": int(common.MIN_HISTORY_POINTS),
        "min_t_obs": int(common.MIN_T_OBS),
        "t_obs_semantics": "last_visible_index",
        "rollout_start_semantics": "start_current_time_equals_t_obs",
        "pfn_max_context": int(pfn_max_context),
        "pfn_max_test_rows_per_task": max_test_rows_per_task,
        "n_support_anchors_built": int(N_SUPPORT_ANCHORS),

        "outcome_name": "outcomes",
        "state_name": "states",
        "action_name": "actions",
        "target_space": pm["target_space"],

        "support_context": strip_private_support_stats(support_context),
        "tasks": tasks,
    }

    del pm, support_raw, support_context
    gc.collect()

    return ready_map, {"skipped": False}


def run_all(
    wanted_domains=None,
    raw_inputs=None,
    pfn_max_context=PFN_MAX_CONTEXT,
    max_test_rows_per_task=PFN_MAX_TEST_ROWS_PER_TASK,
    output_dir=DEFAULT_OUTPUT_DIR,
    seed=RANDOM_SEED,
    overwrite=False,
    ready_build_id=None,
):
    output_dir = Path(output_dir)

    wanted_domains = common.WANTED_DOMAINS if wanted_domains is None else tuple(wanted_domains)

    output_dir = prepare_output_dir(output_dir, overwrite=overwrite)
    ready_build_id = str(ready_build_id or output_dir.name)

    common.configure_torch_runtime(seed=seed)

    raw_files = common.find_raw_pickles(
        common.RawBenchmarkInputs.from_dict(raw_inputs),
    )

    LOGGER.info(
        "Build CausalLongPFN-ready benchmark files | output_dir=%s | "
        "raw_pickles=%s | wanted_domains=%s | pfn_max_context=%s | "
        "max_test_rows_per_task=%s",
        output_dir,
        len(raw_files),
        wanted_domains,
        pfn_max_context,
        max_test_rows_per_task,
    )

    if len(raw_files) == 0:
        raise FileNotFoundError("No raw domain pickles found.")

    LOGGER.info("First raw file: %s", raw_files[0])

    skipped = []
    ready_files = []
    global_dataset_id = 0

    for file_idx, pfile in enumerate(raw_files):
        LOGGER.info("[%s/%s] %s", file_idx + 1, len(raw_files), os.path.basename(pfile))

        ready_map, info = build_ready_map_for_pickle(
            pfile=pfile,
            global_dataset_id=global_dataset_id,
            pfn_max_context=pfn_max_context,
            seed=seed,
            wanted_domains=wanted_domains,
            max_test_rows_per_task=max_test_rows_per_task,
        )

        if info["skipped"]:
            skipped.append({
                "source_file": os.path.basename(pfile),
                "source_path": str(pfile),
                "reason": info["reason"],
            })
            LOGGER.info("Skipped %s: %s", os.path.basename(pfile), info["reason"])
            continue

        out_file = output_dir / (
            f"causal_long_pfn_ready_{ready_map['domain']}_dataset_{global_dataset_id:04d}_"
            f"rawid_{int(ready_map['dataset_id']):03d}.p"
        )

        ready_map["ready_file"] = out_file.name
        ready_map["ready_build_id"] = ready_build_id

        with open(out_file, "wb") as f:
            pickle.dump(ready_map, f, protocol=pickle.HIGHEST_PROTOCOL)

        raw_hash = _sha256_file(pfile)
        ready_hash = _sha256_file(out_file)
        raw_manifest_path = Path(pfile).parent / f"benchmark_dataset_manifest_{ready_map['dataset_uid']}.json"
        if not raw_manifest_path.exists():
            raise FileNotFoundError(f"Raw dataset manifest is required for ready construction: {raw_manifest_path}")
        dataset_manifest = {
            **json.loads(raw_manifest_path.read_text(encoding="utf-8")),
            "ready_build_id": ready_build_id,
            "ready_file": str(out_file), "ready_file_hash": ready_hash,
            "support_patient_count": int(len(np.unique(ready_map["support_context"]["support_patient_ids"]))),
            "query_patient_count": int(len(np.unique(np.concatenate([task["patient_id"] for task in ready_map["tasks"].values()])))),
            "origin_count": int(len(np.unique(np.concatenate([task["origin_uid"] for task in ready_map["tasks"].values()])))),
            "plan_count": int(len(np.unique(np.concatenate([task["plan_uid"] for task in ready_map["tasks"].values()])))),
            "one_step_query_count": int(ready_map["tasks"]["one_step_cf_final"]["n_eval"]),
            "sequence_query_count": int(sum(task["n_eval"] for name, task in ready_map["tasks"].items() if name.startswith("seq_"))),
            "maximum_sequence_horizon": int(common.PROJECTION_HORIZON),
        }
        _write_json(output_dir / f"ready_dataset_manifest_{ready_map['dataset_uid']}.json", dataset_manifest)
        ready_files.append(str(out_file))

        task_summary = ", ".join(
            f"{task_name}: {int(task_value['n_eval'])}"
            for task_name, task_value in ready_map["tasks"].items()
        )

        LOGGER.info(
            "domain=%s | raw_dataset_id=%s | gamma=%s | support=%s | context=%s | d_input=%s | tasks={%s}",
            ready_map["domain"],
            ready_map["dataset_id"],
            ready_map["gamma"],
            ready_map["support_size"],
            ready_map["support_context"]["n_support"],
            ready_map["support_context"]["d_input"],
            task_summary,
        )

        global_dataset_id += 1

        del ready_map
        gc.collect()

    LOGGER.info(
        "Finished CausalLongPFN-ready build | ready_files=%s | skipped=%s | output_dir=%s",
        len(ready_files),
        len(skipped),
        output_dir,
    )

    if len(ready_files) == 0:
        raise RuntimeError("No CausalLongPFN-ready files were produced.")

    _write_json(output_dir / "ready_build_manifest.json", {
        "ready_build_id": ready_build_id,
        "output_dir": str(output_dir),
        "ready_files": ready_files,
        "dataset_manifest_files": sorted(str(path) for path in output_dir.glob("ready_dataset_manifest_*.json")),
    })

    return {
        "ready_files": ready_files,
        "skipped": skipped,
        "output_dir": str(output_dir),
    }
