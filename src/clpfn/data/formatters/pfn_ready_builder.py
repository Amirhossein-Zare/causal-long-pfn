from __future__ import annotations

import gc
import logging
import os
import pickle
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

READY_FORMAT_VERSION = "causal_long_pfn_ready"

DEFAULT_OUTPUT_DIR = Path("outputs/pfn_ready/all_domains")

PFN_MAX_CONTEXT = 250
PFN_MAX_TEST_ROWS_PER_TASK = None
RANDOM_SEED = 2026

TARGET_SENTINEL = HIDDEN_SENTINEL


def clean_output_dir(output_dir=DEFAULT_OUTPUT_DIR) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for stale_file in output_dir.glob("*.p"):
        stale_file.unlink()

    return output_dir


def support_anchor_candidates(Y_i, max_anchor, prefer_min_tobs=True):
    max_anchor = int(max(1, min(max_anchor, common.MAX_TARGET_INDEX)))
    lo = common.MIN_T_OBS + 1 if prefer_min_tobs and max_anchor >= common.MIN_T_OBS + 1 else 1

    candidates = np.arange(lo, max_anchor + 1, dtype=np.int64)
    candidates = candidates[np.isfinite(Y_i[candidates])]

    if candidates.size == 0 and lo > 1:
        candidates = np.arange(1, max_anchor + 1, dtype=np.int64)
        candidates = candidates[np.isfinite(Y_i[candidates])]

    return candidates


def build_features_for_raw(raw, domain, cfg, state_mean, state_std, out_mean, out_std):
    out_std = max(float(out_std), 1e-6)
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
    covariates_norm[~np.isfinite(covariates_norm)] = 0.0

    y_norm = ((outcomes - out_mean) / out_std).astype(np.float32)
    y_norm = np.clip(y_norm, -common.OUTCOME_CLIP_TRAIN, common.OUTCOME_CLIP_TRAIN)
    y_norm[~np.isfinite(y_norm)] = 0.0

    x = np.concatenate([covariates_norm, y_norm[:, :, None]], axis=-1).astype(np.float32)
    d_input = int(x.shape[-1])

    if d_input > D_INPUT_MAX:
        raise ValueError(
            f"{domain} formatted d_input={d_input} exceeds D_INPUT_MAX={D_INPUT_MAX}. "
            "Reduce the active feature set before building PFN-ready files."
        )

    return x, d_input


def make_support_context(raw_support, domain, cfg, rng, max_context=PFN_MAX_CONTEXT):
    lengths = np.asarray(raw_support["sequence_lengths"], dtype=np.int64)
    actions = common.get_actions(raw_support, domain)
    outcomes = common.get_outcome_array(raw_support, domain, cfg)

    n_total = int(lengths.shape[0])
    if n_total == 0:
        raise ValueError("Support data has zero rows.")

    eligible = []

    for row_idx in range(n_total):
        max_anchor = min(int(lengths[row_idx]), outcomes.shape[1] - 1, common.MAX_TARGET_INDEX)

        if max_anchor < 1:
            continue

        candidates = support_anchor_candidates(outcomes[row_idx], max_anchor, prefer_min_tobs=True)

        if candidates.size > 0 and max_anchor >= common.MIN_T_OBS + 1:
            eligible.append(row_idx)

    eligible = np.asarray(eligible, dtype=np.int64)

    if len(eligible) == 0:
        relaxed_eligible = []

        for row_idx in range(n_total):
            max_anchor = min(int(lengths[row_idx]), outcomes.shape[1] - 1, common.MAX_TARGET_INDEX)

            if max_anchor < 1:
                continue

            candidates = support_anchor_candidates(outcomes[row_idx], max_anchor, prefer_min_tobs=False)

            if candidates.size > 0:
                relaxed_eligible.append(row_idx)

        eligible = np.asarray(relaxed_eligible, dtype=np.int64)

    if len(eligible) == 0:
        eligible = np.arange(n_total, dtype=np.int64)

    if len(eligible) > max_context:
        chosen = rng.choice(eligible, size=max_context, replace=False)
    else:
        chosen = eligible.copy()

    chosen = np.sort(chosen)

    model_normalizer_rows = chosen
    state_mean, state_std, out_mean, out_std = common.compute_support_stats(
        raw_support,
        domain,
        cfg,
        model_normalizer_rows,
    )
    eval_normalizer_rows = np.arange(n_total, dtype=np.int64)
    _, _, eval_out_mean, eval_out_std = common.compute_support_stats(
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

    static_all = common.get_static_array(raw_support, n_total)
    support_static = np.zeros((n_support, D_STATIC_MAX), dtype=np.float32)

    for support_idx, row_idx in enumerate(chosen):
        seq_len = int(lengths[row_idx])
        valid_len = min(seq_len, raw_T, common.MAX_SEQ_LEN)

        if valid_len > 0:
            support_x[support_idx, :valid_len, :] = x_all[row_idx, :valid_len, :]

        action_len = min(actions.shape[1], common.MAX_SEQ_LEN)
        support_actions[support_idx, :action_len] = actions[row_idx, :action_len]
        support_static[support_idx] = static_all[row_idx]

        max_anchor = min(seq_len, raw_T - 1, outcomes.shape[1] - 1, common.MAX_TARGET_INDEX)
        max_anchor = max(1, max_anchor)

        candidates = support_anchor_candidates(outcomes[row_idx], max_anchor, prefer_min_tobs=True)

        if candidates.size == 0:
            candidates = support_anchor_candidates(outcomes[row_idx], max_anchor, prefer_min_tobs=False)

        if candidates.size == 0:
            candidates = np.array([max_anchor], dtype=np.int64)

        max_a = int(candidates.max())
        min_a = int(candidates.min())
        mid_a = int(max(min_a, min(max_a, (min_a + max_a) // 2)))

        anchors = [max_a, mid_a, min_a, int(rng.choice(candidates))]

        for anchor_idx, anchor_time in enumerate(anchors[:N_SUPPORT_ANCHORS]):
            anchor_time = int(max(min_a, min(int(anchor_time), max_a)))

            if not np.isfinite(outcomes[row_idx, anchor_time]):
                anchor_time = int(rng.choice(candidates))

            support_anchor_time[support_idx, anchor_idx] = anchor_time
            support_anchor_y[support_idx, anchor_idx] = np.float32(
                np.clip(
                    (float(outcomes[row_idx, anchor_time]) - out_mean) / max(float(out_std), 1e-6),
                    -common.TARGET_NORM_CLIP,
                    common.TARGET_NORM_CLIP,
                )
            )

    return {
        "support_x": support_x,
        "support_actions": support_actions,
        "support_anchor_y": support_anchor_y,
        "support_anchor_time": support_anchor_time,
        "support_static": support_static,

        "n_support": int(n_support),
        "n_ctx": int(n_support),
        "d_input": int(d_input),
        "d": int(d_input),

        "out_mean": np.float32(out_mean),
        "out_std": np.float32(out_std),
        "eval_out_mean": np.float32(eval_out_mean),
        "eval_out_std": np.float32(eval_out_std),

        "state_mean": state_mean,
        "state_std": state_std,

        "support_rows_total": int(n_total),
        "support_rows_eligible": int(len(eligible)),
        "support_rows_used": int(n_support),
        "normalization_rows_used": int(len(model_normalizer_rows)),
        "normalization_scope": "pfn_context",
        "eval_normalization_rows_used": int(len(eval_normalizer_rows)),
        "eval_normalization_scope": "full_support",
    }


def strip_private_support_stats(support_context):
    out = dict(support_context)
    out.pop("state_mean", None)
    out.pop("state_std", None)
    return out


def make_query_task_ready(raw_query, rows, current_times, target_times, domain, cfg, support_context):
    d_input = int(support_context["d_input"])

    if len(rows) == 0:
        return {
            "rows": np.zeros(0, dtype=np.int64),
            "current_time": np.zeros(0, dtype=np.int64),
            "current_t": np.zeros(0, dtype=np.int64),
            "t_obs": np.zeros(0, dtype=np.int64),
            "t_target": np.zeros(0, dtype=np.int64),
            "tau": np.zeros(0, dtype=np.int64),

            "query_x": np.full((0, common.MAX_SEQ_LEN, d_input), TARGET_SENTINEL, dtype=np.float32),
            "query_actions": np.zeros((0, common.MAX_SEQ_LEN), dtype=np.int64),
            "query_static": np.zeros((0, D_STATIC_MAX), dtype=np.float32),

            "target_value": np.zeros(0, dtype=np.float32),
            "target_raw": np.zeros(0, dtype=np.float32),
            "target_y_norm": np.zeros(0, dtype=np.float32),
            "target_model_norm": np.zeros(0, dtype=np.float32),
            "target_eval_norm": np.zeros(0, dtype=np.float32),
            "target_norm": np.zeros(0, dtype=np.float32),

            "out_mean": np.float32(support_context["out_mean"]),
            "out_std": np.float32(support_context["out_std"]),
            "eval_out_mean": np.float32(support_context["eval_out_mean"]),
            "eval_out_std": np.float32(support_context["eval_out_std"]),
            "n_eval": 0,
        }

    actions = common.get_actions(raw_query, domain)
    outcomes = common.get_outcome_array(raw_query, domain, cfg)

    out_mean = float(support_context["out_mean"])
    out_std = max(float(support_context["out_std"]), 1e-6)
    eval_out_mean = float(support_context["eval_out_mean"])
    eval_out_std = max(float(support_context["eval_out_std"]), 1e-6)

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

    static_all = common.get_static_array(raw_query, outcomes.shape[0])

    target_value = np.zeros(n_rows, dtype=np.float32)
    target_model_norm = np.zeros(n_rows, dtype=np.float32)
    target_eval_norm = np.zeros(n_rows, dtype=np.float32)

    t_obs = np.zeros(n_rows, dtype=np.int64)
    t_target = np.zeros(n_rows, dtype=np.int64)
    current_time_out = np.zeros(n_rows, dtype=np.int64)
    tau = np.zeros(n_rows, dtype=np.int64)

    for out_idx, row_id in enumerate(rows):
        row_id = int(row_id)

        current_time = int(current_times[out_idx])
        target_time = int(target_times[out_idx])

        current_time = max(0, min(current_time, raw_T - 1, common.MAX_INPUT_INDEX))
        target_time = max(1, min(target_time, raw_T - 1, common.MAX_TARGET_INDEX))

        if target_time <= current_time:
            target_time = min(current_time + 1, raw_T - 1, common.MAX_TARGET_INDEX)

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

        target_value[out_idx] = np.float32(y_value)
        target_model_norm[out_idx] = np.float32(y_model_norm)
        target_eval_norm[out_idx] = np.float32(y_eval_norm)

        current_time_out[out_idx] = current_time
        t_obs[out_idx] = current_time
        t_target[out_idx] = target_time
        tau[out_idx] = max(1, target_time - current_time)

    return {
        "rows": rows.astype(np.int64),
        "current_time": current_time_out,
        "current_t": current_time_out,
        "t_obs": t_obs,
        "t_target": t_target,
        "tau": tau,

        "query_x": query_x,
        "query_actions": query_actions,
        "query_static": query_static,

        "target_value": target_value,
        "target_raw": target_value,
        "target_y_norm": target_model_norm,
        "target_model_norm": target_model_norm,
        "target_eval_norm": target_eval_norm,
        "target_norm": target_eval_norm,

        "out_mean": np.float32(out_mean),
        "out_std": np.float32(out_std),
        "eval_out_mean": np.float32(eval_out_mean),
        "eval_out_std": np.float32(eval_out_std),
        "n_eval": int(n_rows),
    }


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

    rng = np.random.default_rng(int(seed) + 100000 * int(global_dataset_id) + max(dataset_id, 0))

    support_context = make_support_context(
        raw_support=support_raw,
        domain=domain,
        cfg=cfg,
        rng=rng,
        max_context=pfn_max_context,
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
        return None, {"skipped": True, "reason": "missing_test_data"}

    gamma = pm["gamma"]

    ready_map = {
        "ready_format_version": READY_FORMAT_VERSION,
        "global_dataset_id": int(global_dataset_id),
        "dataset_id": int(dataset_id),
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
        "min_t_obs": int(common.MIN_T_OBS),
        "t_obs_semantics": "last_visible_index",
        "rollout_start_semantics": "start_current_time_equals_t_obs",
        "model_family": "CausalLongPFN",
        "ready_tensor_naming": "manuscript_support_query_names",
        "pfn_max_context": int(pfn_max_context),
        "pfn_max_test_rows_per_task": max_test_rows_per_task,

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
):
    output_dir = Path(output_dir)

    wanted_domains = common.WANTED_DOMAINS if wanted_domains is None else tuple(wanted_domains)

    output_dir = clean_output_dir(output_dir)

    common.configure_torch_runtime(seed=seed)

    raw_files = common.find_raw_pickles(
        common.RawBenchmarkInputs.from_dict(raw_inputs),
    )

    LOGGER.info(
        "Build CausalLongPFN-ready benchmark files | output_dir=%s | "
        "raw_pickles=%s | wanted_domains=%s | pfn_max_context=%s | max_test_rows_per_task=%s",
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

        if info.get("skipped", False):
            skipped.append({
                "source_file": os.path.basename(pfile),
                "source_path": str(pfile),
                "reason": info.get("reason", "skipped"),
            })
            LOGGER.info("Skipped %s: %s", os.path.basename(pfile), info.get("reason", "skipped"))
            continue

        out_file = output_dir / (
            f"causal_long_pfn_ready_{ready_map['domain']}_dataset_{global_dataset_id:04d}_"
            f"rawid_{int(ready_map['dataset_id']):03d}.p"
        )

        ready_map["ready_file"] = out_file.name

        with open(out_file, "wb") as f:
            pickle.dump(ready_map, f, protocol=pickle.HIGHEST_PROTOCOL)

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

    return {
        "ready_files": ready_files,
        "skipped": skipped,
        "output_dir": str(output_dir),
    }
