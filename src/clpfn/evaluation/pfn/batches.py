import numpy as np
import torch

from clpfn.config.defaults import (
    D_INPUT_MAX,
    D_STATIC_MAX,
    MAX_INPUT_INDEX,
    MAX_SEQ_LEN,
    MAX_TARGET_INDEX,
    N_SUPPORT_ANCHORS,
)
from clpfn.evaluation.core import benchmark as common


def move_batch_to_device(batch, device):
    return common.move_tensor_batch_to_device(batch, device)


def support_context_arrays(support_context):
    required = (
        "support_x",
        "support_actions",
        "support_anchor_y",
        "support_anchor_time",
        "support_static",
        "n_support",
        "d_input",
    )
    missing = [key for key in required if key not in support_context]
    if missing:
        raise KeyError(f"Ready support_context is missing required keys: {missing}")

    support_x = np.asarray(support_context["support_x"], dtype=np.float32)
    support_actions = np.asarray(support_context["support_actions"], dtype=np.int64)
    support_anchor_y = np.asarray(support_context["support_anchor_y"], dtype=np.float32)
    support_anchor_time = np.asarray(support_context["support_anchor_time"], dtype=np.int64)
    support_static = support_context["support_static"]
    n_support = int(support_context["n_support"])
    d_input = int(support_context["d_input"])

    return support_x, support_actions, support_anchor_y, support_anchor_time, support_static, n_support, d_input


def task_arrays(task):
    required = (
        "query_x",
        "query_actions",
        "query_static",
        "target_norm",
        "current_time",
        "t_obs",
        "t_target",
        "tau",
    )
    missing = [key for key in required if key not in task]
    if missing:
        raise KeyError(f"Ready task is missing required keys: {missing}")

    query_x = np.asarray(task["query_x"], dtype=np.float32)
    query_actions = np.asarray(task["query_actions"], dtype=np.int64)
    query_static = task["query_static"]

    return query_x, query_actions, query_static


def input_scale_for_d_input(d_input):
    if d_input > D_INPUT_MAX:
        raise ValueError(f"Ready file d_input={d_input} exceeds D_INPUT_MAX={D_INPUT_MAX}")

    return float(np.sqrt(D_INPUT_MAX / d_input)) if d_input < D_INPUT_MAX else 1.0


def init_pfn_batch_tensors(batch_size, n_support, d_input, input_scale):
    return {
        "support_x": torch.zeros(batch_size, n_support, MAX_SEQ_LEN, D_INPUT_MAX),
        "support_actions": torch.zeros(batch_size, n_support, MAX_SEQ_LEN, dtype=torch.long),
        "support_anchor_y": torch.zeros(batch_size, n_support, N_SUPPORT_ANCHORS),
        "support_anchor_time": torch.ones(batch_size, n_support, N_SUPPORT_ANCHORS, dtype=torch.long),
        "support_pad_mask": torch.zeros(batch_size, n_support, dtype=torch.bool),
        "query_x": torch.zeros(batch_size, MAX_SEQ_LEN, D_INPUT_MAX),
        "query_actions": torch.zeros(batch_size, MAX_SEQ_LEN, dtype=torch.long),
        "input_scale": torch.full((batch_size,), float(input_scale), dtype=torch.float32),
        "d_input": torch.full((batch_size,), int(d_input), dtype=torch.long),
    }


def collate_ready_batch(ready_map, task_name, start, end):
    support_context = ready_map["support_context"]
    task = ready_map["tasks"][task_name]

    (
        support_x_np,
        support_actions_np,
        support_anchor_y_np,
        support_anchor_time_np,
        support_static_np,
        n_support,
        d_input,
    ) = support_context_arrays(support_context)

    query_x_np_all, query_actions_np_all, query_static_np_all = task_arrays(task)

    query_x_np = np.asarray(query_x_np_all[start:end], dtype=np.float32)
    query_actions_np = np.asarray(query_actions_np_all[start:end], dtype=np.int64)

    batch_size = int(query_x_np.shape[0])
    input_scale = input_scale_for_d_input(d_input)
    batch = init_pfn_batch_tensors(batch_size, n_support, d_input, input_scale)

    batch["support_x"][:, :, :, :d_input] = torch.from_numpy(support_x_np).float().unsqueeze(0) * input_scale
    batch["support_actions"][:, :, :] = torch.from_numpy(support_actions_np).long().unsqueeze(0)
    batch["support_anchor_y"][:, :, :] = torch.from_numpy(support_anchor_y_np).float().unsqueeze(0)
    batch["support_anchor_time"][:, :, :] = torch.from_numpy(support_anchor_time_np).long().unsqueeze(0)

    batch["query_x"][:, :, :d_input] = torch.from_numpy(query_x_np).float() * input_scale
    batch["query_actions"][:, :] = torch.from_numpy(query_actions_np).long()

    support_static_np = common.fixed_2d_float(support_static_np, rows=n_support, cols=D_STATIC_MAX)

    query_static_np = np.asarray(query_static_np_all[start:end], dtype=np.float32)
    query_static_np = common.fixed_2d_float(query_static_np, rows=batch_size, cols=D_STATIC_MAX)

    batch["support_static"] = torch.from_numpy(support_static_np).float().unsqueeze(0).expand(batch_size, -1, -1).clone()
    batch["query_static"] = torch.from_numpy(query_static_np).float()

    target_model_norm = torch.from_numpy(np.asarray(task["target_model_norm"][start:end], dtype=np.float32)).float()
    target_eval_norm = torch.from_numpy(np.asarray(task["target_eval_norm"][start:end], dtype=np.float32)).float()

    batch.update({
        "t_obs": torch.from_numpy(np.asarray(task["t_obs"][start:end], dtype=np.int64)).long(),
        "t_target": torch.from_numpy(np.asarray(task["t_target"][start:end], dtype=np.int64)).long(),
        "current_time": torch.from_numpy(np.asarray(task["current_time"][start:end], dtype=np.int64)).long(),
        "current_t": torch.from_numpy(np.asarray(task["current_time"][start:end], dtype=np.int64)).long(),
        "tau": torch.from_numpy(np.asarray(task["tau"][start:end], dtype=np.int64)).long(),
        "target_y_norm": target_model_norm,
        "target_model_norm": target_model_norm,
        "target_eval_norm": target_eval_norm,
        "oracle_Y": target_model_norm,
        "oracle_Y_final": target_model_norm,
    })

    return batch


def collate_support_calibration_batch(ready_map, fit_idx, pair_rows):
    support_context = ready_map["support_context"]

    (
        support_x_all,
        support_actions_all,
        support_anchor_y_all,
        support_anchor_time_all,
        support_static_all,
        n_support_total,
        d_input,
    ) = support_context_arrays(support_context)

    fit_idx = np.asarray(fit_idx, dtype=np.int64)
    query_idx = np.asarray([pair[0] for pair in pair_rows], dtype=np.int64)
    anchor_idx = np.asarray([pair[1] for pair in pair_rows], dtype=np.int64)

    batch_size = int(len(pair_rows))
    n_support = int(len(fit_idx))
    input_scale = input_scale_for_d_input(d_input)
    batch = init_pfn_batch_tensors(batch_size, n_support, d_input, input_scale)

    batch["support_x"][:, :, :, :d_input] = torch.from_numpy(support_x_all[fit_idx]).float().unsqueeze(0) * input_scale
    batch["support_actions"][:, :, :] = torch.from_numpy(support_actions_all[fit_idx]).long().unsqueeze(0)
    batch["support_anchor_y"][:, :, :] = torch.from_numpy(support_anchor_y_all[fit_idx]).float().unsqueeze(0)
    batch["support_anchor_time"][:, :, :] = torch.from_numpy(support_anchor_time_all[fit_idx]).long().unsqueeze(0)

    batch["query_x"][:, :, :d_input] = torch.from_numpy(support_x_all[query_idx]).float() * input_scale
    batch["query_actions"][:, :] = torch.from_numpy(support_actions_all[query_idx]).long()

    label_time_np = np.clip(
        support_anchor_time_all[query_idx, anchor_idx].astype(np.int64),
        1,
        MAX_TARGET_INDEX,
    )
    current_time_np = np.clip(label_time_np - 1, 0, MAX_INPUT_INDEX)
    target_norm_np = support_anchor_y_all[query_idx, anchor_idx].astype(np.float32)

    support_static_all = common.fixed_2d_float(support_static_all, rows=n_support_total, cols=D_STATIC_MAX)

    batch["support_static"] = torch.from_numpy(support_static_all[fit_idx]).float().unsqueeze(0).expand(batch_size, -1, -1).clone()
    batch["query_static"] = torch.from_numpy(support_static_all[query_idx]).float()

    batch.update({
        "current_time": torch.from_numpy(current_time_np).long(),
        "current_t": torch.from_numpy(current_time_np).long(),
        "t_obs": torch.from_numpy(current_time_np).long(),
        "t_target": torch.from_numpy(label_time_np).long(),
        "tau": torch.ones(batch_size, dtype=torch.long),
        "target_y_norm": torch.from_numpy(target_norm_np).float(),
        "oracle_Y": torch.from_numpy(target_norm_np).float(),
        "oracle_Y_final": torch.from_numpy(target_norm_np).float(),
    })

    return batch
