import numpy as np
import torch

from clpfn.config.defaults import (
    D_INPUT_MAX,
    D_STATIC_MAX,
    MAX_SEQ_LEN,
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
        "target_eval_norm",
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
    if not 1 <= d_input <= D_INPUT_MAX:
        raise ValueError(f"Ready file d_input={d_input} is outside [1, {D_INPUT_MAX}]")

    return float(np.sqrt(D_INPUT_MAX / d_input)) if d_input < D_INPUT_MAX else 1.0


def resolve_n_support_anchors(n_support_anchors=None):
    """Resolve the number of support anchors a checkpoint expects at inference."""
    resolved = int(N_SUPPORT_ANCHORS if n_support_anchors is None else n_support_anchors)
    if not 1 <= resolved <= int(N_SUPPORT_ANCHORS):
        raise ValueError(
            f"Support-anchor count {resolved} is outside [1, {int(N_SUPPORT_ANCHORS)}]; "
            "ready files cannot supply more anchors than they were built with."
        )
    return resolved


def init_pfn_batch_tensors(
    batch_size,
    n_support,
    d_input,
    input_scale,
    n_support_anchors=None,
):
    n_anchors = resolve_n_support_anchors(n_support_anchors)
    return {
        "support_x": torch.zeros(batch_size, n_support, MAX_SEQ_LEN, D_INPUT_MAX),
        "support_actions": torch.zeros(batch_size, n_support, MAX_SEQ_LEN, dtype=torch.long),
        "support_anchor_y": torch.zeros(batch_size, n_support, n_anchors),
        "support_anchor_time": torch.ones(batch_size, n_support, n_anchors, dtype=torch.long),
        "support_pad_mask": torch.zeros(batch_size, n_support, dtype=torch.bool),
        "query_x": torch.zeros(batch_size, MAX_SEQ_LEN, D_INPUT_MAX),
        "query_actions": torch.zeros(batch_size, MAX_SEQ_LEN, dtype=torch.long),
        "input_scale": torch.full((batch_size,), float(input_scale), dtype=torch.float32),
        "d_input": torch.full((batch_size,), int(d_input), dtype=torch.long),
    }


def collate_ready_batch(ready_map, task_name, start, end, n_support_anchors=None):
    n_anchors = resolve_n_support_anchors(n_support_anchors)
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

    expected_support_shapes = {
        "support_x": (n_support, MAX_SEQ_LEN, d_input),
        "support_actions": (n_support, MAX_SEQ_LEN),
        "support_static": (n_support, D_STATIC_MAX),
    }
    support_shapes = {
        "support_x": support_x_np.shape,
        "support_actions": support_actions_np.shape,
        "support_static": np.asarray(support_static_np).shape,
    }
    invalid_support_shapes = {
        key: (support_shapes[key], expected)
        for key, expected in expected_support_shapes.items()
        if support_shapes[key] != expected
    }
    if invalid_support_shapes:
        raise ValueError(f"Ready support context has noncanonical tensor shapes: {invalid_support_shapes}")

    expected_anchor_shape = (n_support, int(N_SUPPORT_ANCHORS))
    if support_anchor_y_np.shape != expected_anchor_shape:
        raise ValueError(
            f"Ready support-anchor labels have shape {support_anchor_y_np.shape}; "
            f"expected {expected_anchor_shape}."
        )
    if support_anchor_time_np.shape != expected_anchor_shape:
        raise ValueError(
            f"Ready support-anchor times have shape {support_anchor_time_np.shape}; "
            f"expected {expected_anchor_shape}."
        )

    support_anchor_y_np = support_anchor_y_np[:, :n_anchors]
    support_anchor_time_np = support_anchor_time_np[:, :n_anchors]

    batch_size = int(query_x_np.shape[0])
    expected_query_shapes = {
        "query_x": (batch_size, MAX_SEQ_LEN, d_input),
        "query_actions": (batch_size, MAX_SEQ_LEN),
        "query_static": (batch_size, D_STATIC_MAX),
    }
    query_shapes = {
        "query_x": query_x_np.shape,
        "query_actions": query_actions_np.shape,
        "query_static": np.asarray(query_static_np_all[start:end]).shape,
    }
    invalid_query_shapes = {
        key: (query_shapes[key], expected)
        for key, expected in expected_query_shapes.items()
        if query_shapes[key] != expected
    }
    if invalid_query_shapes:
        raise ValueError(f"Ready query task has noncanonical tensor shapes: {invalid_query_shapes}")

    input_scale = input_scale_for_d_input(d_input)
    batch = init_pfn_batch_tensors(
        batch_size,
        n_support,
        d_input,
        input_scale,
        n_support_anchors=n_anchors,
    )

    batch["support_x"][:, :, :, :d_input] = torch.from_numpy(support_x_np).float().unsqueeze(0) * input_scale
    batch["support_actions"][:, :, :] = torch.from_numpy(support_actions_np).long().unsqueeze(0)
    batch["support_anchor_y"][:, :, :] = torch.from_numpy(support_anchor_y_np).float().unsqueeze(0)
    batch["support_anchor_time"][:, :, :] = torch.from_numpy(support_anchor_time_np).long().unsqueeze(0)

    batch["query_x"][:, :, :d_input] = torch.from_numpy(query_x_np).float() * input_scale
    batch["query_actions"][:, :] = torch.from_numpy(query_actions_np).long()

    query_static_np = np.asarray(query_static_np_all[start:end], dtype=np.float32)

    batch["support_static"] = torch.from_numpy(np.asarray(support_static_np, dtype=np.float32)).unsqueeze(0).expand(batch_size, -1, -1).clone()
    batch["query_static"] = torch.from_numpy(query_static_np).float()

    target_eval_norm = torch.from_numpy(np.asarray(task["target_eval_norm"][start:end], dtype=np.float32)).float()

    batch.update({
        "t_obs": torch.from_numpy(np.asarray(task["t_obs"][start:end], dtype=np.int64)).long(),
        "t_target": torch.from_numpy(np.asarray(task["t_target"][start:end], dtype=np.int64)).long(),
        "current_time": torch.from_numpy(np.asarray(task["current_time"][start:end], dtype=np.int64)).long(),
        "tau": torch.from_numpy(np.asarray(task["tau"][start:end], dtype=np.int64)).long(),
        "target_eval_norm": target_eval_norm,
    })

    return batch
