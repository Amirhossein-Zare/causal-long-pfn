import math
import time

import numpy as np
import torch
from torch.utils.data import IterableDataset

from clpfn.config.defaults import (
    D_INPUT_MAX,
    D_STATIC_MAX,
    MAX_SEQ_LEN,
    N_SUPPORT_ANCHORS,
    SEED,
)
from clpfn.data.priors.tscm_episode_generator import TSCMEpisodeGenerator


class OnTheFlyEpisodeDataset(IterableDataset):
    def __init__(self, base_seed: int = SEED):
        self.base_seed = base_seed

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        seed = self.base_seed + (worker.id if worker else 0) + int(time.time() * 1e6) % 100_000

        rng = np.random.default_rng(seed)
        generator = TSCMEpisodeGenerator(rng)

        while True:
            yield generator.sample_episode()


def _as_support_anchor_arrays(sample: dict) -> tuple[np.ndarray, np.ndarray]:
    n_support = int(sample["n_support"])

    anchor_y = np.asarray(sample["support_anchor_y"], dtype=np.float32)

    if anchor_y.ndim == 0:
        anchor_y = np.full((n_support, 1), float(anchor_y), dtype=np.float32)
    elif anchor_y.ndim == 1:
        if anchor_y.size == n_support:
            anchor_y = anchor_y.reshape(n_support, 1)
        elif anchor_y.size % n_support == 0:
            anchor_y = anchor_y.reshape(n_support, anchor_y.size // n_support)
        else:
            fixed = np.zeros(n_support, dtype=np.float32)
            n_copy = min(n_support, anchor_y.size)
            fixed[:n_copy] = anchor_y[:n_copy]
            anchor_y = fixed.reshape(n_support, 1)
    elif anchor_y.ndim == 2:
        if anchor_y.shape[0] != n_support and anchor_y.size % n_support == 0:
            anchor_y = anchor_y.reshape(n_support, anchor_y.size // n_support)
    else:
        anchor_y = anchor_y.reshape(n_support, -1)

    if anchor_y.shape[1] < N_SUPPORT_ANCHORS:
        pad = np.repeat(anchor_y[:, -1:], N_SUPPORT_ANCHORS - anchor_y.shape[1], axis=1)
        anchor_y = np.concatenate([anchor_y, pad], axis=1)
    elif anchor_y.shape[1] > N_SUPPORT_ANCHORS:
        anchor_y = anchor_y[:, :N_SUPPORT_ANCHORS]

    anchor_time = np.asarray(
        sample.get("support_anchor_time", np.full((n_support, 1), MAX_SEQ_LEN)),
        dtype=np.int64,
    )

    if anchor_time.ndim == 0:
        anchor_time = np.full((n_support, 1), int(anchor_time), dtype=np.int64)
    elif anchor_time.ndim == 1:
        if anchor_time.size == n_support:
            anchor_time = anchor_time.reshape(n_support, 1)
        elif anchor_time.size % n_support == 0:
            anchor_time = anchor_time.reshape(n_support, anchor_time.size // n_support)
        else:
            fixed = np.full(n_support, MAX_SEQ_LEN, dtype=np.int64)
            n_copy = min(n_support, anchor_time.size)
            fixed[:n_copy] = anchor_time[:n_copy]
            anchor_time = fixed.reshape(n_support, 1)
    elif anchor_time.ndim == 2:
        if anchor_time.shape[0] != n_support and anchor_time.size % n_support == 0:
            anchor_time = anchor_time.reshape(n_support, anchor_time.size // n_support)
    else:
        anchor_time = anchor_time.reshape(n_support, -1)

    if anchor_time.shape[1] < N_SUPPORT_ANCHORS:
        pad = np.repeat(anchor_time[:, -1:], N_SUPPORT_ANCHORS - anchor_time.shape[1], axis=1)
        anchor_time = np.concatenate([anchor_time, pad], axis=1)
    elif anchor_time.shape[1] > N_SUPPORT_ANCHORS:
        anchor_time = anchor_time[:, :N_SUPPORT_ANCHORS]

    anchor_time = np.clip(anchor_time, 1, MAX_SEQ_LEN)

    return anchor_y.astype(np.float32), anchor_time.astype(np.int64)


def collate_episode_batch(batch: list[dict]) -> dict[str, torch.Tensor]:
    max_d = D_INPUT_MAX
    max_n_support = max(sample["n_support"] for sample in batch)
    batch_size, seq_len = len(batch), MAX_SEQ_LEN

    support_x = torch.zeros(batch_size, max_n_support, seq_len, max_d)
    support_actions = torch.zeros(batch_size, max_n_support, seq_len, dtype=torch.long)
    support_anchor_y = torch.zeros(batch_size, max_n_support, N_SUPPORT_ANCHORS)
    support_anchor_time = torch.ones(batch_size, max_n_support, N_SUPPORT_ANCHORS, dtype=torch.long)
    support_pad_mask = torch.ones(batch_size, max_n_support, dtype=torch.bool)

    query_x = torch.zeros(batch_size, seq_len, max_d)
    query_actions = torch.zeros(batch_size, seq_len, dtype=torch.long)

    support_static = torch.zeros(batch_size, max_n_support, D_STATIC_MAX)
    query_static = torch.zeros(batch_size, D_STATIC_MAX)

    target_y_norm = torch.zeros(batch_size)
    current_time = torch.zeros(batch_size, dtype=torch.long)

    input_scale = torch.ones(batch_size)
    d_input = torch.ones(batch_size, dtype=torch.long)

    for batch_idx, sample in enumerate(batch):
        d_sample = int(sample["d_input"])
        n_support = int(sample["n_support"])

        if d_sample > max_d:
            raise ValueError(f"Sample input dimension d={d_sample} exceeds D_INPUT_MAX={max_d}")

        scale = math.sqrt(max_d / d_sample) if d_sample < max_d else 1.0

        input_scale[batch_idx] = float(scale)
        d_input[batch_idx] = int(d_sample)

        support_x_np = np.asarray(sample["support_x"], dtype=np.float32).reshape(n_support, seq_len, d_sample)
        query_x_np = np.asarray(sample["query_x"], dtype=np.float32).reshape(seq_len, d_sample)

        support_actions_np = np.asarray(sample["support_actions"], dtype=np.int64).reshape(n_support, seq_len)
        query_actions_np = np.asarray(sample["query_actions"], dtype=np.int64).reshape(seq_len)

        support_x[batch_idx, :n_support, :, :d_sample] = torch.from_numpy(support_x_np).float() * scale
        support_actions[batch_idx, :n_support, :] = torch.from_numpy(support_actions_np).long()

        anchor_y, anchor_time = _as_support_anchor_arrays(sample)

        support_anchor_y[batch_idx, :n_support, :] = torch.from_numpy(anchor_y)
        support_anchor_time[batch_idx, :n_support, :] = torch.from_numpy(anchor_time).long()
        support_pad_mask[batch_idx, :n_support] = False

        query_x[batch_idx, :, :d_sample] = torch.from_numpy(query_x_np).float() * scale
        query_actions[batch_idx] = torch.from_numpy(query_actions_np).long()

        if "support_static" in sample:
            support_static_np = np.asarray(sample["support_static"], dtype=np.float32)

            if support_static_np.ndim == 1:
                support_static_np = support_static_np.reshape(n_support, -1)

            static_dim = min(support_static_np.shape[-1], D_STATIC_MAX)
            support_static[batch_idx, :n_support, :static_dim] = torch.from_numpy(
                support_static_np[:, :static_dim]
            )

        if "query_static" in sample:
            query_static_np = np.asarray(sample["query_static"], dtype=np.float32)
            static_dim = min(query_static_np.shape[-1], D_STATIC_MAX)
            query_static[batch_idx, :static_dim] = torch.from_numpy(query_static_np[:static_dim])

        current_time_value = int(sample["current_time"])
        current_time[batch_idx] = max(0, min(MAX_SEQ_LEN - 1, current_time_value))
        target_y_norm[batch_idx] = float(sample["target_y_norm"])

    return {
        "support_x": support_x,
        "support_actions": support_actions,
        "support_anchor_y": support_anchor_y,
        "support_anchor_time": support_anchor_time,
        "support_pad_mask": support_pad_mask,

        "query_x": query_x,
        "query_actions": query_actions,

        "support_static": support_static,
        "query_static": query_static,

        "target_y_norm": target_y_norm,
        "current_time": current_time,

        "input_scale": input_scale,
        "d_input": d_input,
    }
