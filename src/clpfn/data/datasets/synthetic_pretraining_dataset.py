import math
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
    def __init__(self, base_seed: int = SEED, start_index: int = 0):
        self.base_seed = base_seed
        self.start_index = start_index

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1
        episode_index = self.start_index + worker_id

        while True:
            seed_sequence = np.random.SeedSequence([int(self.base_seed), int(episode_index)])
            generator = TSCMEpisodeGenerator(np.random.default_rng(seed_sequence))

            yield generator.sample_episode()
            episode_index += num_workers


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
    t_obs = torch.zeros(batch_size, dtype=torch.long)

    input_scale = torch.ones(batch_size)
    d_input = torch.ones(batch_size, dtype=torch.long)

    for batch_idx, sample in enumerate(batch):
        d_sample = int(sample["d_input"])
        n_support = int(sample["n_support"])

        if not 1 <= d_sample <= max_d:
            raise ValueError(f"Sample input dimension d={d_sample} is outside [1, {max_d}]")

        scale = math.sqrt(max_d / d_sample) if d_sample < max_d else 1.0

        input_scale[batch_idx] = float(scale)
        d_input[batch_idx] = int(d_sample)

        support_x_np = np.asarray(sample["support_x"], dtype=np.float32)
        query_x_np = np.asarray(sample["query_x"], dtype=np.float32)
        support_actions_np = np.asarray(sample["support_actions"], dtype=np.int64)
        query_actions_np = np.asarray(sample["query_actions"], dtype=np.int64)
        anchor_y = np.asarray(sample["support_anchor_y"], dtype=np.float32)
        anchor_time = np.asarray(sample["support_anchor_time"], dtype=np.int64)
        support_static_np = np.asarray(sample["support_static"], dtype=np.float32)
        query_static_np = np.asarray(sample["query_static"], dtype=np.float32)

        expected_shapes = {
            "support_x": (n_support, seq_len, d_sample),
            "query_x": (seq_len, d_sample),
            "support_actions": (n_support, seq_len),
            "query_actions": (seq_len,),
            "support_anchor_y": (n_support, N_SUPPORT_ANCHORS),
            "support_anchor_time": (n_support, N_SUPPORT_ANCHORS),
            "support_static": (n_support, D_STATIC_MAX),
            "query_static": (D_STATIC_MAX,),
        }
        arrays = {
            "support_x": support_x_np,
            "query_x": query_x_np,
            "support_actions": support_actions_np,
            "query_actions": query_actions_np,
            "support_anchor_y": anchor_y,
            "support_anchor_time": anchor_time,
            "support_static": support_static_np,
            "query_static": query_static_np,
        }
        invalid_shapes = {
            key: (value.shape, expected_shapes[key])
            for key, value in arrays.items()
            if value.shape != expected_shapes[key]
        }
        if invalid_shapes:
            raise ValueError(f"Synthetic episode has noncanonical tensor shapes: {invalid_shapes}")
        if np.any((anchor_time < 1) | (anchor_time > MAX_SEQ_LEN)):
            raise ValueError("Synthetic episode support-anchor times are outside the canonical sequence.")

        support_x[batch_idx, :n_support, :, :d_sample] = torch.from_numpy(support_x_np).float() * scale
        support_actions[batch_idx, :n_support, :] = torch.from_numpy(support_actions_np).long()

        support_anchor_y[batch_idx, :n_support, :] = torch.from_numpy(anchor_y)
        support_anchor_time[batch_idx, :n_support, :] = torch.from_numpy(anchor_time).long()
        support_pad_mask[batch_idx, :n_support] = False

        query_x[batch_idx, :, :d_sample] = torch.from_numpy(query_x_np).float() * scale
        query_actions[batch_idx] = torch.from_numpy(query_actions_np).long()

        support_static[batch_idx, :n_support] = torch.from_numpy(support_static_np)
        query_static[batch_idx] = torch.from_numpy(query_static_np)

        current_time_value = int(sample["current_time"])
        t_obs_value = int(sample["t_obs"])
        if not 0 <= current_time_value < MAX_SEQ_LEN:
            raise ValueError(f"Synthetic episode current_time={current_time_value} is outside the sequence.")
        if not 0 <= t_obs_value < MAX_SEQ_LEN:
            raise ValueError(f"Synthetic episode t_obs={t_obs_value} is outside the sequence.")
        current_time[batch_idx] = current_time_value
        t_obs[batch_idx] = t_obs_value
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
        "t_obs": t_obs,

        "input_scale": input_scale,
        "d_input": d_input,
    }
