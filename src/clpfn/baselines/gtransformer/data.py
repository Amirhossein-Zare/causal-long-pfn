from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from clpfn.evaluation.core import benchmark as common

class GTSupportDataset(Dataset):
    def __init__(self, bundle, context_idx):
        idx = np.asarray(context_idx, dtype=np.int64)

        C = bundle["covariates"][idx]
        Yc = bundle["y_norm_clip"][idx]
        Yraw = bundle["y_raw"][idx]
        A = bundle["actions"][idx]
        S = bundle["static"][idx]
        L = bundle["sequence_lengths"][idx]

        T = min(C.shape[1], Yc.shape[1], A.shape[1], common.MAX_SEQ_LEN)
        T_train = max(1, T - 1)

        C = C[:, :T_train, :].astype(np.float32)
        prev_y = Yc[:, :T_train, None].astype(np.float32)
        y_next = Yc[:, 1:T_train + 1, None].astype(np.float32)

        current_actions = A[:, :T_train].astype(np.int64)
        prev_actions = np.zeros_like(current_actions)

        if T_train > 1:
            prev_actions[:, 1:] = current_actions[:, :-1]

        t_grid = np.arange(T_train)[None, :]
        active = ((t_grid + 1) < L[:, None]).astype(np.float32)
        active *= np.isfinite(Yraw[:, 1:T_train + 1]).astype(np.float32)
        active = active[:, :, None]

        keep = active.sum(axis=(1, 2)) > 0

        self.prev_treatments = common.action_onehot_2d(prev_actions[keep]).astype(np.float32)
        self.current_treatments = common.action_onehot_2d(current_actions[keep]).astype(np.float32)
        self.vitals = C[keep].astype(np.float32)
        self.prev_outputs = prev_y[keep].astype(np.float32)
        self.outputs = y_next[keep].astype(np.float32)
        self.static_features = S[keep].astype(np.float32)
        self.active_entries = active[keep].astype(np.float32)

        self.sequence_lengths = np.full((self.prev_treatments.shape[0],), T_train, dtype=np.int64)

        if self.prev_treatments.shape[0] == 0:
            raise ValueError("No active support sequences available for GT training.")

    def __len__(self):
        return int(self.prev_treatments.shape[0])

    def __getitem__(self, idx):
        return {
            "prev_treatments": torch.from_numpy(self.prev_treatments[idx]),
            "current_treatments": torch.from_numpy(self.current_treatments[idx]),
            "vitals": torch.from_numpy(self.vitals[idx]),
            "prev_outputs": torch.from_numpy(self.prev_outputs[idx]),
            "outputs": torch.from_numpy(self.outputs[idx]),
            "static_features": torch.from_numpy(self.static_features[idx]),
            "active_entries": torch.from_numpy(self.active_entries[idx]),
            "sequence_lengths": torch.tensor(self.sequence_lengths[idx], dtype=torch.long),
        }


def move_batch_to_device(batch):
    return common.move_tensor_batch_to_device(batch, float_tensors=True, long_keys=("sequence_lengths",))
