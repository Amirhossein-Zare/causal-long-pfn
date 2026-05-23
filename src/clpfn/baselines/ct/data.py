from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from clpfn.baselines.common.training import move_float_batch_to_device
from clpfn.evaluation.core import benchmark as common

class CTSupportDataset(Dataset):
    def __init__(self, bundle, context_idx):
        idx = np.asarray(context_idx, dtype=np.int64)

        C = bundle["covariates"][idx]
        Yc = bundle["y_norm_clip"][idx]
        A = bundle["actions"][idx]
        S = bundle["static"][idx]
        Lseq = bundle["sequence_lengths"][idx]

        T = min(C.shape[1], Yc.shape[1], A.shape[1], common.MAX_SEQ_LEN)
        T_train = max(1, T - 1)

        actions = A[:, :T_train].astype(np.int64)

        prev_actions = np.zeros_like(actions)
        if T_train > 1:
            prev_actions[:, 1:] = actions[:, :-1]

        self.prev_treatments = common.action_onehot_2d(prev_actions, common.N_ACTIONS).astype(np.float32)
        self.current_treatments = common.action_onehot_2d(actions, common.N_ACTIONS).astype(np.float32)
        self.vitals = C[:, :T_train, :].astype(np.float32)
        self.prev_outputs = Yc[:, :T_train, None].astype(np.float32)
        self.outputs = Yc[:, 1:T_train + 1, None].astype(np.float32)
        self.static_features = S.astype(np.float32)

        t_grid = np.arange(T_train)[None, :]
        active = ((t_grid + 1) < Lseq[:, None]).astype(np.float32)[:, :, None]
        active *= np.isfinite(self.outputs).astype(np.float32)

        keep = active.sum(axis=(1, 2)) > 0

        self.prev_treatments = self.prev_treatments[keep]
        self.current_treatments = self.current_treatments[keep]
        self.vitals = self.vitals[keep]
        self.prev_outputs = self.prev_outputs[keep]
        self.outputs = self.outputs[keep]
        self.static_features = self.static_features[keep]
        self.active_entries = active[keep].astype(np.float32)

        if self.prev_treatments.shape[0] == 0:
            raise ValueError("No active support sequences available for CT training.")

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
        }


def move_batch_to_device(batch):
    return move_float_batch_to_device(batch)
