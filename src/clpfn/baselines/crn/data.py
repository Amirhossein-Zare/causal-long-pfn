from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from clpfn.baselines.common.training import masked_mse_loss, masked_multiclass_ce_loss
from clpfn.baselines.crn.config import MAX_TRAIN_ORIGINS, onehot_actions
from clpfn.evaluation.core import benchmark as common

class CRNEncoderSupportDataset(Dataset):
    def __init__(self, bundle, context_idx):
        idx = np.asarray(context_idx, dtype=np.int64)
        c = bundle["covariates"][idx]
        y = bundle["y_norm_clip"][idx]
        a = bundle["actions"][idx]
        s = bundle["static"][idx]
        lengths = bundle["sequence_lengths"][idx]

        seq_len = min(c.shape[1], y.shape[1], a.shape[1], common.MAX_SEQ_LEN)
        c = c[:, :seq_len, :]
        y = y[:, :seq_len]
        a = a[:, :seq_len]
        t_enc = max(1, seq_len - 1)

        actions = a[:, :t_enc]
        prev_actions = np.zeros_like(actions)
        if t_enc > 1:
            prev_actions[:, 1:] = actions[:, :-1]

        self.prev_treatments = onehot_actions(prev_actions).astype(np.float32)
        self.current_treatments = onehot_actions(actions).astype(np.float32)
        self.current_treatment_idx = actions.astype(np.int64)
        self.vitals = c[:, :t_enc, :].astype(np.float32)
        self.prev_outputs = y[:, :t_enc, None].astype(np.float32)
        self.outputs = y[:, 1:t_enc + 1, None].astype(np.float32)
        self.static_features = s.astype(np.float32)

        t_grid = np.arange(t_enc)[None, :]
        active = ((t_grid + 1) < lengths[:, None]).astype(np.float32)[:, :, None]
        active *= np.isfinite(self.outputs).astype(np.float32)
        keep = active.sum(axis=(1, 2)) > 0

        self.prev_treatments = self.prev_treatments[keep]
        self.current_treatments = self.current_treatments[keep]
        self.current_treatment_idx = self.current_treatment_idx[keep]
        self.vitals = self.vitals[keep]
        self.prev_outputs = self.prev_outputs[keep]
        self.outputs = self.outputs[keep]
        self.static_features = self.static_features[keep]
        self.active_entries = active[keep].astype(np.float32)
        self.raw = {
            "idx": idx[keep],
            "T": int(seq_len),
            "T_enc": int(t_enc),
            "Yc": y[keep].astype(np.float32),
            "A": a[keep].astype(np.int64),
            "C": c[keep].astype(np.float32),
            "static": s[keep].astype(np.float32),
            "lengths": lengths[keep].astype(np.int64),
        }
        if self.vitals.shape[0] == 0:
            raise ValueError("No active support sequences available for CRN encoder training.")

    def __len__(self):
        return int(self.vitals.shape[0])

    def __getitem__(self, idx):
        return {
            "prev_treatments": torch.from_numpy(self.prev_treatments[idx]),
            "current_treatments": torch.from_numpy(self.current_treatments[idx]),
            "current_treatment_idx": torch.from_numpy(self.current_treatment_idx[idx]),
            "vitals": torch.from_numpy(self.vitals[idx]),
            "prev_outputs": torch.from_numpy(self.prev_outputs[idx]),
            "outputs": torch.from_numpy(self.outputs[idx]),
            "static_features": torch.from_numpy(self.static_features[idx]),
            "active_entries": torch.from_numpy(self.active_entries[idx]),
        }


def move_batch_to_device(batch):
    return common.move_tensor_batch_to_device(batch, float_tensors=True, long_keys=("current_treatment_idx",))


def masked_mse(pred, target, active):
    return masked_mse_loss(pred, target, active)


def masked_ce(logits, target_idx, active):
    return masked_multiclass_ce_loss(logits, target_idx, active)


def crn_loss(treatment_pred, outcome_pred, batch, hparams):
    outcome_loss = masked_mse(outcome_pred, batch["outputs"], batch["active_entries"])
    treatment_loss = masked_ce(treatment_pred, batch["current_treatment_idx"], batch["active_entries"])
    return outcome_loss + float(hparams["treatment_loss_weight"]) * treatment_loss, outcome_loss, treatment_loss


@torch.no_grad()
def get_encoder_br_sequence(encoder, encoder_ds):
    encoder.eval()
    batch = {
        "prev_treatments": torch.from_numpy(encoder_ds.prev_treatments).to(common.DEVICE),
        "current_treatments": torch.from_numpy(encoder_ds.current_treatments).to(common.DEVICE),
        "vitals": torch.from_numpy(encoder_ds.vitals).to(common.DEVICE),
        "prev_outputs": torch.from_numpy(encoder_ds.prev_outputs).to(common.DEVICE),
        "static_features": torch.from_numpy(encoder_ds.static_features).to(common.DEVICE),
    }
    _, _, br = encoder(batch)
    return br.detach().float().cpu().numpy().astype(np.float32)


def sample_decoder_origins(raw_train, seed, max_origins=MAX_TRAIN_ORIGINS):
    rng = np.random.default_rng(int(seed))
    lengths = raw_train["lengths"]
    seq_len = int(raw_train["T"])
    pairs = []
    for local_i, length_i in enumerate(lengths):
        max_origin = min(int(length_i) - 2, seq_len - 2, common.MAX_INPUT_INDEX)
        if max_origin < 1:
            continue
        lo = common.MIN_T_OBS if max_origin >= common.MIN_T_OBS else 1
        for origin in range(lo, max_origin + 1):
            pairs.append((local_i, origin))
    if not pairs:
        for local_i, length_i in enumerate(lengths):
            max_origin = min(int(length_i) - 2, seq_len - 2, common.MAX_INPUT_INDEX)
            for origin in range(1, max_origin + 1):
                pairs.append((local_i, origin))
    if not pairs:
        raise ValueError("No valid decoder origins in support context.")
    pairs = np.asarray(pairs, dtype=np.int64)
    if max_origins is not None and len(pairs) > int(max_origins):
        pairs = pairs[rng.choice(len(pairs), size=int(max_origins), replace=False)]
    return pairs


class CRNDecoderOriginDataset(Dataset):
    def __init__(self, raw_train, br_all, origins):
        horizon = common.TAU_MAX
        n = origins.shape[0]
        a = raw_train["A"]
        y = raw_train["Yc"]
        lengths = raw_train["lengths"]
        s = raw_train["static"]

        init_state = np.zeros((n, br_all.shape[-1]), dtype=np.float32)
        prev_actions = np.zeros((n, horizon), dtype=np.int64)
        curr_actions = np.zeros((n, horizon), dtype=np.int64)
        prev_outputs = np.zeros((n, horizon, 1), dtype=np.float32)
        outputs = np.zeros((n, horizon, 1), dtype=np.float32)
        active = np.zeros((n, horizon, 1), dtype=np.float32)
        static_features = np.zeros((n, common.D_STATIC_MAX), dtype=np.float32)

        for j, (i, origin) in enumerate(origins):
            i = int(i)
            origin = int(origin)
            init_state[j] = br_all[i, origin]
            static_features[j] = s[i]
            for h in range(horizon):
                tt = origin + h
                prev_actions[j, h] = int(a[i, tt - 1]) if tt > 0 and tt - 1 < a.shape[1] else 0
                curr_actions[j, h] = int(a[i, tt]) if tt < a.shape[1] else 0
                if tt < y.shape[1]:
                    prev_outputs[j, h, 0] = float(y[i, tt])
                target_t = tt + 1
                if target_t < y.shape[1] and target_t < int(lengths[i]) and np.isfinite(y[i, target_t]):
                    outputs[j, h, 0] = float(y[i, target_t])
                    active[j, h, 0] = 1.0

        keep = active.sum(axis=(1, 2)) > 0
        self.init_state = init_state[keep]
        self.prev_treatments = onehot_actions(prev_actions[keep]).astype(np.float32)
        self.current_treatments = onehot_actions(curr_actions[keep]).astype(np.float32)
        self.current_treatment_idx = curr_actions[keep].astype(np.int64)
        self.prev_outputs = prev_outputs[keep].astype(np.float32)
        self.outputs = outputs[keep].astype(np.float32)
        self.active_entries = active[keep].astype(np.float32)
        self.static_features = static_features[keep].astype(np.float32)
        if self.init_state.shape[0] == 0:
            raise ValueError("No active decoder origins available for CRN decoder training.")

    def __len__(self):
        return int(self.init_state.shape[0])

    def __getitem__(self, idx):
        return {
            "init_state": torch.from_numpy(self.init_state[idx]),
            "prev_treatments": torch.from_numpy(self.prev_treatments[idx]),
            "current_treatments": torch.from_numpy(self.current_treatments[idx]),
            "current_treatment_idx": torch.from_numpy(self.current_treatment_idx[idx]),
            "prev_outputs": torch.from_numpy(self.prev_outputs[idx]),
            "outputs": torch.from_numpy(self.outputs[idx]),
            "active_entries": torch.from_numpy(self.active_entries[idx]),
            "static_features": torch.from_numpy(self.static_features[idx]),
        }
