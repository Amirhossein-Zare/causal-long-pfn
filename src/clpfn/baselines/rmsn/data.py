from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from clpfn.baselines.common.training import move_float_batch_to_device
from clpfn.baselines.rmsn.config import MAX_TRAIN_ORIGINS, action4_to_bits, clip_normalize_weights
from clpfn.evaluation.core import benchmark as common

class RMSNPropensityDataset(Dataset):
    def __init__(self, bundle, context_idx):
        idx = np.asarray(context_idx, dtype=np.int64)

        C = bundle["covariates"][idx]
        Yc = bundle["y_norm_clip"][idx]
        A = bundle["actions"][idx]
        S = bundle["static"][idx]
        L = bundle["sequence_lengths"][idx]

        T = min(C.shape[1], Yc.shape[1], A.shape[1], common.MAX_SEQ_LEN)
        C = C[:, :T, :]
        Yc = Yc[:, :T]
        A = A[:, :T]

        T_use = max(1, T - 1)

        current_bits = action4_to_bits(A[:, :T_use])
        prev_bits = np.zeros_like(current_bits)
        if T_use > 1:
            prev_bits[:, 1:, :] = current_bits[:, :-1, :]

        self.prev_treatments = prev_bits.astype(np.float32)
        self.current_treatments = current_bits.astype(np.float32)
        self.vitals = C[:, :T_use, :].astype(np.float32)
        self.prev_outputs = Yc[:, :T_use, None].astype(np.float32)
        self.static_features = S.astype(np.float32)

        t_grid = np.arange(T_use)[None, :]
        active = ((t_grid + 1) < L[:, None]).astype(np.float32)[:, :, None]
        active *= np.isfinite(self.prev_outputs).astype(np.float32)

        keep = active.sum(axis=(1, 2)) > 0

        self.prev_treatments = self.prev_treatments[keep]
        self.current_treatments = self.current_treatments[keep]
        self.vitals = self.vitals[keep]
        self.prev_outputs = self.prev_outputs[keep]
        self.static_features = self.static_features[keep]
        self.active_entries = active[keep].astype(np.float32)

        if self.current_treatments.shape[0] == 0:
            raise ValueError("No active support sequences available for RMSN propensity training.")

    def __len__(self):
        return int(self.current_treatments.shape[0])

    def __getitem__(self, idx):
        return {
            "prev_treatments": torch.from_numpy(self.prev_treatments[idx]),
            "current_treatments": torch.from_numpy(self.current_treatments[idx]),
            "vitals": torch.from_numpy(self.vitals[idx]),
            "prev_outputs": torch.from_numpy(self.prev_outputs[idx]),
            "static_features": torch.from_numpy(self.static_features[idx]),
            "active_entries": torch.from_numpy(self.active_entries[idx]),
        }


def make_encoder_full_arrays(bundle, context_idx):
    idx = np.asarray(context_idx, dtype=np.int64)

    C = bundle["covariates"][idx]
    Yc = bundle["y_norm_clip"][idx]
    A = bundle["actions"][idx]
    S = bundle["static"][idx]
    L = bundle["sequence_lengths"][idx]

    T = min(C.shape[1], Yc.shape[1], A.shape[1], common.MAX_SEQ_LEN)
    C = C[:, :T, :]
    Yc = Yc[:, :T]
    A = A[:, :T]

    T_enc = max(1, T - 1)

    current_bits = action4_to_bits(A[:, :T_enc])
    prev_bits = np.zeros_like(current_bits)
    if T_enc > 1:
        prev_bits[:, 1:, :] = current_bits[:, :-1, :]

    t_grid = np.arange(T_enc)[None, :]
    active = ((t_grid + 1) < L[:, None]).astype(np.float32)[:, :, None]
    active *= np.isfinite(Yc[:, 1:T_enc + 1, None]).astype(np.float32)

    return {
        "idx": idx,
        "T": int(T),
        "T_enc": int(T_enc),
        "vitals": C[:, :T_enc, :].astype(np.float32),
        "prev_outputs": Yc[:, :T_enc, None].astype(np.float32),
        "current_treatments": current_bits.astype(np.float32),
        "prev_treatments": prev_bits.astype(np.float32),
        "outputs": Yc[:, 1:T_enc + 1, None].astype(np.float32),
        "static_features": S.astype(np.float32),
        "active_entries": active.astype(np.float32),
        "lengths": L.astype(np.int64),
        "Yc": Yc.astype(np.float32),
        "A": A.astype(np.int64),
        "C": C.astype(np.float32),
    }


class RMSNEncoderSupportDataset(Dataset):
    def __init__(self, train_data, sw_tilde_enc):
        self.vitals = train_data["vitals"].astype(np.float32)
        self.prev_outputs = train_data["prev_outputs"].astype(np.float32)
        self.current_treatments = train_data["current_treatments"].astype(np.float32)
        self.outputs = train_data["outputs"].astype(np.float32)
        self.static_features = train_data["static_features"].astype(np.float32)
        self.active_entries = train_data["active_entries"].astype(np.float32)
        self.sw_tilde_enc = sw_tilde_enc.astype(np.float32)

        keep = self.active_entries.sum(axis=(1, 2)) > 0

        self.vitals = self.vitals[keep]
        self.prev_outputs = self.prev_outputs[keep]
        self.current_treatments = self.current_treatments[keep]
        self.outputs = self.outputs[keep]
        self.static_features = self.static_features[keep]
        self.active_entries = self.active_entries[keep]
        self.sw_tilde_enc = self.sw_tilde_enc[keep]

        if self.vitals.shape[0] == 0:
            raise ValueError("No active support sequences available for RMSN encoder training.")

    def __len__(self):
        return int(self.vitals.shape[0])

    def __getitem__(self, idx):
        return {
            "vitals": torch.from_numpy(self.vitals[idx]),
            "prev_outputs": torch.from_numpy(self.prev_outputs[idx]),
            "current_treatments": torch.from_numpy(self.current_treatments[idx]),
            "outputs": torch.from_numpy(self.outputs[idx]),
            "static_features": torch.from_numpy(self.static_features[idx]),
            "active_entries": torch.from_numpy(self.active_entries[idx]),
            "sw_tilde_enc": torch.from_numpy(self.sw_tilde_enc[idx]),
        }


@torch.no_grad()
def get_encoder_representations(encoder, train_data):
    encoder.eval()

    batch = {
        "vitals": torch.from_numpy(train_data["vitals"]).to(common.DEVICE),
        "prev_outputs": torch.from_numpy(train_data["prev_outputs"]).to(common.DEVICE),
        "current_treatments": torch.from_numpy(train_data["current_treatments"]).to(common.DEVICE),
        "static_features": torch.from_numpy(train_data["static_features"]).to(common.DEVICE),
    }

    _, r = encoder(batch)
    return r.detach().float().cpu().numpy()


def sample_decoder_origins(train_data, seed, max_origins=MAX_TRAIN_ORIGINS):
    rng = np.random.default_rng(int(seed))
    lengths = train_data["lengths"]
    timesteps = int(train_data["T"])
    pairs = []

    for local_i, length in enumerate(lengths):
        max_origin = min(int(length) - 2, timesteps - 2, common.MAX_INPUT_INDEX)
        if max_origin < 1:
            continue
        lo = common.MIN_T_OBS if max_origin >= common.MIN_T_OBS else 1
        for origin in range(lo, max_origin + 1):
            pairs.append((local_i, origin))

    if len(pairs) == 0:
        for local_i, length in enumerate(lengths):
            max_origin = min(int(length) - 2, timesteps - 2, common.MAX_INPUT_INDEX)
            for origin in range(1, max_origin + 1):
                pairs.append((local_i, origin))

    if len(pairs) == 0:
        raise ValueError("No valid decoder origins in support context.")

    pairs = np.asarray(pairs, dtype=np.int64)

    if max_origins is not None and len(pairs) > max_origins:
        keep = rng.choice(len(pairs), size=max_origins, replace=False)
        pairs = pairs[keep]

    return pairs


class RMSNDecoderOriginDataset(Dataset):
    def __init__(self, train_data, origins, init_state_all, sw_tilde_enc, hparams):
        horizon = common.TAU_MAX
        n_origins = origins.shape[0]

        actions = train_data["A"]
        y_norm = train_data["Yc"]
        lengths = train_data["lengths"]
        static = train_data["static_features"]

        dec_actions = np.zeros((n_origins, horizon), dtype=np.int64)
        dec_prev_y = np.zeros((n_origins, horizon, 1), dtype=np.float32)
        dec_targets = np.zeros((n_origins, horizon, 1), dtype=np.float32)
        active_dec = np.zeros((n_origins, horizon, 1), dtype=np.float32)
        dec_static = np.zeros((n_origins, common.D_STATIC_MAX), dtype=np.float32)
        init_states = np.zeros((n_origins, init_state_all.shape[-1]), dtype=np.float32)
        dec_sw_raw = np.ones((n_origins, horizon), dtype=np.float32)

        for row, (local_i, origin) in enumerate(origins):
            local_i = int(local_i)
            origin = int(origin)

            dec_static[row] = static[local_i]
            init_states[row] = init_state_all[local_i, origin]

            cumulative_sw = 1.0

            for step in range(horizon):
                action_t = origin + step
                y_t = origin + step
                y_next = origin + step + 1

                if action_t < actions.shape[1]:
                    dec_actions[row, step] = int(actions[local_i, action_t])

                if y_t < y_norm.shape[1]:
                    dec_prev_y[row, step, 0] = float(y_norm[local_i, y_t])

                if action_t < sw_tilde_enc.shape[1]:
                    cumulative_sw *= float(sw_tilde_enc[local_i, action_t])

                dec_sw_raw[row, step] = cumulative_sw

                if y_next < y_norm.shape[1] and y_next < int(lengths[local_i]) and np.isfinite(y_norm[local_i, y_next]):
                    dec_targets[row, step, 0] = float(y_norm[local_i, y_next])
                    active_dec[row, step, 0] = 1.0

        keep = active_dec.sum(axis=(1, 2)) > 0

        self.current_treatments = action4_to_bits(dec_actions[keep])
        self.prev_outputs = dec_prev_y[keep].astype(np.float32)
        self.outputs = dec_targets[keep].astype(np.float32)
        self.active_entries = active_dec[keep].astype(np.float32)
        self.static_features = dec_static[keep].astype(np.float32)
        self.init_state = init_states[keep].astype(np.float32)

        active_2d = self.active_entries.squeeze(-1)
        dec_sw = dec_sw_raw[keep]

        self.sw_tilde_dec = clip_normalize_weights(
            dec_sw,
            active=active_2d,
            quantiles=hparams.get("weight_clip_quantiles", [0.01, 0.99]),
            multiple_horizons=True,
        )

        if self.current_treatments.shape[0] == 0:
            raise ValueError("No active decoder origins available for RMSN decoder training.")

    def __len__(self):
        return int(self.current_treatments.shape[0])

    def __getitem__(self, idx):
        return {
            "current_treatments": torch.from_numpy(self.current_treatments[idx]),
            "prev_outputs": torch.from_numpy(self.prev_outputs[idx]),
            "outputs": torch.from_numpy(self.outputs[idx]),
            "active_entries": torch.from_numpy(self.active_entries[idx]),
            "static_features": torch.from_numpy(self.static_features[idx]),
            "init_state": torch.from_numpy(self.init_state[idx]),
            "sw_tilde_dec": torch.from_numpy(self.sw_tilde_dec[idx]),
        }


def move_batch_to_device(batch):
    return move_float_batch_to_device(batch)
