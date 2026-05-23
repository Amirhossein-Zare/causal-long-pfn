import hashlib
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from clpfn.config.defaults import (
    D_INPUT_MAX,
    D_STATIC_MAX,
    D_MODEL,
    D_FF,
    DROPOUT,
    GMM_K,
    GMM_MAX_SIGMA,
    GMM_MIN_SIGMA,
    GMM_PI_TEMP,
    MAX_INPUT_INDEX,
    MAX_SEQ_LEN,
    MAX_TARGET_INDEX,
    N_ACTIONS,
    N_HEADS,
    N_HISTORY_LAYERS,
    N_PFN_LAYERS,
)


class TimeStepEncoder(nn.Module):
    def __init__(self, d_max: int, d_model: int):
        super().__init__()
        self.covariate_proj = nn.Linear(d_max * 3, d_model)
        self.outcome_proj = nn.Linear(3, d_model)
        self.action_proj = nn.Linear(N_ACTIONS, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, actions, d_input=None):
        batch_size, seq_len, d_max = x.shape

        hidden_mask = x < -90.0
        hidden_flag = hidden_mask.float() * (-2.0)

        x_clean = x.clone()
        x_clean[hidden_mask] = 0.0

        x_diff = torch.zeros_like(x_clean)
        x_diff[:, 1:, :] = x_clean[:, 1:, :] - x_clean[:, :-1, :]

        hidden_boundary = torch.zeros_like(hidden_mask)
        hidden_boundary[:, 1:, :] = hidden_mask[:, 1:, :] | hidden_mask[:, :-1, :]
        hidden_boundary[:, 0, :] = hidden_mask[:, 0, :]

        x_diff[hidden_boundary] = 0.0
        x_diff = x_diff * 0.5

        if d_input is None:
            outcome_index = torch.full((batch_size,), d_max - 1, device=x.device, dtype=torch.long)
        else:
            outcome_index = (d_input.to(x.device).long() - 1).clamp(0, d_max - 1)

        gather_index = outcome_index.view(batch_size, 1, 1).expand(batch_size, seq_len, 1)

        outcome_value = x_clean.gather(-1, gather_index)
        outcome_diff = x_diff.gather(-1, gather_index)
        outcome_hidden = hidden_flag.gather(-1, gather_index)
        outcome_features = torch.cat([outcome_value, outcome_diff, outcome_hidden], dim=-1)

        covariate_value = x_clean.clone()
        covariate_diff = x_diff.clone()
        covariate_hidden = hidden_flag.clone()

        covariate_value.scatter_(-1, gather_index, 0.0)
        covariate_diff.scatter_(-1, gather_index, 0.0)
        covariate_hidden.scatter_(-1, gather_index, 0.0)

        covariate_features = torch.cat(
            [covariate_value, covariate_diff, covariate_hidden],
            dim=-1,
        )

        action_onehot = F.one_hot(
            actions.clamp(0, N_ACTIONS - 1),
            num_classes=N_ACTIONS,
        ).float()

        return self.norm(
            self.covariate_proj(covariate_features)
            + self.outcome_proj(outcome_features)
            + self.action_proj(action_onehot)
        )


class CausalHistoryEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float):
        super().__init__()
        self.time_step_encoder = TimeStepEncoder(D_INPUT_MAX, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.register_buffer("positional_encoding", self._make_positional_encoding(d_model, MAX_SEQ_LEN))
        self.register_buffer("causal_mask", torch.triu(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN), diagonal=1).bool())

        for layer in self.transformer.layers:
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.linear2.weight)

    @staticmethod
    def _make_positional_encoding(d_model: int, max_len: int):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def encode_sequence(self, x, actions, d_input=None):
        seq_len = x.shape[1]
        h = self.time_step_encoder(x, actions, d_input=d_input)
        h = h + self.positional_encoding[:seq_len].to(device=h.device, dtype=h.dtype).unsqueeze(0)
        return self.transformer(h, mask=self.causal_mask[:seq_len, :seq_len], is_causal=True)

    def forward(self, x, actions, current_time, d_input=None):
        h = self.encode_sequence(x, actions, d_input=d_input)
        seq_len = h.shape[1]

        if isinstance(current_time, torch.Tensor) and current_time.dim() > 0:
            index = current_time.clamp(min=0, max=seq_len - 1).long()
            return h[torch.arange(h.shape[0], device=h.device), index, :]

        index = max(0, min(int(current_time), seq_len - 1))
        return h[:, index, :]


class PFNAttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, zero_init: bool = True):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if zero_init:
            nn.init.zeros_(self.attention.out_proj.weight)
            nn.init.zeros_(self.ff2.weight)

    def forward(self, tokens, pad_mask=None):
        attended, _ = self.attention(
            tokens,
            tokens,
            tokens,
            key_padding_mask=pad_mask,
            need_weights=False,
        )
        tokens = self.norm1(tokens + self.dropout(attended))
        ff = self.dropout(F.gelu(self.ff1(tokens)))
        return self.norm2(tokens + self.dropout(self.ff2(ff)))


class GaussianMixtureHead(nn.Module):
    def __init__(self, d_model: int, n_components: int = GMM_K):
        super().__init__()
        self.fc_logit = nn.Linear(d_model, n_components)
        self.fc_mean_delta = nn.Linear(d_model, n_components)
        self.fc_sigma = nn.Linear(d_model, n_components)

    def forward(self, representation):
        log_pi = F.log_softmax(self.fc_logit(representation) / GMM_PI_TEMP, dim=-1)
        mean_delta = 7.0 * torch.tanh(self.fc_mean_delta(representation) / 7.0)
        sigma = (F.softplus(self.fc_sigma(representation)) + GMM_MIN_SIGMA).clamp(max=GMM_MAX_SIGMA)
        return log_pi, mean_delta, sigma


def predictive_mean_from_gmm(log_pi, mu):
    return (log_pi.exp() * mu).sum(dim=-1)


class CausalLongPFN(nn.Module):
    def __init__(self):
        super().__init__()

        self.history_encoder = CausalHistoryEncoder(
            D_MODEL,
            N_HEADS,
            N_HISTORY_LAYERS,
            D_FF,
            DROPOUT,
        )
        self.history_repr_norm = nn.LayerNorm(D_MODEL)

        self.anchor_y_encoder = nn.Linear(2, D_MODEL)
        self.query_label_embedding = nn.Parameter(torch.zeros(D_MODEL))

        self.support_y_stats_encoder = nn.Linear(2, D_MODEL)
        nn.init.normal_(self.support_y_stats_encoder.weight, std=0.005)
        nn.init.zeros_(self.support_y_stats_encoder.bias)

        self.static_encoder = nn.Sequential(
            nn.Linear(D_STATIC_MAX, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, D_MODEL),
        )
        nn.init.zeros_(self.static_encoder[2].weight)
        nn.init.zeros_(self.static_encoder[2].bias)

        self.last_x_proj = nn.Linear(D_INPUT_MAX, D_MODEL)
        nn.init.normal_(self.last_x_proj.weight, std=0.02)
        nn.init.zeros_(self.last_x_proj.bias)

        self.pfn_token_proj = nn.Linear(D_MODEL * 2, D_MODEL)

        self.pfn_layers = nn.ModuleList([
            PFNAttentionLayer(D_MODEL, N_HEADS, D_FF, DROPOUT, zero_init=True)
            for _ in range(N_PFN_LAYERS)
        ])

        self.pfn_output_norm = nn.LayerNorm(D_MODEL)
        self.gmm_head = GaussianMixtureHead(D_MODEL)

        nn.init.zeros_(self.gmm_head.fc_mean_delta.weight)
        nn.init.zeros_(self.gmm_head.fc_mean_delta.bias)

    @staticmethod
    def _compute_support_y_stats(support_anchor_y, support_pad_mask):
        y0 = support_anchor_y[:, :, 0] if support_anchor_y.dim() == 3 else support_anchor_y
        valid = (~support_pad_mask).float()
        n_valid = valid.sum(dim=1, keepdim=True).clamp(min=1.0)

        mean = (y0 * valid).sum(dim=1) / n_valid.squeeze(1)
        sq_mean = ((y0 ** 2) * valid).sum(dim=1) / n_valid.squeeze(1)
        std = (sq_mean - mean ** 2).clamp(min=0.0).sqrt().clamp(min=0.01)

        return mean, std

    @staticmethod
    def _gather_outcome_channel(x_current, d_input, input_scale):
        batch_size = x_current.shape[0]
        outcome_index = (
            d_input.to(x_current.device).long() - 1
        ).clamp(0, x_current.shape[-1] - 1).view(batch_size, 1)

        return (
            x_current.gather(1, outcome_index).squeeze(1)
            / input_scale.to(x_current.device).clamp(min=1e-6)
        ).clamp(-10.0, 10.0)

    @staticmethod
    def _write_outcome_channel(query_x, time_index, pred, d_input, input_scale, active_mask):
        batch_size, seq_len, d_max = query_x.shape
        device = query_x.device

        batch_index = torch.arange(batch_size, device=device)
        time_index = time_index.clamp(0, seq_len - 1).long()
        outcome_index = (d_input.to(device).long() - 1).clamp(0, d_max - 1)

        scaled_pred = pred.to(device=device, dtype=query_x.dtype)
        scaled_pred = scaled_pred * input_scale.to(device=device, dtype=query_x.dtype).clamp(min=1e-6)

        mask = active_mask & (time_index >= 0) & (time_index < seq_len)

        if mask.any():
            query_x[batch_index[mask], time_index[mask], outcome_index[mask]] = scaled_pred[mask]

        return query_x

    def forward_one_step(self, batch, current_time=None, n_layers=None):
        support_x = batch["support_x"]
        support_actions = batch["support_actions"]
        support_anchor_y = batch["support_anchor_y"]
        support_anchor_time = batch.get("support_anchor_time", None)
        support_pad_mask = batch["support_pad_mask"]

        query_x = batch["query_x"]
        query_actions = batch["query_actions"]

        support_static = batch.get("support_static", None)
        query_static = batch.get("query_static", None)

        input_scale = batch.get("input_scale", None)
        d_input = batch.get("d_input", None)

        batch_size, n_support, seq_len, d_max = support_x.shape
        n_anchors = support_anchor_y.shape[-1] if support_anchor_y.dim() == 3 else 1
        device = support_x.device

        if input_scale is None:
            input_scale = torch.ones(batch_size, device=device, dtype=support_x.dtype)
        else:
            input_scale = input_scale.to(device=device, dtype=support_x.dtype).clamp(min=1e-6)

        if d_input is None:
            d_input = torch.full((batch_size,), d_max, device=device, dtype=torch.long)
        else:
            d_input = d_input.to(device=device).long().clamp(1, d_max)

        if current_time is None:
            current_time = batch["current_time"]

        if not isinstance(current_time, torch.Tensor):
            current_time = torch.full((batch_size,), int(current_time), device=device, dtype=torch.long)
        else:
            current_time = current_time.to(device=device).long()

        current_time = current_time.clamp(0, seq_len - 1)
        one_step_target_time = (current_time + 1).clamp(1, MAX_SEQ_LEN)

        if support_anchor_time is None:
            support_anchor_time = one_step_target_time[:, None, None].expand(batch_size, n_support, n_anchors)

        support_anchor_time = support_anchor_time.to(device).long().clamp(1, MAX_SEQ_LEN)

        if support_static is None:
            support_static = torch.zeros(batch_size, n_support, D_STATIC_MAX, device=device, dtype=support_x.dtype)

        if query_static is None:
            query_static = torch.zeros(batch_size, D_STATIC_MAX, device=device, dtype=support_x.dtype)

        real_support_mask_flat = ~support_pad_mask.reshape(batch_size * n_support)
        support_d_input_flat = d_input.repeat_interleave(n_support)

        max_support_time = max(1, min(int(support_anchor_time.max().item()), seq_len))
        max_query_time = max(1, min(int(current_time.max().item()) + 1, seq_len))

        support_x_slice = support_x[:, :, :max_support_time, :].reshape(
            batch_size * n_support,
            max_support_time,
            d_max,
        )
        support_actions_slice = support_actions[:, :, :max_support_time].reshape(
            batch_size * n_support,
            max_support_time,
        )

        support_sequence_repr_flat = torch.zeros(
            batch_size * n_support,
            max_support_time,
            D_MODEL,
            device=device,
            dtype=support_x.dtype,
        )

        if real_support_mask_flat.any():
            encoded_support = self.history_encoder.encode_sequence(
                support_x_slice[real_support_mask_flat],
                support_actions_slice[real_support_mask_flat],
                d_input=support_d_input_flat[real_support_mask_flat],
            )
            support_sequence_repr_flat[real_support_mask_flat] = encoded_support

        support_anchor_flat = support_anchor_time.reshape(batch_size * n_support, n_anchors).clamp(1, max_support_time)
        flat_batch_index = torch.arange(batch_size * n_support, device=device).unsqueeze(1).expand(
            batch_size * n_support,
            n_anchors,
        )

        anchor_history_index = (support_anchor_flat - 1).clamp(0, max_support_time - 1)

        support_history_repr = self.history_repr_norm(
            support_sequence_repr_flat[flat_batch_index, anchor_history_index].reshape(
                batch_size,
                n_support,
                n_anchors,
                D_MODEL,
            )
        )

        query_history_repr = self.history_repr_norm(
            self.history_encoder(
                query_x[:, :max_query_time, :],
                query_actions[:, :max_query_time],
                current_time.clamp(0, max_query_time - 1),
                d_input=d_input,
            )
        )

        batch_index = torch.arange(batch_size, device=device)
        query_current_index = current_time.clamp(0, seq_len - 1)
        query_x_current = query_x[batch_index, query_current_index, :]

        query_x_current_emb = self.last_x_proj(query_x_current)
        last_query_y = self._gather_outcome_channel(query_x_current, d_input, input_scale)

        support_x_at_anchor = support_x_slice[flat_batch_index, anchor_history_index].reshape(
            batch_size,
            n_support,
            n_anchors,
            d_max,
        )
        support_x_anchor_emb = self.last_x_proj(support_x_at_anchor)

        support_static_emb = self.static_encoder(support_static).unsqueeze(2)
        query_static_emb = self.static_encoder(query_static)

        support_y_mean, support_y_std = self._compute_support_y_stats(
            support_anchor_y,
            support_pad_mask,
        )
        support_y_stats_emb = self.support_y_stats_encoder(
            torch.stack([support_y_mean, support_y_std], dim=-1)
        )

        support_global_emb = support_y_stats_emb[:, None, None, :]
        query_global_emb = support_y_stats_emb

        if support_anchor_y.dim() == 2:
            support_anchor_y = support_anchor_y.unsqueeze(-1)

        anchor_indicator = torch.zeros_like(support_anchor_y)
        support_anchor_y_emb = self.anchor_y_encoder(
            torch.stack([support_anchor_y, anchor_indicator], dim=-1)
        )

        support_core = support_history_repr + support_x_anchor_emb + support_static_emb + support_global_emb
        support_tokens = self.pfn_token_proj(
            torch.cat([support_core, support_anchor_y_emb], dim=-1)
        ).reshape(batch_size, n_support * n_anchors, D_MODEL)

        query_core = query_history_repr + query_x_current_emb + query_static_emb + query_global_emb
        query_label_emb = self.query_label_embedding.unsqueeze(0).expand(batch_size, -1)
        query_token = self.pfn_token_proj(
            torch.cat([query_core, query_label_emb], dim=-1)
        ).unsqueeze(1)

        tokens = torch.cat([support_tokens, query_token], dim=1)

        support_token_pad_mask = support_pad_mask.unsqueeze(-1).expand(
            batch_size,
            n_support,
            n_anchors,
        ).reshape(batch_size, n_support * n_anchors)

        pad_mask = torch.cat(
            [support_token_pad_mask, torch.zeros(batch_size, 1, dtype=torch.bool, device=device)],
            dim=1,
        )

        if n_layers is None:
            n_layers = N_PFN_LAYERS

        for layer in self.pfn_layers[:n_layers]:
            tokens = layer(tokens, pad_mask=pad_mask)

        query_repr = self.pfn_output_norm(tokens)[:, -1, :]

        log_pi, mean_delta, sigma = self.gmm_head(query_repr)
        mu = (last_query_y.unsqueeze(-1) + mean_delta).clamp(-12.0, 12.0)

        return log_pi, mu, sigma

    def rollout(self, batch, n_layers=None):
        work = dict(batch)
        work["query_x"] = batch["query_x"].clone()

        batch_size, seq_len, _ = work["query_x"].shape
        device = work["query_x"].device

        t_obs = batch["t_obs"].to(device).long().clamp(0, MAX_INPUT_INDEX)
        t_target = batch["t_target"].to(device).long().clamp(1, MAX_TARGET_INDEX)

        start = t_obs.clamp(0, seq_len - 1)

        d_input = batch.get(
            "d_input",
            torch.full((batch_size,), work["query_x"].shape[-1], device=device, dtype=torch.long),
        ).to(device).long()

        input_scale = batch.get(
            "input_scale",
            torch.ones(batch_size, device=device, dtype=work["query_x"].dtype),
        ).to(device=device, dtype=work["query_x"].dtype)

        final_log_pi = None
        final_mu = None
        final_sigma = None
        final_set = torch.zeros(batch_size, dtype=torch.bool, device=device)

        horizon_len = (t_target - start).clamp(min=1, max=MAX_SEQ_LEN)
        max_horizon = int(horizon_len.max().item())

        for horizon_idx in range(max_horizon):
            current_time = (start + horizon_idx).clamp(0, seq_len - 1)
            active = horizon_idx < horizon_len

            if not active.any():
                continue

            log_pi, mu, sigma = self.forward_one_step(
                work,
                current_time=current_time,
                n_layers=n_layers,
            )

            pred = predictive_mean_from_gmm(log_pi, mu).detach()

            end_now = active & (current_time == (t_target - 1).clamp(0, seq_len - 1))

            if end_now.any():
                if final_log_pi is None:
                    final_log_pi = torch.zeros_like(log_pi)
                    final_mu = torch.zeros_like(mu)
                    final_sigma = torch.ones_like(sigma)

                final_log_pi[end_now] = log_pi[end_now]
                final_mu[end_now] = mu[end_now]
                final_sigma[end_now] = sigma[end_now]
                final_set[end_now] = True

            next_time = current_time + 1

            work["query_x"] = self._write_outcome_channel(
                work["query_x"],
                next_time,
                pred,
                d_input,
                input_scale,
                active & (next_time < seq_len),
            )

        if final_log_pi is None or not bool(final_set.all().item()):
            current_time = (t_target - 1).clamp(0, seq_len - 1)
            final_log_pi, final_mu, final_sigma = self.forward_one_step(
                work,
                current_time=current_time,
                n_layers=n_layers,
            )

        return final_log_pi, final_mu, final_sigma

    def forward(self, batch, n_layers=None):
        return self.forward_one_step(
            batch,
            current_time=batch["current_time"],
            n_layers=n_layers,
        )


def file_sha256_prefix(path, n_hex=16, chunk_size=16 * 1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()[:n_hex]


def tensor_fingerprint_from_state_dict(state_dict, n_hex=16):
    h = hashlib.sha256()
    n_tensors = 0
    n_params = 0
    abs_sum = 0.0
    sq_sum = 0.0

    for key in sorted(state_dict.keys()):
        value = state_dict[key]

        if not torch.is_tensor(value):
            continue

        x = value.detach().cpu().contiguous()
        xf = x.float()

        h.update(key.encode("utf-8"))
        h.update(str(tuple(x.shape)).encode("utf-8"))
        h.update(x.numpy().tobytes())

        n_tensors += 1
        n_params += x.numel()
        abs_sum += float(xf.abs().sum())
        sq_sum += float((xf * xf).sum())

    return {
        "fingerprint": h.hexdigest()[:n_hex],
        "n_tensors": int(n_tensors),
        "n_params": int(n_params),
        "abs_sum": float(abs_sum),
        "sq_sum": float(sq_sum),
    }


def _load_torch_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key], key

        if all(isinstance(k, str) for k in checkpoint.keys()):
            if any(torch.is_tensor(v) for v in checkpoint.values()):
                return checkpoint, "raw_state_dict"

    raise ValueError("Could not find a model state_dict in checkpoint.")


def load_causal_long_pfn_checkpoint(path, device=None, strict=True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = _load_torch_checkpoint(path)
    state_dict, state_dict_key = _extract_state_dict(checkpoint)

    if all(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

    checkpoint_fp = tensor_fingerprint_from_state_dict(state_dict)

    model = CausalLongPFN().to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if strict and (missing or unexpected):
        raise RuntimeError(
            "Checkpoint architecture mismatch. "
            f"Missing={missing[:20]} Unexpected={unexpected[:20]}"
        )

    loaded_fp = tensor_fingerprint_from_state_dict(model.state_dict())
    model.eval()

    meta = {
        "checkpoint_path": str(path),
        "checkpoint_basename": os.path.basename(str(path)),
        "checkpoint_file_size": int(os.path.getsize(path)),
        "checkpoint_file_sha256_prefix": file_sha256_prefix(path),
        "checkpoint_state_dict_key": state_dict_key,

        "checkpoint_tensor_fingerprint": checkpoint_fp["fingerprint"],
        "checkpoint_tensor_n_tensors": checkpoint_fp["n_tensors"],
        "checkpoint_tensor_n_params": checkpoint_fp["n_params"],
        "checkpoint_tensor_abs_sum": checkpoint_fp["abs_sum"],
        "checkpoint_tensor_sq_sum": checkpoint_fp["sq_sum"],

        "loaded_model_fingerprint": loaded_fp["fingerprint"],
        "loaded_model_n_tensors": loaded_fp["n_tensors"],
        "loaded_model_n_params": loaded_fp["n_params"],
        "loaded_model_abs_sum": loaded_fp["abs_sum"],
        "loaded_model_sq_sum": loaded_fp["sq_sum"],

        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }

    if isinstance(checkpoint, dict):
        for key in ("step_count", "global_step", "epoch"):
            if key in checkpoint:
                meta[key] = checkpoint[key]

    return model, meta
