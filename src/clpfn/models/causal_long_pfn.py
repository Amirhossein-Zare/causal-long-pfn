import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_CONFIG_KEYS = (
    "D_INPUT_MAX", "D_STATIC_MAX", "MAX_SEQ_LEN", "N_ACTIONS", "D_MODEL",
    "N_HEADS", "N_HISTORY_LAYERS", "N_PFN_LAYERS", "D_FF", "DROPOUT",
    "GMM_K", "GMM_PI_TEMP", "GMM_MIN_SIGMA", "GMM_MAX_SIGMA",
)

def model_config_from_resolved_config(resolved_config):
    model = resolved_config["model"]
    prior = resolved_config["prior"]
    config = {key: model[key] for key in MODEL_CONFIG_KEYS if key in model}
    config["D_INPUT_MAX"] = int(prior["D_STATE_MAX"]) + int(model["D_OUTCOME"])
    config["MAX_SEQ_LEN"] = int(prior["OBS_TIME_MAX"]) + int(prior["HORIZON_MAX"])
    missing = [key for key in MODEL_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"Resolved model configuration is missing required fields: {missing}")
    return config


class TimeStepEncoder(nn.Module):
    def __init__(self, d_max: int, d_model: int, n_actions: int):
        super().__init__()
        self.n_actions = n_actions
        self.covariate_proj = nn.Linear(d_max * 3, d_model)
        self.outcome_proj = nn.Linear(3, d_model)
        self.action_proj = nn.Linear(n_actions, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, actions, d_input):
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

        outcome_index = d_input.to(x.device).long() - 1

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
            actions.clamp(0, self.n_actions - 1),
            num_classes=self.n_actions,
        ).float()

        return self.norm(
            self.covariate_proj(covariate_features)
            + self.outcome_proj(outcome_features)
            + self.action_proj(action_onehot)
        )


class CausalHistoryEncoder(nn.Module):
    def __init__(self, d_input_max: int, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float, n_actions: int, max_seq_len: int):
        super().__init__()
        self.time_step_encoder = TimeStepEncoder(d_input_max, d_model, n_actions)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.register_buffer("positional_encoding", self._make_positional_encoding(d_model, max_seq_len))
        self.register_buffer("causal_mask", torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool())

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

    def encode_sequence(self, x, actions, d_input):
        seq_len = x.shape[1]
        h = self.time_step_encoder(x, actions, d_input=d_input)
        h = h + self.positional_encoding[:seq_len].to(device=h.device, dtype=h.dtype).unsqueeze(0)
        return self.transformer(h, mask=self.causal_mask[:seq_len, :seq_len], is_causal=True)

    def forward(self, x, actions, current_time, d_input):
        h = self.encode_sequence(x, actions, d_input=d_input)
        seq_len = h.shape[1]
        index = current_time.to(h.device).long()
        return h[torch.arange(h.shape[0], device=h.device), index, :]


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
    def __init__(self, d_model: int, n_components: int, pi_temp: float, min_sigma: float, max_sigma: float):
        super().__init__()
        self.pi_temp = pi_temp
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.fc_logit = nn.Linear(d_model, n_components)
        self.fc_mean_delta = nn.Linear(d_model, n_components)
        self.fc_sigma = nn.Linear(d_model, n_components)

    def forward(self, representation):
        logits = self.fc_logit(representation)
        if self.pi_temp != 1.0:
            logits = logits / self.pi_temp
        log_pi = F.log_softmax(logits, dim=-1)
        mean_delta = 7.0 * torch.tanh(self.fc_mean_delta(representation) / 7.0)
        sigma = (F.softplus(self.fc_sigma(representation)) + self.min_sigma).clamp(max=self.max_sigma)
        return log_pi, mean_delta, sigma


def predictive_mean_from_gmm(log_pi, mu):
    return (log_pi.exp() * mu).sum(dim=-1)


class CausalLongPFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        missing = [key for key in MODEL_CONFIG_KEYS if key not in config]
        if missing:
            raise ValueError(f"Model configuration is missing required fields: {missing}")
        self.config = {key: config[key] for key in MODEL_CONFIG_KEYS}
        self.d_model = int(config["D_MODEL"])
        self.d_static_max = int(config["D_STATIC_MAX"])
        self.max_seq_len = int(config["MAX_SEQ_LEN"])
        self.n_pfn_layers = int(config["N_PFN_LAYERS"])

        self.history_encoder = CausalHistoryEncoder(
            int(config["D_INPUT_MAX"]), self.d_model, int(config["N_HEADS"]),
            int(config["N_HISTORY_LAYERS"]), int(config["D_FF"]), float(config["DROPOUT"]),
            int(config["N_ACTIONS"]), self.max_seq_len,
        )
        self.history_repr_norm = nn.LayerNorm(self.d_model)

        self.anchor_y_encoder = nn.Linear(1, self.d_model)
        self.query_label_embedding = nn.Parameter(torch.zeros(self.d_model))

        self.static_encoder = nn.Sequential(
            nn.Linear(self.d_static_max, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        nn.init.zeros_(self.static_encoder[2].weight)
        nn.init.zeros_(self.static_encoder[2].bias)

        self.pfn_token_proj = nn.Linear(self.d_model * 2, self.d_model)

        self.pfn_layers = nn.ModuleList([
            PFNAttentionLayer(self.d_model, int(config["N_HEADS"]), int(config["D_FF"]), float(config["DROPOUT"]), zero_init=True)
            for _ in range(self.n_pfn_layers)
        ])

        self.pfn_output_norm = nn.LayerNorm(self.d_model)
        self.gmm_head = GaussianMixtureHead(self.d_model, int(config["GMM_K"]), float(config["GMM_PI_TEMP"]), float(config["GMM_MIN_SIGMA"]), float(config["GMM_MAX_SIGMA"]))

        nn.init.zeros_(self.gmm_head.fc_mean_delta.weight)
        nn.init.zeros_(self.gmm_head.fc_mean_delta.bias)

    @classmethod
    def from_config(cls, config):
        return cls(deepcopy(config))

    def get_config(self):
        return deepcopy(self.config)

    @staticmethod
    def _gather_outcome_channel(x_current, d_input, input_scale):
        batch_size = x_current.shape[0]
        outcome_index = (d_input.to(x_current.device).long() - 1).view(batch_size, 1)

        return (
            x_current.gather(1, outcome_index).squeeze(1)
            / input_scale.to(x_current.device).clamp(min=1e-6)
        ).clamp(-10.0, 10.0)

    @staticmethod
    def _write_outcome_channel(query_x, time_index, pred, d_input, input_scale, active_mask):
        batch_size, seq_len, d_max = query_x.shape
        device = query_x.device

        batch_index = torch.arange(batch_size, device=device)
        time_index = time_index.long()
        outcome_index = d_input.to(device).long() - 1

        scaled_pred = pred.to(device=device, dtype=query_x.dtype)
        scaled_pred = scaled_pred * input_scale.to(device=device, dtype=query_x.dtype).clamp(min=1e-6)

        mask = active_mask & (time_index >= 0) & (time_index < seq_len)

        if mask.any():
            query_x[batch_index[mask], time_index[mask], outcome_index[mask]] = scaled_pred[mask]

        return query_x

    def forward_one_step(self, batch, current_time):
        support_x = batch["support_x"]
        support_actions = batch["support_actions"]
        support_anchor_y = batch["support_anchor_y"]
        support_anchor_time = batch["support_anchor_time"]
        support_pad_mask = batch["support_pad_mask"]

        query_x = batch["query_x"]
        query_actions = batch["query_actions"]

        support_static = batch["support_static"]
        query_static = batch["query_static"]
        input_scale = batch["input_scale"]
        d_input = batch["d_input"]

        batch_size, n_support, seq_len, d_max = support_x.shape
        n_anchors = support_anchor_y.shape[-1] if support_anchor_y.dim() == 3 else 1
        device = support_x.device

        input_scale = input_scale.to(device=device, dtype=support_x.dtype)
        d_input = d_input.to(device=device).long()
        current_time = current_time.to(device=device).long()
        if d_input.shape != (batch_size,) or bool(((d_input < 1) | (d_input > d_max)).any()):
            raise ValueError("PFN batch d_input must contain one valid input dimension per episode.")
        if input_scale.shape != (batch_size,) or not bool(torch.isfinite(input_scale).all()) or bool((input_scale <= 0).any()):
            raise ValueError("PFN batch input_scale must contain one finite positive value per episode.")
        if current_time.shape != (batch_size,) or bool(((current_time < 0) | (current_time >= seq_len)).any()):
            raise ValueError("PFN batch current_time must contain one valid sequence index per episode.")
        if support_anchor_y.shape != (batch_size, n_support, n_anchors):
            raise ValueError("PFN support-anchor labels must have shape [batch, support, anchors].")
        if support_anchor_time.shape != support_anchor_y.shape:
            raise ValueError("PFN support-anchor times must match support-anchor label shape.")
        if bool(((support_anchor_time < 1) | (support_anchor_time > seq_len)).any()):
            raise ValueError("PFN support-anchor times are outside the sequence.")

        support_anchor_time = support_anchor_time.to(device).long()

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
            self.d_model,
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

        support_anchor_flat = support_anchor_time.reshape(batch_size * n_support, n_anchors)
        flat_batch_index = torch.arange(batch_size * n_support, device=device).unsqueeze(1).expand(
            batch_size * n_support,
            n_anchors,
        )

        anchor_history_index = support_anchor_flat - 1

        support_history_repr = self.history_repr_norm(
            support_sequence_repr_flat[flat_batch_index, anchor_history_index].reshape(
                batch_size,
                n_support,
                n_anchors,
                self.d_model,
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

        last_query_y = self._gather_outcome_channel(query_x_current, d_input, input_scale)

        support_static_emb = self.static_encoder(support_static).unsqueeze(2)
        query_static_emb = self.static_encoder(query_static)

        support_anchor_y_emb = self.anchor_y_encoder(support_anchor_y.unsqueeze(-1))

        support_core = support_history_repr + support_static_emb

        support_tokens = self.pfn_token_proj(
            torch.cat([support_core, support_anchor_y_emb], dim=-1)
        ).reshape(batch_size, n_support * n_anchors, self.d_model)

        query_core = query_history_repr + query_static_emb
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

        for layer in self.pfn_layers:
            tokens = layer(tokens, pad_mask=pad_mask)

        query_repr = self.pfn_output_norm(tokens)[:, -1, :]

        log_pi, mean_delta, sigma = self.gmm_head(query_repr)
        mu = (last_query_y.unsqueeze(-1) + mean_delta).clamp(-12.0, 12.0)

        return log_pi, mu, sigma

    def rollout(
        self,
        batch,
        return_trace=False,
        rollout_mode="mean",
        feedback_clip: float | None = None,
    ):
        if rollout_mode != "mean":
            raise ValueError(f"Unsupported rollout_mode={rollout_mode!r}; only deterministic mean feedback is available.")
        work = dict(batch)
        work["query_x"] = batch["query_x"].clone()

        batch_size, seq_len, _ = work["query_x"].shape
        device = work["query_x"].device

        t_obs = batch["t_obs"].to(device).long()
        t_target = batch["t_target"].to(device).long()
        if t_obs.shape != (batch_size,) or t_target.shape != (batch_size,):
            raise ValueError("PFN rollout times must contain one value per query.")
        if bool(((t_obs < 0) | (t_obs >= seq_len)).any()):
            raise ValueError("PFN rollout observation times are outside the sequence.")
        if bool(((t_target <= t_obs) | (t_target > seq_len)).any()):
            raise ValueError("PFN rollout target times must follow observation times within the sequence.")

        start = t_obs

        d_input = batch["d_input"].to(device).long()
        input_scale = batch["input_scale"].to(device=device, dtype=work["query_x"].dtype)

        final_log_pi = None
        final_mu = None
        final_sigma = None
        final_set = torch.zeros(batch_size, dtype=torch.bool, device=device)
        trace_log_pi, trace_mu, trace_sigma = [], [], []
        trace_mean, trace_variance, trace_feedback, trace_active = [], [], [], []

        horizon_len = t_target - start
        max_horizon = int(horizon_len.max().item())

        for horizon_idx in range(max_horizon):
            current_time = start + horizon_idx
            active = horizon_idx < horizon_len

            if not active.any():
                continue

            log_pi, mu, sigma = self.forward_one_step(
                work,
                current_time=current_time,
            )

            pred = predictive_mean_from_gmm(log_pi, mu).detach()
            feedback = pred if feedback_clip is None else pred.clamp(-float(feedback_clip), float(feedback_clip))
            variance = (
                log_pi.exp() * (sigma.square() + mu.square())
            ).sum(dim=-1) - pred.square()
            if return_trace:
                trace_log_pi.append(log_pi.detach())
                trace_mu.append(mu.detach())
                trace_sigma.append(sigma.detach())
                trace_mean.append(pred)
                trace_variance.append(variance.clamp_min(0.0).detach())
                trace_feedback.append(feedback)
                trace_active.append(active.detach())

            end_now = active & (current_time == t_target - 1)

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
                feedback,
                d_input,
                input_scale,
                active & (next_time < seq_len),
            )

        if final_log_pi is None or not bool(final_set.all().item()):
            raise RuntimeError("PFN rollout did not produce every requested endpoint.")

        if not return_trace:
            return final_log_pi, final_mu, final_sigma
        return final_log_pi, final_mu, final_sigma, {
            "log_pi": torch.stack(trace_log_pi, dim=1),
            "mu": torch.stack(trace_mu, dim=1),
            "sigma": torch.stack(trace_sigma, dim=1),
            "mixture_mean": torch.stack(trace_mean, dim=1),
            "mixture_variance": torch.stack(trace_variance, dim=1),
            "feedback_outcome": torch.stack(trace_feedback, dim=1),
            "active": torch.stack(trace_active, dim=1),
            "rollout_mode": rollout_mode,
        }

    def forward(self, batch):
        return self.forward_one_step(batch, current_time=batch["current_time"])
