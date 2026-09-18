from __future__ import annotations

import gc
import math
import time
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch

from clpfn.baselines.common.api import BaselineAdapter, Prediction, baseline_port_metadata, canonical_hparams
from clpfn.baselines.common.api import (
    single_model_train_final,
    train_diag_record_fields,
)
from clpfn.baselines.common.features import (
    encode_actions,
    has_dynamic_vitals,
    prepare_baseline_bundle,
    treatment_dim,
    treatment_mode_for_bundle,
)
from clpfn.baselines.common.training import train_loader
from clpfn.baselines.common.tuning import sample_random_hparams
from clpfn.baselines.common.training import (
    masked_mse_loss,
    masked_sequence_loss,
    rmse_from_predictions,
    support_val_one_step_candidates,
    targets_for_candidates,
)
from clpfn.baselines.ct import config as ct_config
from clpfn.baselines.ct.config import (
    CT_EVAL_BATCH_SIZE,
    CT_SPACE,
    DEFAULT_HPARAMS,
    MAX_VAL_ORIGINS,
    OUTPUT_DIR,
)
from clpfn.baselines.ct.data import CTSupportDataset, move_batch_to_device
from clpfn.baselines.models.ct import CT
from clpfn.evaluation.core import benchmark as common


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def _ceil_to_multiple(value, multiple):
    value = max(1, int(round(float(value))))
    multiple = max(1, int(multiple))
    return int(math.ceil(value / multiple) * multiple)


def _ct_input_dim(bundle):
    bundle = prepare_baseline_bundle(bundle)
    d_vitals = int(bundle["covariates"].shape[-1]) if has_dynamic_vitals(bundle) else 0
    d_static = int(bundle["static"].shape[-1])
    mode = treatment_mode_for_bundle(bundle, cancer_mode="multiclass")
    return int(max(treatment_dim(mode), d_static, d_vitals, 1))


def ct_hyperparameter_space(bundle):
    space = deepcopy(CT_SPACE)
    input_dim = _ct_input_dim(bundle)
    space["_input_dim"] = [int(input_dim)]
    return space, {
        "ct_input_dim": int(input_dim),
        "width_strategy": "reference_input_scaled",
        "balancing": "domain_confusion_ema",
    }


def _space_sample_to_hparams(sample):
    hp = dict(DEFAULT_HPARAMS)
    input_dim = int(sample["_input_dim"])
    num_heads = int(sample["num_heads"])
    head_multiple = int(np.lcm(num_heads, 2))
    seq_hidden = _ceil_to_multiple(input_dim * float(sample["seq_hidden_multiplier"]), head_multiple)
    br_size = max(2, int(round(input_dim * float(sample["br_multiplier"]))))
    fc_hidden = max(2, int(round(br_size * float(sample["fc_hidden_multiplier"]))))

    hp.update(
        {
            "seq_hidden_units": int(seq_hidden),
            "br_size": int(br_size),
            "fc_hidden_units": int(fc_hidden),
            "num_layers": int(sample["num_layers"]),
            "num_heads": int(num_heads),
            "dropout": float(sample["dropout"]),
            "lr": float(sample["lr"]),
            "batch_size": int(sample["batch_size"]),
            "grad_clip": float(sample["grad_clip"]),
            "ct_epochs": int(sample["ct_epochs"]),
            "max_position_len": int(hp.get("max_position_len", common.MAX_SEQ_LEN)),
            "max_relative_position": int(hp.get("max_relative_position", 15)),
            "trainable_positional_encoding": bool(hp.get("trainable_positional_encoding", True)),
            "balancing": hp.get("balancing", "domain_confusion"),
            "alpha_max": float(hp.get("alpha_max", 0.01)),
            "update_alpha": bool(hp.get("update_alpha", True)),
            "alpha_schedule": hp.get("alpha_schedule", "sigmoid"),
            "weights_ema": bool(hp.get("weights_ema", True)),
            "ema_beta": float(hp.get("ema_beta", 0.99)),
            "weight_decay": float(hp.get("weight_decay", 0.0)),
            "optimizer_cls": hp.get("optimizer_cls", "adam"),
        }
    )
    return canonical_hparams(hp)


def sample_random_candidates(space, n, seed):
    return sample_random_hparams(
        space,
        n,
        seed,
        default_hparams=DEFAULT_HPARAMS,
        canonical_hparams=canonical_hparams,
        transform_sample=_space_sample_to_hparams,
        is_valid=lambda hp: int(hp["seq_hidden_units"]) % int(hp["num_heads"]) == 0,
    )


def make_ct_args(d_vitals, hparams, d_static, treatment_mode):
    seq_hidden_units = int(hparams["seq_hidden_units"])
    num_heads = int(hparams["num_heads"])
    if seq_hidden_units % num_heads != 0:
        raise ValueError(f"seq_hidden_units={seq_hidden_units} must be divisible by num_heads={num_heads}")

    d_static = int(d_static)
    balancing = str(hparams["balancing"])
    alpha_max = float(hparams["alpha_max"])
    update_alpha = bool(hparams["update_alpha"])
    weights_ema = bool(hparams["weights_ema"])
    ema_beta = float(hparams["ema_beta"])

    multi = ns(
        seq_hidden_units=seq_hidden_units,
        br_size=int(hparams["br_size"]),
        fc_hidden_units=int(hparams["fc_hidden_units"]),
        dropout_rate=float(hparams["dropout"]),
        num_layer=int(hparams["num_layers"]),
        num_heads=num_heads,
        head_size=int(seq_hidden_units // num_heads),
        alpha=alpha_max,
        update_alpha=update_alpha,
        balancing=balancing,
        batch_size=int(hparams["batch_size"]),
        max_grad_norm=float(hparams["grad_clip"]),
        max_position_len=int(hparams["max_position_len"]),
        max_seq_length=int(hparams["max_position_len"]),
        self_positional_encoding={
            "absolute": False,
            "trainable": bool(hparams["trainable_positional_encoding"]),
            "max_relative_position": int(hparams["max_relative_position"]),
        },
        trainable_positional_encoding=bool(hparams["trainable_positional_encoding"]),
        max_relative_position=int(hparams["max_relative_position"]),
        attn_dropout=True,
        disable_cross_attention=False,
        isolate_subnetwork="",
        augment_with_masked_vitals=bool(hparams["augment_with_masked_vitals"]),
        optimizer={
            "learning_rate": float(hparams["lr"]),
            "weight_decay": float(hparams["weight_decay"]),
            "optimizer_cls": str(hparams["optimizer_cls"]),
            "lr_scheduler": False,
        },
    )
    return ns(
        model=ns(
            dim_treatments=treatment_dim(treatment_mode),
            dim_vitals=int(d_vitals),
            dim_static_features=d_static,
            dim_outcomes=1,
            multi=multi,
        ),
        dataset=ns(
            val_batch_size=int(hparams["batch_size"]),
            projection_horizon=common.PROJECTION_HORIZON,
            treatment_mode=str(treatment_mode),
            holdout_ratio=0.0,
        ),
        exp=ns(
            unscale_rmse=False,
            percentage_rmse=False,
            bce_weight=False,
            gpus="[]",
            max_epochs=int(hparams["ct_epochs"]),
            alpha_rate=str(hparams["alpha_schedule"]),
            update_alpha=update_alpha,
            balancing=balancing,
            alpha=alpha_max,
            weights_ema=weights_ema,
            beta=ema_beta,
        ),
    )


class _ParameterEMA:
    def __init__(self, named_parameters, decay):
        self.decay = float(decay)
        self.num_updates = 0
        self.named_parameters = [(name, param) for name, param in named_parameters]
        self.shadow = {
            name: param.detach().clone()
            for name, param in self.named_parameters
        }

    @torch.no_grad()
    def update(self):
        self.num_updates += 1
        decay = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        for name, param in self.named_parameters:
            self.shadow[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)

    @contextmanager
    def average_parameters(self):
        backup = {
            name: param.detach().clone()
            for name, param in self.named_parameters
        }
        try:
            with torch.no_grad():
                for name, param in self.named_parameters:
                    param.copy_(self.shadow[name])
            yield
        finally:
            with torch.no_grad():
                for name, param in self.named_parameters:
                    param.copy_(backup[name])

    @torch.no_grad()
    def copy_to_model(self):
        for name, param in self.named_parameters:
            param.copy_(self.shadow[name])


def _set_requires_grad(named_parameters, enabled):
    for _, param in named_parameters:
        param.requires_grad_(bool(enabled))


def _repeat_ct_targets(batch, pred_batch):
    base_batch = int(batch["outputs"].shape[0])
    if int(pred_batch) % base_batch != 0:
        raise ValueError(f"Unexpected CT augmentation batch size: {pred_batch} vs {base_batch}")
    repeat_factor = int(pred_batch) // base_batch
    return (
        batch["outputs"].repeat((repeat_factor, 1, 1)),
        batch["current_treatments"].repeat((repeat_factor, 1, 1)),
        batch["active_entries"].repeat((repeat_factor, 1, 1)),
    )


def _ct_losses(model, batch, *, treatment_kind):
    treatment_pred, outcome_pred, _ = model(batch)
    outputs, current_treatments, active_entries = _repeat_ct_targets(batch, outcome_pred.shape[0])
    outcome_loss = masked_mse_loss(outcome_pred, outputs, active_entries)
    treatment_loss = model.bce_loss(
        treatment_pred,
        current_treatments.float(),
        kind=treatment_kind,
    )
    treatment_loss = masked_sequence_loss(treatment_loss, active_entries)
    return outcome_loss, treatment_loss


def _sigmoid_alpha(epoch_completed, total_epochs, alpha_max):
    progress = float(epoch_completed) / float(max(int(total_epochs), 1))
    return float((2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0) * float(alpha_max))


def _ct_parameter_groups(model):
    treatment_prefixes = tuple(
        f"br_treatment_outcome_head.{name}"
        for name in model.br_treatment_outcome_head.treatment_head_params
    )
    treatment_named = [
        (name, param)
        for name, param in model.named_parameters()
        if name.startswith(treatment_prefixes)
    ]
    non_treatment_named = [
        (name, param)
        for name, param in model.named_parameters()
        if not name.startswith(treatment_prefixes)
    ]
    if not treatment_named or not non_treatment_named:
        raise RuntimeError("Failed to split CT treatment and non-treatment parameter groups.")
    return non_treatment_named, treatment_named


def _train_ct_domain_confusion(model, loader, hparams):
    non_treatment_named, treatment_named = _ct_parameter_groups(model)
    non_treatment_optimizer = model._get_optimizer(non_treatment_named)
    treatment_optimizer = model._get_optimizer(treatment_named)

    use_ema = bool(hparams["weights_ema"])
    ema_beta = float(hparams["ema_beta"])
    ema_non_treatment = _ParameterEMA(non_treatment_named, ema_beta) if use_ema else None
    ema_treatment = _ParameterEMA(treatment_named, ema_beta) if use_ema else None

    epochs = int(hparams["ct_epochs"])
    alpha_max = float(hparams["alpha_max"])
    head = model.br_treatment_outcome_head
    head.alpha_max = alpha_max
    head.alpha = 0.0 if bool(hparams["update_alpha"]) else alpha_max

    last_loss = float("nan")
    last_outcome = float("nan")
    last_predict = float("nan")
    last_confuse = float("nan")
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_outcome = 0.0
        epoch_predict = 0.0
        epoch_confuse = 0.0
        denom = 0

        for batch in loader:
            batch = move_batch_to_device(batch)
            base_batch = int(batch["outputs"].shape[0])

            # Outcome/representation update. The treatment classifier is held at
            # its EMA parameters while the representation is optimized to make
            # the predicted treatment distribution uniform.
            _set_requires_grad(treatment_named, False)
            _set_requires_grad(non_treatment_named, True)
            non_treatment_optimizer.zero_grad(set_to_none=True)
            treatment_context = ema_treatment.average_parameters() if use_ema else nullcontext()
            with treatment_context:
                outcome_loss, confuse_loss = _ct_losses(model, batch, treatment_kind="confuse")
                representation_loss = outcome_loss + float(head.alpha) * confuse_loss
                representation_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [param for _, param in non_treatment_named],
                    float(hparams["grad_clip"]),
                )
                non_treatment_optimizer.step()
            if use_ema:
                ema_non_treatment.update()

            # Treatment-classifier update. The non-treatment network is held at
            # its EMA parameters and the balanced representation is detached.
            _set_requires_grad(non_treatment_named, False)
            _set_requires_grad(treatment_named, True)
            treatment_optimizer.zero_grad(set_to_none=True)
            non_treatment_context = ema_non_treatment.average_parameters() if use_ema else nullcontext()
            with non_treatment_context:
                treatment_pred, _, _ = model(batch, detach_treatment=True)
                _, current_treatments, active_entries = _repeat_ct_targets(batch, treatment_pred.shape[0])
                predict_loss = model.bce_loss(
                    treatment_pred,
                    current_treatments.float(),
                    kind="predict",
                )
                predict_loss = masked_sequence_loss(predict_loss, active_entries)
                classifier_loss = float(head.alpha) * predict_loss
                classifier_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [param for _, param in treatment_named],
                    float(hparams["grad_clip"]),
                )
                treatment_optimizer.step()
            if use_ema:
                ema_treatment.update()

            _set_requires_grad(non_treatment_named, True)
            _set_requires_grad(treatment_named, True)

            epoch_loss += float(representation_loss.detach().cpu()) * base_batch
            epoch_outcome += float(outcome_loss.detach().cpu()) * base_batch
            epoch_predict += float(predict_loss.detach().cpu()) * base_batch
            epoch_confuse += float(confuse_loss.detach().cpu()) * base_batch
            denom += base_batch

        last_loss = epoch_loss / max(denom, 1)
        last_outcome = epoch_outcome / max(denom, 1)
        last_predict = epoch_predict / max(denom, 1)
        last_confuse = epoch_confuse / max(denom, 1)
        if bool(hparams["update_alpha"]):
            head.alpha = _sigmoid_alpha(epoch + 1, epochs, alpha_max)

    if use_ema:
        ema_non_treatment.copy_to_model()
        ema_treatment.copy_to_model()

    return {
        "ct_loss": float(last_loss),
        "outcome_loss": float(last_outcome),
        "treatment_loss": float(last_predict),
        "confusion_loss": float(last_confuse),
        "final_alpha": float(head.alpha),
        "fit_time_sec": float(time.time() - t0),
    }


def train_ct_dataset(bundle, hparams, context_idx, seed):
    common.seed_everything(seed)
    bundle = prepare_baseline_bundle(bundle)
    has_vitals = has_dynamic_vitals(bundle)

    treatment_mode = treatment_mode_for_bundle(bundle, cancer_mode="multiclass")
    ds = CTSupportDataset(bundle, context_idx, treatment_mode=treatment_mode)
    d_vitals = int(bundle["covariates"].shape[-1])
    args = make_ct_args(d_vitals=d_vitals, hparams=hparams, d_static=int(bundle["static"].shape[-1]), treatment_mode=treatment_mode)

    model = CT(
        args,
        dataset_collection=None,
        autoregressive=True,
        has_vitals=has_vitals,
        projection_horizon=common.PROJECTION_HORIZON,
        bce_weights=None,
    ).to(common.DEVICE)

    loader = train_loader(ds, int(hparams["batch_size"]))
    if str(hparams["balancing"]) != "domain_confusion":
        raise ValueError("The corrected CT primary implementation requires domain_confusion balancing.")
    metrics = _train_ct_domain_confusion(model, loader, hparams)

    return model, {
        **metrics,
        "train_loss": float(metrics["ct_loss"]),
        "n_train_sequences": int(len(ds)),
        "weights_ema": int(bool(hparams["weights_ema"])),
        "has_dynamic_vitals": int(has_vitals),
    }


def build_ct_rollout_arrays_for_rows(query_bundle, rows, current_ts, target_ts):
    query_bundle = prepare_baseline_bundle(query_bundle)
    rows = np.asarray(rows, dtype=np.int64)
    current_ts = np.asarray(current_ts, dtype=np.int64)
    target_ts = np.asarray(target_ts, dtype=np.int64)
    C, Yc, A, S = query_bundle["covariates"], query_bundle["y_norm_clip"], query_bundle["actions"], query_bundle["static"]
    B = int(len(rows))
    if B == 0:
        raise ValueError("Cannot build empty rollout batch.")
    d = int(C.shape[-1])
    L = int(max(1, min(int(np.max(target_ts)), common.MAX_SEQ_LEN)))
    current_actions = np.zeros((B, L), dtype=np.int64)
    vitals = np.zeros((B, L, d), dtype=np.float32)
    prev_outputs = np.zeros((B, L, 1), dtype=np.float32)
    static_features = np.zeros((B, S.shape[-1]), dtype=np.float32)
    future_past_split = np.zeros(B, dtype=np.int64)
    for j, row_id in enumerate(rows):
        row_id = int(row_id)
        t_obs = max(0, min(int(current_ts[j]), common.MAX_INPUT_INDEX, L - 1))
        for t in range(L):
            if t < A.shape[1]:
                current_actions[j, t] = int(A[row_id, t])
        visible_len = min(t_obs + 1, L, C.shape[1], Yc.shape[1])
        if visible_len > 0:
            vitals[j, :visible_len, :] = C[row_id, :visible_len, :]
            prev_outputs[j, :visible_len, 0] = Yc[row_id, :visible_len]
        static_features[j] = S[row_id]
        future_past_split[j] = min(t_obs + 1, L)
    treatment_mode = treatment_mode_for_bundle(query_bundle, cancer_mode="multiclass")
    current_treatments = encode_actions(current_actions, treatment_mode).astype(np.float32)
    prev_treatments = np.zeros_like(current_treatments)
    prev_treatments[:, 1:] = current_treatments[:, :-1]
    return {
        "prev_treatments": prev_treatments,
        "current_treatments": current_treatments,
        "vitals": vitals,
        "prev_outputs": prev_outputs,
        "active_entries": np.ones((B, L, 1), dtype=np.float32),
        "static_features": static_features,
        "future_past_split": future_past_split,
    }


@torch.no_grad()
def rollout_ct_batch(model, arrays, t_obs_np, t_target_np, return_unclipped=False):
    model.eval()
    prev_treatments = torch.from_numpy(arrays["prev_treatments"]).to(common.DEVICE)
    current_treatments = torch.from_numpy(arrays["current_treatments"]).to(common.DEVICE)
    vitals = torch.from_numpy(arrays["vitals"]).to(common.DEVICE)
    prev_outputs = torch.from_numpy(arrays["prev_outputs"]).to(common.DEVICE)
    active_entries = torch.from_numpy(arrays["active_entries"]).to(common.DEVICE)
    static_features = torch.from_numpy(arrays["static_features"]).to(common.DEVICE)
    future_past_split = torch.from_numpy(arrays["future_past_split"]).long().to(common.DEVICE)
    B, L, _ = prev_outputs.shape
    b_idx = torch.arange(B, device=common.DEVICE)
    t_obs = torch.tensor(np.asarray(t_obs_np, dtype=np.int64), dtype=torch.long, device=common.DEVICE).clamp(0, L - 1)
    t_target = torch.tensor(np.asarray(t_target_np, dtype=np.int64), dtype=torch.long, device=common.DEVICE).clamp(1, common.MAX_TARGET_INDEX)
    horizon = (t_target - t_obs).clamp(min=1, max=common.MAX_SEQ_LEN)
    final_pred = torch.full((B,), float("nan"), dtype=torch.float32, device=common.DEVICE)
    paths = [[] for _ in range(B)]
    unclipped_paths = [[] for _ in range(B)]

    for h in range(int(horizon.max().item())):
        batch = {
            "prev_treatments": prev_treatments,
            "current_treatments": current_treatments,
            "vitals": vitals,
            "prev_outputs": prev_outputs,
            "static_features": static_features,
            "active_entries": active_entries,
            "future_past_split": future_past_split,
        }
        _, y_pred, _ = model(batch)
        cur = (t_obs + h).clamp(0, L - 1)
        active = h < horizon
        pred_unclipped = y_pred[b_idx, cur, 0]
        pred = pred_unclipped.clamp(-common.PRED_CLIP_REPORT, common.PRED_CLIP_REPORT)
        pred_np = pred.detach().float().cpu().numpy()
        pred_unclipped_np = pred_unclipped.detach().float().cpu().numpy()
        active_np = active.detach().cpu().numpy().astype(bool)
        for j in range(B):
            if active_np[j]:
                paths[j].append(float(pred_np[j]))
                unclipped_paths[j].append(float(pred_unclipped_np[j]))
        end_now = active & (cur == (t_target - 1).clamp(0, L - 1))
        if end_now.any():
            final_pred[end_now] = pred[end_now]
        next_time = cur + 1
        write_mask = active & (next_time < L)
        if write_mask.any():
            prev_outputs[b_idx[write_mask], next_time[write_mask], 0] = pred[write_mask].clamp(
                -common.OUTCOME_CLIP_TRAIN,
                common.OUTCOME_CLIP_TRAIN,
            )

    missing = ~torch.isfinite(final_pred)
    if missing.any():
        _, y_pred, _ = model(
            {
                "prev_treatments": prev_treatments,
                "current_treatments": current_treatments,
                "vitals": vitals,
                "prev_outputs": prev_outputs,
                "static_features": static_features,
                "active_entries": active_entries,
                "future_past_split": future_past_split,
            }
        )
        end_pos = (t_target - 1).clamp(0, L - 1)
        missing_idx = torch.nonzero(missing, as_tuple=False).squeeze(-1)
        fallback_unclipped = y_pred[b_idx[missing], end_pos[missing], 0]
        fallback_clipped = fallback_unclipped.clamp(-common.PRED_CLIP_REPORT, common.PRED_CLIP_REPORT)
        final_pred[missing] = fallback_clipped
        for k, j in enumerate(missing_idx.tolist()):
            paths[j].append(float(fallback_clipped[k].item()))
            unclipped_paths[j].append(float(fallback_unclipped[k].item()))
    final = final_pred.detach().float().cpu().numpy().astype(np.float32)
    if return_unclipped:
        return final, paths, unclipped_paths
    return final, paths


def predict_ct_rows(model, query_bundle, rows, current_ts, target_ts, batch_size, return_unclipped=False):
    rows = np.asarray(rows, dtype=np.int64)
    current_ts = np.asarray(current_ts, dtype=np.int64)
    target_ts = np.asarray(target_ts, dtype=np.int64)
    pred = np.zeros(len(rows), dtype=np.float32)
    paths = [None for _ in range(len(rows))]
    unclipped_paths = [None for _ in range(len(rows))]
    elapsed = np.zeros(len(rows), dtype=np.float32)
    for start in range(0, len(rows), int(batch_size)):
        end = min(start + int(batch_size), len(rows))
        arrays = build_ct_rollout_arrays_for_rows(query_bundle, rows[start:end], current_ts[start:end], target_ts[start:end])
        t0 = time.time()
        result = rollout_ct_batch(model, arrays, current_ts[start:end], target_ts[start:end], return_unclipped=return_unclipped)
        if return_unclipped:
            pred_b, paths_b, unclipped_paths_b = result
        else:
            pred_b, paths_b = result
        pred[start:end] = pred_b
        elapsed[start:end] = float((time.time() - t0) / max(1, end - start))
        for k, p in enumerate(paths_b):
            paths[start + k] = p
            if return_unclipped:
                unclipped_paths[start + k] = unclipped_paths_b[k]
    if return_unclipped:
        return pred, paths, unclipped_paths, elapsed
    return pred, paths, elapsed


def evaluate_support_val_rmse(bundle, model, val_idx, seed):
    candidates = support_val_one_step_candidates(bundle, val_idx, seed, MAX_VAL_ORIGINS)
    if not candidates:
        return float("nan")
    rows = np.asarray([c[0] for c in candidates], dtype=np.int64)
    cur = np.asarray([c[1] for c in candidates], dtype=np.int64)
    tgt = np.asarray([c[2] for c in candidates], dtype=np.int64)
    pred, _, _ = predict_ct_rows(model, bundle, rows, cur, tgt, batch_size=CT_EVAL_BATCH_SIZE)
    return rmse_from_predictions(pred, targets_for_candidates(bundle, candidates))


def evaluate_candidate_on_support(bundle, candidate, train_idx, val_idx, seed):
    model, diag = train_ct_dataset(bundle, candidate, train_idx, seed=seed)
    val_rmse = evaluate_support_val_rmse(bundle, model, val_idx, seed=seed + 99)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(val_rmse), diag


def predict_rows(model, query_bundle, rows, current_ts, target_ts):
    pred_norm, pred_paths, pred_unclipped_paths, pred_times = predict_ct_rows(
        model,
        query_bundle,
        rows,
        current_ts,
        target_ts,
        batch_size=CT_EVAL_BATCH_SIZE,
        return_unclipped=True,
    )
    return [
        Prediction(float(pred_norm[idx]), float(pred_times[idx]), path=np.asarray(pred_paths[idx]), pred_norm_unclipped=float(np.asarray(pred_unclipped_paths[idx])[-1]), unclipped_path=np.asarray(pred_unclipped_paths[idx]))
        for idx in range(len(rows))
    ]


def tuning_candidate_label(candidate):
    return (
        f"d={candidate['seq_hidden_units']} br={candidate['br_size']} "
        f"fc={candidate['fc_hidden_units']} layers={candidate['num_layers']} "
        f"heads={candidate['num_heads']} drop={candidate['dropout']} "
        f"lr={candidate['lr']} batch={candidate['batch_size']} "
        f"alpha={candidate['alpha_max']} ema={candidate['weights_ema']}"
    )


ADAPTER = BaselineAdapter(
    method_name="ct",
    method_family="CausalTransformer",
    title="CausalTransformer benchmark evaluation",
    default_hparams=DEFAULT_HPARAMS,
    hyperparameter_space=ct_hyperparameter_space,
    sample_candidates=sample_random_candidates,
    canonical_hparams=canonical_hparams,
    evaluate_candidate=evaluate_candidate_on_support,
    train_final=single_model_train_final(train_ct_dataset),
    predict_rows=predict_rows,
    extra_record_fields=train_diag_record_fields(
        float_keys=("ct_loss", "outcome_loss", "treatment_loss", "confusion_loss", "final_alpha"),
        int_keys=("n_train_sequences", "weights_ema", "has_dynamic_vitals"),
    ),
    extra_meta_fields=lambda _meta: baseline_port_metadata(),
    tuning_candidate_label=tuning_candidate_label,
    output_dir=OUTPUT_DIR,
)


def configure_from_eval_config(baseline_config):
    global CT_EVAL_BATCH_SIZE, MAX_VAL_ORIGINS, OUTPUT_DIR

    ct_config.apply_config(baseline_config)
    CT_EVAL_BATCH_SIZE = ct_config.CT_EVAL_BATCH_SIZE
    MAX_VAL_ORIGINS = ct_config.MAX_VAL_ORIGINS
    OUTPUT_DIR = ct_config.OUTPUT_DIR
    ADAPTER.default_hparams = DEFAULT_HPARAMS
    ADAPTER.output_dir = OUTPUT_DIR
