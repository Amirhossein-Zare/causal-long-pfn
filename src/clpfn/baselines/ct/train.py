from __future__ import annotations

import gc
import time
from types import SimpleNamespace

import numpy as np
import torch

from clpfn.baselines.common.api import BaselineAdapter, Prediction, baseline_port_metadata, canonical_hparams
from clpfn.baselines.common.api import (
    single_model_train_final,
    train_diag_record_fields,
)
from clpfn.baselines.common.training import run_epoch_training, train_loader
from clpfn.baselines.common.tuning import sample_random_hparams
from clpfn.baselines.common.training import (
    masked_mse_loss,
    masked_sequence_loss,
    rmse_from_predictions,
    support_val_candidates,
    targets_for_candidates,
)
from clpfn.baselines.ct import config as ct_config
from clpfn.baselines.ct.config import (
    CT_EVAL_BATCH_SIZE,
    CT_SPACE,
    DEFAULT_HPARAMS,
    MAX_VAL_ORIGINS,
    OUTPUT_DIR,
    TUNING_CACHE,
)
from clpfn.baselines.ct.data import CTSupportDataset, move_batch_to_device
from clpfn.baselines.models.ct import CT
from clpfn.evaluation.core import benchmark as common


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def sample_random_candidates(space, n, seed):
    def transform(sample):
        hp = dict(DEFAULT_HPARAMS)
        hp.update(sample)
        hp["max_position_len"] = int(common.MAX_SEQ_LEN)
        hp["grl_alpha"] = float(hp.get("grl_alpha", 0.10))
        hp["weight_decay"] = float(hp.get("weight_decay", 1e-5))
        hp["optimizer_cls"] = str(hp.get("optimizer_cls", "adamw"))
        return hp

    return sample_random_hparams(
        space,
        n,
        seed,
        default_hparams=DEFAULT_HPARAMS,
        canonical_hparams=canonical_hparams,
        transform_sample=transform,
        is_valid=lambda hp: int(hp["seq_hidden_units"]) % int(hp["num_heads"]) == 0,
    )


def make_ct_args(d_vitals, hparams):
    seq_hidden_units = int(hparams["seq_hidden_units"])
    num_heads = int(hparams["num_heads"])
    if seq_hidden_units % num_heads != 0:
        raise ValueError(f"seq_hidden_units={seq_hidden_units} must be divisible by num_heads={num_heads}")
    multi = ns(
        seq_hidden_units=seq_hidden_units,
        br_size=int(hparams["br_size"]),
        fc_hidden_units=int(hparams["fc_hidden_units"]),
        dropout_rate=float(hparams["dropout"]),
        num_layer=int(hparams["num_layers"]),
        num_heads=num_heads,
        head_size=int(seq_hidden_units // num_heads),
        alpha=float(hparams.get("grl_alpha", 0.0)),
        update_alpha=False,
        balancing="grad_reverse",
        treatment_loss_weight=float(hparams["treatment_loss_weight"]),
        batch_size=int(hparams["batch_size"]),
        max_grad_norm=float(hparams["grad_clip"]),
        max_position_len=int(hparams.get("max_position_len", common.MAX_SEQ_LEN)),
        self_positional_encoding="none",
        trainable_positional_encoding=False,
        max_relative_position=int(common.MAX_SEQ_LEN),
        attn_dropout=True,
        disable_cross_attention=False,
        isolate_subnetwork="",
        augment_with_masked_vitals=bool(hparams.get("augment_with_masked_vitals", False)),
        optimizer={
            "learning_rate": float(hparams["lr"]),
            "weight_decay": float(hparams.get("weight_decay", 1e-5)),
            "optimizer_cls": str(hparams.get("optimizer_cls", "adamw")),
            "lr_scheduler": False,
        },
    )
    return ns(
        model=ns(
            dim_treatments=common.N_ACTIONS,
            dim_vitals=int(d_vitals),
            dim_static_features=common.D_STATIC_MAX,
            dim_outcomes=1,
            multi=multi,
        ),
        dataset=ns(
            val_batch_size=int(hparams["batch_size"]),
            projection_horizon=common.PROJECTION_HORIZON,
            treatment_mode="multiclass",
            holdout_ratio=0.0,
        ),
        exp=ns(
            unscale_rmse=False,
            percentage_rmse=False,
            bce_weight=False,
            gpus="[]",
            max_epochs=int(hparams["ct_epochs"]),
            alpha_rate="exp",
            update_alpha=False,
        ),
    )


def _ct_step(hparams):
    def _step(model, batch, _batch_ind):
        treatment_pred, outcome_pred, _ = model(batch)

        outcome_loss = masked_mse_loss(outcome_pred, batch["outputs"], batch["active_entries"])

        treatment_loss = model.bce_loss(
            treatment_pred,
            batch["current_treatments"].float(),
            kind="predict",
        )
        treatment_loss = masked_sequence_loss(treatment_loss, batch["active_entries"])

        loss = outcome_loss + float(hparams["treatment_loss_weight"]) * treatment_loss
        return loss, {
            "outcome_loss": outcome_loss,
            "treatment_loss": treatment_loss,
        }

    return _step


def train_ct_dataset(bundle, hparams, context_idx, seed):
    common.seed_everything(seed)

    ds = CTSupportDataset(bundle, context_idx)
    d_vitals = int(bundle["covariates"].shape[-1])
    args = make_ct_args(d_vitals=d_vitals, hparams=hparams)

    model = CT(
        args,
        dataset_collection=None,
        autoregressive=True,
        has_vitals=True,
        projection_horizon=common.PROJECTION_HORIZON,
        bce_weights=None,
    ).to(common.DEVICE)

    opt = model.configure_optimizers()
    loader = train_loader(ds, int(hparams["batch_size"]))
    last_loss, last_metrics, fit_time_sec = run_epoch_training(
        model,
        loader,
        opt,
        epochs=int(hparams["ct_epochs"]),
        grad_clip=float(hparams["grad_clip"]),
        move_batch_to_device=move_batch_to_device,
        step_fn=_ct_step(hparams),
    )

    return model, {
        "ct_loss": float(last_loss),
        "outcome_loss": float(last_metrics.get("outcome_loss", float("nan"))),
        "treatment_loss": float(last_metrics.get("treatment_loss", float("nan"))),
        "train_loss": float(last_loss),
        "fit_time_sec": float(fit_time_sec),
        "n_train_sequences": int(len(ds)),
    }


def build_ct_rollout_arrays_for_rows(query_bundle, rows, current_ts, target_ts):
    rows = np.asarray(rows, dtype=np.int64)
    current_ts = np.asarray(current_ts, dtype=np.int64)
    target_ts = np.asarray(target_ts, dtype=np.int64)
    C, Yc, A, S = query_bundle["covariates"], query_bundle["y_norm_clip"], query_bundle["actions"], query_bundle["static"]
    B = int(len(rows))
    if B == 0:
        raise ValueError("Cannot build empty rollout batch.")
    d = int(C.shape[-1])
    L = int(max(1, min(int(np.max(target_ts)), common.MAX_SEQ_LEN)))
    prev_actions = np.zeros((B, L), dtype=np.int64)
    current_actions = np.zeros((B, L), dtype=np.int64)
    vitals = np.zeros((B, L, d), dtype=np.float32)
    prev_outputs = np.zeros((B, L, 1), dtype=np.float32)
    static_features = np.zeros((B, common.D_STATIC_MAX), dtype=np.float32)
    future_past_split = np.zeros(B, dtype=np.int64)
    for j, row_id in enumerate(rows):
        row_id = int(row_id)
        t_obs = max(0, min(int(current_ts[j]), common.MAX_INPUT_INDEX, L - 1))
        for t in range(L):
            if t > 0 and (t - 1) < A.shape[1]:
                prev_actions[j, t] = int(A[row_id, t - 1])
            if t < A.shape[1]:
                current_actions[j, t] = int(A[row_id, t])
        visible_len = min(t_obs + 1, L, C.shape[1], Yc.shape[1])
        if visible_len > 0:
            vitals[j, :visible_len, :] = C[row_id, :visible_len, :]
            prev_outputs[j, :visible_len, 0] = Yc[row_id, :visible_len]
        static_features[j] = S[row_id]
        future_past_split[j] = min(t_obs + 1, L)
    return {
        "prev_treatments": common.action_onehot_2d(prev_actions, common.N_ACTIONS).astype(np.float32),
        "current_treatments": common.action_onehot_2d(current_actions, common.N_ACTIONS).astype(np.float32),
        "vitals": vitals,
        "prev_outputs": prev_outputs,
        "active_entries": np.ones((B, L, 1), dtype=np.float32),
        "static_features": static_features,
        "future_past_split": future_past_split,
    }


@torch.no_grad()
def rollout_ct_batch(model, arrays, t_obs_np, t_target_np):
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
        pred = y_pred[b_idx, cur, 0].clamp(-common.PRED_CLIP_REPORT, common.PRED_CLIP_REPORT)
        pred_np = pred.detach().float().cpu().numpy()
        active_np = active.detach().cpu().numpy().astype(bool)
        for j in range(B):
            if active_np[j]:
                paths[j].append(float(pred_np[j]))
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
        final_pred[missing] = y_pred[b_idx[missing], end_pos[missing], 0].clamp(
            -common.PRED_CLIP_REPORT,
            common.PRED_CLIP_REPORT,
        )
    return final_pred.detach().float().cpu().numpy().astype(np.float32), paths


def predict_ct_rows(model, query_bundle, rows, current_ts, target_ts, batch_size=CT_EVAL_BATCH_SIZE):
    rows = np.asarray(rows, dtype=np.int64)
    current_ts = np.asarray(current_ts, dtype=np.int64)
    target_ts = np.asarray(target_ts, dtype=np.int64)
    pred = np.zeros(len(rows), dtype=np.float32)
    paths = [None for _ in range(len(rows))]
    elapsed = np.zeros(len(rows), dtype=np.float32)
    for start in range(0, len(rows), int(batch_size)):
        end = min(start + int(batch_size), len(rows))
        arrays = build_ct_rollout_arrays_for_rows(query_bundle, rows[start:end], current_ts[start:end], target_ts[start:end])
        t0 = time.time()
        pred_b, paths_b = rollout_ct_batch(model, arrays, current_ts[start:end], target_ts[start:end])
        pred[start:end] = pred_b
        elapsed[start:end] = float((time.time() - t0) / max(1, end - start))
        for k, p in enumerate(paths_b):
            paths[start + k] = p
    return pred, paths, elapsed


def evaluate_support_val_rmse(bundle, model, val_idx, seed):
    candidates = support_val_candidates(bundle, val_idx, seed, MAX_VAL_ORIGINS)
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
    pred_norm, pred_paths, pred_times = predict_ct_rows(
        model,
        query_bundle,
        rows,
        current_ts,
        target_ts,
        batch_size=CT_EVAL_BATCH_SIZE,
    )
    return [
        Prediction(float(pred_norm[idx]), float(pred_times[idx]), path=np.asarray(pred_paths[idx]))
        for idx in range(len(rows))
    ]


def tuning_candidate_label(candidate):
    return (
        f"d={candidate['seq_hidden_units']} br={candidate['br_size']} "
        f"fc={candidate['fc_hidden_units']} layers={candidate['num_layers']} "
        f"heads={candidate['num_heads']} drop={candidate['dropout']} "
        f"lr={candidate['lr']} batch={candidate['batch_size']} "
        f"tw={candidate['treatment_loss_weight']}"
    )


ADAPTER = BaselineAdapter(
    method_name="ct",
    method_family="CausalTransformer",
    title="CausalTransformer benchmark evaluation",
    default_hparams=DEFAULT_HPARAMS,
    tuning_cache=TUNING_CACHE,
    hyperparameter_space=lambda _bundle: (CT_SPACE, {}),
    sample_candidates=sample_random_candidates,
    canonical_hparams=canonical_hparams,
    evaluate_candidate=evaluate_candidate_on_support,
    train_final=single_model_train_final(train_ct_dataset),
    predict_rows=predict_rows,
    extra_record_fields=train_diag_record_fields(
        float_keys=("ct_loss", "outcome_loss", "treatment_loss"),
        int_keys=("n_train_sequences",),
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
