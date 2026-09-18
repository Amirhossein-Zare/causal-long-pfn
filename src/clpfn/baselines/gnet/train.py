from __future__ import annotations

import gc
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from clpfn.baselines.common.features import (
    encode_actions,
    has_dynamic_vitals,
    prepare_baseline_bundle,
    treatment_dim,
    treatment_mode_for_bundle,
)
from clpfn.baselines.common.api import BaselineAdapter, baseline_port_metadata, canonical_hparams
from clpfn.baselines.common.api import (
    single_model_train_final,
    single_rollout_predict_rows,
    train_diag_record_fields,
)
from clpfn.baselines.common.training import (
    rmse_from_predictions,
    support_val_one_step_candidates,
    targets_for_candidates,
)
from clpfn.baselines.common.training import run_epoch_training, train_loader
from clpfn.baselines.common.tuning import sample_random_hparams
from clpfn.baselines.gnet import config as gnet_config
from clpfn.baselines.gnet.config import (
    DEFAULT_HPARAMS,
    GNET_SPACE,
    MAX_VAL_ORIGINS,
    OUTPUT_DIR,
)
from clpfn.baselines.gnet.data import GNetSupportDataset, move_batch_to_device
from clpfn.baselines.models.gnet import GNet
from clpfn.evaluation.core import benchmark as common


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def _has_dynamic_vitals(bundle):
    return has_dynamic_vitals(bundle)


def _input_dim_for_bundle(bundle):
    bundle = prepare_baseline_bundle(bundle)
    d_vitals = int(bundle["covariates"].shape[-1]) if _has_dynamic_vitals(bundle) else 0
    d_static = int(bundle["static"].shape[-1])
    mode = treatment_mode_for_bundle(bundle, cancer_mode="multiclass")
    return int(treatment_dim(mode) + d_static + 1 + d_vitals)


def space_sample_to_hparams(sample):
    hp = dict(DEFAULT_HPARAMS)
    input_dim = int(sample["_input_dim"])
    max_width = int(sample["max_width"])
    seq_hidden = max(2, min(max_width, int(input_dim * float(sample["hidden_multiplier"]))))
    r_size = max(2, min(max_width, int(input_dim * float(sample["r_size_multiplier"]))))
    fc_hidden = max(2, min(max_width, int(seq_hidden * float(sample["fc_hidden_multiplier"]))))
    hp["seq_hidden_units"] = int(seq_hidden)
    hp["r_size"] = int(r_size)
    hp["fc_hidden_units"] = int(fc_hidden)
    hp["num_layer"] = int(sample["num_layers"])
    hp["dropout_rate"] = float(sample["dropout"])
    hp["learning_rate"] = float(sample["lr"])
    hp["batch_size"] = int(sample["batch_size"])
    hp["epochs"] = int(sample["epochs"])
    hp["optimizer_cls"] = "adam"
    hp["weight_decay"] = 0.0
    hp["mc_samples"] = int(DEFAULT_HPARAMS["mc_samples"])
    hp["residual_holdout_ratio"] = float(DEFAULT_HPARAMS["residual_holdout_ratio"])
    return canonical_hparams(hp)


def sample_random_candidates(space, n, seed):
    return sample_random_hparams(
        space,
        n,
        seed,
        default_hparams=DEFAULT_HPARAMS,
        canonical_hparams=canonical_hparams,
        transform_sample=space_sample_to_hparams,
    )


def make_gnet_args(d_vitals, hparams, d_static, treatment_mode):
    output_size = 1 + int(d_vitals)
    d_static = int(d_static)
    gnet = ns(
        dropout_rate=float(hparams["dropout_rate"]),
        seq_hidden_units=int(hparams["seq_hidden_units"]),
        r_size=int(hparams["r_size"]),
        num_layer=int(hparams["num_layer"]),
        comp_sizes=[output_size],
        num_comp=1,
        fc_hidden_units=int(hparams["fc_hidden_units"]),
        mc_samples=int(hparams["mc_samples"]),
        fit_vitals=bool(d_vitals > 0),
        vitals_loss_weight=float(hparams["vitals_loss_weight"]),
        batch_size=int(hparams["batch_size"]),
        optimizer={
            "learning_rate": float(hparams["learning_rate"]),
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
            g_net=gnet,
        ),
        dataset=ns(
            val_batch_size=int(hparams["batch_size"]),
            projection_horizon=common.PROJECTION_HORIZON,
            treatment_mode=str(treatment_mode),
            holdout_ratio=float(hparams["residual_holdout_ratio"]),
        ),
        exp=ns(
            unscale_rmse=False,
            percentage_rmse=False,
            bce_weight=False,
            gpus="[]",
            max_epochs=int(hparams["epochs"]),
            alpha_rate="exp",
            update_alpha=False,
        ),
    )


def _split_fit_and_residual_indices(context_idx, holdout_ratio, seed):
    idx = np.asarray(context_idx, dtype=np.int64)
    if idx.size < 6:
        raise ValueError("GNet requires at least six support sequences.")
    if not 0.0 < float(holdout_ratio) < 1.0:
        raise ValueError("GNet residual_holdout_ratio must be between zero and one.")

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(idx)
    n_holdout = max(1, int(round(idx.size * float(holdout_ratio))))
    n_holdout = min(n_holdout, max(1, idx.size - 5))
    return perm[n_holdout:].astype(np.int64), perm[:n_holdout].astype(np.int64)


@torch.no_grad()
def _fit_empirical_residuals(model, bundle, holdout_idx, *, has_vitals, batch_size):
    if len(holdout_idx) == 0:
        raise ValueError("GNet residual fitting requires non-empty holdout indices.")

    ds = GNetSupportDataset(
        bundle, holdout_idx, has_vitals=has_vitals,
        treatment_mode=treatment_mode_for_bundle(bundle, cancer_mode="multiclass"),
    )
    loader = DataLoader(ds, batch_size=max(1, min(int(batch_size), len(ds))), shuffle=False)
    residual_rows = []
    residual_lengths = []

    model.eval()
    for batch in loader:
        device_batch = move_batch_to_device(batch)
        pred = model(device_batch).detach().float().cpu().numpy()
        target_y = batch["outputs"].numpy()
        active = batch["active_entries"].numpy()[:, :, 0]

        if has_vitals:
            target_vitals = batch["next_vitals"].numpy()
            joint_steps = min(pred.shape[1] - 1, target_vitals.shape[1])
            resid = np.concatenate(
                (
                    target_y[:, :joint_steps, :] - pred[:, :joint_steps, :1],
                    target_vitals[:, :joint_steps, :] - pred[:, :joint_steps, 1:],
                ),
                axis=-1,
            )
            lengths = np.minimum(active.sum(axis=1).astype(np.int64) - 1, joint_steps)
        else:
            resid = target_y - pred[:, :, :1]
            lengths = np.minimum(active.sum(axis=1).astype(np.int64), resid.shape[1])

        lengths = np.maximum(lengths, 0)
        residual_rows.append(resid.astype(np.float32))
        residual_lengths.append(lengths.astype(np.int64))

    if not residual_rows:
        raise RuntimeError("GNet residual fitting produced no residual rows.")

    model.holdout_resid = np.concatenate(residual_rows, axis=0)
    model.holdout_resid_len = np.concatenate(residual_lengths, axis=0)
    valid = model.holdout_resid_len > 0
    if not np.any(valid):
        raise RuntimeError("GNet residual fitting produced no valid residual sequences.")

    model.holdout_resid = model.holdout_resid[valid]
    model.holdout_resid_len = model.holdout_resid_len[valid]
    return int(valid.sum())


def train_gnet_model(bundle, hparams, context_idx, seed):
    common.seed_everything(seed)
    bundle = prepare_baseline_bundle(bundle)

    has_vitals = _has_dynamic_vitals(bundle)
    d_vitals = int(bundle["covariates"].shape[-1]) if has_vitals else 0
    fit_idx, residual_idx = _split_fit_and_residual_indices(
        context_idx,
        hparams["residual_holdout_ratio"],
        seed + 17,
    )

    treatment_mode = treatment_mode_for_bundle(bundle, cancer_mode="multiclass")
    ds = GNetSupportDataset(bundle, fit_idx, has_vitals=has_vitals, treatment_mode=treatment_mode)
    args = make_gnet_args(
        d_vitals=d_vitals, hparams=hparams, d_static=int(bundle["static"].shape[-1]),
        treatment_mode=treatment_mode,
    )

    model = GNet(
        args,
        dataset_collection=None,
        autoregressive=True,
        has_vitals=has_vitals,
        projection_horizon=common.PROJECTION_HORIZON,
        bce_weights=None,
    ).to(common.DEVICE)

    opt = model.configure_optimizers()
    loader = train_loader(ds, int(hparams["batch_size"]))
    last_loss, _, fit_time_sec = run_epoch_training(
        model,
        loader,
        opt,
        epochs=int(hparams["epochs"]),
        grad_clip=float(hparams["grad_clip"]),
        move_batch_to_device=move_batch_to_device,
    )

    n_residual_sequences = _fit_empirical_residuals(
        model,
        bundle,
        residual_idx,
        has_vitals=has_vitals,
        batch_size=int(hparams["batch_size"]),
    )
    model.rollout_seed = int(seed)
    model.mc_samples = int(hparams["mc_samples"])
    if model.mc_samples < 1:
        raise ValueError("GNet mc_samples must be positive.")
    model.has_dynamic_vitals = bool(has_vitals)

    return model, {
        "train_loss": float(last_loss),
        "fit_time_sec": float(fit_time_sec),
        "n_train_sequences": int(len(ds)),
        "n_residual_sequences": int(n_residual_sequences),
        "has_dynamic_vitals": int(has_vitals),
    }


def _observed_value(arr, row_id, t):
    t = int(t)
    if t < 0 or t >= arr.shape[1]:
        raise IndexError(f"Time index {t} is outside array length {arr.shape[1]}.")
    return arr[row_id, t]


def _sample_residual(model, rng, step, output_size):
    residuals = model.holdout_resid
    lengths = model.holdout_resid_len
    if len(residuals) == 0:
        raise RuntimeError("GNet has no fitted residual sequences.")

    ridx = int(rng.integers(0, len(residuals)))
    rlen = max(1, int(lengths[ridx]))
    tidx = min(int(step), rlen - 1, residuals.shape[1] - 1)
    resid = np.asarray(residuals[ridx, tidx], dtype=np.float32)
    if resid.shape[0] != output_size:
        raise RuntimeError(
            f"GNet residual width mismatch: expected {output_size}, got {resid.shape[0]}."
        )
    return resid


@torch.no_grad()
def predict_single_rollout(model, bundle, row_id, t_obs, t_target, return_unclipped=False):
    bundle = prepare_baseline_bundle(bundle)
    C, Yc, A, S = bundle["covariates"], bundle["y_norm_clip"], bundle["actions"], bundle["static"]
    row_id, t_obs, t_target = int(row_id), int(t_obs), int(t_target)
    tau = int(t_target - t_obs)
    if tau < 1:
        raise ValueError("GNet target time must be after observation time.")
    has_vitals = bool(model.has_dynamic_vitals)
    d = int(C.shape[-1]) if has_vitals else 0
    n_mc = 1 if tau == 1 else int(model.mc_samples)
    seq_len = t_obs + tau
    if seq_len > A.shape[1]:
        raise ValueError("GNet query actions do not cover the requested rollout.")

    action_idx = np.zeros((1, seq_len), dtype=np.int64)
    action_idx[0] = A[row_id, :seq_len]
    treatments_np = np.repeat(
        encode_actions(action_idx, treatment_mode_for_bundle(bundle, cancer_mode="multiclass")).astype(np.float32), n_mc, axis=0
    )

    prev_outputs_np = np.zeros((n_mc, seq_len, 1), dtype=np.float32)
    vitals_np = np.zeros((n_mc, seq_len, d), dtype=np.float32)
    visible = min(t_obs + 1, seq_len)
    for t in range(seq_len):
        if t < visible:
            y_t = float(_observed_value(Yc, row_id, t))
            cov_t = _observed_value(C, row_id, t) if has_vitals else None
        else:
            y_t = float(_observed_value(Yc, row_id, t_obs))
            cov_t = _observed_value(C, row_id, t_obs) if has_vitals else None
        prev_outputs_np[:, t, 0] = y_t
        if has_vitals:
            vitals_np[:, t, :] = np.asarray(cov_t, dtype=np.float32)

    treatments = torch.from_numpy(treatments_np).to(common.DEVICE)
    prev_outputs = torch.from_numpy(prev_outputs_np).to(common.DEVICE)
    vitals = torch.from_numpy(vitals_np).to(common.DEVICE)
    static = torch.from_numpy(
        np.repeat(S[row_id].astype(np.float32).reshape(1, -1), n_mc, axis=0)
    ).to(common.DEVICE)

    rngs = [
        np.random.default_rng(
            int(model.rollout_seed)
            + 1000003 * row_id
            + 1009 * t_obs
            + 97 * t_target
            + mc
        )
        for mc in range(n_mc)
    ]

    preds_unclipped = np.zeros((n_mc, tau), dtype=np.float32)

    model.eval()
    for h in range(tau):
        eval_t = t_obs + h
        out = model(
            {
                "current_treatments": treatments[:, : eval_t + 1],
                "vitals": vitals[:, : eval_t + 1],
                "prev_outputs": prev_outputs[:, : eval_t + 1],
                "static_features": static,
            }
        )[:, eval_t, : 1 + d]
        out_np = out.detach().float().cpu().numpy()

        preds_unclipped[:, h] = out_np[:, 0]

        next_t = eval_t + 1
        if tau > 1 and next_t < seq_len:
            residual = np.stack(
                [_sample_residual(model, rngs[mc], eval_t, 1 + d) for mc in range(n_mc)], axis=0
            )
            sampled = out_np + residual
            prev_outputs[:, next_t, 0] = torch.from_numpy(
                np.clip(sampled[:, 0], -common.OUTCOME_CLIP_TRAIN, common.OUTCOME_CLIP_TRAIN)
                .astype(np.float32)
            ).to(common.DEVICE)
            if has_vitals:
                vitals[:, next_t, :] = torch.from_numpy(
                    np.clip(sampled[:, 1 : 1 + d], -common.STATE_CLIP_TRAIN, common.STATE_CLIP_TRAIN)
                    .astype(np.float32)
                ).to(common.DEVICE)

    pred_path_unclipped = preds_unclipped.mean(axis=0).astype(np.float32)
    pred_path = np.clip(
        pred_path_unclipped, -common.PRED_CLIP_REPORT, common.PRED_CLIP_REPORT
    ).astype(np.float32)
    if return_unclipped:
        return float(pred_path[-1]), pred_path, pred_path_unclipped
    return float(pred_path[-1]), pred_path


def evaluate_candidate_on_support(bundle, candidate, train_idx, val_idx, seed):
    model, diag = train_gnet_model(bundle, candidate, train_idx, seed=seed)
    final_mc_samples = int(model.mc_samples)
    model.mc_samples = 1
    candidates = support_val_one_step_candidates(bundle, val_idx, seed + 99, MAX_VAL_ORIGINS)
    predictions = [
        predict_single_rollout(model, bundle, row_id, t_obs, t_target)[0]
        for row_id, t_obs, t_target in candidates
    ]
    val_rmse = rmse_from_predictions(predictions, targets_for_candidates(bundle, candidates)) if candidates else float("nan")
    model.mc_samples = final_mc_samples
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(val_rmse), diag


def hyperparameter_space(bundle):
    input_dim = _input_dim_for_bundle(bundle)
    max_width = int(GNET_SPACE["max_width"][0])
    space = {
        "_input_dim": [int(input_dim)],
        "hidden_multiplier": list(GNET_SPACE["seq_hidden_multiplier"]),
        "r_size_multiplier": list(GNET_SPACE["r_size_multiplier"]),
        "fc_hidden_multiplier": list(GNET_SPACE["fc_hidden_multiplier"]),
        "max_width": [int(max_width)],
        "num_layers": list(GNET_SPACE["num_layers"]),
        "dropout": list(GNET_SPACE["dropout"]),
        "lr": list(GNET_SPACE["lr"]),
        "batch_size": list(GNET_SPACE["batch_size"]),
        "epochs": list(GNET_SPACE["epochs"]),
    }
    return space, {
        "space": space,
        "input_dim": int(input_dim),
        "has_dynamic_vitals": bool(_has_dynamic_vitals(bundle)),
        "n_possible_combinations": int(np.prod([len(values) for values in space.values()])),
    }


def tuning_candidate_label(candidate):
    return (
        f"hidden={candidate['seq_hidden_units']} r={candidate['r_size']} "
        f"fc={candidate['fc_hidden_units']} layers={candidate['num_layer']} "
        f"drop={candidate['dropout_rate']} lr={candidate['learning_rate']} "
        f"batch={candidate['batch_size']} epochs={candidate['epochs']}"
    )


ADAPTER = BaselineAdapter(
    method_name="gnet",
    method_family="GNet",
    title="GNet benchmark evaluation",
    default_hparams=DEFAULT_HPARAMS,
    hyperparameter_space=hyperparameter_space,
    sample_candidates=sample_random_candidates,
    canonical_hparams=canonical_hparams,
    evaluate_candidate=evaluate_candidate_on_support,
    train_final=single_model_train_final(train_gnet_model),
    predict_rows=single_rollout_predict_rows(predict_single_rollout),
    extra_record_fields=train_diag_record_fields(
        int_keys=("n_train_sequences", "n_residual_sequences", "has_dynamic_vitals"),
    ),
    extra_meta_fields=lambda _meta: baseline_port_metadata(
        gnet_output_components=1,
        gnet_vitals_loss_weight=float(DEFAULT_HPARAMS["vitals_loss_weight"]),
        gnet_residual_holdout_ratio=float(DEFAULT_HPARAMS["residual_holdout_ratio"]),
        gnet_mc_samples_final=int(DEFAULT_HPARAMS["mc_samples"]),
        gnet_mc_samples_tuning=1,
        gnet_cancer_dynamic_vitals=False,
    ),
    tuning_candidate_label=tuning_candidate_label,
    output_dir=OUTPUT_DIR,
)


def configure_from_eval_config(baseline_config):
    global MAX_VAL_ORIGINS, OUTPUT_DIR

    gnet_config.apply_config(baseline_config)
    MAX_VAL_ORIGINS = gnet_config.MAX_VAL_ORIGINS
    OUTPUT_DIR = gnet_config.OUTPUT_DIR
    ADAPTER.default_hparams = DEFAULT_HPARAMS
    ADAPTER.output_dir = OUTPUT_DIR
