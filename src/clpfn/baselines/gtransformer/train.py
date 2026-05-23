from __future__ import annotations

import gc
from types import SimpleNamespace

import numpy as np
import torch

from clpfn.baselines.common.api import BaselineAdapter, baseline_port_metadata, canonical_hparams
from clpfn.baselines.common.api import (
    single_model_train_final,
    single_rollout_predict_rows,
)
from clpfn.baselines.common.tuning import sample_random_hparams
from clpfn.baselines.common.training import evaluate_single_rollout_val_rmse
from clpfn.baselines.gtransformer import config as gt_config
from clpfn.baselines.gtransformer.config import (
    DEFAULT_HPARAMS,
    GTRANSFORMER_SPACE,
    GT_PROJECTION_HORIZON,
    MAX_VAL_ORIGINS,
    OUTPUT_DIR,
    TUNING_CACHE,
)
from clpfn.baselines.gtransformer.data import GTSupportDataset, move_batch_to_device
from clpfn.baselines.models.gt import GT
from clpfn.baselines.common.training import run_epoch_training, train_loader
from clpfn.evaluation.core import benchmark as common


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def valid_gt_hparams(hp):
    d_model = int(hp["d_model"])
    num_heads = int(hp["num_heads"])
    return d_model % num_heads == 0 and d_model % 2 == 0


def sample_random_candidates(space, n, seed):
    return sample_random_hparams(
        space,
        n,
        seed,
        default_hparams=DEFAULT_HPARAMS,
        canonical_hparams=canonical_hparams,
        is_valid=valid_gt_hparams,
        min_attempts=1000,
        attempts_per_candidate=200,
    )


def make_gt_args(d_vitals, hparams, projection_horizon=None):
    projection_horizon = GT_PROJECTION_HORIZON if projection_horizon is None else projection_horizon
    treatment_sequence = np.zeros((common.PROJECTION_HORIZON + 1, common.N_ACTIONS), dtype=np.float32)
    treatment_sequence[:, 0] = 1.0
    gt = ns(
        max_seq_length=int(common.MAX_SEQ_LEN),
        hr_size=int(hparams["br_size"]),
        seq_hidden_units=int(hparams["d_model"]),
        fc_hidden_units=int(hparams["fc_hidden_units"]),
        dropout_rate=float(hparams["dropout"]),
        num_layer=int(hparams["num_layers"]),
        num_heads=int(hparams["num_heads"]),
        projection_horizon=int(projection_horizon),
        attn_dropout=True,
        disable_cross_attention=False,
        isolate_subnetwork="",
        self_positional_encoding=ns(absolute=True, trainable=False, max_relative_position=15),
        batch_size=int(hparams["batch_size"]),
        optimizer={
            "learning_rate": float(hparams["lr"]),
            "weight_decay": float(hparams["weight_decay"]),
            "optimizer_cls": "adamw",
            "lr_scheduler": False,
        },
    )
    return ns(
        model=ns(
            dim_treatments=common.N_ACTIONS,
            dim_vitals=int(d_vitals),
            dim_static_features=common.D_STATIC_MAX,
            dim_outcomes=1,
            gt=gt,
        ),
        dataset=ns(
            val_batch_size=int(hparams["batch_size"]),
            projection_horizon=int(common.PROJECTION_HORIZON),
            treatment_mode="multiclass",
            holdout_ratio=0.0,
            treatment_sequence=treatment_sequence.tolist(),
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


def train_gt_model(bundle, hparams, context_idx, seed):
    common.seed_everything(seed)

    ds = GTSupportDataset(bundle, context_idx)
    d_vitals = int(bundle["covariates"].shape[-1])

    args = make_gt_args(d_vitals=d_vitals, hparams=hparams)

    model = GT(
        args,
        dataset_collection=None,
        autoregressive=True,
        has_vitals=True,
        projection_horizon=GT_PROJECTION_HORIZON,
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

    return model, {
        "gt_loss": float(last_loss),
        "train_loss": float(last_loss),
        "fit_time_sec": float(fit_time_sec),
        "n_train_sequences": int(len(ds)),
        "projection_horizon": int(GT_PROJECTION_HORIZON),
    }


def predict_single_rollout(model, bundle, row_id, t_obs, t_target):
    model.eval()
    C, Yc, A, S = bundle["covariates"], bundle["y_norm_clip"], bundle["actions"], bundle["static"]
    row_id, t_obs, t_target = int(row_id), int(t_obs), int(t_target)
    tau = max(1, int(t_target - t_obs))
    l_roll = max(1, min(int(max(t_obs + 1, t_target)), common.MAX_SEQ_LEN))
    d_base = int(C.shape[-1])
    prev_actions = np.zeros((1, l_roll), dtype=np.int64)
    current_actions = np.zeros((1, l_roll), dtype=np.int64)
    for pos in range(l_roll):
        if pos > 0 and (pos - 1) < A.shape[1]:
            prev_actions[0, pos] = int(A[row_id, pos - 1])
        if pos < A.shape[1]:
            current_actions[0, pos] = int(A[row_id, pos])

    vitals = np.zeros((1, l_roll, d_base), dtype=np.float32)
    visible_len = min(t_obs + 1, C.shape[1], l_roll)
    if visible_len > 0:
        vitals[0, :visible_len, :] = C[row_id, :visible_len, :]

    prev_outputs = np.zeros((1, l_roll, 1), dtype=np.float32)
    y_visible = min(t_obs + 1, Yc.shape[1], l_roll)
    if y_visible > 0:
        prev_outputs[0, :y_visible, 0] = Yc[row_id, :y_visible]

    pt = torch.from_numpy(common.action_onehot_2d(prev_actions)).to(common.DEVICE)
    ct = torch.from_numpy(common.action_onehot_2d(current_actions)).to(common.DEVICE)
    vi = torch.from_numpy(vitals).to(common.DEVICE)
    po = torch.from_numpy(prev_outputs).to(common.DEVICE)
    sf = torch.from_numpy(S[row_id:row_id + 1].astype(np.float32)).to(common.DEVICE)
    ae = torch.ones((1, l_roll, 1), dtype=torch.float32, device=common.DEVICE)
    preds = []

    for h in range(tau):
        cur = min(int(t_obs + h), l_roll - 1)
        hr = model.build_hr(
            prev_treatments=pt,
            vitals=vi,
            prev_outputs=po,
            static_features=sf,
            active_entries=ae,
        )
        pred = torch.clamp(
            model.G_comp_heads[0].build_outcome(hr, ct)[:, cur, 0].detach(),
            -common.PRED_CLIP_REPORT,
            common.PRED_CLIP_REPORT,
        )
        preds.append(float(pred[0].detach().cpu()))
        next_time = cur + 1
        if next_time < l_roll:
            po[:, next_time, 0] = pred

    pred_path = np.asarray(preds, dtype=np.float32)
    return float(pred_path[-1]), pred_path


def evaluate_candidate_on_support(bundle, candidate, train_idx, val_idx, seed):
    model, diag = train_gt_model(bundle, candidate, train_idx, seed=seed)
    val_rmse = evaluate_single_rollout_val_rmse(
        bundle,
        model,
        val_idx,
        seed + 99,
        MAX_VAL_ORIGINS,
        predict_single_rollout,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(val_rmse), diag


def extra_record_fields(train_diag, _prediction, _tune_info, _meta):
    return {
        "gt_loss": float(train_diag.get("gt_loss", np.nan)),
        "projection_horizon": int(train_diag.get("projection_horizon", GT_PROJECTION_HORIZON)),
        "n_train_sequences": int(train_diag.get("n_train_sequences", 0)),
    }


def tuning_candidate_label(candidate):
    return (
        f"d_model={candidate['d_model']} br={candidate['br_size']} "
        f"fc={candidate['fc_hidden_units']} layers={candidate['num_layers']} "
        f"heads={candidate['num_heads']} drop={candidate['dropout']} "
        f"lr={candidate['lr']} wd={candidate['weight_decay']} bs={candidate['batch_size']}"
    )


ADAPTER = BaselineAdapter(
    method_name="gtransformer",
    method_family="G-Transformer",
    title="G-Transformer benchmark evaluation",
    default_hparams=DEFAULT_HPARAMS,
    tuning_cache=TUNING_CACHE,
    hyperparameter_space=lambda _bundle: (GTRANSFORMER_SPACE, {}),
    sample_candidates=sample_random_candidates,
    canonical_hparams=canonical_hparams,
    evaluate_candidate=evaluate_candidate_on_support,
    train_final=single_model_train_final(train_gt_model),
    predict_rows=single_rollout_predict_rows(predict_single_rollout),
    extra_record_fields=extra_record_fields,
    extra_meta_fields=lambda _meta: baseline_port_metadata(projection_horizon=int(GT_PROJECTION_HORIZON)),
    tuning_candidate_label=tuning_candidate_label,
    output_dir=OUTPUT_DIR,
)


def configure_from_eval_config(baseline_config):
    global GT_PROJECTION_HORIZON, MAX_VAL_ORIGINS, OUTPUT_DIR

    gt_config.apply_config(baseline_config)
    GT_PROJECTION_HORIZON = gt_config.GT_PROJECTION_HORIZON
    MAX_VAL_ORIGINS = gt_config.MAX_VAL_ORIGINS
    OUTPUT_DIR = gt_config.OUTPUT_DIR
    ADAPTER.default_hparams = DEFAULT_HPARAMS
    ADAPTER.output_dir = OUTPUT_DIR
