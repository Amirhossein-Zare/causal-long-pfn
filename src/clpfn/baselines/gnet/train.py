from __future__ import annotations

import gc
from types import SimpleNamespace

import numpy as np
import torch

from clpfn.baselines.common.api import BaselineAdapter, baseline_port_metadata, canonical_hparams
from clpfn.baselines.common.api import (
    single_model_train_final,
    single_rollout_predict_rows,
    train_diag_record_fields,
)
from clpfn.baselines.common.training import run_epoch_training, train_loader
from clpfn.baselines.common.tuning import sample_random_hparams
from clpfn.baselines.common.training import evaluate_single_rollout_val_rmse
from clpfn.baselines.gnet import config as gnet_config
from clpfn.baselines.gnet.config import (
    DEFAULT_HPARAMS,
    GNET_SPACE,
    MAX_VAL_ORIGINS,
    OUTPUT_DIR,
    TUNING_CACHE,
)
from clpfn.baselines.gnet.data import GNetSupportDataset, move_batch_to_device
from clpfn.baselines.models.gnet import GNet
from clpfn.evaluation.core import benchmark as common


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def space_sample_to_hparams(sample):
    hp = dict(DEFAULT_HPARAMS)
    hp["seq_hidden_units"] = int(sample["hidden_size"])
    hp["r_size"] = int(sample["r_size"])
    hp["fc_hidden_units"] = int(sample["r_size"])
    hp["num_layer"] = int(sample["num_layers"])
    hp["dropout_rate"] = float(sample["dropout"])
    hp["learning_rate"] = float(sample["lr"])
    hp["batch_size"] = int(sample["batch_size"])
    hp["epochs"] = int(sample["epochs"])
    hp["vitals_loss_weight"] = float(sample["vitals_loss_weight"])
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


def make_gnet_args(d_vitals, hparams):
    gnet = ns(
        dropout_rate=float(hparams["dropout_rate"]),
        seq_hidden_units=int(hparams["seq_hidden_units"]),
        r_size=int(hparams["r_size"]),
        num_layer=int(hparams["num_layer"]),
        comp_sizes=[1, int(d_vitals)],
        num_comp=2,
        fc_hidden_units=int(hparams["fc_hidden_units"]),
        mc_samples=int(hparams["mc_samples"]),
        fit_vitals=True,
        vitals_loss_weight=float(hparams.get("vitals_loss_weight", 1.0)),
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
            dim_treatments=common.N_ACTIONS,
            dim_vitals=int(d_vitals),
            dim_static_features=common.D_STATIC_MAX,
            dim_outcomes=1,
            g_net=gnet,
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
            max_epochs=int(hparams["epochs"]),
            alpha_rate="exp",
            update_alpha=False,
        ),
    )


def train_gnet_model(bundle, hparams, context_idx, seed):
    common.seed_everything(seed)

    ds = GNetSupportDataset(bundle, context_idx)
    d_vitals = int(bundle["covariates"].shape[-1])
    args = make_gnet_args(d_vitals=d_vitals, hparams=hparams)

    model = GNet(
        args,
        dataset_collection=None,
        autoregressive=True,
        has_vitals=True,
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

    return model, {
        "train_loss": float(last_loss),
        "fit_time_sec": float(fit_time_sec),
        "n_train_sequences": int(len(ds)),
    }


def _observed_or_last(arr, row_id, t):
    t = int(t)
    return arr[row_id, t] if t < arr.shape[1] else arr[row_id, arr.shape[1] - 1]


@torch.no_grad()
def predict_single_rollout(model, bundle, row_id, t_obs, t_target):
    C, Yc, A, S = bundle["covariates"], bundle["y_norm_clip"], bundle["actions"], bundle["static"]
    row_id, t_obs, t_target = int(row_id), int(t_obs), int(t_target)
    tau = max(1, int(t_target - t_obs))
    d = int(C.shape[-1])
    static = S[row_id].astype(np.float32)
    future_y, future_cov, preds = {}, {}, []

    model.eval()
    for h in range(tau):
        eval_t = t_obs + h
        seq_len = eval_t + 1
        treatments = np.zeros((1, seq_len, common.N_ACTIONS), dtype=np.float32)
        vitals = np.zeros((1, seq_len, d), dtype=np.float32)
        prev_outputs = np.zeros((1, seq_len, 1), dtype=np.float32)

        for t in range(seq_len):
            treatments[0, t] = common.onehot_action(int(A[row_id, t]) if t < A.shape[1] else 0)
            if t <= t_obs:
                cov_t = _observed_or_last(C, row_id, t).astype(np.float32)
                y_t = float(_observed_or_last(Yc, row_id, t))
            else:
                cov_t = future_cov.get(t)
                y_t = future_y.get(t)
                if cov_t is None:
                    cov_t = _observed_or_last(C, row_id, min(t_obs, C.shape[1] - 1)).astype(np.float32)
                if y_t is None:
                    y_t = float(_observed_or_last(Yc, row_id, min(t_obs, Yc.shape[1] - 1)))
            vitals[0, t] = np.asarray(cov_t, dtype=np.float32)
            prev_outputs[0, t, 0] = float(y_t)

        out = model(
            {
                "current_treatments": torch.from_numpy(treatments).to(common.DEVICE),
                "vitals": torch.from_numpy(vitals).to(common.DEVICE),
                "prev_outputs": torch.from_numpy(prev_outputs).to(common.DEVICE),
                "static_features": torch.from_numpy(static.reshape(1, -1)).to(common.DEVICE),
            }
        ).detach().float().cpu().numpy()[0, eval_t]

        next_y = float(np.clip(out[0], -common.PRED_CLIP_REPORT, common.PRED_CLIP_REPORT))
        preds.append(next_y)
        next_t = eval_t + 1
        future_y[next_t] = float(np.clip(next_y, -common.OUTCOME_CLIP_TRAIN, common.OUTCOME_CLIP_TRAIN))
        future_cov[next_t] = np.clip(out[1:1 + d], -common.STATE_CLIP_TRAIN, common.STATE_CLIP_TRAIN).astype(np.float32)

    pred_path = np.asarray(preds, dtype=np.float32)
    return float(pred_path[-1]), pred_path


def evaluate_candidate_on_support(bundle, candidate, train_idx, val_idx, seed):
    model, diag = train_gnet_model(bundle, candidate, train_idx, seed=seed)
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


def hyperparameter_space(_bundle):
    return GNET_SPACE, {
        "space": GNET_SPACE,
        "n_possible_combinations": int(np.prod([len(values) for values in GNET_SPACE.values()])),
    }


def tuning_candidate_label(candidate):
    return (
        f"hidden={candidate['seq_hidden_units']} r={candidate['r_size']} "
        f"layers={candidate['num_layer']} drop={candidate['dropout_rate']} "
        f"lr={candidate['learning_rate']} batch={candidate['batch_size']} "
        f"epochs={candidate['epochs']} vitals_w={candidate['vitals_loss_weight']}"
    )


ADAPTER = BaselineAdapter(
    method_name="gnet",
    method_family="GNet",
    title="GNet benchmark evaluation",
    default_hparams=DEFAULT_HPARAMS,
    tuning_cache=TUNING_CACHE,
    hyperparameter_space=hyperparameter_space,
    sample_candidates=sample_random_candidates,
    canonical_hparams=canonical_hparams,
    evaluate_candidate=evaluate_candidate_on_support,
    train_final=single_model_train_final(train_gnet_model),
    predict_rows=single_rollout_predict_rows(predict_single_rollout),
    extra_record_fields=train_diag_record_fields(int_keys=("n_train_sequences",)),
    extra_meta_fields=lambda _meta: baseline_port_metadata(
        gnet_extension="vitals_loss_weight is an explicit CLPFN tuning parameter",
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
