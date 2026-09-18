from __future__ import annotations

import time

import numpy as np

from clpfn.baselines.common.features import prepare_baseline_bundle, treatment_mode_for_bundle
from clpfn.baselines.common.api import (
    BaselineAdapter,
    Prediction,
    TrainedArtifacts,
    baseline_port_metadata,
    canonical_hparams,
)
from clpfn.baselines.common.tuning import all_grid_candidates as build_grid_candidates
from clpfn.baselines.msm import config as msm_config
from clpfn.baselines.msm.data import max_anchor_for_tau
from clpfn.baselines.msm.config import (
    DEFAULT_HPARAMS,
    MAX_VAL_ORIGINS,
    MSM_SPACE,
    OUTPUT_DIR,
)
from clpfn.baselines.models.msm import MSMRegressor
from clpfn.evaluation.core import benchmark as common


def fit_msm_dataset(bundle, hparams, context_indices):
    bundle = prepare_baseline_bundle(bundle)
    treatment_mode = treatment_mode_for_bundle(bundle, cancer_mode="multilabel")
    model = MSMRegressor.from_hparams(
        hparams, dim_static_features=bundle["static"].shape[-1], treatment_mode=treatment_mode
    ).fit_bundle(bundle, context_indices)
    return model, model.train_diag


def predict_msm_single(model, bundle, row_id, t_obs, t_target):
    bundle = prepare_baseline_bundle(bundle)
    return model.predict_single(bundle, row_id, t_obs, t_target)


def evaluate_support_val_rmse(bundle, model, val_idx, seed):
    bundle = prepare_baseline_bundle(bundle)
    if len(val_idx) == 0:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    Y, Yraw = bundle["y_norm_clip"], bundle["y_raw"]
    candidates = []
    for i in np.asarray(val_idx, dtype=np.int64):
        for tau in (1,):
            max_anchor = max_anchor_for_tau(bundle, int(i), int(tau))
            if max_anchor < common.MIN_T_OBS:
                continue
            for t in range(common.MIN_T_OBS, max_anchor + 1):
                target_t = t + int(tau)
                if target_t < Y.shape[1] and np.isfinite(Yraw[i, target_t]):
                    candidates.append((int(i), int(t), int(target_t)))
    if not candidates:
        return float("nan")
    if len(candidates) > MAX_VAL_ORIGINS:
        keep = rng.choice(len(candidates), size=MAX_VAL_ORIGINS, replace=False)
        candidates = [candidates[k] for k in keep]
    sq = []
    for i, t_obs, t_target in candidates:
        pred, _ = predict_msm_single(model, bundle, i, t_obs, t_target)
        target = float(Y[i, t_target])
        if np.isfinite(pred) and np.isfinite(target):
            sq.append((pred - target) ** 2)
    return float(np.sqrt(np.mean(sq))) if sq else float("nan")


def evaluate_candidate_on_support(bundle, candidate, train_idx, val_idx, seed):
    bundle = prepare_baseline_bundle(bundle)
    common.seed_everything(seed)
    model, diag = fit_msm_dataset(bundle, candidate, train_idx)
    val_rmse = evaluate_support_val_rmse(bundle, model, val_idx, seed=seed + 99)
    return float(val_rmse), diag


def sample_random_candidates(space, n, seed):
    rng = np.random.default_rng(int(seed))
    candidates = build_grid_candidates(
        space,
        default_hparams=DEFAULT_HPARAMS,
        canonical_hparams=canonical_hparams,
    )
    if len(candidates) <= int(n):
        return [candidates[int(i)] for i in rng.permutation(len(candidates))]
    return [candidates[int(i)] for i in rng.choice(len(candidates), size=int(n), replace=False)]


def train_final(bundle, hparams, context_idx, seed):
    bundle = prepare_baseline_bundle(bundle)
    common.seed_everything(seed)
    model, train_diag = fit_msm_dataset(bundle, hparams, context_idx)
    return TrainedArtifacts(payload=model, train_diag=train_diag)


def predict_rows(payload, query_bundle, rows, current_ts, target_ts):
    out = []
    for row_id, current_t, target_t in zip(rows, current_ts, target_ts):
        started = time.time()
        horizon = int(target_t - current_t)
        if horizon < 1:
            raise ValueError("MSM target time must be after observation time.")
        if horizon == 1:
            pred_norm, info = predict_msm_single(payload, query_bundle, int(row_id), int(current_t), int(target_t))
            out.append(Prediction(float(pred_norm), float(time.time() - started), pred_norm_unclipped=float(info["prediction_model_norm_unclipped"]), info=info))
            continue
        reported_path, unclipped_path, info = [], [], None
        for tau in range(1, horizon + 1):
            pred_norm, step_info = predict_msm_single(payload, query_bundle, int(row_id), int(current_t), int(current_t + tau))
            reported_path.append(float(pred_norm))
            unclipped_path.append(float(step_info["prediction_model_norm_unclipped"]))
            info = step_info
        out.append(Prediction(float(reported_path[-1]), float(time.time() - started), path=np.asarray(reported_path, dtype=np.float32), pred_norm_unclipped=float(unclipped_path[-1]), unclipped_path=np.asarray(unclipped_path, dtype=np.float32), info=info))
    return out


def extra_record_fields(train_diag, prediction, _tune_info, _meta):
    info = prediction.info
    return {
        "train_rmse_norm": float(info["train_rmse_norm"]),
        "mean_weight": float(info["mean_weight"]),
        "pred_status": str(info["pred_status"]),
        "regressor_status": str(info["regressor_status"]),
        "propensity_status": str(train_diag["propensity_status"]),
        "n_propensity": int(train_diag["n_propensity"]),
        "mean_weight_raw": float(train_diag["mean_weight_raw"]),
        "direct_msm_horizon_model": True,
    }


def tuning_diag_fields(diag):
    return {
        "train_loss": float(diag["train_loss"]),
        "propensity_status": str(diag["propensity_status"]),
        "n_propensity": int(diag["n_propensity"]),
    }


def tuning_candidate_label(candidate):
    return (
        f"lag={candidate['lag_features']} reg={candidate['regressor']} alpha={candidate['ridge_alpha']}"
    )


ADAPTER = BaselineAdapter(
    method_name="msm",
    method_family="MSM",
    title="MSM benchmark evaluation",
    default_hparams=DEFAULT_HPARAMS,
    hyperparameter_space=lambda _bundle: (MSM_SPACE, {}),
    sample_candidates=sample_random_candidates,
    canonical_hparams=canonical_hparams,
    evaluate_candidate=evaluate_candidate_on_support,
    train_final=train_final,
    predict_rows=predict_rows,
    extra_record_fields=extra_record_fields,
    extra_meta_fields=lambda _meta: baseline_port_metadata(
        device="cpu/sklearn",
        msm_multilabel_propensity="observed_treatment_likelihood",
    ),
    tuning_diag_fields=tuning_diag_fields,
    tuning_candidate_label=tuning_candidate_label,
    output_dir=OUTPUT_DIR,
    device_label="cpu/sklearn",
)


def configure_from_eval_config(baseline_config):
    global MAX_VAL_ORIGINS, OUTPUT_DIR

    msm_config.apply_config(baseline_config)
    MAX_VAL_ORIGINS = msm_config.MAX_VAL_ORIGINS
    OUTPUT_DIR = msm_config.OUTPUT_DIR
    ADAPTER.default_hparams = DEFAULT_HPARAMS
    ADAPTER.output_dir = OUTPUT_DIR
