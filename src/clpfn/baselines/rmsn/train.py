from __future__ import annotations

import gc
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from clpfn.baselines.common.api import BaselineAdapter, baseline_port_metadata
from clpfn.baselines.common.api import (
    paired_model_train_final,
    single_rollout_predict_rows,
    train_diag_record_fields,
)
from clpfn.baselines.rmsn import config as rmsn_config
from clpfn.baselines.rmsn.config import (
    BASE_RMSN_SPACE,
    DEFAULT_HPARAMS,
    MAX_TRAIN_ORIGINS,
    MAX_VAL_ORIGINS,
    OUTPUT_DIR,
    PROP_BATCH_SIZE,
    TUNING_CACHE,
    action4_to_bits,
    build_group_scaled_space,
    canonical_hparams,
    clip_normalize_weights,
    make_rmsn_args,
    sample_random_candidates,
)
from clpfn.baselines.rmsn.data import (
    RMSNDecoderOriginDataset,
    RMSNEncoderSupportDataset,
    RMSNPropensityDataset,
    get_encoder_representations,
    make_encoder_full_arrays,
    move_batch_to_device,
    sample_decoder_origins,
)
from clpfn.baselines.models.rmsn import (
    RMSNDecoder,
    RMSNEncoder,
    RMSNPropensityNetworkHistory,
    RMSNPropensityNetworkTreatment,
)
from clpfn.baselines.common.training import evaluate_paired_rollout_val_rmse
from clpfn.evaluation.core import benchmark as common

def train_propensity_models(bundle, hparams, context_idx, seed):
    common.seed_everything(seed)

    d_vitals = int(bundle["covariates"].shape[-1])
    args = make_rmsn_args(d_vitals=d_vitals, hparams=hparams)

    ds = RMSNPropensityDataset(bundle, context_idx)
    loader = DataLoader(
        ds,
        batch_size=min(PROP_BATCH_SIZE, len(ds)),
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    prop_treat = RMSNPropensityNetworkTreatment(
        args,
        dataset_collection=None,
        autoregressive=True,
        has_vitals=True,
        bce_weights=None,
    ).to(common.DEVICE)

    prop_hist = RMSNPropensityNetworkHistory(
        args,
        dataset_collection=None,
        autoregressive=True,
        has_vitals=True,
        bce_weights=None,
    ).to(common.DEVICE)

    opt_treat = prop_treat.configure_optimizers()
    opt_hist = prop_hist.configure_optimizers()

    last_treat_loss = float("nan")
    last_hist_loss = float("nan")

    for _ in range(int(hparams["propensity_epochs"])):
        treat_losses = []
        hist_losses = []

        for batch_ind, batch in enumerate(loader):
            batch = move_batch_to_device(batch)

            prop_treat.train()
            loss_t = prop_treat.training_step(batch, batch_ind)

            opt_treat.zero_grad(set_to_none=True)
            loss_t.backward()
            torch.nn.utils.clip_grad_norm_(prop_treat.parameters(), float(hparams["max_grad_norm"]))
            opt_treat.step()

            prop_hist.train()
            loss_h = prop_hist.training_step(batch, batch_ind)

            opt_hist.zero_grad(set_to_none=True)
            loss_h.backward()
            torch.nn.utils.clip_grad_norm_(prop_hist.parameters(), float(hparams["max_grad_norm"]))
            opt_hist.step()

            treat_losses.append(float(loss_t.detach().cpu()))
            hist_losses.append(float(loss_h.detach().cpu()))

        last_treat_loss = float(np.mean(treat_losses)) if treat_losses else float("nan")
        last_hist_loss = float(np.mean(hist_losses)) if hist_losses else float("nan")

    return prop_treat, prop_hist, {
        "propensity_treatment_loss": last_treat_loss,
        "propensity_history_loss": last_hist_loss,
    }


@torch.no_grad()
def compute_stabilized_weights(prop_treat, prop_hist, train_data, hparams):
    active = train_data["active_entries"].squeeze(-1).astype(np.float32)

    prop_treat.eval()
    prop_hist.eval()

    batch = {
        "prev_treatments": torch.from_numpy(train_data["prev_treatments"]).to(common.DEVICE),
        "current_treatments": torch.from_numpy(train_data["current_treatments"]).to(common.DEVICE),
        "vitals": torch.from_numpy(train_data["vitals"]).to(common.DEVICE),
        "prev_outputs": torch.from_numpy(train_data["prev_outputs"]).to(common.DEVICE),
        "static_features": torch.from_numpy(train_data["static_features"]).to(common.DEVICE),
        "active_entries": torch.from_numpy(train_data["active_entries"]).to(common.DEVICE),
    }

    logits_treat = prop_treat(batch)
    logits_hist = prop_hist(batch)

    p_treat = torch.sigmoid(logits_treat).detach().cpu().numpy()
    p_hist = torch.sigmoid(logits_hist).detach().cpu().numpy()

    obs = train_data["current_treatments"].astype(np.float32)

    eps = 1e-4
    p_treat = np.clip(p_treat, eps, 1.0 - eps)
    p_hist = np.clip(p_hist, eps, 1.0 - eps)

    prob_num_bits = p_treat * obs + (1.0 - p_treat) * (1.0 - obs)
    prob_den_bits = p_hist * obs + (1.0 - p_hist) * (1.0 - obs)

    prob_num = np.prod(prob_num_bits, axis=-1)
    prob_den = np.prod(prob_den_bits, axis=-1)

    sw = prob_num / np.maximum(prob_den, eps)
    sw = clip_normalize_weights(
        sw,
        active=active,
        quantiles=hparams.get("weight_clip_quantiles", [0.01, 0.99]),
        multiple_horizons=False,
    )

    return sw.astype(np.float32)


def train_rmsn_models(bundle, hparams, context_idx, seed):
    common.seed_everything(seed)

    d_vitals = int(bundle["covariates"].shape[-1])
    args = make_rmsn_args(d_vitals=d_vitals, hparams=hparams)

    t0 = time.time()

    train_data = make_encoder_full_arrays(bundle, context_idx)

    prop_treat, prop_hist, prop_diag = train_propensity_models(
        bundle,
        hparams,
        context_idx,
        seed=seed + 11,
    )

    sw_tilde_enc = compute_stabilized_weights(
        prop_treat,
        prop_hist,
        train_data,
        hparams,
    )

    encoder_ds = RMSNEncoderSupportDataset(train_data, sw_tilde_enc)
    encoder_loader = DataLoader(
        encoder_ds,
        batch_size=min(int(hparams["batch_size_encoder"]), len(encoder_ds)),
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    encoder = RMSNEncoder(
        args,
        propensity_treatment=prop_treat,
        propensity_history=prop_hist,
        dataset_collection=None,
        autoregressive=True,
        has_vitals=True,
        bce_weights=None,
    ).to(common.DEVICE)

    opt_enc = encoder.configure_optimizers()

    encoder.train()
    last_enc_loss = float("nan")

    for _ in range(int(hparams["encoder_epochs"])):
        epoch_loss = 0.0
        denom = 0

        for batch_ind, batch in enumerate(encoder_loader):
            batch = move_batch_to_device(batch)

            loss = encoder.training_step(batch, batch_ind)

            opt_enc.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), float(hparams["max_grad_norm"]))
            opt_enc.step()

            bs = batch["outputs"].shape[0]
            epoch_loss += float(loss.detach().cpu()) * bs
            denom += bs

        last_enc_loss = epoch_loss / max(denom, 1)

    r_all = get_encoder_representations(encoder, train_data)

    origins = sample_decoder_origins(train_data, seed=seed + 17, max_origins=MAX_TRAIN_ORIGINS)
    decoder_ds = RMSNDecoderOriginDataset(train_data, origins, r_all, sw_tilde_enc, hparams)

    decoder_loader = DataLoader(
        decoder_ds,
        batch_size=min(int(hparams["batch_size_decoder"]), len(decoder_ds)),
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    decoder = RMSNDecoder(
        args,
        encoder=None,
        dataset_collection=None,
        encoder_r_size=int(encoder.seq_hidden_units),
        autoregressive=True,
        has_vitals=True,
        bce_weights=None,
    ).to(common.DEVICE)

    opt_dec = decoder.configure_optimizers()

    decoder.train()
    last_dec_loss = float("nan")

    for _ in range(int(hparams["decoder_epochs"])):
        epoch_loss = 0.0
        denom = 0

        for batch_ind, batch in enumerate(decoder_loader):
            batch = move_batch_to_device(batch)

            loss = decoder.training_step(batch, batch_ind)

            opt_dec.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), float(hparams["max_grad_norm"]))
            opt_dec.step()

            bs = batch["outputs"].shape[0]
            epoch_loss += float(loss.detach().cpu()) * bs
            denom += bs

        last_dec_loss = epoch_loss / max(denom, 1)

    diagnostics = {
        "propensity_treatment_loss": float(prop_diag.get("propensity_treatment_loss", np.nan)),
        "propensity_history_loss": float(prop_diag.get("propensity_history_loss", np.nan)),
        "encoder_loss": float(last_enc_loss),
        "decoder_loss": float(last_dec_loss),
        "train_loss": float(last_dec_loss),
        "fit_time_sec": float(time.time() - t0),
        "n_encoder_sequences": int(len(encoder_ds)),
        "n_decoder_origins": int(len(decoder_ds)),
    }

    del prop_treat, prop_hist

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return encoder, decoder, diagnostics


def predict_single_rollout(encoder, decoder, bundle, row_id, t_obs, t_target):
    C, Yc, A, S = bundle["covariates"], bundle["y_norm_clip"], bundle["actions"], bundle["static"]
    row_id, t_obs, t_target = int(row_id), int(t_obs), int(t_target)
    tau = max(1, int(t_target - t_obs))
    hist_end = max(min(t_obs + 1, C.shape[1], Yc.shape[1], A.shape[1]), 1)
    q_hist = {
        "vitals": torch.from_numpy(C[row_id:row_id + 1, :hist_end, :].astype(np.float32)).to(common.DEVICE),
        "prev_outputs": torch.from_numpy(Yc[row_id:row_id + 1, :hist_end, None].astype(np.float32)).to(common.DEVICE),
        "current_treatments": torch.from_numpy(action4_to_bits(A[row_id:row_id + 1, :hist_end])).to(common.DEVICE),
        "static_features": torch.from_numpy(S[row_id:row_id + 1].astype(np.float32)).to(common.DEVICE),
    }
    encoder.eval()
    decoder.eval()
    _, r_hist = encoder(q_hist)
    init_state = r_hist[:, -1, :]
    seed_y = float(Yc[row_id, t_obs]) if t_obs < Yc.shape[1] else float(Yc[row_id, hist_end - 1])
    pred_values, prev_values = [], [seed_y]
    static_features = torch.from_numpy(S[row_id:row_id + 1].astype(np.float32)).to(common.DEVICE)

    for h in range(tau):
        cur_len = h + 1
        actions = np.zeros((1, cur_len), dtype=np.int64)
        prev_outputs = np.zeros((1, cur_len, 1), dtype=np.float32)
        for k in range(cur_len):
            action_t = t_obs + k
            if action_t < A.shape[1]:
                actions[0, k] = int(A[row_id, action_t])
            prev_outputs[0, k, 0] = float(prev_values[k])
        out_seq = decoder(
            {
                "current_treatments": torch.from_numpy(action4_to_bits(actions)).to(common.DEVICE),
                "prev_outputs": torch.from_numpy(prev_outputs).to(common.DEVICE),
                "static_features": static_features,
                "init_state": init_state,
            }
        ).detach().float().cpu().numpy()[0, :, 0]
        next_y = float(np.clip(out_seq[-1], -common.PRED_CLIP_REPORT, common.PRED_CLIP_REPORT))
        pred_values.append(next_y)
        prev_values.append(float(np.clip(next_y, -common.OUTCOME_CLIP_TRAIN, common.OUTCOME_CLIP_TRAIN)))

    pred_path = np.asarray(pred_values, dtype=np.float32)
    return float(pred_path[-1]), pred_path


def evaluate_candidate_on_support(bundle, candidate, train_idx, val_idx, seed):
    encoder, decoder, diag = train_rmsn_models(bundle, candidate, train_idx, seed=seed)
    val_rmse = evaluate_paired_rollout_val_rmse(
        bundle,
        encoder,
        decoder,
        val_idx,
        seed=seed + 99,
        max_val_origins=MAX_VAL_ORIGINS,
        predict_fn=predict_single_rollout,
    )
    del encoder, decoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(val_rmse), diag


def hyperparameter_space(bundle):
    return build_group_scaled_space(int(bundle["covariates"].shape[-1]))


def tuning_candidate_label(candidate):
    return (
        f"enc={candidate['hidden_units_encoder']} dec={candidate['hidden_units_decoder']} "
        f"layers={candidate['num_layers']} drop={candidate['dropout']} "
        f"lr_p/e/d={candidate['lr_prop']}/{candidate['lr_enc']}/{candidate['lr_dec']}"
    )


ADAPTER = BaselineAdapter(
    method_name="rmsn",
    method_family="RMSN",
    title="RMSN benchmark evaluation",
    default_hparams=DEFAULT_HPARAMS,
    tuning_cache=TUNING_CACHE,
    hyperparameter_space=hyperparameter_space,
    sample_candidates=sample_random_candidates,
    canonical_hparams=canonical_hparams,
    evaluate_candidate=evaluate_candidate_on_support,
    train_final=paired_model_train_final(train_rmsn_models),
    predict_rows=single_rollout_predict_rows(predict_single_rollout, unpack_pair_payload=True),
    extra_record_fields=train_diag_record_fields(
        float_keys=(
            "propensity_treatment_loss",
            "propensity_history_loss",
            "encoder_loss",
            "decoder_loss",
        ),
        int_keys=("n_encoder_sequences", "n_decoder_origins"),
    ),
    extra_meta_fields=lambda _meta: baseline_port_metadata(),
    tuning_candidate_label=tuning_candidate_label,
    output_dir=OUTPUT_DIR,
)


def configure_from_eval_config(baseline_config):
    global MAX_TRAIN_ORIGINS, MAX_VAL_ORIGINS, OUTPUT_DIR, PROP_BATCH_SIZE

    rmsn_config.apply_config(baseline_config)
    MAX_TRAIN_ORIGINS = rmsn_config.MAX_TRAIN_ORIGINS
    MAX_VAL_ORIGINS = rmsn_config.MAX_VAL_ORIGINS
    PROP_BATCH_SIZE = rmsn_config.PROP_BATCH_SIZE
    OUTPUT_DIR = rmsn_config.OUTPUT_DIR
    ADAPTER.default_hparams = DEFAULT_HPARAMS
    ADAPTER.output_dir = OUTPUT_DIR
