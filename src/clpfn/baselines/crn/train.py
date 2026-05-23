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
from clpfn.baselines.crn import config as crn_config
from clpfn.baselines.crn.config import (
    BASE_CRN_SPACE,
    DEFAULT_HPARAMS,
    MAX_TRAIN_ORIGINS,
    MAX_VAL_ORIGINS,
    OUTPUT_DIR,
    TUNING_CACHE,
    build_group_scaled_space,
    canonical_hparams,
    make_crn_args,
    onehot_actions,
    sample_random_candidates,
)
from clpfn.baselines.crn.data import (
    CRNDecoderOriginDataset,
    CRNEncoderSupportDataset,
    crn_loss,
    get_encoder_br_sequence,
    move_batch_to_device,
    sample_decoder_origins,
)
from clpfn.baselines.models.crn import CRNDecoder, CRNEncoder
from clpfn.baselines.common.training import evaluate_paired_rollout_val_rmse
from clpfn.evaluation.core import benchmark as common

def train_crn_models(bundle, hparams, context_idx, seed):
    common.seed_everything(seed)
    d_vitals = int(bundle["covariates"].shape[-1])
    args = make_crn_args(d_vitals=d_vitals, hparams=hparams)
    t0 = time.time()

    encoder_ds = CRNEncoderSupportDataset(bundle, context_idx)
    encoder_loader = DataLoader(
        encoder_ds,
        batch_size=min(int(hparams["batch_size_encoder"]), len(encoder_ds)),
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    encoder = CRNEncoder(args, dataset_collection=None, autoregressive=True, has_vitals=True, bce_weights=None).to(common.DEVICE)
    opt_enc = encoder.configure_optimizers()
    last_enc_loss = last_enc_outcome_loss = last_enc_treatment_loss = float("nan")
    for _ in range(int(hparams["encoder_epochs"])):
        losses, outcome_losses, treatment_losses = [], [], []
        encoder.train()
        for batch in encoder_loader:
            batch = move_batch_to_device(batch)
            treatment_pred, outcome_pred, _ = encoder(batch)
            loss, outcome_loss, treatment_loss = crn_loss(treatment_pred, outcome_pred, batch, hparams)
            opt_enc.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), float(hparams["grad_clip"]))
            opt_enc.step()
            losses.append(float(loss.detach().cpu()))
            outcome_losses.append(float(outcome_loss.detach().cpu()))
            treatment_losses.append(float(treatment_loss.detach().cpu()))
        last_enc_loss = float(np.mean(losses)) if losses else float("nan")
        last_enc_outcome_loss = float(np.mean(outcome_losses)) if outcome_losses else float("nan")
        last_enc_treatment_loss = float(np.mean(treatment_losses)) if treatment_losses else float("nan")

    br_all = get_encoder_br_sequence(encoder, encoder_ds)
    origins = sample_decoder_origins(encoder_ds.raw, seed=seed + 17, max_origins=MAX_TRAIN_ORIGINS)
    decoder_ds = CRNDecoderOriginDataset(encoder_ds.raw, br_all, origins)
    decoder_loader = DataLoader(
        decoder_ds,
        batch_size=min(int(hparams["batch_size_decoder"]), len(decoder_ds)),
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    decoder = CRNDecoder(
        args,
        encoder=None,
        dataset_collection=None,
        encoder_r_size=int(encoder.br_size),
        autoregressive=True,
        has_vitals=True,
        bce_weights=None,
    ).to(common.DEVICE)
    opt_dec = decoder.configure_optimizers()
    last_dec_loss = last_dec_outcome_loss = last_dec_treatment_loss = float("nan")
    for _ in range(int(hparams["decoder_epochs"])):
        losses, outcome_losses, treatment_losses = [], [], []
        decoder.train()
        for batch in decoder_loader:
            batch = move_batch_to_device(batch)
            treatment_pred, outcome_pred, _ = decoder(batch)
            loss, outcome_loss, treatment_loss = crn_loss(treatment_pred, outcome_pred, batch, hparams)
            opt_dec.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), float(hparams["grad_clip"]))
            opt_dec.step()
            losses.append(float(loss.detach().cpu()))
            outcome_losses.append(float(outcome_loss.detach().cpu()))
            treatment_losses.append(float(treatment_loss.detach().cpu()))
        last_dec_loss = float(np.mean(losses)) if losses else float("nan")
        last_dec_outcome_loss = float(np.mean(outcome_losses)) if outcome_losses else float("nan")
        last_dec_treatment_loss = float(np.mean(treatment_losses)) if treatment_losses else float("nan")

    diagnostics = {
        "encoder_loss": float(last_enc_loss),
        "encoder_outcome_loss": float(last_enc_outcome_loss),
        "encoder_treatment_loss": float(last_enc_treatment_loss),
        "decoder_loss": float(last_dec_loss),
        "decoder_outcome_loss": float(last_dec_outcome_loss),
        "decoder_treatment_loss": float(last_dec_treatment_loss),
        "train_loss": float(last_dec_loss),
        "fit_time_sec": float(time.time() - t0),
        "n_encoder_sequences": int(len(encoder_ds)),
        "n_decoder_origins": int(len(decoder_ds)),
    }
    return encoder, decoder, diagnostics


def predict_single_rollout(encoder, decoder, bundle, row_id, t_obs, t_target):
    c, y, a, s = bundle["covariates"], bundle["y_norm_clip"], bundle["actions"], bundle["static"]
    row_id, t_obs, t_target = int(row_id), int(t_obs), int(t_target)
    tau = max(1, int(t_target - t_obs))
    hist_end = max(min(t_obs + 1, c.shape[1], y.shape[1], a.shape[1]), 1)
    hist_actions = a[row_id:row_id + 1, :hist_end].astype(np.int64)
    prev_actions = np.zeros_like(hist_actions)
    if hist_end > 1:
        prev_actions[:, 1:] = hist_actions[:, :-1]

    q_hist = {
        "prev_treatments": torch.from_numpy(onehot_actions(prev_actions)).to(common.DEVICE),
        "current_treatments": torch.from_numpy(onehot_actions(hist_actions)).to(common.DEVICE),
        "vitals": torch.from_numpy(c[row_id:row_id + 1, :hist_end, :].astype(np.float32)).to(common.DEVICE),
        "prev_outputs": torch.from_numpy(y[row_id:row_id + 1, :hist_end, None].astype(np.float32)).to(common.DEVICE),
        "static_features": torch.from_numpy(s[row_id:row_id + 1].astype(np.float32)).to(common.DEVICE),
    }
    encoder.eval()
    decoder.eval()
    _, _, br_hist = encoder(q_hist)
    init_state = br_hist[:, -1, :]
    seed_y = float(y[row_id, t_obs]) if t_obs < y.shape[1] else float(y[row_id, hist_end - 1])
    pred_values, prev_values = [], [seed_y]
    q_static = s[row_id:row_id + 1].astype(np.float32)

    for h in range(tau):
        cur_len = h + 1
        prev_action_seq = np.zeros((1, cur_len), dtype=np.int64)
        curr_action_seq = np.zeros((1, cur_len), dtype=np.int64)
        prev_outputs = np.zeros((1, cur_len, 1), dtype=np.float32)
        for k in range(cur_len):
            tt = t_obs + k
            prev_action_seq[0, k] = int(a[row_id, tt - 1]) if tt > 0 and tt - 1 < a.shape[1] else 0
            curr_action_seq[0, k] = int(a[row_id, tt]) if tt < a.shape[1] else 0
            prev_outputs[0, k, 0] = float(prev_values[k])
        _, outcome_pred, _ = decoder(
            {
                "prev_treatments": torch.from_numpy(onehot_actions(prev_action_seq)).to(common.DEVICE),
                "current_treatments": torch.from_numpy(onehot_actions(curr_action_seq)).to(common.DEVICE),
                "prev_outputs": torch.from_numpy(prev_outputs).to(common.DEVICE),
                "static_features": torch.from_numpy(q_static).to(common.DEVICE),
                "init_state": init_state,
            }
        )
        next_y = float(outcome_pred.detach().float().cpu().numpy()[0, -1, 0])
        next_y = float(np.clip(next_y, -common.PRED_CLIP_REPORT, common.PRED_CLIP_REPORT))
        pred_values.append(next_y)
        prev_values.append(float(np.clip(next_y, -common.OUTCOME_CLIP_TRAIN, common.OUTCOME_CLIP_TRAIN)))
    pred_path = np.asarray(pred_values, dtype=np.float32)
    return float(pred_path[-1]), pred_path


def evaluate_candidate_on_support(bundle, candidate, train_idx, val_idx, seed):
    encoder, decoder, diag = train_crn_models(bundle, candidate, train_idx, seed=seed)
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
        f"hidden={candidate['hidden_units']} br={candidate['br_size']} "
        f"fc={candidate['fc_hidden_units']} layers={candidate['num_layers']} "
        f"drop={candidate['dropout']}"
    )


ADAPTER = BaselineAdapter(
    method_name="crn",
    method_family="CRN",
    title="CRN benchmark evaluation",
    default_hparams=DEFAULT_HPARAMS,
    tuning_cache=TUNING_CACHE,
    hyperparameter_space=hyperparameter_space,
    sample_candidates=sample_random_candidates,
    canonical_hparams=canonical_hparams,
    evaluate_candidate=evaluate_candidate_on_support,
    train_final=paired_model_train_final(train_crn_models),
    predict_rows=single_rollout_predict_rows(predict_single_rollout, unpack_pair_payload=True),
    extra_record_fields=train_diag_record_fields(
        float_keys=(
            "encoder_loss",
            "encoder_outcome_loss",
            "encoder_treatment_loss",
            "decoder_loss",
            "decoder_outcome_loss",
            "decoder_treatment_loss",
        ),
        int_keys=("n_encoder_sequences", "n_decoder_origins"),
    ),
    extra_meta_fields=lambda _meta: baseline_port_metadata(),
    tuning_candidate_label=tuning_candidate_label,
    output_dir=OUTPUT_DIR,
)


def configure_from_eval_config(baseline_config):
    global MAX_TRAIN_ORIGINS, MAX_VAL_ORIGINS, OUTPUT_DIR

    crn_config.apply_config(baseline_config)
    MAX_TRAIN_ORIGINS = crn_config.MAX_TRAIN_ORIGINS
    MAX_VAL_ORIGINS = crn_config.MAX_VAL_ORIGINS
    OUTPUT_DIR = crn_config.OUTPUT_DIR
    ADAPTER.default_hparams = DEFAULT_HPARAMS
    ADAPTER.output_dir = OUTPUT_DIR
