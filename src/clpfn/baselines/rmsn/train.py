from __future__ import annotations

import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from clpfn.baselines.common.api import BaselineAdapter, baseline_port_metadata
from clpfn.baselines.common.api import paired_model_train_final, single_rollout_predict_rows, train_diag_record_fields
from clpfn.baselines.common.features import encode_actions, has_dynamic_vitals, prepare_baseline_bundle
from clpfn.baselines.common.stagewise import (
    cached_stagewise_result,
    candidates_for_horizon,
    cleanup_torch,
    component_space,
    finalize_stagewise_result,
    prepare_stagewise_context,
    rmse_for_candidates,
    saved_or_run_trial,
    teacher_forced_decoder_rmse,
)
from clpfn.baselines.common.tuning import sample_random_hparams
from clpfn.baselines.common.persistence import sha256_json
from clpfn.baselines.common.training import evaluate_paired_rollout_val_rmse
from clpfn.baselines.rmsn import config as rmsn_config
from clpfn.baselines.rmsn.config import (
    DEFAULT_HPARAMS,
    MAX_TRAIN_ORIGINS,
    MAX_VAL_ORIGINS,
    OUTPUT_DIR,
    build_group_scaled_space,
    canonical_hparams,
    clip_normalize_weights,
    make_rmsn_args,
    sample_random_candidates,
    treatment_spec_for_bundle,
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
from clpfn.evaluation.core import benchmark as common

LOGGER = logging.getLogger(__name__)


def _model_loader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=min(int(batch_size), len(dataset)),
        shuffle=bool(shuffle),
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _train_model(model, dataset, *, epochs, batch_size, max_grad_norm):
    loader = _model_loader(dataset, batch_size, shuffle=True)
    optimizer = model.configure_optimizers()
    last_loss = float("nan")
    for _ in range(int(epochs)):
        losses = []
        model.train()
        for batch_ind, batch in enumerate(loader):
            batch = move_batch_to_device(batch)
            loss = model.training_step(batch, batch_ind)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(losses)) if losses else float("nan")
    return last_loss


@torch.no_grad()
def _evaluate_model_loss(model, dataset, batch_size):
    loader = _model_loader(dataset, batch_size, shuffle=False)
    model.eval()
    total, denom = 0.0, 0.0
    for batch_ind, batch in enumerate(loader):
        batch = move_batch_to_device(batch)
        loss = model.training_step(batch, batch_ind)
        weight = float(batch["active_entries"].sum().detach().cpu())
        total += float(loss.detach().cpu()) * weight
        denom += weight
    return float(total / max(denom, 1e-8))


def _make_propensity_model(bundle, hparams, component):
    bundle = prepare_baseline_bundle(bundle)
    has_vitals = has_dynamic_vitals(bundle)
    treatment_mode, _ = treatment_spec_for_bundle(bundle)
    args = make_rmsn_args(int(bundle["covariates"].shape[-1]), hparams, treatment_mode, d_static=int(bundle["static"].shape[-1]))
    cls = RMSNPropensityNetworkTreatment if component == "propensity_treatment" else RMSNPropensityNetworkHistory
    return cls(args, dataset_collection=None, autoregressive=True, has_vitals=has_vitals, bce_weights=None).to(common.DEVICE), args


def train_propensity_model(bundle, hparams, context_idx, seed, component):
    common.seed_everything(seed)
    bundle = prepare_baseline_bundle(bundle)
    dataset = RMSNPropensityDataset(bundle, context_idx)
    model, args = _make_propensity_model(bundle, hparams, component)
    cfg = getattr(args.model, component)
    loss = _train_model(
        model,
        dataset,
        epochs=hparams["propensity_epochs"],
        batch_size=cfg.batch_size,
        max_grad_norm=cfg.max_grad_norm,
    )
    return model, dataset, float(loss)


def _observed_treatment_probability(probabilities, observed, treatment_mode):
    if treatment_mode == "multiclass":
        return np.sum(probabilities * observed, axis=-1)
    if treatment_mode == "multilabel":
        obs = np.asarray(observed, dtype=np.float64)
        return np.prod(probabilities * obs + (1.0 - probabilities) * (1.0 - obs), axis=-1)
    raise ValueError(f"Unknown RMSN treatment mode: {treatment_mode!r}")


@torch.no_grad()
def compute_stabilized_weights(prop_treat, prop_hist, train_data, hparams):
    active = train_data["active_entries"].squeeze(-1).astype(np.float32)
    treatment_mode = str(train_data["treatment_mode"])
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
    logits_num = prop_treat(batch)
    logits_den = prop_hist(batch)
    if treatment_mode == "multiclass":
        p_num = torch.softmax(logits_num, dim=-1).cpu().numpy()
        p_den = torch.softmax(logits_den, dim=-1).cpu().numpy()
    else:
        p_num = torch.sigmoid(logits_num).cpu().numpy()
        p_den = torch.sigmoid(logits_den).cpu().numpy()
    observed = train_data["current_treatments"].astype(np.float32)
    eps = 1e-4
    p_num = np.clip(p_num, eps, 1.0 - eps)
    p_den = np.clip(p_den, eps, 1.0 - eps)
    prob_num = _observed_treatment_probability(p_num, observed, treatment_mode)
    prob_den = _observed_treatment_probability(p_den, observed, treatment_mode)
    sw_raw = prob_num / np.maximum(prob_den, eps)
    sw_raw[active <= 0] = 1.0
    if not np.isfinite(sw_raw[active > 0]).all():
        raise RuntimeError("RMSN produced non-finite stabilized weights on active entries.")
    sw_tilde_enc = clip_normalize_weights(
        sw_raw,
        active=active,
        quantiles=hparams["weight_clip_quantiles"],
        multiple_horizons=False,
    )
    active_values = sw_raw[active > 0]
    diag = {
        "stabilized_weight_raw_mean": float(np.mean(active_values)) if active_values.size else float("nan"),
        "stabilized_weight_raw_max": float(np.max(active_values)) if active_values.size else float("nan"),
    }
    return sw_raw.astype(np.float32), sw_tilde_enc.astype(np.float32), diag


def train_rmsn_encoder(bundle, hparams, context_idx, seed, train_data=None, sw_tilde_enc=None):
    common.seed_everything(seed)
    bundle = prepare_baseline_bundle(bundle)
    has_vitals = has_dynamic_vitals(bundle)
    treatment_mode, _ = treatment_spec_for_bundle(bundle)
    args = make_rmsn_args(int(bundle["covariates"].shape[-1]), hparams, treatment_mode, d_static=int(bundle["static"].shape[-1]))
    train_data = train_data if train_data is not None else make_encoder_full_arrays(bundle, context_idx)
    if sw_tilde_enc is None:
        raise ValueError("RMSN encoder training requires precomputed stabilized weights.")
    dataset = RMSNEncoderSupportDataset(train_data, sw_tilde_enc)
    encoder = RMSNEncoder(
        args,
        propensity_treatment=None,
        propensity_history=None,
        dataset_collection=None,
        autoregressive=True,
        has_vitals=has_vitals,
        bce_weights=None,
    ).to(common.DEVICE)
    loss = _train_model(
        encoder,
        dataset,
        epochs=hparams["encoder_epochs"],
        batch_size=args.model.encoder.batch_size,
        max_grad_norm=args.model.encoder.max_grad_norm,
    )
    return encoder, dataset, float(loss)


def train_rmsn_decoder(bundle, hparams, context_idx, seed, encoder, train_data, sw_raw):
    common.seed_everything(seed)
    bundle = prepare_baseline_bundle(bundle)
    has_vitals = has_dynamic_vitals(bundle)
    treatment_mode, _ = treatment_spec_for_bundle(bundle)
    args = make_rmsn_args(int(bundle["covariates"].shape[-1]), hparams, treatment_mode, d_static=int(bundle["static"].shape[-1]))
    representations = get_encoder_representations(encoder, train_data)
    origins = sample_decoder_origins(train_data, seed=seed + 17, max_origins=MAX_TRAIN_ORIGINS)
    dataset = RMSNDecoderOriginDataset(train_data, origins, representations, sw_raw, hparams)
    decoder = RMSNDecoder(
        args,
        encoder=None,
        dataset_collection=None,
        encoder_r_size=int(encoder.seq_hidden_units),
        autoregressive=True,
        has_vitals=has_vitals,
        bce_weights=None,
    ).to(common.DEVICE)
    loss = _train_model(
        decoder,
        dataset,
        epochs=hparams["decoder_epochs"],
        batch_size=args.model.decoder.batch_size,
        max_grad_norm=args.model.decoder.max_grad_norm,
    )
    return decoder, dataset, float(loss)


def train_rmsn_models(bundle, hparams, context_idx, seed):
    bundle = prepare_baseline_bundle(bundle)
    t0 = time.time()
    prop_treat, _, pnum_loss = train_propensity_model(bundle, hparams, context_idx, seed + 11, "propensity_treatment")
    prop_hist, _, pden_loss = train_propensity_model(bundle, hparams, context_idx, seed + 22, "propensity_history")
    train_data = make_encoder_full_arrays(bundle, context_idx)
    sw_raw, sw_tilde_enc, weight_diag = compute_stabilized_weights(prop_treat, prop_hist, train_data, hparams)
    cleanup_torch(prop_treat, prop_hist)
    encoder, encoder_ds, enc_loss = train_rmsn_encoder(bundle, hparams, context_idx, seed + 33, train_data, sw_tilde_enc)
    decoder, decoder_ds, dec_loss = train_rmsn_decoder(bundle, hparams, context_idx, seed + 44, encoder, train_data, sw_raw)
    diagnostics = {
        "propensity_treatment_loss": pnum_loss,
        "propensity_history_loss": pden_loss,
        "encoder_loss": enc_loss,
        "decoder_loss": dec_loss,
        "train_loss": dec_loss,
        "fit_time_sec": float(time.time() - t0),
        "n_encoder_sequences": int(len(encoder_ds)),
        "n_decoder_origins": int(len(decoder_ds)),
        "treatment_mode": str(train_data["treatment_mode"]),
        **weight_diag,
    }
    return encoder, decoder, diagnostics


def _encoder_query_batch(bundle, row_id, end_exclusive, treatment_mode):
    bundle = prepare_baseline_bundle(bundle)
    C, Yc, A, S = bundle["covariates"], bundle["y_norm_clip"], bundle["actions"], bundle["static"]
    end_exclusive = max(min(int(end_exclusive), C.shape[1], Yc.shape[1], A.shape[1]), 1)
    return {
        "vitals": torch.from_numpy(C[row_id:row_id + 1, :end_exclusive, :].astype(np.float32)).to(common.DEVICE),
        "prev_outputs": torch.from_numpy(Yc[row_id:row_id + 1, :end_exclusive, None].astype(np.float32)).to(common.DEVICE),
        "current_treatments": torch.from_numpy(encode_actions(A[row_id:row_id + 1, :end_exclusive], treatment_mode)).to(common.DEVICE),
        "static_features": torch.from_numpy(S[row_id:row_id + 1].astype(np.float32)).to(common.DEVICE),
    }


@torch.no_grad()
def predict_encoder_one_step(encoder, bundle, row_id, t_obs):
    treatment_mode, _ = treatment_spec_for_bundle(bundle)
    encoder.eval()
    outcome_pred, _ = encoder(_encoder_query_batch(bundle, int(row_id), int(t_obs) + 1, treatment_mode))
    return float(outcome_pred[0, -1, 0].detach().cpu())


@torch.no_grad()
def predict_single_rollout(encoder, decoder, bundle, row_id, t_obs, t_target, return_unclipped=False):
    bundle = prepare_baseline_bundle(bundle)
    A, S = bundle["actions"], bundle["static"]
    row_id, t_obs, t_target = int(row_id), int(t_obs), int(t_target)
    tau = max(1, int(t_target - t_obs))
    treatment_mode, _ = treatment_spec_for_bundle(bundle)
    encoder.eval()
    decoder.eval()
    q_hist = _encoder_query_batch(bundle, row_id, t_obs + 1, treatment_mode)
    outcome_hist, r_hist = encoder(q_hist)
    h1_unclipped = float(outcome_hist[0, -1, 0].detach().cpu())
    h1 = float(np.clip(h1_unclipped, -common.PRED_CLIP_REPORT, common.PRED_CLIP_REPORT))
    pred_values, raw_values = [h1], [h1_unclipped]
    if tau == 1:
        path = np.asarray(pred_values, dtype=np.float32)
        raw = np.asarray(raw_values, dtype=np.float32)
        return (h1, path, raw) if return_unclipped else (h1, path)
    init_state = r_hist[:, -1, :]
    prev_values = [float(np.clip(h1, -common.OUTCOME_CLIP_TRAIN, common.OUTCOME_CLIP_TRAIN))]
    static_features = torch.from_numpy(S[row_id:row_id + 1].astype(np.float32)).to(common.DEVICE)
    for h in range(1, tau):
        cur_len = h
        actions = np.zeros((1, cur_len), dtype=np.int64)
        prev_outputs = np.zeros((1, cur_len, 1), dtype=np.float32)
        for k in range(cur_len):
            action_t = t_obs + 1 + k
            if action_t < A.shape[1]:
                actions[0, k] = int(A[row_id, action_t])
            prev_outputs[0, k, 0] = float(prev_values[k])
        out_seq = decoder({
            "current_treatments": torch.from_numpy(encode_actions(actions, treatment_mode)).to(common.DEVICE),
            "prev_outputs": torch.from_numpy(prev_outputs).to(common.DEVICE),
            "static_features": static_features,
            "init_state": init_state,
        }).detach().float().cpu().numpy()[0, :, 0]
        raw = float(out_seq[-1])
        pred = float(np.clip(raw, -common.PRED_CLIP_REPORT, common.PRED_CLIP_REPORT))
        pred_values.append(pred)
        raw_values.append(raw)
        prev_values.append(float(np.clip(pred, -common.OUTCOME_CLIP_TRAIN, common.OUTCOME_CLIP_TRAIN)))
    path = np.asarray(pred_values, dtype=np.float32)
    return (float(path[-1]), path, np.asarray(raw_values, dtype=np.float32)) if return_unclipped else (float(path[-1]), path)


def evaluate_candidate_on_support(bundle, candidate, train_idx, val_idx, seed):
    encoder, decoder, diag = train_rmsn_models(bundle, candidate, train_idx, seed)
    score = evaluate_paired_rollout_val_rmse(bundle, encoder, decoder, val_idx, seed + 99, MAX_VAL_ORIGINS, predict_single_rollout)
    cleanup_torch(encoder, decoder)
    return float(score), diag


def hyperparameter_space(bundle):
    bundle = prepare_baseline_bundle(bundle)
    treatment_mode, _ = treatment_spec_for_bundle(bundle)
    return build_group_scaled_space(int(bundle["covariates"].shape[-1]), treatment_mode=treatment_mode, d_static=int(bundle["static"].shape[-1]))


def _sample_stage(space, base_hparams, n, seed):
    return sample_random_hparams(space, n, seed, default_hparams=base_hparams, canonical_hparams=canonical_hparams)


def _propensity_stage_space(space, component):
    suffix = f"_{component}"
    return component_space(space, lambda key: key.endswith(suffix))


def select_stagewise_hparams(
    adapter,
    bundle,
    meta,
    source_file,
    trial_store,
    dataset_hash,
    raw_dataset_hash,
):
    bundle = prepare_baseline_bundle(bundle)
    seeds, train_idx, val_idx = prepare_stagewise_context(adapter, bundle, meta)
    space, space_info = hyperparameter_space(bundle)
    trials = dict(rmsn_config.STAGEWISE_TRIALS)
    tuning_protocol_hash = sha256_json({
        "method": "rmsn",
        "strategy": "task_specific_stagewise_search",
        "stagewise_trials": trials,
        "space": space,
        "space_info": space_info,
        "seeds": seeds,
    })
    cached = cached_stagewise_result(
        adapter, trial_store=trial_store, meta=meta,
        dataset_hash=dataset_hash, raw_dataset_hash=raw_dataset_hash,
        tuning_protocol_hash=tuning_protocol_hash,
    )
    if cached is not None:
        return cached
    started = time.time()
    trial_rows = []
    selected = dict(DEFAULT_HPARAMS)
    global_offset = 0

    for stage_no, component in enumerate(("propensity_treatment", "propensity_history")):
        stage_space = _propensity_stage_space(space, component)
        candidates = _sample_stage(
            stage_space, selected, int(trials[component]),
            seeds["plan_seed"] + 10000 * (stage_no + 1),
        )
        results = []
        val_dataset = RMSNPropensityDataset(bundle, val_idx)
        stage_seed = seeds["train_seed"] + 10000 * (stage_no + 1) + 1000
        for local_ci, candidate in enumerate(candidates):
            ci = global_offset + local_ci
            try:
                score, diag, row, resumed = saved_or_run_trial(
                    trial_store=trial_store, adapter=adapter, meta=meta,
                    stage=component,
                    candidate_index=ci, candidate=candidate, seed=stage_seed,
                    objective="heldout_treatment_negative_log_likelihood", seeds=seeds,
                    tuning_protocol_hash=tuning_protocol_hash,
                    run=lambda candidate=candidate, component=component: _run_rmsn_propensity_candidate(
                        bundle, candidate, train_idx, stage_seed, component, val_dataset
                    ),
                )
                trial_rows.append(row)
                results.append((score, candidate, ci, diag))
                if resumed:
                    LOGGER.info("resume %s cand=%02d val_nll=%.4f", component, ci, score)
            except Exception:
                cached_row = trial_store.load_trial(
                    dataset_uid=str(meta["dataset_uid"]), stage=component,
                    candidate_index=ci, hparams=candidate,
                ) if trial_store is not None else None
                if cached_row is not None:
                    trial_rows.append(cached_row)
                cleanup_torch()
        finite = [row for row in results if np.isfinite(row[0])]
        if finite:
            _, selected, _, _ = min(finite, key=lambda x: x[0])
        else:
            raise RuntimeError(
                f"All RMSN {component} candidates failed or were non-finite for "
                f"dataset_uid={meta['dataset_uid']}. Saved trial records can be resumed "
                "after correcting the failure."
            )
        global_offset += len(candidates)

    prop_treat, _, _ = train_propensity_model(
        bundle, selected, train_idx, seeds["train_seed"] + 31000, "propensity_treatment"
    )
    prop_hist, _, _ = train_propensity_model(
        bundle, selected, train_idx, seeds["train_seed"] + 32000, "propensity_history"
    )
    train_data = make_encoder_full_arrays(bundle, train_idx)
    sw_raw, sw_tilde_enc, _ = compute_stabilized_weights(prop_treat, prop_hist, train_data, selected)
    cleanup_torch(prop_treat, prop_hist)

    encoder_space = component_space(space, lambda key: key.endswith("_encoder"))
    encoder_candidates = _sample_stage(
        encoder_space, selected, int(trials["encoder"]), seeds["plan_seed"] + 40000
    )
    encoder_results = []
    one_step_candidates = candidates_for_horizon(
        bundle, val_idx, horizon=1, seed=seeds["validation_seed"] + 40100,
        max_val_origins=MAX_VAL_ORIGINS,
    )
    encoder_seed = seeds["train_seed"] + 41000
    for local_ci, candidate in enumerate(encoder_candidates):
        ci = global_offset + local_ci
        try:
            score, diag, row, resumed = saved_or_run_trial(
                trial_store=trial_store, adapter=adapter, meta=meta,
                stage="encoder",
                candidate_index=ci, candidate=candidate, seed=encoder_seed,
                objective="one_step_normalized_rmse", seeds=seeds,
                tuning_protocol_hash=tuning_protocol_hash,
                run=lambda candidate=candidate: _run_rmsn_encoder_candidate(
                    bundle, candidate, train_idx, encoder_seed, train_data,
                    sw_tilde_enc, one_step_candidates
                ),
            )
            trial_rows.append(row)
            encoder_results.append((score, candidate, ci, diag))
            if resumed:
                LOGGER.info("resume encoder cand=%02d val_rmse=%.4f", ci, score)
        except Exception:
            cached_row = trial_store.load_trial(
                dataset_uid=str(meta["dataset_uid"]), stage="encoder",
                candidate_index=ci, hparams=candidate,
            ) if trial_store is not None else None
            if cached_row is not None:
                trial_rows.append(cached_row)
            cleanup_torch()
    finite_enc = [row for row in encoder_results if np.isfinite(row[0])]
    if finite_enc:
        _, selected, _, _ = min(finite_enc, key=lambda x: x[0])
    else:
        raise RuntimeError(
            f"All RMSN encoder candidates failed or were non-finite for dataset_uid={meta['dataset_uid']}. "
            "Saved trial records can be resumed after correcting the failure."
        )
    global_offset += len(encoder_candidates)

    fixed_encoder, _, _ = train_rmsn_encoder(
        bundle, selected, train_idx, seeds["train_seed"] + 50000, train_data, sw_tilde_enc
    )
    decoder_space = component_space(space, lambda key: key.endswith("_decoder"))
    decoder_candidates = _sample_stage(
        decoder_space, selected, int(trials["decoder"]), seeds["plan_seed"] + 51000
    )
    decoder_results = []
    val_train_data = make_encoder_full_arrays(bundle, val_idx)
    val_representations = get_encoder_representations(fixed_encoder, val_train_data)
    val_origins = sample_decoder_origins(
        val_train_data, seed=seeds["validation_seed"] + 51100,
        max_origins=MAX_VAL_ORIGINS,
    )
    val_decoder_ds = RMSNDecoderOriginDataset(
        val_train_data, val_origins, val_representations, None, selected
    )
    decoder_seed = seeds["train_seed"] + 52000
    for local_ci, candidate in enumerate(decoder_candidates):
        ci = global_offset + local_ci
        try:
            score, diag, row, resumed = saved_or_run_trial(
                trial_store=trial_store, adapter=adapter, meta=meta,
                stage="decoder",
                candidate_index=ci, candidate=candidate, seed=decoder_seed,
                objective="teacher_forced_decoder_masked_rmse", seeds=seeds,
                tuning_protocol_hash=tuning_protocol_hash,
                run=lambda candidate=candidate: _run_rmsn_decoder_candidate(
                    bundle, candidate, train_idx, decoder_seed, fixed_encoder,
                    train_data, sw_raw, val_decoder_ds
                ),
            )
            trial_rows.append(row)
            decoder_results.append((score, candidate, ci, diag))
            if resumed:
                LOGGER.info("resume decoder cand=%02d val_rmse=%.4f", ci, score)
        except Exception:
            cached_row = trial_store.load_trial(
                dataset_uid=str(meta["dataset_uid"]), stage="decoder",
                candidate_index=ci, hparams=candidate,
            ) if trial_store is not None else None
            if cached_row is not None:
                trial_rows.append(cached_row)
            cleanup_torch()
    cleanup_torch(fixed_encoder)

    finite_dec = [row for row in decoder_results if np.isfinite(row[0])]
    if finite_dec:
        best_score, best_hparams, selected_candidate, _ = min(finite_dec, key=lambda x: x[0])
    else:
        raise RuntimeError(
            f"All RMSN decoder candidates failed or were non-finite for dataset_uid={meta['dataset_uid']}. "
            "Saved trial records can be resumed after correcting the failure."
        )
    return finalize_stagewise_result(
        adapter, meta=meta, source_file=source_file,
        seeds=seeds, train_idx=train_idx, val_idx=val_idx, space_info=space_info,
        best_hparams=best_hparams, final_score=best_score,
        selected_candidate=selected_candidate, search_started=started,
        trial_rows=trial_rows, stagewise_trials=trials, trial_store=trial_store,
        dataset_hash=dataset_hash, raw_dataset_hash=raw_dataset_hash,
        tuning_protocol_hash=tuning_protocol_hash,
    )


def _run_rmsn_propensity_candidate(
    bundle, candidate, train_idx, seed, component, val_dataset
):
    model, _, train_loss = train_propensity_model(
        bundle, candidate, train_idx, seed, component
    )
    try:
        treatment_mode, _ = treatment_spec_for_bundle(bundle)
        args = make_rmsn_args(
            int(bundle["covariates"].shape[-1]), candidate, treatment_mode,
            d_static=int(bundle["static"].shape[-1]),
        )
        val_loss = _evaluate_model_loss(
            model, val_dataset, getattr(args.model, component).batch_size
        )
        return float(val_loss), {"train_loss": float(train_loss)}
    finally:
        cleanup_torch(model)


def _run_rmsn_encoder_candidate(
    bundle, candidate, train_idx, seed, train_data, sw_tilde_enc, one_step_candidates
):
    encoder, _, train_loss = train_rmsn_encoder(
        bundle, candidate, train_idx, seed, train_data, sw_tilde_enc
    )
    try:
        score = rmse_for_candidates(
            bundle, one_step_candidates,
            lambda r, t, _: predict_encoder_one_step(encoder, bundle, r, t),
        )
        return float(score), {"train_loss": float(train_loss)}
    finally:
        cleanup_torch(encoder)


def _run_rmsn_decoder_candidate(
    bundle, candidate, train_idx, seed, fixed_encoder, train_data, sw_raw,
    val_decoder_ds,
):
    decoder, _, train_loss = train_rmsn_decoder(
        bundle, candidate, train_idx, seed, fixed_encoder, train_data, sw_raw
    )
    try:
        score = teacher_forced_decoder_rmse(
            decoder,
            val_decoder_ds,
            batch_size=int(candidate["batch_size_decoder"]),
            move_batch_to_device=move_batch_to_device,
            outcome_of=lambda model, batch: model(batch),
        )
        return float(score), {"train_loss": float(train_loss)}
    finally:
        cleanup_torch(decoder)


def tuning_candidate_label(candidate):
    return (
        "pnum/phist/enc/dec="
        f"{candidate['hidden_units_propensity_treatment']}/"
        f"{candidate['hidden_units_propensity_history']}/"
        f"{candidate['hidden_units_encoder']}/"
        f"{candidate['hidden_units_decoder']}"
    )


ADAPTER = BaselineAdapter(
    method_name="rmsn",
    method_family="RMSN",
    title="RMSN benchmark evaluation",
    default_hparams=DEFAULT_HPARAMS,
    hyperparameter_space=hyperparameter_space,
    sample_candidates=sample_random_candidates,
    canonical_hparams=canonical_hparams,
    evaluate_candidate=evaluate_candidate_on_support,
    train_final=paired_model_train_final(train_rmsn_models),
    predict_rows=single_rollout_predict_rows(predict_single_rollout, unpack_pair_payload=True),
    extra_record_fields=train_diag_record_fields(
        float_keys=("propensity_treatment_loss", "propensity_history_loss", "encoder_loss", "decoder_loss", "stabilized_weight_raw_mean", "stabilized_weight_raw_max"),
        int_keys=("n_encoder_sequences", "n_decoder_origins"),
    ),
    extra_meta_fields=lambda _meta: baseline_port_metadata(
        rmsn_treatment_models="multilabel_cancer_mimic_multiclass_hiv_warfarin",
        rmsn_multilabel_propensity="observed_treatment_likelihood",
        rmsn_decoder_training_alignment="encoder_state_origin_minus_1_then_Y_origin_A_origin",
        rmsn_rollout_alignment="encoder_h1_then_decoder_h2_onward_from_state_t_obs",
        rmsn_one_step_source="weighted_encoder",
        rmsn_tuning="task_specific_stagewise_atomic_resume",
    ),
    select_hparams=select_stagewise_hparams,
    tuning_candidate_label=tuning_candidate_label,
    tuning_strategy="task_specific_stagewise_atomic_resume",
    output_dir=OUTPUT_DIR,
)


def configure_from_eval_config(baseline_config):
    global MAX_TRAIN_ORIGINS, MAX_VAL_ORIGINS, OUTPUT_DIR
    rmsn_config.apply_config(baseline_config)
    MAX_TRAIN_ORIGINS = rmsn_config.MAX_TRAIN_ORIGINS
    MAX_VAL_ORIGINS = rmsn_config.MAX_VAL_ORIGINS
    OUTPUT_DIR = rmsn_config.OUTPUT_DIR
    ADAPTER.default_hparams = DEFAULT_HPARAMS
    ADAPTER.output_dir = OUTPUT_DIR
