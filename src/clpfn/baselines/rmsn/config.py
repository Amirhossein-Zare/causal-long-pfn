from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from clpfn.baselines.common.api import canonical_hparams
from clpfn.baselines.common.api import replace_mapping, require_baseline_config
from clpfn.baselines.common.tuning import sample_random_hparams
from clpfn.evaluation.core import benchmark as common


OUTPUT_DIR = Path("outputs/eval/rmsn")

N_ACTION_BITS = 2
MAX_TRAIN_ORIGINS = 0
MAX_VAL_ORIGINS = 0
PROP_BATCH_SIZE = 1
ENCODER_BATCH_SIZE = 1
DECODER_BATCH_SIZE = 1

DEFAULT_HPARAMS = {}
BASE_RMSN_SPACE = {}
SCALED_SPACE = {}

TUNING_CACHE = {}


def apply_config(config):
    global N_ACTION_BITS, MAX_TRAIN_ORIGINS, MAX_VAL_ORIGINS
    global PROP_BATCH_SIZE, ENCODER_BATCH_SIZE, DECODER_BATCH_SIZE

    config = require_baseline_config(config, "rmsn")
    limits = config["limits"]
    N_ACTION_BITS = int(limits["n_action_bits"])
    MAX_TRAIN_ORIGINS = int(limits["max_train_origins"])
    MAX_VAL_ORIGINS = int(limits["max_val_origins"])
    PROP_BATCH_SIZE = int(limits["prop_batch_size"])
    ENCODER_BATCH_SIZE = int(limits["encoder_batch_size"])
    DECODER_BATCH_SIZE = int(limits["decoder_batch_size"])
    replace_mapping(DEFAULT_HPARAMS, config["default_hparams"])
    replace_mapping(BASE_RMSN_SPACE, config["search_space"])
    replace_mapping(SCALED_SPACE, config.get("scaled_space", {}))
    TUNING_CACHE.clear()


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def round_to_valid(values, min_value=None, max_value=None, multiple=None):
    min_value = int(SCALED_SPACE.get("min_width", 32) if min_value is None else min_value)
    max_value = int(SCALED_SPACE.get("max_width", 160) if max_value is None else max_value)
    multiple = int(SCALED_SPACE.get("width_multiple", 16) if multiple is None else multiple)
    out = []
    for value in values:
        x = int(round(float(value) / multiple) * multiple)
        out.append(max(int(min_value), min(int(max_value), x)))
    return sorted(set(out))


def size_grid(width, min_value=None, max_value=None):
    multipliers = SCALED_SPACE.get("width_multipliers", [0.5, 1.0, 2.0, 4.0])
    return round_to_valid([float(multiplier) * width for multiplier in multipliers], min_value=min_value, max_value=max_value)


def build_group_scaled_space(d_base):
    c_hist = N_ACTION_BITS + int(d_base) + 1 + common.D_STATIC_MAX
    c_dec = N_ACTION_BITS + 1 + common.D_STATIC_MAX
    enc_sizes = size_grid(c_hist)
    dec_sizes = size_grid(c_dec)
    practical = [int(value) for value in SCALED_SPACE.get("practical_widths", [])]
    enc_sizes = sorted(set(enc_sizes + practical))
    dec_sizes = sorted(set(dec_sizes + practical))
    space = dict(BASE_RMSN_SPACE)
    space["hidden_units_encoder"] = enc_sizes
    space["hidden_units_decoder"] = dec_sizes
    return space, {
        "C_hist": int(c_hist),
        "C_dec": int(c_dec),
        "encoder_size_grid": enc_sizes,
        "decoder_size_grid": dec_sizes,
    }


def make_rmsn_args(d_vitals, hparams):
    def submodel_cfg(hidden_units, lr, batch_size):
        return ns(
            seq_hidden_units=int(hidden_units),
            dropout_rate=float(hparams["dropout"]),
            num_layer=int(hparams["num_layers"]),
            batch_size=int(batch_size),
            max_grad_norm=float(hparams["max_grad_norm"]),
            optimizer={
                "learning_rate": float(lr),
                "weight_decay": float(hparams.get("weight_decay", 1e-5)),
                "optimizer_cls": str(hparams.get("optimizer_cls", "adamw")),
                "lr_scheduler": False,
            },
        )

    propensity_treatment = submodel_cfg(
        hparams["hidden_units_encoder"],
        hparams["lr_prop"],
        hparams["batch_size_encoder"],
    )
    propensity_history = submodel_cfg(
        hparams["hidden_units_encoder"],
        hparams["lr_prop"],
        hparams["batch_size_encoder"],
    )
    encoder = submodel_cfg(
        hparams["hidden_units_encoder"],
        hparams["lr_enc"],
        hparams["batch_size_encoder"],
    )
    decoder = submodel_cfg(
        hparams["hidden_units_decoder"],
        hparams["lr_dec"],
        hparams["batch_size_decoder"],
    )

    return ns(
        model=ns(
            dim_treatments=N_ACTION_BITS,
            dim_vitals=int(d_vitals),
            dim_static_features=common.D_STATIC_MAX,
            dim_outcomes=1,
            encoder=encoder,
            decoder=decoder,
            propensity_treatment=propensity_treatment,
            propensity_history=propensity_history,
        ),
        dataset=ns(
            val_batch_size=int(hparams["batch_size_encoder"]),
            projection_horizon=common.PROJECTION_HORIZON,
            treatment_mode="multilabel",
            holdout_ratio=0.0,
        ),
        exp=ns(
            unscale_rmse=False,
            percentage_rmse=False,
            bce_weight=False,
            gpus="[]",
            max_epochs=max(
                int(hparams["propensity_epochs"]),
                int(hparams["encoder_epochs"]),
                int(hparams["decoder_epochs"]),
            ),
            alpha_rate="exp",
            update_alpha=False,
        ),
    )


def action4_to_bits(actions):
    actions = np.asarray(actions, dtype=np.int64)
    return np.stack([(actions & 1), ((actions >> 1) & 1)], axis=-1).astype(np.float32)


def sample_random_candidates(space, n, seed):
    return sample_random_hparams(
        space,
        n,
        seed,
        default_hparams=DEFAULT_HPARAMS,
        canonical_hparams=canonical_hparams,
    )


def clip_normalize_weights(weights, active, quantiles=(0.01, 0.99), multiple_horizons=False):
    weights = np.asarray(weights, dtype=np.float32).copy()
    active_bool = np.asarray(active).astype(bool)
    weights[~active_bool] = np.nan

    finite = np.isfinite(weights)
    if finite.sum() == 0:
        out = np.zeros_like(weights, dtype=np.float32)
        out[active_bool] = 1.0
        return out

    lo, hi = quantiles
    qlo = np.nanquantile(weights, float(lo))
    qhi = np.nanquantile(weights, float(hi))
    weights = np.clip(weights, qlo, qhi)

    if multiple_horizons:
        denom = np.nanmean(weights, axis=0, keepdims=True)
        denom = np.where(np.isfinite(denom) & (np.abs(denom) > 1e-8), denom, 1.0)
        weights = weights / denom
    else:
        denom = np.nanmean(weights)
        if not np.isfinite(denom) or abs(denom) < 1e-8:
            denom = 1.0
        weights = weights / denom

    weights[~active_bool] = 0.0
    weights[~np.isfinite(weights)] = 0.0
    return weights.astype(np.float32)
