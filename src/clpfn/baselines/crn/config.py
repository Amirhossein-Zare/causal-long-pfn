from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from clpfn.baselines.common.api import canonical_hparams
from clpfn.baselines.common.api import replace_mapping, require_baseline_config
from clpfn.baselines.common.tuning import sample_random_hparams
from clpfn.evaluation.core import benchmark as common


OUTPUT_DIR = Path("outputs/eval/crn")

MAX_TRAIN_ORIGINS = 0
MAX_VAL_ORIGINS = 0

DEFAULT_HPARAMS = {}
BASE_CRN_SPACE = {}
SCALED_SPACE = {}

TUNING_CACHE = {}


def apply_config(config):
    global MAX_TRAIN_ORIGINS, MAX_VAL_ORIGINS

    config = require_baseline_config(config, "crn")
    limits = config["limits"]
    MAX_TRAIN_ORIGINS = int(limits["max_train_origins"])
    MAX_VAL_ORIGINS = int(limits["max_val_origins"])
    replace_mapping(DEFAULT_HPARAMS, config["default_hparams"])
    replace_mapping(BASE_CRN_SPACE, config["search_space"])
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
    return round_to_valid([float(multiplier) * width for multiplier in multipliers], min_value, max_value)


def build_group_scaled_space(d_base):
    c_hist = int(common.N_ACTIONS + int(d_base) + 1 + common.D_STATIC_MAX)
    c_dec = int(common.N_ACTIONS + 1 + common.D_STATIC_MAX)
    hist_sizes = size_grid(c_hist)
    dec_sizes = size_grid(c_dec)
    practical_hidden = [int(value) for value in SCALED_SPACE.get("practical_hidden", [])]
    practical_fc = [int(value) for value in SCALED_SPACE.get("practical_fc_hidden", [])]

    space = dict(BASE_CRN_SPACE)
    space["hidden_units"] = sorted(set(hist_sizes + dec_sizes + practical_hidden))
    space["br_size"] = sorted(set(hist_sizes + dec_sizes + practical_hidden))
    space["fc_hidden_units"] = sorted(set(hist_sizes + dec_sizes + practical_fc))
    return space, {
        "C_hist": c_hist,
        "C_dec": c_dec,
        "hist_size_grid": hist_sizes,
        "decoder_size_grid": dec_sizes,
        "hidden_units_grid": space["hidden_units"],
        "br_size_grid": space["br_size"],
        "fc_hidden_units_grid": space["fc_hidden_units"],
    }


def make_crn_args(d_vitals, hparams):
    def submodel_cfg(lr, batch_size, seq_hidden_units=None):
        return ns(
            seq_hidden_units=int(seq_hidden_units if seq_hidden_units is not None else hparams["hidden_units"]),
            br_size=int(hparams["br_size"]),
            fc_hidden_units=int(hparams["fc_hidden_units"]),
            dropout_rate=float(hparams["dropout"]),
            num_layer=int(hparams["num_layers"]),
            batch_size=int(batch_size),
            optimizer={
                "learning_rate": float(lr),
                "weight_decay": float(hparams.get("weight_decay", 1e-5)),
                "optimizer_cls": str(hparams.get("optimizer_cls", "adamw")),
                "lr_scheduler": False,
            },
        )

    encoder = submodel_cfg(hparams["lr_encoder"], hparams["batch_size_encoder"], hparams["hidden_units"])
    decoder = submodel_cfg(hparams["lr_decoder"], hparams["batch_size_decoder"], hparams["br_size"])
    return ns(
        model=ns(
            dim_treatments=common.N_ACTIONS,
            dim_vitals=int(d_vitals),
            dim_static_features=common.D_STATIC_MAX,
            dim_outcomes=1,
            alpha=float(hparams["grl_alpha"]),
            update_alpha=False,
            balancing="grad_reverse",
            treatment_loss_weight=float(hparams["treatment_loss_weight"]),
            encoder=encoder,
            decoder=decoder,
        ),
        dataset=ns(
            val_batch_size=int(hparams["batch_size_encoder"]),
            projection_horizon=common.PROJECTION_HORIZON,
            treatment_mode="multiclass",
            holdout_ratio=0.0,
        ),
        exp=ns(
            unscale_rmse=False,
            percentage_rmse=False,
            bce_weight=False,
            gpus="[]",
            max_epochs=max(int(hparams["encoder_epochs"]), int(hparams["decoder_epochs"])),
            alpha_rate="exp",
            update_alpha=False,
        ),
    )


def onehot_actions(actions):
    return common.action_onehot_2d(np.asarray(actions, dtype=np.int64), common.N_ACTIONS)


def sample_random_candidates(space, n, seed):
    return sample_random_hparams(
        space,
        n,
        seed,
        default_hparams=DEFAULT_HPARAMS,
        canonical_hparams=canonical_hparams,
    )
