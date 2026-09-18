from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from clpfn.baselines.common.api import canonical_hparams, replace_mapping, require_baseline_config
from clpfn.baselines.common.features import treatment_dim
from clpfn.baselines.common.tuning import sample_random_hparams
from clpfn.evaluation.core import benchmark as common


OUTPUT_DIR = Path("outputs/eval/rmsn")
N_ACTIONS = 4
MAX_TRAIN_ORIGINS = None
MAX_VAL_ORIGINS = 0

DEFAULT_HPARAMS: dict = {}
BASE_RMSN_SPACE: dict = {}
SCALED_SPACE: dict = {}
STAGEWISE_TRIALS = {
    "propensity_treatment": 40,
    "propensity_history": 40,
    "encoder": 40,
    "decoder": 20,
}
TREATMENT_MODE_BY_DOMAIN: dict[str, str] = {
    "cancer": "multilabel",
    "hiv": "multiclass",
    "warfarin": "multiclass",
    "mimic": "multilabel",
}
_COMPONENTS = ("propensity_treatment", "propensity_history", "encoder", "decoder")


def apply_config(config):
    global N_ACTIONS, MAX_TRAIN_ORIGINS, MAX_VAL_ORIGINS
    config = require_baseline_config(config, "rmsn")
    limits = config["limits"]
    N_ACTIONS = int(limits["n_actions"])
    max_train_origins = limits.get("max_train_origins")
    MAX_TRAIN_ORIGINS = None if max_train_origins is None else int(max_train_origins)
    MAX_VAL_ORIGINS = int(limits["max_val_origins"])
    replace_mapping(DEFAULT_HPARAMS, config["default_hparams"])
    replace_mapping(BASE_RMSN_SPACE, config["search_space"])
    replace_mapping(SCALED_SPACE, config["scaled_space"])
    replace_mapping(TREATMENT_MODE_BY_DOMAIN, config["treatment_mode_by_domain"])
    for domain, mode in TREATMENT_MODE_BY_DOMAIN.items():
        if str(mode) not in {"multilabel", "multiclass"}:
            raise ValueError(f"Invalid RMSN treatment mode for {domain}: {mode!r}")
    stagewise = config["stagewise_tuning"]
    STAGEWISE_TRIALS["propensity_treatment"] = int(stagewise["propensity_treatment_trials"])
    STAGEWISE_TRIALS["propensity_history"] = int(stagewise["propensity_history_trials"])
    STAGEWISE_TRIALS["encoder"] = int(stagewise["encoder_trials"])
    STAGEWISE_TRIALS["decoder"] = int(stagewise["decoder_trials"])
    if min(STAGEWISE_TRIALS.values()) < 1:
        raise ValueError("RMSN stagewise trial counts must be positive.")


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def treatment_mode_for_domain(domain: str) -> str:
    key = str(domain).strip().lower()
    if key not in TREATMENT_MODE_BY_DOMAIN:
        raise ValueError(f"RMSN has no treatment mode for domain {domain!r}.")
    return str(TREATMENT_MODE_BY_DOMAIN[key])


def treatment_spec_for_bundle(bundle) -> tuple[str, int]:
    mode = treatment_mode_for_domain(bundle["domain"])
    return mode, treatment_dim(mode)


def round_to_valid(values):
    min_value = int(SCALED_SPACE["min_width"])
    max_value = int(SCALED_SPACE["max_width"])
    multiple = int(SCALED_SPACE["width_multiple"])
    if multiple < 1:
        raise ValueError("RMSN width_multiple must be positive.")
    out = []
    for value in values:
        if multiple == 1:
            x = int(float(value))
        else:
            x = int(round(float(value) / multiple) * multiple)
        out.append(max(min_value, min(max_value, x)))
    return sorted(set(out))


def component_multipliers(component):
    if component == "decoder":
        return list(SCALED_SPACE["decoder_width_multipliers"])
    return list(SCALED_SPACE["width_multipliers"])


def component_practical_widths(component):
    return [int(value) for value in SCALED_SPACE["practical_widths"]]


def size_grid(width, component):
    multipliers = component_multipliers(component)
    return round_to_valid([float(multiplier) * width for multiplier in multipliers])


def build_group_scaled_space(d_base, treatment_mode, d_static):
    dim_treatments = treatment_dim(treatment_mode)
    d_static = int(d_static)
    sizes = {
        "propensity_treatment": dim_treatments,
        "propensity_history": dim_treatments + int(d_base) + 1 + d_static,
        "encoder": dim_treatments + int(d_base) + 1 + d_static,
        "decoder": dim_treatments + 1 + d_static,
    }
    grids = {
        name: sorted(set(size_grid(width, component=name) + component_practical_widths(name)))
        for name, width in sizes.items()
    }
    space = dict(BASE_RMSN_SPACE)
    for component in _COMPONENTS:
        space[f"hidden_units_{component}"] = grids[component]
    return space, {
        "treatment_mode": treatment_mode,
        "dim_treatments": int(dim_treatments),
        **{f"C_{name}": int(width) for name, width in sizes.items()},
        **{f"{name}_size_grid": values for name, values in grids.items()},
    }


def make_rmsn_args(d_vitals, hparams, treatment_mode, d_static):
    def submodel_cfg(component):
        return ns(
            seq_hidden_units=int(hparams[f"hidden_units_{component}"]),
            dropout_rate=float(hparams[f"dropout_{component}"]),
            num_layer=int(hparams[f"num_layers_{component}"]),
            batch_size=int(hparams[f"batch_size_{component}"]),
            max_grad_norm=float(hparams[f"max_grad_norm_{component}"]),
            optimizer={
                "learning_rate": float(hparams[f"lr_{component}"]),
                "weight_decay": float(hparams["weight_decay"]),
                "optimizer_cls": str(hparams["optimizer_cls"]),
                "lr_scheduler": False,
            },
        )

    propensity_treatment = submodel_cfg("propensity_treatment")
    propensity_history = submodel_cfg("propensity_history")
    encoder = submodel_cfg("encoder")
    decoder = submodel_cfg("decoder")
    dim_treatments = treatment_dim(treatment_mode)
    d_static = int(d_static)
    return ns(
        model=ns(
            dim_treatments=dim_treatments,
            dim_vitals=int(d_vitals),
            dim_static_features=d_static,
            dim_outcomes=1,
            encoder=encoder,
            decoder=decoder,
            propensity_treatment=propensity_treatment,
            propensity_history=propensity_history,
        ),
        dataset=ns(
            val_batch_size=int(encoder.batch_size),
            projection_horizon=common.PROJECTION_HORIZON,
            treatment_mode=str(treatment_mode),
            holdout_ratio=0.0,
        ),
        exp=ns(
            unscale_rmse=False,
            percentage_rmse=False,
            bce_weight=False,
            gpus="[]",
            max_epochs=max(int(hparams["propensity_epochs"]), int(hparams["encoder_epochs"]), int(hparams["decoder_epochs"])),
            alpha_rate="exp",
            update_alpha=False,
        ),
    )


def sample_random_candidates(space, n, seed):
    return sample_random_hparams(space, n, seed, default_hparams=DEFAULT_HPARAMS, canonical_hparams=canonical_hparams)


def clip_normalize_weights(weights, active, quantiles=(0.01, 0.99), multiple_horizons=False):
    weights = np.asarray(weights, dtype=np.float32).copy()
    active_bool = np.asarray(active).astype(bool)
    weights[~active_bool] = np.nan
    if np.isfinite(weights).sum() == 0:
        raise RuntimeError("RMSN has no finite active stabilized weights.")
    lo, hi = quantiles
    weights = np.clip(weights, np.nanquantile(weights, float(lo)), np.nanquantile(weights, float(hi)))
    if multiple_horizons:
        denom = np.ones((1, weights.shape[1]), dtype=np.float32)
        for horizon in range(weights.shape[1]):
            values = weights[:, horizon]
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            mean = float(values.mean())
            if not np.isfinite(mean) or abs(mean) <= 1e-8:
                raise RuntimeError(f"RMSN weights have an invalid mean at horizon {horizon}.")
            denom[0, horizon] = mean
        weights = weights / denom
    else:
        denom = float(np.nanmean(weights))
        if not np.isfinite(denom) or abs(denom) < 1e-8:
            raise RuntimeError("RMSN weights have an invalid mean.")
        weights = weights / denom
    if not np.isfinite(weights[active_bool]).all():
        raise RuntimeError("RMSN produced non-finite normalized weights.")
    weights[~active_bool] = 0.0
    return weights.astype(np.float32)
