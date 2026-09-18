from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

MAIN_EXPERIMENT = "causal_long_pfn"
CANONICAL_REFERENCE_EXPERIMENT = "canonical_reference"
ABLATIONS = {
    "no_motifs": {
        "prior_variant": "tscm_no_motifs",
        "change": ("ENABLE_MOTIFS", False),
    },
    "no_latent_heterogeneity": {
        "prior_variant": "tscm_no_latent_heterogeneity",
        "change": ("ENABLE_LATENT_HETEROGENEITY", False),
    },
    "no_confounding": {
        "prior_variant": "tscm_no_confounding",
        "change": ("ENABLE_CONFOUNDING", False),
    },
    "immediate_effects_only": {
        "prior_variant": "tscm_immediate_effects_only",
        "change": ("ENABLE_EXPLICIT_DELAYED_TREATMENT_EFFECTS", False),
    },
    "factual_pretraining_only": {
        "prior_variant": "tscm_factual_pretraining_only",
        "change": ("OBSERVATIONAL_QUERY_PROB", 1.0),
    },
    "counterfactual_pretraining_only": {
        "prior_variant": "tscm_counterfactual_pretraining_only",
        "change": ("OBSERVATIONAL_QUERY_PROB", 0.0),
    },
    "single_anchor_support": {
        "prior_variant": "tscm_single_anchor_support",
        "change": ("N_SUPPORT_ANCHORS", 1),
    },
}
EXPERIMENTS = (MAIN_EXPERIMENT, CANONICAL_REFERENCE_EXPERIMENT, *ABLATIONS)

D_OUTCOME = 1
D_STATIC_MAX = 5
N_ACTIONS = 4
D_MODEL = 256
N_HEADS = 8
N_HISTORY_LAYERS = 4
N_PFN_LAYERS = 4
D_FF = 1024
DROPOUT = 0.1
GMM_K = 5
GMM_PI_TEMP = 1.0
GMM_MIN_SIGMA = 0.02
GMM_MAX_SIGMA = 2.0

OBS_TIME_MIN = 1
OBS_TIME_MAX = 60
HORIZON_MIN = 1
HORIZON_MAX = 5
D_STATE_MIN = 1
D_STATE_MAX = 10
N_SUPPORT_MIN = 3
N_SUPPORT_MAX = 500
N_SUPPORT_ANCHORS = 4
OBSERVATIONAL_QUERY_PROB = 0.50
SUPPORT_FUTURE_COVARIATE_MASK_PROB = 0.35
ENABLE_MOTIFS = True
ENABLE_LATENT_HETEROGENEITY = True
ENABLE_EXPLICIT_DELAYED_TREATMENT_EFFECTS = True
ENABLE_CONFOUNDING = True

CLIP_NORM = 1.0
BATCH_SIZE = 16
ACCUM_STEPS = 16
TRAIN_NUM_WORKERS = 0
LR = 0.0003
WEIGHT_DECAY = 1.0e-05
MIN_LR_SCALE = 0.02
LOG_EVERY = 100
PRIOR_VALIDATION_EVERY = 1000
PRIOR_VALIDATION_SIZE = 100
PRIOR_VALIDATION_SEED = 2026
SESSION_TIMEOUT = 999999
DETERMINISTIC = False
ALLOW_TF32 = True
WARMUP_STEPS = 400
MAX_STEPS = 10000
SCHED_TOTAL_STEPS = 10000
CHECKPOINT_EVERY = 500
INFERENCE_MILESTONE_UPDATES = [1000, 2500, 5000, 7500, 10000]
SEED = 42

LATENT_UNIT_DIM = 3
HIDDEN_SENTINEL = -99.0

CANONICAL_MODEL = {
    "D_OUTCOME": D_OUTCOME,
    "D_STATIC_MAX": D_STATIC_MAX,
    "N_ACTIONS": N_ACTIONS,
    "D_MODEL": D_MODEL,
    "N_HEADS": N_HEADS,
    "N_HISTORY_LAYERS": N_HISTORY_LAYERS,
    "N_PFN_LAYERS": N_PFN_LAYERS,
    "D_FF": D_FF,
    "DROPOUT": DROPOUT,
    "GMM_K": GMM_K,
    "GMM_PI_TEMP": GMM_PI_TEMP,
    "GMM_MIN_SIGMA": GMM_MIN_SIGMA,
    "GMM_MAX_SIGMA": GMM_MAX_SIGMA,
}
CANONICAL_PRIOR = {
    "OBS_TIME_MIN": OBS_TIME_MIN,
    "OBS_TIME_MAX": OBS_TIME_MAX,
    "HORIZON_MIN": HORIZON_MIN,
    "HORIZON_MAX": HORIZON_MAX,
    "D_STATE_MIN": D_STATE_MIN,
    "D_STATE_MAX": D_STATE_MAX,
    "N_SUPPORT_MIN": N_SUPPORT_MIN,
    "N_SUPPORT_MAX": N_SUPPORT_MAX,
    "N_SUPPORT_ANCHORS": N_SUPPORT_ANCHORS,
    "OBSERVATIONAL_QUERY_PROB": OBSERVATIONAL_QUERY_PROB,
    "SUPPORT_FUTURE_COVARIATE_MASK_PROB": SUPPORT_FUTURE_COVARIATE_MASK_PROB,
    "ENABLE_MOTIFS": ENABLE_MOTIFS,
    "ENABLE_LATENT_HETEROGENEITY": ENABLE_LATENT_HETEROGENEITY,
    "ENABLE_EXPLICIT_DELAYED_TREATMENT_EFFECTS": ENABLE_EXPLICIT_DELAYED_TREATMENT_EFFECTS,
    "ENABLE_CONFOUNDING": ENABLE_CONFOUNDING,
}
CANONICAL_STABILITY = {"CLIP_NORM": CLIP_NORM}
COMMON_TRAINING = {
    "BATCH_SIZE": BATCH_SIZE,
    "ACCUM_STEPS": ACCUM_STEPS,
    "TRAIN_NUM_WORKERS": TRAIN_NUM_WORKERS,
    "LR": LR,
    "WEIGHT_DECAY": WEIGHT_DECAY,
    "MIN_LR_SCALE": MIN_LR_SCALE,
    "LOG_EVERY": LOG_EVERY,
    "SESSION_TIMEOUT": SESSION_TIMEOUT,
    "DETERMINISTIC": DETERMINISTIC,
    "ALLOW_TF32": ALLOW_TF32,
}
MAIN_TRAINING_SCHEDULE = {
    "WARMUP_STEPS": WARMUP_STEPS,
    "MAX_STEPS": MAX_STEPS,
    "SCHED_TOTAL_STEPS": SCHED_TOTAL_STEPS,
    "CHECKPOINT_EVERY": CHECKPOINT_EVERY,
    "INFERENCE_MILESTONE_UPDATES": INFERENCE_MILESTONE_UPDATES,
    "PRIOR_VALIDATION_EVERY": PRIOR_VALIDATION_EVERY,
    "PRIOR_VALIDATION_SIZE": PRIOR_VALIDATION_SIZE,
    "PRIOR_VALIDATION_SEED": PRIOR_VALIDATION_SEED,
}
ABLATION_TRAINING_SCHEDULE = {
    "WARMUP_STEPS": 100,
    "MAX_STEPS": 2500,
    "SCHED_TOTAL_STEPS": 2500,
    "CHECKPOINT_EVERY": 100,
    "INFERENCE_MILESTONE_UPDATES": [500, 1000, 1500, 2000, 2500],
}
SUPPORTED_SEEDS = (42, 43, 44)
CANONICAL_TRAINING = COMMON_TRAINING | MAIN_TRAINING_SCHEDULE | {"SEED": SEED}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_training_config_path() -> Path:
    return project_root() / "configs" / "train" / "causal_long_pfn.yaml"


def _update_derived_defaults() -> None:
    global MAX_SEQ_LEN, MAX_INPUT_INDEX, MAX_TARGET_INDEX, D_INPUT_MAX

    MAX_SEQ_LEN = OBS_TIME_MAX + HORIZON_MAX
    MAX_INPUT_INDEX = MAX_SEQ_LEN - 1
    MAX_TARGET_INDEX = MAX_SEQ_LEN
    D_INPUT_MAX = D_STATE_MAX + D_OUTCOME


def load_training_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else default_training_config_path()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Training configuration must be a mapping: {config_path}")
    return config


def apply_training_config(config: dict[str, Any]) -> None:
    validate_training_config(config)

    global D_OUTCOME, D_STATIC_MAX, N_ACTIONS, D_MODEL, N_HEADS
    global N_HISTORY_LAYERS, N_PFN_LAYERS, D_FF, DROPOUT
    global GMM_K, GMM_PI_TEMP, GMM_MIN_SIGMA, GMM_MAX_SIGMA
    global BATCH_SIZE, ACCUM_STEPS, TRAIN_NUM_WORKERS, LR, WEIGHT_DECAY
    global WARMUP_STEPS, MAX_STEPS, SCHED_TOTAL_STEPS, MIN_LR_SCALE
    global CHECKPOINT_EVERY, LOG_EVERY, INFERENCE_MILESTONE_UPDATES
    global SESSION_TIMEOUT, SEED, DETERMINISTIC, ALLOW_TF32
    global OBS_TIME_MIN, OBS_TIME_MAX, HORIZON_MIN, HORIZON_MAX
    global D_STATE_MIN, D_STATE_MAX, N_SUPPORT_MIN, N_SUPPORT_MAX
    global N_SUPPORT_ANCHORS, OBSERVATIONAL_QUERY_PROB
    global SUPPORT_FUTURE_COVARIATE_MASK_PROB, ENABLE_MOTIFS
    global ENABLE_LATENT_HETEROGENEITY
    global ENABLE_EXPLICIT_DELAYED_TREATMENT_EFFECTS, ENABLE_CONFOUNDING
    global CLIP_NORM

    model = config["model"]
    training = config["training"]
    prior = config["prior"]
    stability = config["stability"]

    D_OUTCOME = model["D_OUTCOME"]
    D_STATIC_MAX = model["D_STATIC_MAX"]
    N_ACTIONS = model["N_ACTIONS"]
    D_MODEL = model["D_MODEL"]
    N_HEADS = model["N_HEADS"]
    N_HISTORY_LAYERS = model["N_HISTORY_LAYERS"]
    N_PFN_LAYERS = model["N_PFN_LAYERS"]
    D_FF = model["D_FF"]
    DROPOUT = model["DROPOUT"]
    GMM_K = model["GMM_K"]
    GMM_PI_TEMP = model["GMM_PI_TEMP"]
    GMM_MIN_SIGMA = model["GMM_MIN_SIGMA"]
    GMM_MAX_SIGMA = model["GMM_MAX_SIGMA"]

    BATCH_SIZE = training["BATCH_SIZE"]
    ACCUM_STEPS = training["ACCUM_STEPS"]
    TRAIN_NUM_WORKERS = training["TRAIN_NUM_WORKERS"]
    LR = training["LR"]
    WEIGHT_DECAY = training["WEIGHT_DECAY"]
    WARMUP_STEPS = training["WARMUP_STEPS"]
    MAX_STEPS = training["MAX_STEPS"]
    SCHED_TOTAL_STEPS = training["SCHED_TOTAL_STEPS"]
    MIN_LR_SCALE = training["MIN_LR_SCALE"]
    CHECKPOINT_EVERY = training["CHECKPOINT_EVERY"]
    LOG_EVERY = training["LOG_EVERY"]
    INFERENCE_MILESTONE_UPDATES = list(training["INFERENCE_MILESTONE_UPDATES"])
    SESSION_TIMEOUT = training["SESSION_TIMEOUT"]
    SEED = training["SEED"]
    DETERMINISTIC = training["DETERMINISTIC"]
    ALLOW_TF32 = training["ALLOW_TF32"]

    OBS_TIME_MIN = prior["OBS_TIME_MIN"]
    OBS_TIME_MAX = prior["OBS_TIME_MAX"]
    HORIZON_MIN = prior["HORIZON_MIN"]
    HORIZON_MAX = prior["HORIZON_MAX"]
    D_STATE_MIN = prior["D_STATE_MIN"]
    D_STATE_MAX = prior["D_STATE_MAX"]
    N_SUPPORT_MIN = prior["N_SUPPORT_MIN"]
    N_SUPPORT_MAX = prior["N_SUPPORT_MAX"]
    N_SUPPORT_ANCHORS = prior["N_SUPPORT_ANCHORS"]
    OBSERVATIONAL_QUERY_PROB = prior["OBSERVATIONAL_QUERY_PROB"]
    SUPPORT_FUTURE_COVARIATE_MASK_PROB = prior["SUPPORT_FUTURE_COVARIATE_MASK_PROB"]
    ENABLE_MOTIFS = prior["ENABLE_MOTIFS"]
    ENABLE_LATENT_HETEROGENEITY = prior["ENABLE_LATENT_HETEROGENEITY"]
    ENABLE_EXPLICIT_DELAYED_TREATMENT_EFFECTS = prior["ENABLE_EXPLICIT_DELAYED_TREATMENT_EFFECTS"]
    ENABLE_CONFOUNDING = prior["ENABLE_CONFOUNDING"]

    CLIP_NORM = stability["CLIP_NORM"]
    _update_derived_defaults()

def configure_from_file(path: str | Path | None = None) -> dict[str, Any]:
    config = load_training_config(path)
    apply_training_config(config)
    return config


def expected_prior(experiment: str) -> dict[str, Any]:
    prior = dict(CANONICAL_PRIOR)
    if experiment in ABLATIONS:
        key, value = ABLATIONS[experiment]["change"]
        prior[key] = value
    return prior


def validate_training_config(config: dict[str, Any]) -> None:
    required_sections = {"identity", "runtime", "model", "training", "prior", "stability"}
    missing = sorted(required_sections.difference(config))
    if missing:
        raise ValueError(f"Training configuration is missing sections: {missing}")

    identity = config["identity"]
    experiment = identity["model_variant"]
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unsupported experiment {experiment!r}; choose from {EXPERIMENTS}")
    if config["model"] != CANONICAL_MODEL:
        raise ValueError("Model section differs from the supported architecture.")
    if config["prior"] != expected_prior(experiment):
        raise ValueError(
            f"Prior section for {experiment!r} must differ from the main model "
            "only in its declared mechanism."
        )
    if config["stability"] != CANONICAL_STABILITY:
        raise ValueError("Stability section must use gradient clipping at 1.0.")

    schedule = (
        MAIN_TRAINING_SCHEDULE
        if experiment == MAIN_EXPERIMENT
        else ABLATION_TRAINING_SCHEDULE
    )
    training = config["training"]
    seed = training["SEED"]
    if seed not in SUPPORTED_SEEDS:
        raise ValueError(
            f"training.SEED must be one of {SUPPORTED_SEEDS}; found {seed}."
        )
    expected_training = COMMON_TRAINING | schedule | {"SEED": seed}
    if training != expected_training:
        raise ValueError(
            f"Training section for {experiment!r} differs from the supported configuration."
        )

    prior_variant = (
        "tscm"
        if experiment in (MAIN_EXPERIMENT, CANONICAL_REFERENCE_EXPERIMENT)
        else ABLATIONS[experiment]["prior_variant"]
    )
    if identity["prior_variant"] != prior_variant:
        raise ValueError(
            f"identity.prior_variant must be {prior_variant!r} for {experiment!r}."
        )


_update_derived_defaults()
