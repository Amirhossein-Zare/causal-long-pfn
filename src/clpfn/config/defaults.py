from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_training_config_path() -> Path:
    return project_root() / "configs" / "train" / "causal_long_pfn.yaml"


OBS_TIME_MIN = 3
OBS_TIME_MAX = 60
HORIZON_MIN = 1
HORIZON_MAX = 5
MAX_SEQ_LEN = OBS_TIME_MAX + HORIZON_MAX
MAX_INPUT_INDEX = MAX_SEQ_LEN - 1
MAX_TARGET_INDEX = MAX_SEQ_LEN

D_STATE_MIN = 1
D_STATE_MAX = 10
D_OUTCOME = 1
D_INPUT_MAX = D_STATE_MAX + D_OUTCOME
D_STATIC_MAX = 5

N_ACTIONS = 4
LATENT_UNIT_DIM = 3

N_SUPPORT_MIN = 3
N_SUPPORT_MAX = 500
N_SUPPORT_ANCHORS = 4

D_MODEL = 256
N_HEADS = 8
N_HISTORY_LAYERS = 4
N_PFN_LAYERS = 6
D_FF = 1024
DROPOUT = 0.1

GMM_K = 5
GMM_PI_TEMP = 1.0
GMM_MIN_SIGMA = 0.02
GMM_MAX_SIGMA = 2.0

BATCH_SIZE = 16
ACCUM_STEPS = 16
LR = 3e-4
WEIGHT_DECAY = 1e-5
WARMUP_STEPS = 400

MAX_STEPS = 10000
SCHED_TOTAL_STEPS = 10000
MIN_LR_SCALE = 0.02

CHECKPOINT_EVERY = 500
SESSION_TIMEOUT = 42000
SEED = 42

OBSERVATIONAL_QUERY_PROB = 0.30
SUPPORT_LABEL_NOISE_PROB = 0.15
SUPPORT_FUTURE_COVARIATE_MASK_PROB = 0.35

NLL_HUBER_THRESHOLD = 15.0
NLL_HUBER_SLOPE = 0.01
MEAN_HUBER_DELTA = 3.0
MEAN_LOSS_WEIGHT = 0.25
CONC_PEN_WEIGHT = 0.03
CONC_PEN_MAX_PI = 0.90

CLIP_BASE = 0.50
CLIP_MAX = 1.50
CLIP_RAMP_STEPS = 4000


HIDDEN_SENTINEL = -99.0

_CONFIG_SECTIONS = ("model", "training", "prior", "stability")


def _update_derived_defaults() -> None:
    global MAX_SEQ_LEN, MAX_INPUT_INDEX, MAX_TARGET_INDEX, D_INPUT_MAX

    MAX_SEQ_LEN = OBS_TIME_MAX + HORIZON_MAX
    MAX_INPUT_INDEX = MAX_SEQ_LEN - 1
    MAX_TARGET_INDEX = MAX_SEQ_LEN
    D_INPUT_MAX = D_STATE_MAX + D_OUTCOME


def load_training_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else default_training_config_path()
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def apply_training_config(config: Mapping[str, Any] | None) -> None:
    if not config:
        return

    for section in _CONFIG_SECTIONS:
        values = config.get(section, {}) or {}
        for key, value in values.items():
            globals()[key] = value

    _update_derived_defaults()


def configure_from_file(path: str | Path | None = None) -> dict[str, Any]:
    config = load_training_config(path)
    apply_training_config(config)
    return config


_update_derived_defaults()
