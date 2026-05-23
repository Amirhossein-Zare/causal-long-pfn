from pathlib import Path

from clpfn.baselines.common.api import replace_mapping, require_baseline_config


OUTPUT_DIR = Path("outputs/eval/gtransformer")

GT_PROJECTION_HORIZON = 0
MAX_VAL_ORIGINS = 0

DEFAULT_HPARAMS = {}
GTRANSFORMER_SPACE = {}

TUNING_CACHE = {}


def apply_config(config):
    global GT_PROJECTION_HORIZON, MAX_VAL_ORIGINS

    config = require_baseline_config(config, "gtransformer")
    limits = config["limits"]
    GT_PROJECTION_HORIZON = int(limits["projection_horizon"])
    MAX_VAL_ORIGINS = int(limits["max_val_origins"])
    replace_mapping(DEFAULT_HPARAMS, config["default_hparams"])
    replace_mapping(GTRANSFORMER_SPACE, config["search_space"])
    TUNING_CACHE.clear()
