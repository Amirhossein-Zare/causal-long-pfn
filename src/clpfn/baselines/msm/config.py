from pathlib import Path

from clpfn.baselines.common.api import replace_mapping, require_baseline_config


OUTPUT_DIR = Path("outputs/eval/msm")

N_ACTION_BITS = 2
MAX_VAL_ORIGINS = 0

DEFAULT_HPARAMS = {}
MSM_SPACE = {}

TUNING_CACHE = {}


def apply_config(config):
    global N_ACTION_BITS, MAX_VAL_ORIGINS

    config = require_baseline_config(config, "msm")
    limits = config["limits"]
    N_ACTION_BITS = int(limits["n_action_bits"])
    MAX_VAL_ORIGINS = int(limits["max_val_origins"])
    replace_mapping(DEFAULT_HPARAMS, config["default_hparams"])
    replace_mapping(MSM_SPACE, config["search_space"])
    TUNING_CACHE.clear()
