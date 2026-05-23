from pathlib import Path

from clpfn.baselines.common.api import replace_mapping, require_baseline_config


OUTPUT_DIR = Path("outputs/eval/ct")

MAX_VAL_ORIGINS = 0
CT_EVAL_BATCH_SIZE = 1

DEFAULT_HPARAMS = {}
CT_SPACE = {}

TUNING_CACHE = {}


def apply_config(config):
    global MAX_VAL_ORIGINS, CT_EVAL_BATCH_SIZE

    config = require_baseline_config(config, "ct")
    limits = config["limits"]
    MAX_VAL_ORIGINS = int(limits["max_val_origins"])
    CT_EVAL_BATCH_SIZE = int(limits["eval_batch_size"])
    replace_mapping(DEFAULT_HPARAMS, config["default_hparams"])
    replace_mapping(CT_SPACE, config["search_space"])
    TUNING_CACHE.clear()
