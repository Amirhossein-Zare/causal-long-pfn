import os
import random
import logging

import numpy as np
import torch


LOGGER = logging.getLogger(__name__)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint(model, optimizer, scheduler, step, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload = {
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step_count": int(step),
        "torch_rng_state": torch.get_rng_state(),
        "np_rng_state": np.random.get_state(),
        "py_rng_state": random.getstate(),
    }

    torch.save(payload, path)
    LOGGER.info("[ckpt] saved %s (step %s)", path, step)


def load_latest_checkpoint(model, optimizer, scheduler, ckpt_dir):
    candidates = [
        name
        for name in os.listdir(ckpt_dir)
        if name.startswith("ckpt_") and name.endswith(".pt")
    ]

    if not candidates:
        return 0

    def checkpoint_sort_key(name):
        value = name.replace("ckpt_step_", "").replace("ckpt_final", "99999999").replace(".pt", "")

        try:
            return int(value)
        except Exception:
            return -1

    path = os.path.join(ckpt_dir, max(candidates, key=checkpoint_sort_key))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    state_dict = checkpoint["model_state_dict"]

    if all(key.startswith("module.") for key in state_dict.keys()):
        LOGGER.info("[ckpt] stripping 'module.' prefix from checkpoint")
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

    unwrap_model(model).load_state_dict(state_dict)

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    LOGGER.info("[ckpt] optimizer + scheduler restored")

    try:
        torch.set_rng_state(checkpoint["torch_rng_state"])
        np.random.set_state(checkpoint["np_rng_state"])
        random.setstate(checkpoint["py_rng_state"])
    except Exception:
        pass

    step = int(checkpoint.get("step_count", 0))
    LOGGER.info("[ckpt] resumed from step %s", step)

    return step
