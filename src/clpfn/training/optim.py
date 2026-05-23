import math
import logging

import torch


LOGGER = logging.getLogger(__name__)


def build_adamw_optimizer(model, lr, weight_decay):
    decay_params, no_decay_params = [], []

    no_decay_names = (
        "bias",
        "norm",
        "query_label_embedding",
        "support_y_stats_encoder",
        "static_encoder",
    )

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(key in name for key in no_decay_names):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    n_decay = sum(param.numel() for param in decay_params)
    n_no_decay = sum(param.numel() for param in no_decay_params)

    LOGGER.info("AdamW: %s wd=%s | %s wd=0", f"{n_decay:,}", weight_decay, f"{n_no_decay:,}")

    return torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay_params, "weight_decay": 0.0, "lr": lr},
    ])


def build_cosine_warmup_scheduler(optimizer, warmup, total, min_lr_scale=0.02):
    total = max(total, warmup + 1)

    def lr_fn(step):
        if step < warmup:
            return step / max(1, warmup)

        progress = min((step - warmup) / max(1, total - warmup), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
