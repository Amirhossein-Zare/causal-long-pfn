import math
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from clpfn.config.defaults import (
    ACCUM_STEPS,
    BATCH_SIZE,
    CHECKPOINT_EVERY,
    CLIP_BASE,
    CLIP_MAX,
    CLIP_RAMP_STEPS,
    LR,
    MAX_STEPS,
    MIN_LR_SCALE,
    N_PFN_LAYERS,
    SCHED_TOTAL_STEPS,
    SEED,
    SESSION_TIMEOUT,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from clpfn.data.datasets.synthetic_pretraining_dataset import OnTheFlyEpisodeDataset, collate_episode_batch
from clpfn.models.causal_long_pfn import CausalLongPFN
from clpfn.training.checkpointing import load_latest_checkpoint, save_checkpoint
from clpfn.training.losses import gaussian_mixture_loss
from clpfn.training.optim import build_adamw_optimizer, build_cosine_warmup_scheduler


CKPT_INPUT_DIR = os.environ.get("CAUSALLONGPFN_CKPT_INPUT_DIR", "")
OUTPUT_DIR = os.environ.get("CAUSALLONGPFN_OUTPUT_DIR", "./outputs")

LOGGER = logging.getLogger(__name__)


def configure_runtime():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        torch.set_float32_matmul_precision("high")


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    configure_runtime()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpu = torch.cuda.device_count()

    model = CausalLongPFN().to(device)

    if n_gpu > 1:
        model = nn.DataParallel(model)

    n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    LOGGER.info("Device: %s | GPUs: %s | Params: %s", device, n_gpu, f"{n_params:,}")
    LOGGER.info("CausalLongPFN synthetic pretraining: one-step autoregressive PFN")
    LOGGER.info("Output dir: %s", OUTPUT_DIR)
    LOGGER.info("Checkpoint input dir: %s", CKPT_INPUT_DIR if CKPT_INPUT_DIR else "<none>")

    optimizer = build_adamw_optimizer(model, LR, WEIGHT_DECAY)
    scheduler = build_cosine_warmup_scheduler(
        optimizer,
        WARMUP_STEPS,
        SCHED_TOTAL_STEPS,
        min_lr_scale=MIN_LR_SCALE,
    )

    LOGGER.info("Optimizer group LRs at init: %s", [group["lr"] for group in optimizer.param_groups])

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    step = load_latest_checkpoint(
        model,
        optimizer,
        scheduler,
        CKPT_INPUT_DIR,
    ) if os.path.isdir(CKPT_INPUT_DIR) else 0

    start_step = step

    loader_num_workers = 4 if torch.cuda.is_available() else 0

    loader_kwargs = dict(
        dataset=OnTheFlyEpisodeDataset(base_seed=SEED + step),
        batch_size=BATCH_SIZE,
        num_workers=loader_num_workers,
        collate_fn=collate_episode_batch,
        pin_memory=torch.cuda.is_available(),
    )

    if loader_num_workers > 0:
        loader_kwargs.update(prefetch_factor=4, persistent_workers=True)

    loader = DataLoader(**loader_kwargs)

    start_time = time.time()

    running_loss = 0.0
    running_sigma = 0.0
    running_grad_norm = 0.0
    running_pred_std = 0.0
    running_conc_pen = 0.0

    recent_skips = 0
    recent_bad_losses = 0
    micro_step = 0

    model.train()

    for batch in loader:
        elapsed = time.time() - start_time

        if elapsed > SESSION_TIMEOUT or step >= MAX_STEPS:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                step,
                os.path.join(OUTPUT_DIR, "ckpt_final.pt"),
            )

            return

        batch = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

        n_active_pfn_layers = random.randint(N_PFN_LAYERS // 2, N_PFN_LAYERS)

        with torch.amp.autocast(amp_device, enabled=torch.cuda.is_available()):
            log_pi, mu, sigma = model(batch, n_layers=n_active_pfn_layers)
            loss, aux = gaussian_mixture_loss(log_pi, mu, sigma, batch["target_y_norm"])

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)

            if scaler.is_enabled():
                current_scale = float(scaler.get_scale())
                scaler.update(new_scale=max(current_scale / 2.0, 1.0))

            recent_skips += 1
            recent_bad_losses += 1
            step += 1
            micro_step = 0

            continue

        scaler.scale(loss / ACCUM_STEPS).backward()

        with torch.no_grad():
            running_pred_std += aux["pred_mean"].std().item() if aux["pred_mean"].shape[0] > 1 else 0.0
            running_sigma += aux["sigma_mean"].item()
            running_loss += loss.item()
            running_conc_pen += aux["concentration_penalty"].item()

        micro_step += 1

        if micro_step % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)

            total_grad_sq = 0.0
            has_bad_grad = False

            for param in model.parameters():
                if param.grad is not None:
                    grad_norm_param = param.grad.detach().norm(2).item()

                    if math.isfinite(grad_norm_param):
                        total_grad_sq += grad_norm_param ** 2
                    else:
                        has_bad_grad = True

            grad_norm = math.sqrt(total_grad_sq) if not has_bad_grad else float("inf")
            clip_value = CLIP_BASE + min(CLIP_MAX - CLIP_BASE, step / float(CLIP_RAMP_STEPS))
            skip_step = not math.isfinite(grad_norm)

            if skip_step:
                optimizer.zero_grad(set_to_none=True)

                if scaler.is_enabled():
                    current_scale = float(scaler.get_scale())
                    scaler.update(new_scale=max(current_scale / 2.0, 1.0))

                recent_skips += 1
                step += 1

            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

                if scaler.is_enabled():
                    scale_before = float(scaler.get_scale())
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = float(scaler.get_scale())

                    if scale_after >= scale_before:
                        scheduler.step()
                else:
                    optimizer.step()
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)

                running_grad_norm += min(grad_norm, 1e4)
                step += 1

            if step % 100 == 0:
                denom = 100 * ACCUM_STEPS

                last_loss = running_loss / denom
                last_sigma = running_sigma / denom
                last_pred_std = running_pred_std / denom
                last_grad_norm = running_grad_norm / max(1, 100 - recent_skips)
                last_conc_pen = running_conc_pen / denom

                base_lr = optimizer.param_groups[0]["lr"]
                speed = max(step - start_step, 1) / max(elapsed, 1e-6)
                eta_hours = (MAX_STEPS - step) / max(speed, 1e-9) / 3600.0
                grad_norm_str = f"{last_grad_norm:.2f}" if math.isfinite(last_grad_norm) else "inf"

                LOGGER.info(
                    "step %7d | loss %.4f | lr %.2e | %.3f it/s | ETA %.1fh",
                    step,
                    last_loss,
                    base_lr,
                    speed,
                    eta_hours,
                )
                LOGGER.info(
                    "sig:%.3f | p_std:%.4f | conc:%.4f | skip:%s | badloss:%s | gn:%s clip:%.2f",
                    last_sigma,
                    last_pred_std,
                    last_conc_pen,
                    recent_skips,
                    recent_bad_losses,
                    grad_norm_str,
                    clip_value,
                )

                running_loss = 0.0
                running_sigma = 0.0
                running_grad_norm = 0.0
                running_pred_std = 0.0
                running_conc_pen = 0.0

                recent_skips = 0
                recent_bad_losses = 0

            if step % CHECKPOINT_EVERY == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    step,
                    os.path.join(OUTPUT_DIR, f"ckpt_step_{step}.pt"),
                )

                kept = sorted(
                    [
                        name
                        for name in os.listdir(OUTPUT_DIR)
                        if name.startswith("ckpt_step_") and name.endswith(".pt")
                    ],
                    key=lambda name: int(name.replace("ckpt_step_", "").replace(".pt", "")),
                )

                for checkpoint_name in kept[:-3]:
                    os.remove(os.path.join(OUTPUT_DIR, checkpoint_name))

    save_checkpoint(
        model,
        optimizer,
        scheduler,
        step,
        os.path.join(OUTPUT_DIR, "ckpt_final.pt"),
    )

    LOGGER.info("Done. Steps: %s", step)


if __name__ == "__main__":
    train()
