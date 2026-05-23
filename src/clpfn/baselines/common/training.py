from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from clpfn.evaluation.core import benchmark as common


def train_loader(dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=min(int(batch_size), len(dataset)),
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch_training(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    grad_clip: float,
    move_batch_to_device: Callable[[dict[str, Any]], dict[str, Any]],
    step_fn: Callable[[torch.nn.Module, dict[str, Any], int], tuple[torch.Tensor, dict[str, Any]]] | None = None,
) -> tuple[float, dict[str, float], float]:

    def _default_step(current_model, batch, batch_ind):
        return current_model.training_step(batch, batch_ind), {}

    step = step_fn or _default_step
    t0 = time.time()
    last_loss = float("nan")
    last_metrics: dict[str, float] = {}

    for _ in range(int(epochs)):
        epoch_loss = 0.0
        denom = 0
        model.train()

        for batch_ind, batch in enumerate(loader):
            batch = move_batch_to_device(batch)
            loss, metrics = step(model, batch, batch_ind)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

            bs = int(batch["outputs"].shape[0])
            epoch_loss += float(loss.detach().cpu()) * bs
            denom += bs
            last_metrics = {
                key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
                for key, value in metrics.items()
            }

        last_loss = epoch_loss / max(denom, 1)

    return float(last_loss), last_metrics, float(time.time() - t0)


def move_float_batch_to_device(batch: dict[str, Any]) -> dict[str, Any]:
    return common.move_tensor_batch_to_device(batch, float_tensors=True)


def masked_mean(values: torch.Tensor, active_entries: torch.Tensor) -> torch.Tensor:
    return (values * active_entries).sum() / active_entries.sum().clamp(min=1.0)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, active_entries: torch.Tensor) -> torch.Tensor:
    return masked_mean((pred - target) ** 2, active_entries)


def masked_weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    active_entries: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return masked_mean(F.mse_loss(pred, target, reduction="none") * weights.unsqueeze(-1), active_entries)


def masked_sequence_loss(loss_by_timestep: torch.Tensor, active_entries: torch.Tensor) -> torch.Tensor:
    return (
        active_entries.squeeze(-1) * loss_by_timestep
    ).sum() / active_entries.sum().clamp(min=1.0)


def masked_multiclass_ce_loss(
    logits: torch.Tensor,
    target_idx: torch.Tensor,
    active_entries: torch.Tensor,
) -> torch.Tensor:
    active_flat = active_entries.squeeze(-1).reshape(-1) > 0.5
    if not active_flat.any():
        return logits.sum() * 0.0
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1])[active_flat],
        target_idx.reshape(-1)[active_flat].long(),
    )


def support_val_candidates(bundle: dict[str, Any], val_idx, seed: int, max_val_origins: int):
    rng = np.random.default_rng(int(seed))
    y_norm = bundle["y_norm_clip"]
    lengths = bundle["sequence_lengths"]
    candidates = []

    for i in np.asarray(val_idx, dtype=np.int64):
        max_origin = min(int(lengths[i]) - 2, y_norm.shape[1] - 2, common.MAX_INPUT_INDEX)
        if max_origin < common.MIN_T_OBS:
            continue
        for origin in range(common.MIN_T_OBS, max_origin + 1):
            candidates.append((int(i), int(origin), int(origin + 1)))
            horizon_target = origin + common.PROJECTION_HORIZON
            if horizon_target < int(lengths[i]) and horizon_target < y_norm.shape[1]:
                candidates.append((int(i), int(origin), int(horizon_target)))

    if len(candidates) > max_val_origins:
        keep = rng.choice(len(candidates), size=max_val_origins, replace=False)
        candidates = [candidates[k] for k in keep]
    return candidates


def targets_for_candidates(bundle: dict[str, Any], candidates) -> np.ndarray:
    y_norm = bundle["y_norm_clip"]
    return np.asarray(
        [
            float(np.clip(y_norm[row_id, target_t], -common.TARGET_NORM_CLIP, common.TARGET_NORM_CLIP))
            for row_id, _, target_t in candidates
        ],
        dtype=np.float32,
    )


def rmse_from_predictions(predictions, targets) -> float:
    pred = np.asarray(predictions, dtype=np.float32)
    target = np.asarray(targets, dtype=np.float32)
    mask = np.isfinite(pred) & np.isfinite(target)
    return float(np.sqrt(np.mean((pred[mask] - target[mask]) ** 2))) if mask.any() else float("nan")


def evaluate_single_rollout_val_rmse(
    bundle: dict[str, Any],
    model: Any,
    val_idx,
    seed: int,
    max_val_origins: int,
    predict_fn: Callable[..., tuple[float, Any]],
) -> float:
    candidates = support_val_candidates(bundle, val_idx, seed, max_val_origins)
    if not candidates:
        return float("nan")

    predictions = [
        predict_fn(model, bundle, row_id, t_obs, t_target)[0]
        for row_id, t_obs, t_target in candidates
    ]
    return rmse_from_predictions(predictions, targets_for_candidates(bundle, candidates))


def evaluate_paired_rollout_val_rmse(
    bundle: dict[str, Any],
    encoder: Any,
    decoder: Any,
    val_idx,
    seed: int,
    max_val_origins: int,
    predict_fn: Callable[..., tuple[float, Any]],
) -> float:
    candidates = support_val_candidates(bundle, val_idx, seed, max_val_origins)
    if not candidates:
        return float("nan")

    predictions = [
        predict_fn(encoder, decoder, bundle, row_id, t_obs, t_target)[0]
        for row_id, t_obs, t_target in candidates
    ]
    return rmse_from_predictions(predictions, targets_for_candidates(bundle, candidates))
