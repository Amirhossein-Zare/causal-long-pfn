import hashlib
import importlib.metadata
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from clpfn.config.defaults import (
    ACCUM_STEPS,
    ALLOW_TF32,
    BATCH_SIZE,
    CHECKPOINT_EVERY,
    CLIP_NORM,
    DETERMINISTIC,
    INFERENCE_MILESTONE_UPDATES,
    LOG_EVERY,
    LR,
    MAX_STEPS,
    MIN_LR_SCALE,
    N_PFN_LAYERS,
    SCHED_TOTAL_STEPS,
    SEED,
    SESSION_TIMEOUT,
    TRAIN_NUM_WORKERS,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from clpfn.config.defaults import validate_training_config
from clpfn.data.datasets.synthetic_pretraining_dataset import (
    OnTheFlyEpisodeDataset,
    collate_episode_batch,
)
from clpfn.models.causal_long_pfn import (
    CausalLongPFN,
    model_config_from_resolved_config,
)
from clpfn.training.checkpointing import (
    load_latest_checkpoint,
    restore_training_state,
    save_checkpoint,
    tensor_fingerprint_from_state_dict,
)
from clpfn.training.losses import gaussian_mixture_loss
from clpfn.training.optim import build_adamw_optimizer, build_cosine_warmup_scheduler


LOGGER = logging.getLogger(__name__)
PRETRAINING_RUNTIME_FILENAME = "pretraining_runtime.json"
TRAINING_HISTORY_FILENAME = "training_history.parquet"
RUN_MANIFEST_FILENAME = "run_manifest.json"


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _config_hash(config):
    return hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()


def _resume_config(config):
    """Scientific/training state that must remain identical across sessions."""
    signature = deepcopy(config)
    signature.pop("runtime", None)
    for key in (
        "SESSION_TIMEOUT",
        "CHECKPOINT_EVERY",
        "LOG_EVERY",
        "INFERENCE_MILESTONE_UPDATES",
    ):
        signature["training"].pop(key, None)
    return signature


def _update_scaler_after_skipped_unscaled_step(scaler):
    if scaler.is_enabled():
        scaler.update()


def _code_commit_or_hash():
    root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip(), result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        digest = hashlib.sha256()
        for path in sorted(root.glob("src/**/*.py")):
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
        return None, f"source_sha256:{digest.hexdigest()}"


def _environment_versions():
    packages = {}
    for package in ("torch", "numpy", "pandas", "pyarrow", "PyYAML"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version,
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_names": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
    }


def configure_runtime():
    if DETERMINISTIC:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:128",
    )
    torch.manual_seed(int(SEED))
    np.random.seed(int(SEED))
    random.seed(int(SEED))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(SEED))
        torch.backends.cuda.matmul.allow_tf32 = bool(ALLOW_TF32)
        torch.backends.cudnn.allow_tf32 = bool(ALLOW_TF32)
        torch.backends.cudnn.benchmark = not bool(DETERMINISTIC)
        torch.backends.cudnn.deterministic = bool(DETERMINISTIC)
        torch.set_float32_matmul_precision("high" if ALLOW_TF32 else "highest")
    if DETERMINISTIC:
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_loader_worker(worker_id):
    worker = torch.utils.data.get_worker_info()
    dataset = worker.dataset
    seed_sequence = np.random.SeedSequence(
        [int(dataset.base_seed), int(dataset.start_index), int(worker_id)]
    )
    worker_seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _validate_in_prior(model, device, amp_device, *, size, seed):
    was_training = model.training
    model.eval()
    iterator = iter(OnTheFlyEpisodeDataset(base_seed=int(seed), start_index=0))
    total_nll = 0.0
    total_squared_error = 0.0
    seen = 0
    try:
        with torch.inference_mode():
            while seen < int(size):
                batch_size = min(int(BATCH_SIZE), int(size) - seen)
                batch = collate_episode_batch([next(iterator) for _ in range(batch_size)])
                batch = {
                    key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                with torch.amp.autocast(amp_device, enabled=torch.cuda.is_available()):
                    log_pi, mu, sigma = model(batch)
                    loss, aux = gaussian_mixture_loss(
                        log_pi, mu, sigma, batch["target_y_norm"]
                    )
                errors = aux["pred_mean"].float() - batch["target_y_norm"].float()
                total_nll += float(loss.item()) * batch_size
                total_squared_error += float(errors.square().sum().item())
                seen += batch_size
    finally:
        model.train(was_training)
    return {
        "prior_val_nll": total_nll / seen,
        "prior_val_rmse": math.sqrt(total_squared_error / seen),
        "prior_val_size": seen,
    }

def _write_json(path, value):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def _load_training_history(output_dir):
    path = os.path.join(output_dir, TRAINING_HISTORY_FILENAME)
    if not os.path.exists(path):
        return []
    return pd.read_parquet(path).to_dict("records")


def _append_training_history(output_dir, row, history):
    finite_fields = (
        "loss_total",
        "loss_nll",
        "predictive_sigma_mean",
        "predictive_std_mean",
        "gradient_norm_pre_clip",
        "clip_threshold",
        "learning_rate",
        "active_pfn_layers",
        "episodes_per_second",
        "wall_time",
    )
    invalid = [key for key in finite_fields if not math.isfinite(float(row[key]))]
    if invalid:
        raise RuntimeError(f"Refusing to write non-finite training-history metrics: {invalid}")
    if int(row["successful_update"]) > int(row["attempted_update"]):
        raise RuntimeError("successful_update cannot exceed attempted_update")
    if any(
        int(previous["successful_update"]) == int(row["successful_update"])
        for previous in history
    ):
        raise RuntimeError(
            f"training_history already contains successful update {row['successful_update']}"
        )
    history.append(row)
    path = os.path.join(output_dir, TRAINING_HISTORY_FILENAME)
    tmp_path = f"{path}.tmp"
    pd.DataFrame(history).to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def _checkpoint_metadata(resolved_config, launch_command, started_at):
    identity = resolved_config["identity"]
    config_hash = _config_hash(resolved_config)
    resume_config_hash = _config_hash(_resume_config(resolved_config))
    git_commit, code_commit_or_hash = _code_commit_or_hash()
    return {
        "experiment_id": identity["experiment_id"],
        "pretraining_seed": int(resolved_config["training"]["SEED"]),
        "model_variant": identity["model_variant"],
        "prior_variant": identity["prior_variant"],
        "resolved_config": deepcopy(resolved_config),
        "model_config": model_config_from_resolved_config(resolved_config),
        "prior_config": deepcopy(resolved_config["prior"]),
        "training_config": {
            "training": deepcopy(resolved_config["training"]),
            "stability": deepcopy(resolved_config["stability"]),
        },
        "config_hash": config_hash,
        "config_sha256": config_hash,
        "resume_config_hash": resume_config_hash,
        "git_commit": git_commit,
        "code_commit_or_hash": code_commit_or_hash,
        "environment_versions": _environment_versions(),
        "launch_command": launch_command,
        "started_at": started_at,
    }


def _write_pretraining_runtime(
    output_dir,
    *,
    training_wall_time,
    successful_updates,
    episodes_seen,
    device,
    final_checkpoint_path,
    trainable_parameters,
):
    out = {
        "wall_time_sec": float(training_wall_time),
        "gpu_hours": float(training_wall_time) * torch.cuda.device_count() / 3600.0,
        "optimizer_steps": int(successful_updates),
        "effective_batch_size": int(BATCH_SIZE) * int(ACCUM_STEPS),
        "synthetic_episodes_processed": int(episodes_seen),
        "mean_step_time_sec": float(training_wall_time) / max(int(successful_updates), 1),
        "episodes_per_sec": int(episodes_seen) / max(float(training_wall_time), 1e-9),
        "peak_gpu_memory_gb": (
            float(torch.cuda.max_memory_allocated() / 1024**3)
            if torch.cuda.is_available()
            else 0.0
        ),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else str(device)
        ),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda or ""),
        "final_checkpoint_basename": os.path.basename(final_checkpoint_path),
        "checkpoint_step_count": int(successful_updates),
        "trainable_parameters": int(trainable_parameters),
    }
    _write_json(os.path.join(output_dir, PRETRAINING_RUNTIME_FILENAME), out)


def train(*, resolved_config, launch_command, ckpt_input_dir="", output_dir=None):
    validate_training_config(resolved_config)
    output_dir = str(output_dir or resolved_config["runtime"]["output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    configure_runtime()

    checkpoint_metadata = _checkpoint_metadata(resolved_config, launch_command, _utc_now())
    _write_json(os.path.join(output_dir, RUN_MANIFEST_FILENAME), checkpoint_metadata)
    model_config = checkpoint_metadata["model_config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_device = "cuda" if torch.cuda.is_available() else "cpu"
    resume_checkpoint = None
    if ckpt_input_dir:
        if not os.path.isdir(ckpt_input_dir):
            raise FileNotFoundError(f"Checkpoint input directory not found: {ckpt_input_dir}")
        resume_checkpoint, _ = load_latest_checkpoint(
            ckpt_input_dir,
            model_config,
            expected_resume_config_hash=checkpoint_metadata["resume_config_hash"],
        )
        model = CausalLongPFN.from_config(resume_checkpoint["model_config"]).to(device)
    else:
        model = CausalLongPFN.from_config(model_config).to(device)

    initial_fingerprint = tensor_fingerprint_from_state_dict(model.state_dict())
    checkpoint_metadata.update({
        "initial_model_fingerprint": initial_fingerprint["fingerprint"],
        "initial_model_n_params": initial_fingerprint["n_params"],
        "initial_model_abs_sum": initial_fingerprint["abs_sum"],
        "initial_model_sq_sum": initial_fingerprint["sq_sum"],
    })
    _write_json(os.path.join(output_dir, RUN_MANIFEST_FILENAME), checkpoint_metadata)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = build_adamw_optimizer(model, LR, WEIGHT_DECAY)
    scheduler = build_cosine_warmup_scheduler(
        optimizer,
        WARMUP_STEPS,
        SCHED_TOTAL_STEPS,
        min_lr_scale=MIN_LR_SCALE,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    successful_updates = attempted_updates = episodes_seen = 0
    prior_training_wall_time = 0.0
    if resume_checkpoint is not None:
        restore_training_state(model, optimizer, scheduler, scaler, resume_checkpoint)
        successful_updates = int(resume_checkpoint["successful_optimizer_updates"])
        attempted_updates = int(resume_checkpoint["attempted_updates"])
        episodes_seen = int(resume_checkpoint["episodes_seen"])
        prior_training_wall_time = float(resume_checkpoint["training_wall_time"])

    LOGGER.info(
        "Device: %s | GPUs: %s | Params: %s",
        device,
        torch.cuda.device_count(),
        f"{n_params:,}",
    )
    LOGGER.info("CausalLongPFN pretraining: four-layer PFN, GMM K=5")
    LOGGER.info("Output dir: %s | resume dir: %s", output_dir, ckpt_input_dir or "<none>")

    loader_generator = torch.Generator().manual_seed(int(SEED))
    loader_kwargs = dict(
        dataset=OnTheFlyEpisodeDataset(base_seed=int(SEED), start_index=episodes_seen),
        batch_size=BATCH_SIZE,
        num_workers=int(TRAIN_NUM_WORKERS),
        collate_fn=collate_episode_batch,
        pin_memory=torch.cuda.is_available(),
        generator=loader_generator,
        worker_init_fn=seed_loader_worker,
    )
    if int(TRAIN_NUM_WORKERS) > 0:
        loader_kwargs.update(prefetch_factor=4, persistent_workers=True)
    loader = DataLoader(**loader_kwargs)

    started = time.time()
    interval_started = started
    interval_episodes = episodes_seen
    running = {
        "loss_total": 0.0,
        "loss_nll": 0.0,
        "predictive_sigma_mean": 0.0,
        "predictive_std_mean": 0.0,
        "gradient_norm_pre_clip": 0.0,
        "loss_count": 0,
        "grad_count": 0,
    }
    skip_counts = {"nonfinite_loss": 0, "nonfinite_gradient": 0, "amp_overflow": 0}
    micro_step = 0
    training_history = _load_training_history(output_dir)
    validation_every = int(resolved_config["training"].get("PRIOR_VALIDATION_EVERY", 0))
    validation_size = int(resolved_config["training"].get("PRIOR_VALIDATION_SIZE", 0))
    validation_seed = int(resolved_config["training"].get("PRIOR_VALIDATION_SEED", 0))
    model.train()

    def wall_time():
        return prior_training_wall_time + time.time() - started

    def save(path, include_training_state):
        if not 0 <= successful_updates <= attempted_updates:
            raise RuntimeError("Checkpoint counters violate successful_updates <= attempted_updates")
        core = model.module if hasattr(model, "module") else model
        if core.get_config() != model_config:
            raise RuntimeError("Runtime model configuration diverged from checkpoint model_config")
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            successful_optimizer_updates=successful_updates,
            attempted_updates=attempted_updates,
            episodes_seen=episodes_seen,
            training_wall_time=wall_time(),
            path=path,
            checkpoint_metadata=checkpoint_metadata,
            include_training_state=include_training_state,
        )

    for batch in loader:
        if time.time() - started > SESSION_TIMEOUT or successful_updates >= MAX_STEPS:
            break
        episodes_seen += int(batch["target_y_norm"].shape[0])
        batch = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

        with torch.amp.autocast(amp_device, enabled=torch.cuda.is_available()):
            log_pi, mu, sigma = model(batch)
            loss, aux = gaussian_mixture_loss(log_pi, mu, sigma, batch["target_y_norm"])

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            attempted_updates += 1
            skip_counts["nonfinite_loss"] += 1
            micro_step = 0
            continue

        scaler.scale(loss / ACCUM_STEPS).backward()
        for name in (
            "loss_total",
            "loss_nll",
            "predictive_sigma_mean",
            "predictive_std_mean",
        ):
            running[name] += float(aux[name].item())
        running["loss_count"] += 1
        micro_step += 1
        if micro_step % ACCUM_STEPS:
            continue

        attempted_updates += 1
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(CLIP_NORM)))
        if not math.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            _update_scaler_after_skipped_unscaled_step(scaler)
            skip_counts["nonfinite_gradient"] += 1
            continue

        if scaler.is_enabled():
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            updated = float(scaler.get_scale()) >= scale_before
        else:
            optimizer.step()
            updated = True
        optimizer.zero_grad(set_to_none=True)
        if not updated:
            skip_counts["amp_overflow"] += 1
            continue

        scheduler.step()
        successful_updates += 1
        running["gradient_norm_pre_clip"] += grad_norm
        running["grad_count"] += 1

        prior_validation = None
        if validation_every and successful_updates % validation_every == 0:
            prior_validation = _validate_in_prior(
                model,
                device,
                amp_device,
                size=validation_size,
                seed=validation_seed,
            )
            LOGGER.info(
                "prior validation %7d | nll %.4f | rmse %.4f | n %d",
                successful_updates,
                prior_validation["prior_val_nll"],
                prior_validation["prior_val_rmse"],
                prior_validation["prior_val_size"],
            )

        if successful_updates % LOG_EVERY == 0:
            loss_count = max(running["loss_count"], 1)
            elapsed = max(time.time() - interval_started, 1e-9)
            row = {
                "successful_update": successful_updates,
                "attempted_update": attempted_updates,
                "episodes_seen": episodes_seen,
                "loss_total": running["loss_total"] / loss_count,
                "loss_nll": running["loss_nll"] / loss_count,
                "predictive_sigma_mean": running["predictive_sigma_mean"] / loss_count,
                "predictive_std_mean": running["predictive_std_mean"] / loss_count,
                "gradient_norm_pre_clip": running["gradient_norm_pre_clip"] / max(running["grad_count"], 1),
                "clip_threshold": float(CLIP_NORM),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "active_pfn_layers": int(N_PFN_LAYERS),
                "prior_val_nll": (
                    prior_validation["prior_val_nll"] if prior_validation is not None else None
                ),
                "prior_val_rmse": (
                    prior_validation["prior_val_rmse"] if prior_validation is not None else None
                ),
                "prior_val_size": (
                    prior_validation["prior_val_size"] if prior_validation is not None else None
                ),
                "skip_counts": json.dumps(skip_counts, sort_keys=True),
                "episodes_per_second": (episodes_seen - interval_episodes) / elapsed,
                "wall_time": wall_time(),
                "gpu_memory_current": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
                "gpu_memory_peak": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            }
            _append_training_history(output_dir, row, training_history)
            LOGGER.info(
                "update %7d | loss %.4f | lr %.2e | %.1f episodes/s",
                successful_updates,
                row["loss_total"],
                row["learning_rate"],
                row["episodes_per_second"],
            )
            running = {key: 0.0 for key in running}
            running["loss_count"] = running["grad_count"] = 0
            skip_counts = {key: 0 for key in skip_counts}
            interval_started, interval_episodes = time.time(), episodes_seen

        if successful_updates in set(INFERENCE_MILESTONE_UPDATES):
            save(
                os.path.join(output_dir, f"ckpt_inference_step_{successful_updates}.pt"),
                include_training_state=False,
            )
        if successful_updates % CHECKPOINT_EVERY == 0:
            save(
                os.path.join(output_dir, f"ckpt_train_step_{successful_updates}.pt"),
                include_training_state=True,
            )
            train_checkpoints = sorted(
                Path(output_dir).glob("ckpt_train_step_*.pt"),
                key=lambda path: int(path.stem.removeprefix("ckpt_train_step_")),
            )
            for checkpoint in train_checkpoints[:-3]:
                checkpoint.unlink()

    final_checkpoint_path = os.path.join(output_dir, "ckpt_final.pt")
    save(final_checkpoint_path, include_training_state=True)
    _write_pretraining_runtime(
        output_dir,
        training_wall_time=wall_time(),
        successful_updates=successful_updates,
        episodes_seen=episodes_seen,
        device=device,
        final_checkpoint_path=final_checkpoint_path,
        trainable_parameters=n_params,
    )
    manifest_path = os.path.join(output_dir, RUN_MANIFEST_FILENAME)
    manifest = deepcopy(checkpoint_metadata)
    manifest.update({
        "finished_at": _utc_now(),
        "successful_optimizer_updates": successful_updates,
        "attempted_updates": attempted_updates,
        "episodes_seen": episodes_seen,
        "training_wall_time": wall_time(),
        "final_checkpoint_basename": os.path.basename(final_checkpoint_path),
    })
    _write_json(manifest_path, manifest)
    LOGGER.info("Done. Successful optimizer updates: %s", successful_updates)
