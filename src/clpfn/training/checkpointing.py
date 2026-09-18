import hashlib
import json
import logging
import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch


LOGGER = logging.getLogger(__name__)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def assert_model_config_matches(checkpoint_config, expected_config):
    if _canonical_json(checkpoint_config) != _canonical_json(expected_config):
        raise RuntimeError(
            "Checkpoint model_config does not match the supported runtime model."
        )


def file_sha256_prefix(path, n_hex=16, chunk_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()[:n_hex]


def tensor_fingerprint_from_state_dict(state_dict, n_hex=16):
    digest = hashlib.sha256()
    n_tensors = 0
    n_params = 0
    abs_sum = 0.0
    sq_sum = 0.0

    for key in sorted(state_dict):
        value = state_dict[key]
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        float_tensor = tensor.float()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
        n_tensors += 1
        n_params += tensor.numel()
        abs_sum += float(float_tensor.abs().sum())
        sq_sum += float((float_tensor * float_tensor).sum())

    return {
        "fingerprint": digest.hexdigest()[:n_hex],
        "n_tensors": int(n_tensors),
        "n_params": int(n_params),
        "abs_sum": float(abs_sum),
        "sq_sum": float(sq_sum),
    }


def resolve_checkpoint_path(path):
    resolved = Path(path)
    if resolved.is_dir():
        config_path = resolved / "config.json"
        if config_path.is_file():
            with config_path.open("r", encoding="utf-8") as handle:
                weights_name = json.load(handle).get("weights_file")
            if weights_name and (resolved / weights_name).is_file():
                return resolved / weights_name
        candidates = sorted(resolved.glob("*.safetensors"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected exactly one .safetensors file in {resolved}; found {len(candidates)}."
            )
        return candidates[0]
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    return resolved


LEGACY_PRIOR_VARIANTS = {"tscm_paper": "tscm"}


def _reconcile_released_config(resolved_config):
    from clpfn.config.defaults import expected_prior

    identity = resolved_config["identity"]
    renamed = LEGACY_PRIOR_VARIANTS.get(identity["prior_variant"])
    if renamed:
        LOGGER.info("[ckpt] prior_variant %s -> %s", identity["prior_variant"], renamed)
        identity["prior_variant"] = renamed

    prior = resolved_config["prior"]
    supported = expected_prior(identity["model_variant"])
    legacy = sorted(key for key in prior if key not in supported)
    enabled = [key for key in legacy if prior[key]]
    if enabled:
        raise ValueError(
            f"Released prior enables mechanisms this version does not implement: {enabled}"
        )
    for key in legacy:
        prior.pop(key)
    if legacy:
        LOGGER.info("[ckpt] ignoring disabled legacy prior switches: %s", ", ".join(legacy))


def _release_checkpoint_payload(weights_path):
    try:
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ImportError as error:
        raise ImportError("Loading .safetensors weights requires: pip install safetensors") from error

    import yaml

    from clpfn.config.defaults import load_training_config
    from clpfn.models.causal_long_pfn import model_config_from_resolved_config

    with safe_open(str(weights_path), framework="pt") as handle:
        header = handle.metadata() or {}

    directory = weights_path.parent
    release_config = {}
    config_path = directory / "config.json"
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            release_config = json.load(handle)

    resolved_config = None
    for train_config_path in sorted(directory.glob("*_train_config.yaml")):
        with train_config_path.open("r", encoding="utf-8") as handle:
            released = yaml.safe_load(handle)
        resolved_config = {
            section: released[section]
            for section in ("identity", "model", "training", "prior", "stability")
        }
        break
    if resolved_config is None:
        resolved_config = load_training_config()
    else:
        _reconcile_released_config(resolved_config)
    resolved_config["runtime"] = {"ckpt_input_dir": "", "output_dir": ""}

    seed = int(release_config.get(
        "pretraining_seed", header.get("seed", resolved_config["training"]["SEED"])
    ))
    step = int(release_config.get("checkpoint_step", header.get("step", 0)))
    config_sha256 = str(
        release_config.get("release_metadata", {}).get(
            "config_sha256", header.get("config_sha256", "")
        )
    )
    resolved_config["training"]["SEED"] = seed
    identity = resolved_config["identity"]

    return {
        "model_state_dict": load_file(str(weights_path), device="cpu"),
        "model_config": release_config.get(
            "model", model_config_from_resolved_config(resolved_config)
        ),
        "resolved_config": resolved_config,
        "prior_config": resolved_config["prior"],
        "training_config": {
            "training": resolved_config["training"],
            "stability": resolved_config["stability"],
        },
        "step_count": step,
        "successful_optimizer_updates": step,
        "attempted_updates": step,
        "episodes_seen": int(
            release_config.get("training", {}).get("synthetic_episodes_processed", 0)
        ),
        "training_wall_time": 0.0,
        "experiment_id": identity["experiment_id"],
        "pretraining_seed": seed,
        "model_variant": identity["model_variant"],
        "prior_variant": identity["prior_variant"],
        "config_hash": config_sha256,
        "config_sha256": config_sha256,
        "resume_config_hash": "",
    }


def load_causal_long_pfn_checkpoint(path, device):
    from clpfn.config.defaults import validate_training_config
    from clpfn.models.causal_long_pfn import (
        CausalLongPFN,
        model_config_from_resolved_config,
    )

    path = resolve_checkpoint_path(path)
    if path.suffix == ".safetensors":
        checkpoint = _release_checkpoint_payload(path)
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("CausalLongPFN checkpoint must be a dictionary.")

    required = {
        "model_state_dict",
        "model_config",
        "resolved_config",
        "prior_config",
        "training_config",
        "step_count",
        "successful_optimizer_updates",
        "attempted_updates",
        "episodes_seen",
        "training_wall_time",
        "experiment_id",
        "pretraining_seed",
        "model_variant",
        "prior_variant",
        "config_hash",
        "config_sha256",
        "resume_config_hash",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"CausalLongPFN checkpoint is missing required fields: {missing}")

    resolved_config = checkpoint["resolved_config"]
    validate_training_config(resolved_config)
    expected_model_config = model_config_from_resolved_config(resolved_config)
    if checkpoint["model_config"] != expected_model_config:
        raise ValueError("Checkpoint model_config differs from its resolved configuration.")
    if checkpoint["prior_config"] != resolved_config["prior"]:
        raise ValueError("Checkpoint prior_config differs from its resolved configuration.")
    expected_training_config = {
        "training": resolved_config["training"],
        "stability": resolved_config["stability"],
    }
    if checkpoint["training_config"] != expected_training_config:
        raise ValueError("Checkpoint training_config differs from its resolved configuration.")

    identity = resolved_config["identity"]
    metadata_identity = {
        "experiment_id": checkpoint["experiment_id"],
        "pretraining_seed": checkpoint["pretraining_seed"],
        "model_variant": checkpoint["model_variant"],
        "prior_variant": checkpoint["prior_variant"],
    }
    expected_identity = {
        "experiment_id": identity["experiment_id"],
        "pretraining_seed": resolved_config["training"]["SEED"],
        "model_variant": identity["model_variant"],
        "prior_variant": identity["prior_variant"],
    }
    if metadata_identity != expected_identity:
        raise ValueError("Checkpoint identity metadata differs from its resolved configuration.")
    if int(checkpoint["step_count"]) != int(checkpoint["successful_optimizer_updates"]):
        raise ValueError("Checkpoint step_count differs from successful_optimizer_updates.")
    if checkpoint["config_hash"] != checkpoint["config_sha256"]:
        raise ValueError("Checkpoint configuration hashes disagree.")

    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint model_state_dict must be a dictionary.")
    checkpoint_fp = tensor_fingerprint_from_state_dict(state_dict)
    model = CausalLongPFN.from_config(checkpoint["model_config"]).to(device)
    model.load_state_dict(state_dict, strict=True)
    loaded_fp = tensor_fingerprint_from_state_dict(model.state_dict())
    if loaded_fp["fingerprint"] != checkpoint_fp["fingerprint"]:
        raise RuntimeError("Loaded model fingerprint differs from the checkpoint state dictionary.")
    model.eval()

    meta = {
        "checkpoint_path": str(path),
        "checkpoint_basename": os.path.basename(str(path)),
        "checkpoint_step_count": int(checkpoint["step_count"]),
        "checkpoint_file_size": int(os.path.getsize(path)),
        "checkpoint_file_sha256_prefix": file_sha256_prefix(path),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    for prefix, fingerprint in (
        ("checkpoint_tensor", checkpoint_fp),
        ("loaded_model", loaded_fp),
    ):
        meta[f"{prefix}_fingerprint"] = fingerprint["fingerprint"]
        meta[f"{prefix}_n_tensors"] = fingerprint["n_tensors"]
        meta[f"{prefix}_n_params"] = fingerprint["n_params"]
        meta[f"{prefix}_abs_sum"] = fingerprint["abs_sum"]
        meta[f"{prefix}_sq_sum"] = fingerprint["sq_sum"]
    for key in (
        "model_config",
        "step_count",
        "experiment_id",
        "pretraining_seed",
        "model_variant",
        "prior_variant",
        "config_hash",
        "config_sha256",
        "successful_optimizer_updates",
        "attempted_updates",
        "episodes_seen",
        "training_wall_time",
        "prior_config",
        "training_config",
    ):
        meta[key] = checkpoint[key]
    return model, meta


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    *,
    successful_optimizer_updates,
    attempted_updates,
    episodes_seen,
    training_wall_time,
    path,
    checkpoint_metadata,
    include_training_state,
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model_state_dict": unwrap_model(model).state_dict(),
        "successful_optimizer_updates": int(successful_optimizer_updates),
        "attempted_updates": int(attempted_updates),
        "episodes_seen": int(episodes_seen),
        "training_wall_time": float(training_wall_time),
        "step_count": int(successful_optimizer_updates),
        "data_index": int(episodes_seen),
        **deepcopy(checkpoint_metadata),
    }
    if include_training_state:
        payload.update({
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "np_rng_state": np.random.get_state(),
            "py_rng_state": random.getstate(),
        })

    torch.save(payload, path)
    kind = "training" if include_training_state else "inference"
    LOGGER.info("[ckpt] saved %s checkpoint: %s (successful update %s)", kind, path, successful_optimizer_updates)


def load_latest_checkpoint(ckpt_dir, expected_model_config, expected_resume_config_hash):
    candidates = [
        name for name in os.listdir(ckpt_dir)
        if name.startswith("ckpt_train_step_") and name.endswith(".pt")
    ]
    final_path = os.path.join(ckpt_dir, "ckpt_final.pt")
    if os.path.exists(final_path):
        candidates.append("ckpt_final.pt")
    if not candidates:
        raise FileNotFoundError(f"No resumable training checkpoint found in {ckpt_dir}")

    def checkpoint_sort_key(name):
        if name == "ckpt_final.pt":
            return float("inf")
        return int(name.removeprefix("ckpt_train_step_").removesuffix(".pt"))

    path = os.path.join(ckpt_dir, max(candidates, key=checkpoint_sort_key))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "model_config", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict",
        "torch_rng_state", "cuda_rng_state_all", "np_rng_state", "py_rng_state",
        "successful_optimizer_updates", "attempted_updates", "episodes_seen", "training_wall_time",
        "config_hash", "resume_config_hash", "resolved_config",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"Resumable checkpoint {path} is missing required fields: {missing}")
    assert_model_config_matches(checkpoint["model_config"], expected_model_config)
    if checkpoint["resume_config_hash"] != expected_resume_config_hash:
        raise RuntimeError(
            "Resume checkpoint was created from a different resolved configuration. "
            "Only an exact run may be resumed."
        )
    return checkpoint, path


def restore_training_state(model, optimizer, scheduler, scaler, checkpoint):
    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    np.random.set_state(checkpoint["np_rng_state"])
    random.setstate(checkpoint["py_rng_state"])
    LOGGER.info(
        "[ckpt] resumed from successful update %s | episodes seen %s",
        checkpoint["successful_optimizer_updates"], checkpoint["episodes_seen"],
    )
