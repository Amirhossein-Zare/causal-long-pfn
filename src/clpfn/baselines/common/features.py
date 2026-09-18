from __future__ import annotations

from collections.abc import Mapping

import numpy as np


_DOMAIN_STATIC_WIDTHS = {
    "cancer": 1,
    "hiv": 5,
    "warfarin": 4,
    "mimic": 5,
}


def static_feature_width(bundle: Mapping) -> int:
    static = np.asarray(bundle["static"])
    if static.ndim != 2:
        raise ValueError(f"Expected a 2-D static feature array, got shape={static.shape}.")
    domain = str(bundle["domain"]).strip().lower()
    if domain not in _DOMAIN_STATIC_WIDTHS:
        raise ValueError(f"Unknown benchmark domain {domain!r}.")
    return min(int(static.shape[-1]), int(_DOMAIN_STATIC_WIDTHS[domain]))


def has_dynamic_vitals(bundle: Mapping) -> bool:
    domain = str(bundle["domain"]).strip().lower()
    if domain not in _DOMAIN_STATIC_WIDTHS:
        raise ValueError(f"Unknown benchmark domain {domain!r}.")
    covariates = np.asarray(bundle["covariates"])
    return domain != "cancer" and covariates.ndim >= 3 and int(covariates.shape[-1]) > 0


def prepare_baseline_bundle(bundle: Mapping) -> dict:
    out = dict(bundle)

    static = np.asarray(bundle["static"])
    d_static = static_feature_width(bundle)
    out["static"] = static[:, :d_static].astype(np.float32, copy=False)

    if not has_dynamic_vitals(bundle):
        covariates = np.asarray(bundle["covariates"])
        out["covariates"] = np.zeros((*covariates.shape[:-1], 0), dtype=np.float32)
    return out


def treatment_mode_for_bundle(bundle: Mapping, *, cancer_mode: str = "multiclass") -> str:
    domain = str(bundle["domain"]).strip().lower()
    if domain == "mimic":
        return "multilabel"
    if domain in {"hiv", "warfarin"}:
        return "multiclass"
    if domain == "cancer":
        return str(cancer_mode)
    raise ValueError(f"Unknown benchmark domain {domain!r}.")


def treatment_dim(mode: str) -> int:
    if str(mode) == "multilabel":
        return 2
    if str(mode) == "multiclass":
        return 4
    raise ValueError(f"Unknown treatment mode: {mode!r}")


def encode_actions(actions, mode: str) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.int64)
    if str(mode) == "multilabel":
        return np.stack([(actions & 1), ((actions >> 1) & 1)], axis=-1).astype(np.float32)
    if str(mode) == "multiclass":
        if np.any((actions < 0) | (actions >= 4)):
            raise ValueError("Multiclass actions must be in [0, 3].")
        eye = np.eye(4, dtype=np.float32)
        return eye[actions]
    raise ValueError(f"Unknown treatment mode: {mode!r}")
