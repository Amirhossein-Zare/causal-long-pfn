from __future__ import annotations

import numpy as np

from clpfn.baselines.common.features import encode_actions, prepare_baseline_bundle, treatment_mode_for_bundle
from clpfn.evaluation.core import benchmark as common

def action_window(actions, start, end, treatment_mode):
    vals = np.asarray(actions[max(0, int(start)):max(0, int(end))], dtype=np.int64)
    dim = 2 if treatment_mode == "multilabel" else common.N_ACTIONS
    if vals.size == 0:
        return np.zeros(dim, dtype=np.float32)
    return encode_actions(vals, treatment_mode).sum(axis=0).astype(np.float32)


def action_bits_window(actions, start, end):
    return action_window(actions, start, end, "multilabel")


def safe_lag_slice(arr, t, lag_features):
    start = int(t) - int(lag_features)
    if start < 0:
        raise ValueError("t must be >= lag_features")
    return arr[start:int(t) + 1]


def history_feature(states_n, y_n, actions, static, t, lag_features, treatment_mode="multilabel"):
    prev_treat_sum = action_window(actions, 0, int(t), treatment_mode)
    lagged_states = safe_lag_slice(states_n, t, lag_features).reshape(-1).astype(np.float32)
    lagged_y = safe_lag_slice(y_n, t, lag_features).reshape(-1).astype(np.float32)
    feat = np.concatenate([prev_treat_sum, lagged_states, lagged_y, static.astype(np.float32)], axis=0)
    if not np.isfinite(feat).all():
        raise ValueError("MSM history features contain non-finite values.")
    return feat.astype(np.float32)


def msm_feature(states_n, y_n, actions, static, t, tau, lag_features, treatment_mode="multilabel"):
    hist = history_feature(states_n, y_n, actions, static, t=t, lag_features=lag_features, treatment_mode=treatment_mode)
    future_treat_sequence = action_window(actions, int(t), int(t) + int(tau), treatment_mode)
    features = np.concatenate([hist, future_treat_sequence], axis=0)
    if not np.isfinite(features).all():
        raise ValueError("MSM features contain non-finite values.")
    return features.astype(np.float32)


def build_propensity_training_data(bundle, hparams, context_indices):
    bundle = prepare_baseline_bundle(bundle)
    X_num, X_den, Y_treat, pairs = [], [], [], []
    lag_features = int(hparams["lag_features"])
    treatment_mode = treatment_mode_for_bundle(bundle, cancer_mode="multilabel")
    C, Y, A, S, L = bundle["covariates"], bundle["y_norm_clip"], bundle["actions"], bundle["static"], bundle["sequence_lengths"]
    T = min(C.shape[1], Y.shape[1], A.shape[1], common.MAX_SEQ_LEN)
    preferred_start = int(lag_features)

    def collect_with_start(start_t):
        for i in np.asarray(context_indices, dtype=np.int64):
            max_t = min(int(L[i]) - 1, T - 1, common.MAX_INPUT_INDEX)
            if max_t < start_t:
                continue
            for t in range(start_t, max_t + 1):
                if t >= Y.shape[1] or not np.isfinite(Y[i, t]):
                    continue
                den_feat = history_feature(
                    C[i],
                    Y[i],
                    A[i],
                    S[i],
                    t=t,
                    lag_features=lag_features,
                    treatment_mode=treatment_mode,
                )
                num_feat = action_window(A[i], 0, t, treatment_mode)
                target = encode_actions(np.asarray([A[i, t]], dtype=np.int64), treatment_mode)[0] if treatment_mode == "multilabel" else int(A[i, t])
                X_num.append(num_feat)
                X_den.append(den_feat)
                Y_treat.append(target)
                pairs.append((int(i), int(t)))

    collect_with_start(preferred_start)
    if len(Y_treat) == 0:
        return None
    return np.asarray(X_num, dtype=np.float32), np.asarray(X_den, dtype=np.float32), np.asarray(Y_treat), pairs


def max_anchor_for_tau(bundle, i, tau):
    bundle = prepare_baseline_bundle(bundle)
    C, Y, A, L = bundle["covariates"], bundle["y_norm_clip"], bundle["actions"], bundle["sequence_lengths"]
    tau = int(tau)
    return min(int(L[i]) - tau, Y.shape[1] - tau - 1, A.shape[1] - tau, C.shape[1] - 1, common.MAX_INPUT_INDEX)


def build_regression_data_for_tau(bundle, sw_matrix, tau, hparams, context_indices):
    bundle = prepare_baseline_bundle(bundle)
    X, y, weights = [], [], []
    lag_features = int(hparams["lag_features"])
    treatment_mode = treatment_mode_for_bundle(bundle, cancer_mode="multilabel")
    C, Y, Yraw, A, S = bundle["covariates"], bundle["y_norm_clip"], bundle["y_raw"], bundle["actions"], bundle["static"]
    preferred_start = int(lag_features)

    def collect_with_start(start_t):
        for i in np.asarray(context_indices, dtype=np.int64):
            max_anchor = max_anchor_for_tau(bundle, int(i), int(tau))
            if max_anchor < start_t:
                continue
            for t in range(start_t, max_anchor + 1):
                target_t = t + int(tau)
                if target_t >= Y.shape[1] or not np.isfinite(Yraw[i, target_t]):
                    continue
                feat = msm_feature(
                    C[i],
                    Y[i],
                    A[i],
                    S[i],
                    t=t,
                    tau=int(tau),
                    lag_features=lag_features,
                    treatment_mode=treatment_mode,
                )
                target = float(Y[i, target_t])
                weight = float(np.prod(sw_matrix[i, t:t + int(tau)]))
                if not np.isfinite(weight):
                    raise RuntimeError("MSM produced a non-finite regression weight.")
                X.append(feat)
                y.append(target)
                weights.append(weight)

    collect_with_start(preferred_start)
    if len(y) == 0:
        return None, None, None
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float64)
    q = hparams["weight_clip_quantiles"]
    lo, hi = np.nanquantile(weights, [float(q[0]), float(q[1])])
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise RuntimeError("MSM weight quantiles are non-finite.")
    if hi > lo:
        weights = np.clip(weights, lo, hi)
    weights = np.maximum(weights, 1e-6)
    return X, y, weights
