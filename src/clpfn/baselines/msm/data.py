from __future__ import annotations

import numpy as np

from clpfn.baselines.msm.config import N_ACTION_BITS
from clpfn.evaluation.core import benchmark as common


def action4_to_bits(actions):
    actions = np.asarray(actions, dtype=np.int64)
    return np.stack([(actions & 1), ((actions >> 1) & 1)], axis=-1).astype(np.float32)


def action_bits_window(actions, start, end):
    vals = np.asarray(actions[max(0, int(start)):max(0, int(end))], dtype=np.int64)
    if vals.size == 0:
        return np.zeros(N_ACTION_BITS, dtype=np.float32)
    return action4_to_bits(vals).sum(axis=0).astype(np.float32)


def safe_lag_slice(arr, t, lag_features):
    start = int(t) - int(lag_features)
    if start < 0:
        raise ValueError("t must be >= lag_features")
    return arr[start:int(t) + 1]


def history_feature(states_n, y_n, actions, static, t, lag_features):
    prev_treat_sum = action_bits_window(actions, 0, int(t))
    lagged_states = safe_lag_slice(states_n, t, lag_features).reshape(-1).astype(np.float32)
    lagged_y = safe_lag_slice(y_n, t, lag_features).reshape(-1).astype(np.float32)
    feat = np.concatenate([prev_treat_sum, lagged_states, lagged_y, static.astype(np.float32)], axis=0)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def msm_feature(states_n, y_n, actions, static, t, tau, lag_features):
    hist = history_feature(states_n, y_n, actions, static, t=t, lag_features=lag_features)
    future_treat_sum = action_bits_window(actions, int(t), int(t) + int(tau))
    return np.nan_to_num(
        np.concatenate([hist, future_treat_sum], axis=0),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def build_propensity_training_data(bundle, hparams, context_indices):
    X_num, X_den, Y_bits, pairs = [], [], [], []
    lag_features = int(hparams["lag_features"])
    C, Y, A, S, L = bundle["covariates"], bundle["y_norm_clip"], bundle["actions"], bundle["static"], bundle["sequence_lengths"]
    T = min(C.shape[1], Y.shape[1], A.shape[1], common.MAX_SEQ_LEN)
    preferred_start = max(lag_features, common.MIN_T_OBS)

    def collect_with_start(start_t):
        for i in np.asarray(context_indices, dtype=np.int64):
            max_t = min(int(L[i]) - 1, T - 1, common.MAX_INPUT_INDEX)
            if max_t < start_t:
                continue
            for t in range(start_t, max_t + 1):
                if t >= Y.shape[1] or not np.isfinite(Y[i, t]):
                    continue
                try:
                    den_feat = history_feature(C[i], Y[i], A[i], S[i], t=t, lag_features=lag_features)
                except Exception:
                    continue
                num_feat = action_bits_window(A[i], 0, t)
                target_bits = action4_to_bits(np.asarray([A[i, t]], dtype=np.int64))[0]
                X_num.append(num_feat)
                X_den.append(den_feat)
                Y_bits.append(target_bits)
                pairs.append((int(i), int(t)))

    collect_with_start(preferred_start)
    if len(Y_bits) == 0 and preferred_start > lag_features:
        collect_with_start(lag_features)
    if len(Y_bits) == 0:
        return None
    return np.asarray(X_num, dtype=np.float32), np.asarray(X_den, dtype=np.float32), np.asarray(Y_bits, dtype=np.float32), pairs


def max_anchor_for_tau(bundle, i, tau):
    C, Y, A, L = bundle["covariates"], bundle["y_norm_clip"], bundle["actions"], bundle["sequence_lengths"]
    tau = int(tau)
    return min(int(L[i]) - tau - 1, Y.shape[1] - tau - 1, A.shape[1] - tau, C.shape[1] - 1, common.MAX_INPUT_INDEX)


def build_regression_data_for_tau(bundle, sw_matrix, tau, hparams, context_indices):
    X, y, weights = [], [], []
    lag_features = int(hparams["lag_features"])
    C, Y, Yraw, A, S = bundle["covariates"], bundle["y_norm_clip"], bundle["y_raw"], bundle["actions"], bundle["static"]
    preferred_start = max(lag_features, common.MIN_T_OBS)

    def collect_with_start(start_t):
        for i in np.asarray(context_indices, dtype=np.int64):
            max_anchor = max_anchor_for_tau(bundle, int(i), int(tau))
            if max_anchor < start_t:
                continue
            for t in range(start_t, max_anchor + 1):
                target_t = t + int(tau)
                if target_t >= Y.shape[1] or not np.isfinite(Yraw[i, target_t]):
                    continue
                try:
                    feat = msm_feature(C[i], Y[i], A[i], S[i], t=t, tau=int(tau), lag_features=lag_features)
                except Exception:
                    continue
                target = float(Y[i, target_t])
                weight = float(np.prod(sw_matrix[i, t:t + int(tau)]))
                X.append(feat)
                y.append(target)
                weights.append(weight if np.isfinite(weight) else 1.0)

    collect_with_start(preferred_start)
    if len(y) == 0 and preferred_start > lag_features:
        collect_with_start(lag_features)
    if len(y) == 0:
        return None, None, None
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.size > 10:
        q = hparams.get("weight_clip_quantiles", [0.01, 0.99])
        lo, hi = np.nanquantile(weights, [float(q[0]), float(q[1])])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            weights = np.clip(weights, lo, hi)
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    weights = np.maximum(weights, 1e-6)
    return X, y, weights
