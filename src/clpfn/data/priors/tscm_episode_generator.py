import numpy as np

from clpfn.config.defaults import (
    D_OUTCOME,
    D_STATIC_MAX,
    D_STATE_MIN,
    D_STATE_MAX,
    HIDDEN_SENTINEL,
    HORIZON_MAX,
    HORIZON_MIN,
    LATENT_UNIT_DIM,
    MAX_SEQ_LEN,
    N_ACTIONS,
    N_SUPPORT_ANCHORS,
    N_SUPPORT_MAX,
    N_SUPPORT_MIN,
    OBSERVATIONAL_QUERY_PROB,
    OBS_TIME_MAX,
    OBS_TIME_MIN,
    SUPPORT_FUTURE_COVARIATE_MASK_PROB,
    SUPPORT_LABEL_NOISE_PROB,
)


STATIC_COVARIATE_EPISODE_PROB = 0.30
STATIC_CONTINUOUS_NOISE = 0.35
STATIC_CATEGORY_TEMPERATURE = 1.0


class TSCMEpisodeGenerator:
    """
    Synthetic episode generator for the CausalLongPFN TSCM prior.

    Each episode samples a fresh temporal structural causal model and then draws
    support/query trajectories from it. The prior combines sparse nonlinear
    lagged state dynamics, latent unit heterogeneity, confounded behavior
    policies, autoregressive outcomes, and optional generic dynamical motifs.
    """

    ACTIVATION_NAMES = ["identity", "tanh", "sin", "cos", "abs", "square", "relu", "softplus"]

    @staticmethod
    def _activation(name: str, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -10.0, 10.0)

        if name == "identity":
            return x
        if name == "tanh":
            return np.tanh(x)
        if name == "sin":
            return np.sin(x)
        if name == "cos":
            return np.cos(x)
        if name == "abs":
            return np.abs(x)
        if name == "square":
            return np.clip(x ** 2, -5.0, 5.0)
        if name == "relu":
            return np.maximum(0.0, x)
        if name == "softplus":
            return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

        return x

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def _sample_instantaneous_dag(self, d_state: int, edge_prob: float) -> np.ndarray:
        graph = np.zeros((d_state, d_state), dtype=np.float32)

        for i in range(1, d_state):
            for j in range(i):
                if self.rng.random() < edge_prob:
                    graph[i, j] = 1.0

        return graph

    def _sample_lagged_adjacency(self, d_state: int, edge_prob: float) -> np.ndarray:
        return (self.rng.random((d_state, d_state)) < edge_prob).astype(np.float32)

    def _sample_mechanism(
        self,
        d_state: int,
        lag_order: int,
        instantaneous_graph: np.ndarray,
        lagged_graphs: list[np.ndarray],
        weight_scale: float,
    ) -> dict:
        instantaneous_weights = (
            self.rng.standard_normal((d_state, d_state)).astype(np.float32)
            * weight_scale
            * instantaneous_graph
        )

        lagged_weights = [
            self.rng.standard_normal((d_state, d_state)).astype(np.float32)
            * weight_scale
            * 0.7
            * lagged_graphs[k]
            for k in range(lag_order)
        ]

        activation_names = [
            self.ACTIVATION_NAMES[int(self.rng.integers(0, len(self.ACTIVATION_NAMES)))]
            for _ in range(d_state)
        ]

        return {
            "instantaneous_weights": instantaneous_weights,
            "lagged_weights": lagged_weights,
            "activation_names": activation_names,
        }

    def sample_tscm(self) -> dict:
        rng = self.rng
        d_state = int(rng.integers(D_STATE_MIN, D_STATE_MAX + 1))

        edge_prob = float(rng.beta(2.0, 2.0) * 0.5 + 0.1)
        lag_decay = float(rng.uniform(0.4, 0.8))
        lag_order = int(rng.integers(1, 3))

        instantaneous_graph = self._sample_instantaneous_dag(d_state, edge_prob)
        lagged_graphs = [
            self._sample_lagged_adjacency(d_state, edge_prob * (lag_decay ** (k + 1)))
            for k in range(lag_order)
        ]

        weight_scale = float(rng.uniform(0.3, 1.0))
        primary_mechanism = self._sample_mechanism(
            d_state=d_state,
            lag_order=lag_order,
            instantaneous_graph=instantaneous_graph,
            lagged_graphs=lagged_graphs,
            weight_scale=weight_scale,
        )

        state_ar_coeff = ((rng.random(d_state) < 0.5) * rng.uniform(0.5, 1.0, size=d_state)).astype(np.float32)

        noise_base = float(rng.uniform(0.00, 0.05)) if rng.random() < 0.60 else float(rng.uniform(0.05, 0.20))
        state_noise_family = [int(rng.integers(0, 3)) for _ in range(d_state)]
        state_noise_scale = []

        for _ in range(d_state):
            if rng.random() < 0.50:
                state_noise_scale.append(0.0)
            else:
                state_noise_scale.append(float(rng.uniform(0.5, 1.25)) * noise_base)

        policy_state_weights_bit0 = rng.standard_normal(d_state).astype(np.float32) * float(rng.uniform(0.5, 2.0))
        policy_state_weights_bit1 = rng.standard_normal(d_state).astype(np.float32) * float(rng.uniform(0.5, 2.0))
        policy_bias_bit0 = float(rng.standard_normal())
        policy_bias_bit1 = float(rng.standard_normal())

        policy_memory_decay_bit0 = float(rng.uniform(0.5, 0.95))
        policy_memory_decay_bit1 = float(rng.uniform(0.5, 0.95))
        policy_memory_weight_bit0 = float(rng.uniform(-2.0, -0.5))
        policy_memory_weight_bit1 = float(rng.uniform(-2.0, -0.5))

        strength_draw = rng.random()
        if strength_draw < 0.08:
            policy_strength = 0
        elif strength_draw < 0.28:
            policy_strength = 1
        else:
            policy_strength = int(rng.integers(2, 6))

        policy_state_weights_bit0 *= policy_strength
        policy_state_weights_bit1 *= policy_strength

        policy_loading_scale = float(rng.uniform(0.1, 1.0))
        initial_state_loading_scale = float(rng.uniform(0.1, 1.0))

        latent_policy_loading_bit0 = (
            rng.standard_normal((d_state, LATENT_UNIT_DIM)).astype(np.float32) * policy_loading_scale
        )
        latent_policy_loading_bit1 = (
            rng.standard_normal((d_state, LATENT_UNIT_DIM)).astype(np.float32) * policy_loading_scale
        )
        latent_initial_state_loading = (
            rng.standard_normal((d_state, LATENT_UNIT_DIM)).astype(np.float32) * initial_state_loading_scale
        )

        regime_switch_config = None
        if rng.random() < 0.12:
            secondary_mechanism = self._sample_mechanism(
                d_state=d_state,
                lag_order=lag_order,
                instantaneous_graph=instantaneous_graph,
                lagged_graphs=lagged_graphs,
                weight_scale=weight_scale,
            )
            switch_time = int(rng.integers(OBS_TIME_MIN // 3, 2 * OBS_TIME_MAX // 3 + 1))
            regime_switch_config = {
                "switch_time": switch_time,
                "mechanism": secondary_mechanism,
            }

        use_action_memory_channel = bool(rng.random() < 0.25)
        use_saturating_channel = bool(rng.random() < 0.25)
        use_homeostatic_channel = bool(rng.random() < 0.25)
        use_feedback_channel = bool(rng.random() < 0.25)
        use_readout_channel = bool(rng.random() < 0.20)

        remaining_coordinates = list(rng.permutation(d_state))

        def take_coordinates(count: int) -> np.ndarray:
            nonlocal remaining_coordinates
            count = min(int(count), len(remaining_coordinates))

            if count <= 0:
                return np.array([], dtype=np.int32)

            out = np.array(sorted(remaining_coordinates[:count]), dtype=np.int32)
            remaining_coordinates = remaining_coordinates[count:]
            return out

        action_memory_indices = take_coordinates(1 if use_action_memory_channel else 0)

        n_saturating = 1 + int(rng.random() < 0.30) if use_saturating_channel else 0
        saturating_indices = take_coordinates(n_saturating)
        homeostatic_indices = take_coordinates(1 if use_homeostatic_channel else 0)
        feedback_indices = take_coordinates(1 if use_feedback_channel else 0)
        readout_indices = take_coordinates(1 if use_readout_channel else 0)

        action_memory_decay = np.zeros(len(action_memory_indices), dtype=np.float32)
        action_memory_action_weights = np.zeros((len(action_memory_indices), 2), dtype=np.float32)
        action_memory_history_weights = np.zeros((len(action_memory_indices), 2), dtype=np.float32)

        for j in range(len(action_memory_indices)):
            action_memory_decay[j] = float(rng.uniform(0.72, 0.97))
            action_memory_action_weights[j] = rng.normal(0.0, 0.7, size=2).astype(np.float32)
            action_memory_history_weights[j] = rng.normal(0.0, 0.4, size=2).astype(np.float32)

        saturation_baseline = np.zeros(len(saturating_indices), dtype=np.float32)
        saturation_rate = np.zeros(len(saturating_indices), dtype=np.float32)
        saturation_gain = np.zeros(len(saturating_indices), dtype=np.float32)
        saturation_half_level = np.zeros(len(saturating_indices), dtype=np.float32)
        saturation_feedback_gain = np.zeros(len(saturating_indices), dtype=np.float32)
        saturation_signal_weights = np.zeros((len(saturating_indices), 2), dtype=np.float32)

        for j in range(len(saturating_indices)):
            saturation_baseline[j] = float(rng.uniform(0.5, 1.5))
            saturation_rate[j] = float(rng.uniform(0.02, 0.15))
            saturation_gain[j] = float(rng.uniform(0.25, 0.95))
            saturation_half_level[j] = float(rng.uniform(0.3, 2.0))
            saturation_feedback_gain[j] = float(rng.uniform(-0.20, 0.30))
            saturation_signal_weights[j] = rng.normal(0.0, 0.45, size=2).astype(np.float32)

        homeostatic_baseline = np.zeros(len(homeostatic_indices), dtype=np.float32)
        homeostatic_reversion_rate = np.zeros(len(homeostatic_indices), dtype=np.float32)
        homeostatic_action_weights = np.zeros((len(homeostatic_indices), 2), dtype=np.float32)

        for j in range(len(homeostatic_indices)):
            homeostatic_baseline[j] = float(rng.uniform(-0.5, 0.5))
            homeostatic_reversion_rate[j] = float(rng.uniform(0.03, 0.15))
            homeostatic_action_weights[j] = rng.normal(0.0, 0.35, size=2).astype(np.float32)

        feedback_decay = np.zeros(len(feedback_indices), dtype=np.float32)
        feedback_gain = np.zeros(len(feedback_indices), dtype=np.float32)
        feedback_setpoint = np.zeros(len(feedback_indices), dtype=np.float32)
        feedback_action_weights = np.zeros((len(feedback_indices), 2), dtype=np.float32)
        feedback_source_index = np.full(len(feedback_indices), -1, dtype=np.int32)

        feedback_candidates = []
        feedback_candidates.extend(saturating_indices.tolist())
        feedback_candidates.extend(homeostatic_indices.tolist())
        feedback_candidates.extend(action_memory_indices.tolist())

        blocked_for_feedback = set(
            action_memory_indices.tolist()
            + saturating_indices.tolist()
            + homeostatic_indices.tolist()
            + feedback_indices.tolist()
            + readout_indices.tolist()
        )
        feedback_candidates.extend([i for i in range(d_state) if i not in blocked_for_feedback])

        for j in range(len(feedback_indices)):
            feedback_decay[j] = float(rng.uniform(0.65, 0.95))
            feedback_gain[j] = float(rng.uniform(0.10, 0.90))
            feedback_setpoint[j] = float(rng.uniform(-0.5, 0.5))
            feedback_action_weights[j] = rng.normal(0.0, 0.25, size=2).astype(np.float32)

            if len(feedback_candidates) > 0:
                feedback_source_index[j] = int(feedback_candidates[j % len(feedback_candidates)])

        readout_smoothing_decay = np.zeros(len(readout_indices), dtype=np.float32)
        readout_source_index = np.full(len(readout_indices), -1, dtype=np.int32)

        readout_sources = []
        readout_sources.extend(saturating_indices.tolist())
        readout_sources.extend(feedback_indices.tolist())
        readout_sources.extend(homeostatic_indices.tolist())
        readout_sources.extend(action_memory_indices.tolist())
        readout_sources.extend([i for i in range(d_state) if i not in set(readout_indices.tolist())])

        for j in range(len(readout_indices)):
            readout_smoothing_decay[j] = float(rng.uniform(0.70, 0.97))

            if len(readout_sources) > 0:
                readout_source_index[j] = int(readout_sources[j % len(readout_sources)])

        use_state_outcome = bool(rng.random() < 0.40)

        outcome_coordinate_weights = np.ones(d_state, dtype=np.float32)

        if len(saturating_indices) > 0:
            outcome_coordinate_weights[saturating_indices] += 0.8
        if len(readout_indices) > 0:
            outcome_coordinate_weights[readout_indices] += 0.6
        if len(feedback_indices) > 0:
            outcome_coordinate_weights[feedback_indices] += 0.3
        if len(homeostatic_indices) > 0:
            outcome_coordinate_weights[homeostatic_indices] += 0.3
        if len(action_memory_indices) > 0:
            outcome_coordinate_weights[action_memory_indices] += 0.2

        outcome_coordinate_weights /= outcome_coordinate_weights.sum()
        outcome_state_index = int(rng.choice(d_state, p=outcome_coordinate_weights))

        outcome_readout_weights = (
            rng.standard_normal(d_state).astype(np.float32)
            if rng.random() < 0.5
            else np.zeros(d_state, dtype=np.float32)
        )
        outcome_readout_bias = float(rng.standard_normal())

        has_any_motif = bool(
            len(action_memory_indices)
            or len(saturating_indices)
            or len(homeostatic_indices)
            or len(feedback_indices)
            or len(readout_indices)
        )

        outcome_ar_coeff = float(rng.uniform(0.35, 0.90))
        outcome_state_gain = float(rng.uniform(0.35, 1.20))
        outcome_action_weights = rng.normal(0.0, 0.25, size=2).astype(np.float32)

        if has_any_motif:
            outcome_action_weights += rng.normal(0.0, 0.25, size=2).astype(np.float32)

        outcome_action_memory_weights = rng.normal(0.0, 0.10, size=2).astype(np.float32)
        outcome_noise_scale = (
            float(rng.uniform(0.00, 0.06))
            if rng.random() < 0.7
            else float(rng.uniform(0.06, 0.16))
        )
        outcome_trend = float(rng.normal(0.0, 0.015))

        motif_coordinate_mask = np.zeros(d_state, dtype=np.bool_)

        for indices in (
            action_memory_indices,
            saturating_indices,
            homeostatic_indices,
            feedback_indices,
            readout_indices,
        ):
            if len(indices) > 0:
                motif_coordinate_mask[indices] = True

        return {
            "d_state": d_state,
            "lag_order": lag_order,
            "primary_mechanism": primary_mechanism,
            "state_ar_coeff": state_ar_coeff,
            "state_noise_family": state_noise_family,
            "state_noise_scale": state_noise_scale,
            "outcome_readout_weights": outcome_readout_weights,
            "outcome_readout_bias": outcome_readout_bias,
            "use_state_outcome": use_state_outcome,
            "outcome_state_index": outcome_state_index,
            "policy_state_weights_bit0": policy_state_weights_bit0,
            "policy_state_weights_bit1": policy_state_weights_bit1,
            "policy_bias_bit0": policy_bias_bit0,
            "policy_bias_bit1": policy_bias_bit1,
            "policy_memory_decay_bit0": policy_memory_decay_bit0,
            "policy_memory_decay_bit1": policy_memory_decay_bit1,
            "policy_memory_weight_bit0": policy_memory_weight_bit0,
            "policy_memory_weight_bit1": policy_memory_weight_bit1,
            "latent_policy_loading_bit0": latent_policy_loading_bit0,
            "latent_policy_loading_bit1": latent_policy_loading_bit1,
            "latent_initial_state_loading": latent_initial_state_loading,
            "regime_switch_config": regime_switch_config,
            "action_memory_indices": action_memory_indices,
            "saturating_indices": saturating_indices,
            "homeostatic_indices": homeostatic_indices,
            "feedback_indices": feedback_indices,
            "readout_indices": readout_indices,
            "motif_coordinate_mask": motif_coordinate_mask,
            "action_memory_decay": action_memory_decay,
            "action_memory_action_weights": action_memory_action_weights,
            "action_memory_history_weights": action_memory_history_weights,
            "saturation_baseline": saturation_baseline,
            "saturation_rate": saturation_rate,
            "saturation_gain": saturation_gain,
            "saturation_half_level": saturation_half_level,
            "saturation_feedback_gain": saturation_feedback_gain,
            "saturation_signal_weights": saturation_signal_weights,
            "homeostatic_baseline": homeostatic_baseline,
            "homeostatic_reversion_rate": homeostatic_reversion_rate,
            "homeostatic_action_weights": homeostatic_action_weights,
            "feedback_decay": feedback_decay,
            "feedback_gain": feedback_gain,
            "feedback_setpoint": feedback_setpoint,
            "feedback_action_weights": feedback_action_weights,
            "feedback_source_index": feedback_source_index,
            "readout_smoothing_decay": readout_smoothing_decay,
            "readout_source_index": readout_source_index,
            "outcome_ar_coeff": outcome_ar_coeff,
            "outcome_state_gain": outcome_state_gain,
            "outcome_action_weights": outcome_action_weights,
            "outcome_action_memory_weights": outcome_action_memory_weights,
            "outcome_noise_scale": outcome_noise_scale,
            "outcome_trend": outcome_trend,
        }

    def _state_readout(self, tscm: dict, state_path: np.ndarray, mix: dict | None = None) -> np.ndarray:
        if tscm["use_state_outcome"]:
            return state_path[..., tscm["outcome_state_index"]]

        primary = state_path @ tscm["outcome_readout_weights"] + tscm["outcome_readout_bias"]

        if mix is None:
            return primary

        secondary = state_path @ mix["secondary_outcome_readout_weights"] + mix["secondary_outcome_readout_bias"]

        return mix["primary_mix_weight"] * primary + (1.0 - mix["primary_mix_weight"]) * secondary

    def _build_outcome_path(
        self,
        tscm: dict,
        state_path: np.ndarray,
        actions: np.ndarray,
        mix: dict | None = None,
    ) -> np.ndarray:
        batch_size, n_times_plus_one, _ = state_path.shape
        n_transitions = n_times_plus_one - 1

        base_readout = self._state_readout(tscm, state_path, mix=mix).astype(np.float32)

        outcome_path = np.zeros((batch_size, n_times_plus_one), dtype=np.float32)
        outcome_path[:, 0] = base_readout[:, 0] + self.rng.normal(
            0.0,
            tscm["outcome_noise_scale"],
            size=batch_size,
        ).astype(np.float32)

        action_memory_bit0 = np.zeros(batch_size, dtype=np.float32)
        action_memory_bit1 = np.zeros(batch_size, dtype=np.float32)

        for time_idx in range(n_transitions):
            action = actions[:, time_idx].astype(np.int32)
            action_bits = np.stack([action & 1, (action >> 1) & 1], axis=1).astype(np.float32)

            action_memory_bit0 = tscm["policy_memory_decay_bit0"] * action_memory_bit0 + action_bits[:, 0]
            action_memory_bit1 = tscm["policy_memory_decay_bit1"] * action_memory_bit1 + action_bits[:, 1]
            action_memory = np.stack([action_memory_bit0, action_memory_bit1], axis=1).astype(np.float32)

            action_effect = (
                action_bits @ tscm["outcome_action_weights"]
                + action_memory @ tscm["outcome_action_memory_weights"]
            )

            innovation = (
                tscm["outcome_state_gain"] * base_readout[:, time_idx + 1]
                + action_effect
                + tscm["outcome_trend"] * (time_idx + 1)
            )

            noise = self.rng.normal(0.0, tscm["outcome_noise_scale"], size=batch_size).astype(np.float32)

            outcome_path[:, time_idx + 1] = (
                tscm["outcome_ar_coeff"] * outcome_path[:, time_idx]
                + (1.0 - tscm["outcome_ar_coeff"]) * innovation
                + noise
            )

        np.clip(outcome_path, -1e4, 1e4, out=outcome_path)
        return outcome_path.astype(np.float32)

    def _continue_outcome_from_history(
        self,
        tscm: dict,
        state_path: np.ndarray,
        actions: np.ndarray,
        observed_outcome_prefix: np.ndarray,
        mix: dict | None = None,
        add_future_noise: bool = False,
    ) -> np.ndarray:
        batch_size, n_times_plus_one, _ = state_path.shape
        n_transitions = n_times_plus_one - 1

        prefix_len = int(observed_outcome_prefix.shape[1])
        if prefix_len < 1:
            raise ValueError("observed_outcome_prefix must contain at least Y_0")

        base_readout = self._state_readout(tscm, state_path, mix=mix).astype(np.float32)

        outcome_path = np.zeros((batch_size, n_times_plus_one), dtype=np.float32)
        prefix_clip = min(prefix_len, n_times_plus_one)
        outcome_path[:, :prefix_clip] = observed_outcome_prefix[:, :prefix_clip]
        start_transition = max(0, prefix_clip - 1)

        action_memory_bit0 = np.zeros(batch_size, dtype=np.float32)
        action_memory_bit1 = np.zeros(batch_size, dtype=np.float32)

        for time_idx in range(n_transitions):
            action = actions[:, time_idx].astype(np.int32)
            action_bits = np.stack([action & 1, (action >> 1) & 1], axis=1).astype(np.float32)

            action_memory_bit0 = tscm["policy_memory_decay_bit0"] * action_memory_bit0 + action_bits[:, 0]
            action_memory_bit1 = tscm["policy_memory_decay_bit1"] * action_memory_bit1 + action_bits[:, 1]
            action_memory = np.stack([action_memory_bit0, action_memory_bit1], axis=1).astype(np.float32)

            if time_idx < start_transition:
                continue

            action_effect = (
                action_bits @ tscm["outcome_action_weights"]
                + action_memory @ tscm["outcome_action_memory_weights"]
            )

            innovation = (
                tscm["outcome_state_gain"] * base_readout[:, time_idx + 1]
                + action_effect
                + tscm["outcome_trend"] * (time_idx + 1)
            )

            noise = (
                self.rng.normal(0.0, tscm["outcome_noise_scale"], size=batch_size).astype(np.float32)
                if add_future_noise
                else 0.0
            )

            outcome_path[:, time_idx + 1] = (
                tscm["outcome_ar_coeff"] * outcome_path[:, time_idx]
                + (1.0 - tscm["outcome_ar_coeff"]) * innovation
                + noise
            )

        np.clip(outcome_path, -1e4, 1e4, out=outcome_path)
        return outcome_path.astype(np.float32)

    def _sample_state_noise(self, tscm: dict, batch_size: int, n_transitions: int) -> np.ndarray:
        d_state = tscm["d_state"]
        noise = np.zeros((n_transitions, d_state, batch_size), dtype=np.float32)

        for state_idx in range(d_state):
            scale = tscm["state_noise_scale"][state_idx]

            if scale < 1e-6:
                continue

            family = tscm["state_noise_family"][state_idx]

            if family == 0:
                noise[:, state_idx, :] = self.rng.standard_normal((n_transitions, batch_size)) * scale
            elif family == 1:
                noise[:, state_idx, :] = self.rng.uniform(
                    -scale * 1.73,
                    scale * 1.73,
                    (n_transitions, batch_size),
                )
            else:
                noise[:, state_idx, :] = self.rng.laplace(0.0, scale, (n_transitions, batch_size))

        return noise

    def _sample_behavior_action(
        self,
        tscm: dict,
        state: np.ndarray,
        memory_bit0: np.ndarray,
        memory_bit1: np.ndarray,
        policy_weights_bit0: np.ndarray,
        policy_weights_bit1: np.ndarray,
    ) -> np.ndarray:
        logit_bit0 = (
            np.sum(policy_weights_bit0 * state, axis=1)
            + tscm["policy_bias_bit0"]
            + tscm["policy_memory_weight_bit0"] * memory_bit0
        )
        logit_bit1 = (
            np.sum(policy_weights_bit1 * state, axis=1)
            + tscm["policy_bias_bit1"]
            + tscm["policy_memory_weight_bit1"] * memory_bit1
        )

        prob_bit0 = 1.0 / (1.0 + np.exp(-np.clip(logit_bit0, -10.0, 10.0)))
        prob_bit1 = 1.0 / (1.0 + np.exp(-np.clip(logit_bit1, -10.0, 10.0)))

        bit0 = (self.rng.random(state.shape[0]) < prob_bit0).astype(np.int32)
        bit1 = (self.rng.random(state.shape[0]) < prob_bit1).astype(np.int32)

        return 2 * bit1 + bit0

    @staticmethod
    def _update_policy_action_memory(
        tscm: dict,
        memory_bit0: np.ndarray,
        memory_bit1: np.ndarray,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            tscm["policy_memory_decay_bit0"] * memory_bit0 + (action & 1),
            tscm["policy_memory_decay_bit1"] * memory_bit1 + ((action >> 1) & 1),
        )

    def transition_state(
        self,
        tscm: dict,
        state: np.ndarray,
        lag_buffer: list[np.ndarray | None],
        action: np.ndarray,
        action_memory: np.ndarray,
        noise_t: np.ndarray,
        time_idx: int,
    ) -> np.ndarray:
        batch_size = state.shape[0]
        d_state = tscm["d_state"]
        lag_order = tscm["lag_order"]

        regime_switch = tscm["regime_switch_config"]
        mechanism = (
            regime_switch["mechanism"]
            if (regime_switch is not None and time_idx >= regime_switch["switch_time"])
            else tscm["primary_mechanism"]
        )

        previous_state = (
            lag_buffer[0]
            if (len(lag_buffer) > 0 and lag_buffer[0] is not None)
            else np.zeros((batch_size, d_state), dtype=np.float32)
        )

        lagged_contribution = np.zeros((batch_size, d_state), dtype=np.float32)

        for lag_idx in range(lag_order):
            if lag_idx < len(lag_buffer) and lag_buffer[lag_idx] is not None:
                lagged_contribution += lag_buffer[lag_idx] @ mechanism["lagged_weights"][lag_idx].T

        action_bits = np.stack([action & 1, (action >> 1) & 1], axis=1).astype(np.float32)
        next_state = np.zeros((batch_size, d_state), dtype=np.float32)

        if len(tscm["action_memory_indices"]) > 0:
            for motif_idx, state_idx in enumerate(tscm["action_memory_indices"]):
                previous_value = previous_state[:, state_idx]
                drive = (
                    action_bits @ tscm["action_memory_action_weights"][motif_idx]
                    + action_memory @ tscm["action_memory_history_weights"][motif_idx]
                )
                next_state[:, state_idx] = np.clip(
                    tscm["action_memory_decay"][motif_idx] * previous_value + drive + noise_t[state_idx, :],
                    -8.0,
                    8.0,
                )

        if len(tscm["homeostatic_indices"]) > 0:
            for motif_idx, state_idx in enumerate(tscm["homeostatic_indices"]):
                previous_value = previous_state[:, state_idx]
                drive = action_bits @ tscm["homeostatic_action_weights"][motif_idx]
                reversion = tscm["homeostatic_reversion_rate"][motif_idx] * (
                    tscm["homeostatic_baseline"][motif_idx] - previous_value
                )
                next_state[:, state_idx] = np.clip(
                    previous_value + drive + reversion + noise_t[state_idx, :],
                    -8.0,
                    8.0,
                )

        if len(tscm["feedback_indices"]) > 0:
            for motif_idx, state_idx in enumerate(tscm["feedback_indices"]):
                previous_value = previous_state[:, state_idx]
                source_idx = int(tscm["feedback_source_index"][motif_idx])
                source_value = previous_state[:, source_idx] if source_idx >= 0 else 0.0
                error = tscm["feedback_setpoint"][motif_idx] - source_value
                drive = (
                    tscm["feedback_gain"][motif_idx] * error
                    + action_bits @ tscm["feedback_action_weights"][motif_idx]
                )
                next_state[:, state_idx] = np.clip(
                    tscm["feedback_decay"][motif_idx] * previous_value + drive + noise_t[state_idx, :],
                    -8.0,
                    8.0,
                )

        if len(tscm["saturating_indices"]) > 0:
            feedback_signal = (
                previous_state[:, tscm["feedback_indices"][0]]
                if len(tscm["feedback_indices"]) > 0
                else 0.0
            )

            for motif_idx, state_idx in enumerate(tscm["saturating_indices"]):
                previous_value = np.maximum(previous_state[:, state_idx], 0.0)

                memory_signal = np.abs(action_memory @ tscm["saturation_signal_weights"][motif_idx])

                if len(tscm["action_memory_indices"]) > 0:
                    latent_memory_signal = np.abs(next_state[:, tscm["action_memory_indices"][0]])
                    signal = 0.65 * latent_memory_signal + 0.35 * memory_signal
                else:
                    signal = memory_signal

                saturation_mod = (
                    tscm["saturation_gain"][motif_idx]
                    * signal
                    / (tscm["saturation_half_level"][motif_idx] + signal + 1e-6)
                )
                base_flow = tscm["saturation_rate"][motif_idx] * tscm["saturation_baseline"][motif_idx] * (
                    1.0 + tscm["saturation_feedback_gain"][motif_idx] * np.tanh(feedback_signal)
                )

                updated = (
                    previous_value
                    + base_flow * (1.0 - saturation_mod)
                    - tscm["saturation_rate"][motif_idx] * previous_value
                    + noise_t[state_idx, :]
                )

                next_state[:, state_idx] = np.clip(updated, 0.0, 6.0)

        for state_idx in range(d_state):
            if tscm["motif_coordinate_mask"][state_idx]:
                continue

            instantaneous = next_state @ mechanism["instantaneous_weights"][state_idx]
            transformed = self._activation(
                mechanism["activation_names"][state_idx],
                instantaneous + lagged_contribution[:, state_idx],
            )
            previous_value = previous_state[:, state_idx]

            next_state[:, state_idx] = (
                tscm["state_ar_coeff"][state_idx] * previous_value
                + transformed
                + noise_t[state_idx, :]
            )

        if len(tscm["readout_indices"]) > 0:
            for motif_idx, state_idx in enumerate(tscm["readout_indices"]):
                previous_value = previous_state[:, state_idx]
                source_idx = int(tscm["readout_source_index"][motif_idx])
                source_value = next_state[:, source_idx] if source_idx >= 0 else 0.0

                next_state[:, state_idx] = np.clip(
                    tscm["readout_smoothing_decay"][motif_idx] * previous_value
                    + (1.0 - tscm["readout_smoothing_decay"][motif_idx]) * source_value
                    + noise_t[state_idx, :],
                    -8.0,
                    8.0,
                )

        np.clip(next_state, -10.0, 10.0, out=next_state)
        return next_state

    def _sample_support_size(self) -> int:
        return int(self.rng.integers(N_SUPPORT_MIN, N_SUPPORT_MAX + 1))

    def _initialize_motif_coordinates(self, tscm: dict, state: np.ndarray) -> np.ndarray:
        if len(tscm["action_memory_indices"]) > 0:
            state[:, tscm["action_memory_indices"]] = self.rng.normal(
                0.0,
                0.05,
                size=(state.shape[0], len(tscm["action_memory_indices"])),
            ).astype(np.float32)

        if len(tscm["saturating_indices"]) > 0:
            for motif_idx, state_idx in enumerate(tscm["saturating_indices"]):
                state[:, state_idx] = self.rng.normal(
                    tscm["saturation_baseline"][motif_idx],
                    0.05,
                    size=state.shape[0],
                ).astype(np.float32)

        if len(tscm["homeostatic_indices"]) > 0:
            for motif_idx, state_idx in enumerate(tscm["homeostatic_indices"]):
                state[:, state_idx] = self.rng.normal(
                    tscm["homeostatic_baseline"][motif_idx],
                    0.05,
                    size=state.shape[0],
                ).astype(np.float32)

        if len(tscm["feedback_indices"]) > 0:
            state[:, tscm["feedback_indices"]] = self.rng.normal(
                0.0,
                0.05,
                size=(state.shape[0], len(tscm["feedback_indices"])),
            ).astype(np.float32)

        if len(tscm["readout_indices"]) > 0:
            state[:, tscm["readout_indices"]] = self.rng.normal(
                0.0,
                0.05,
                size=(state.shape[0], len(tscm["readout_indices"])),
            ).astype(np.float32)

        return state

    def _sample_support_anchor_times(self, obs_time: int, one_step_target_time: int) -> np.ndarray:
        """
        Support anchor at time s represents:
            history through s - 1, action A[s - 1], label Y[s].
        """
        one_step_target_time = int(np.clip(one_step_target_time, 1, MAX_SEQ_LEN))
        earliest_anchor = int(np.clip(obs_time + 1, 1, one_step_target_time))

        anchor_times = np.full(N_SUPPORT_ANCHORS, one_step_target_time, dtype=np.int32)

        if N_SUPPORT_ANCHORS >= 2:
            anchor_times[1] = earliest_anchor
        if N_SUPPORT_ANCHORS >= 3:
            anchor_times[2] = max(earliest_anchor, (earliest_anchor + one_step_target_time) // 2)

        for anchor_idx in range(3, N_SUPPORT_ANCHORS):
            anchor_times[anchor_idx] = int(self.rng.integers(earliest_anchor, one_step_target_time + 1))

        return np.clip(anchor_times, 1, MAX_SEQ_LEN).astype(np.int32)

    def _sample_observed_static_from_latent(
        self,
        latent: np.ndarray,
        enabled: bool,
    ) -> np.ndarray:
        """
        Generate observed static covariates from latent unit factors.

        Layout for D_STATIC_MAX=5:
            0: continuous latent proxy
            1: binary latent proxy
            2: complementary binary latent proxy
            3: categorical latent proxy indicator 1
            4: categorical latent proxy indicator 2
        """
        n_units = int(latent.shape[0])
        static = np.zeros((n_units, D_STATIC_MAX), dtype=np.float32)

        if not enabled:
            return static

        continuous_proxy = 0.8 * latent[:, 0] + STATIC_CONTINUOUS_NOISE * self.rng.standard_normal(n_units)
        continuous_scale = float(np.sqrt(0.8 ** 2 + STATIC_CONTINUOUS_NOISE ** 2))
        continuous_proxy = continuous_proxy / max(continuous_scale, 0.1)
        static[:, 0] = np.clip(continuous_proxy, -3.0, 3.0).astype(np.float32)

        if D_STATIC_MAX >= 3:
            binary_logit = 0.9 * latent[:, 1] + 0.25 * self.rng.standard_normal(n_units)
            binary_proxy = (binary_logit > 0.0).astype(np.float32)
            static[:, 1] = binary_proxy
            static[:, 2] = 1.0 - binary_proxy

        if D_STATIC_MAX >= 5:
            logits = np.stack(
                [
                    0.7 * latent[:, 2],
                    -0.4 * latent[:, 0] + 0.4 * latent[:, 1],
                    -0.3 * latent[:, 2] + 0.2 * self.rng.standard_normal(n_units),
                ],
                axis=1,
            ).astype(np.float32)

            logits = logits / STATIC_CATEGORY_TEMPERATURE
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs = probs / probs.sum(axis=1, keepdims=True)

            draws = np.array(
                [self.rng.choice(3, p=probs[i]) for i in range(n_units)],
                dtype=np.int32,
            )

            static[:, 3] = (draws == 0).astype(np.float32)
            static[:, 4] = (draws == 1).astype(np.float32)

        return static.astype(np.float32)

    def sample_episode(self) -> dict:
        while True:
            tscm = self.sample_tscm()
            d_state = tscm["d_state"]
            lag_order = tscm["lag_order"]
            n_support = self._sample_support_size()

            obs_time = int(self.rng.integers(OBS_TIME_MIN, OBS_TIME_MAX + 1))
            horizon = int(self.rng.integers(HORIZON_MIN, HORIZON_MAX + 1))
            final_target_time = obs_time + horizon

            if final_target_time > MAX_SEQ_LEN:
                continue

            is_observational_query = bool(self.rng.random() < OBSERVATIONAL_QUERY_PROB)
            mask_future_support_covariates = bool(self.rng.random() < SUPPORT_FUTURE_COVARIATE_MASK_PROB)
            use_observed_static = bool(self.rng.random() < STATIC_COVARIATE_EPISODE_PROB)

            planned_actions = np.zeros(MAX_SEQ_LEN, dtype=np.int32)

            if not is_observational_query:
                current_plan_time = obs_time

                while current_plan_time < final_target_time:
                    lo = max(1, horizon // 4)
                    hi = max(2, horizon // 2)
                    block_len = int(self.rng.integers(lo, hi))

                    planned_actions[
                        current_plan_time:min(current_plan_time + block_len, final_target_time)
                    ] = int(self.rng.integers(0, N_ACTIONS))

                    current_plan_time += block_len

            outcome_mix = None

            if (not tscm["use_state_outcome"]) and (self.rng.random() < 0.25):
                outcome_mix = {
                    "secondary_outcome_readout_weights": self.rng.standard_normal(d_state).astype(np.float32),
                    "secondary_outcome_readout_bias": float(self.rng.standard_normal()),
                    "primary_mix_weight": float(self.rng.uniform(0.35, 0.65)),
                }

            support_latent = self.rng.standard_normal((n_support, LATENT_UNIT_DIM)).astype(np.float32)
            support_static = self._sample_observed_static_from_latent(
                support_latent,
                enabled=use_observed_static,
            )
            support_policy_weights_bit0 = (
                tscm["policy_state_weights_bit0"] + support_latent @ tscm["latent_policy_loading_bit0"].T
            )
            support_policy_weights_bit1 = (
                tscm["policy_state_weights_bit1"] + support_latent @ tscm["latent_policy_loading_bit1"].T
            )

            support_state = (
                self.rng.standard_normal((n_support, d_state)).astype(np.float32) * 0.1
                + support_latent @ tscm["latent_initial_state_loading"].T
            )
            support_state = self._initialize_motif_coordinates(tscm, support_state)

            support_memory_bit0 = np.zeros(n_support, dtype=np.float32)
            support_memory_bit1 = np.zeros(n_support, dtype=np.float32)

            support_state_path = np.zeros((n_support, final_target_time + 1, d_state), dtype=np.float32)
            support_actions = np.zeros((n_support, MAX_SEQ_LEN), dtype=np.int32)
            support_noise = self._sample_state_noise(tscm, n_support, final_target_time)
            support_lag_buffer: list[np.ndarray | None] = [None] * lag_order

            for time_idx in range(final_target_time):
                action = self._sample_behavior_action(
                    tscm,
                    support_state,
                    support_memory_bit0,
                    support_memory_bit1,
                    support_policy_weights_bit0,
                    support_policy_weights_bit1,
                )

                support_state_path[:, time_idx, :] = support_state
                support_actions[:, time_idx] = action

                next_memory_bit0, next_memory_bit1 = self._update_policy_action_memory(
                    tscm,
                    support_memory_bit0,
                    support_memory_bit1,
                    action,
                )
                action_memory = np.stack([next_memory_bit0, next_memory_bit1], axis=1).astype(np.float32)

                next_state = self.transition_state(
                    tscm,
                    support_state,
                    support_lag_buffer,
                    action,
                    action_memory,
                    support_noise[time_idx],
                    time_idx,
                )

                support_lag_buffer = [support_state] + support_lag_buffer[:-1]
                support_state = next_state
                support_memory_bit0, support_memory_bit1 = next_memory_bit0, next_memory_bit1

            support_state_path[:, final_target_time, :] = support_state

            if not np.isfinite(support_state_path).all():
                continue

            support_outcome_path = self._build_outcome_path(
                tscm,
                support_state_path,
                support_actions[:, :final_target_time],
                mix=outcome_mix,
            )

            if not np.isfinite(support_outcome_path).all():
                continue

            query_latent = self.rng.standard_normal(LATENT_UNIT_DIM).astype(np.float32)
            query_static = self._sample_observed_static_from_latent(
                query_latent.reshape(1, -1),
                enabled=use_observed_static,
            )[0]

            query_policy_weights_bit0 = (
                tscm["policy_state_weights_bit0"] + tscm["latent_policy_loading_bit0"] @ query_latent
            ).reshape(1, d_state)
            query_policy_weights_bit1 = (
                tscm["policy_state_weights_bit1"] + tscm["latent_policy_loading_bit1"] @ query_latent
            ).reshape(1, d_state)

            query_state = (
                self.rng.standard_normal((1, d_state)).astype(np.float32) * 0.1
                + (tscm["latent_initial_state_loading"] @ query_latent).reshape(1, d_state)
            )
            query_state = self._initialize_motif_coordinates(tscm, query_state)

            query_memory_bit0 = np.zeros(1, dtype=np.float32)
            query_memory_bit1 = np.zeros(1, dtype=np.float32)

            query_factual_state_path = np.zeros((MAX_SEQ_LEN + 1, d_state), dtype=np.float32)
            query_actions = np.zeros(MAX_SEQ_LEN, dtype=np.int32)
            query_noise = self._sample_state_noise(tscm, 1, MAX_SEQ_LEN)
            query_lag_buffer: list[np.ndarray | None] = [None] * lag_order

            state_at_obs = None
            memory_bit0_at_obs = None
            memory_bit1_at_obs = None
            lag_buffer_at_obs = None

            for time_idx in range(MAX_SEQ_LEN):
                if time_idx == obs_time:
                    state_at_obs = query_state.copy()
                    memory_bit0_at_obs = query_memory_bit0.copy()
                    memory_bit1_at_obs = query_memory_bit1.copy()
                    lag_buffer_at_obs = [
                        value.copy() if value is not None else None
                        for value in query_lag_buffer
                    ]

                if time_idx < obs_time or is_observational_query:
                    action = self._sample_behavior_action(
                        tscm,
                        query_state,
                        query_memory_bit0,
                        query_memory_bit1,
                        query_policy_weights_bit0,
                        query_policy_weights_bit1,
                    )
                else:
                    action = np.array([planned_actions[time_idx]], dtype=np.int32)

                query_factual_state_path[time_idx] = query_state[0].copy()
                query_actions[time_idx] = action[0]

                next_memory_bit0, next_memory_bit1 = self._update_policy_action_memory(
                    tscm,
                    query_memory_bit0,
                    query_memory_bit1,
                    action,
                )
                action_memory = np.stack([next_memory_bit0, next_memory_bit1], axis=1).astype(np.float32)

                next_state = self.transition_state(
                    tscm,
                    query_state,
                    query_lag_buffer,
                    action,
                    action_memory,
                    query_noise[time_idx],
                    time_idx,
                )

                query_lag_buffer = [query_state] + query_lag_buffer[:-1]
                query_state = next_state
                query_memory_bit0, query_memory_bit1 = next_memory_bit0, next_memory_bit1

            query_factual_state_path[MAX_SEQ_LEN] = query_state[0].copy()

            if state_at_obs is None or not np.isfinite(query_factual_state_path).all():
                continue

            if is_observational_query:
                query_intervened_state_path = query_factual_state_path[:final_target_time + 1].copy()
                intervention_actions = query_actions[:final_target_time].reshape(1, -1).copy()
            else:
                query_intervened_state_path = query_factual_state_path[:final_target_time + 1].copy()

                intervention_actions_1d = query_actions[:final_target_time].copy()
                intervention_actions_1d[obs_time:final_target_time] = planned_actions[obs_time:final_target_time]

                replay_state = state_at_obs.copy()
                replay_memory_bit0 = memory_bit0_at_obs.copy()
                replay_memory_bit1 = memory_bit1_at_obs.copy()
                replay_lag_buffer = [
                    value.copy() if value is not None else None
                    for value in lag_buffer_at_obs
                ]

                mean_future_noise = np.zeros((HORIZON_MAX + 1, d_state, 1), dtype=np.float32)

                for replay_idx, time_idx in enumerate(range(obs_time, final_target_time)):
                    action_t = np.array([planned_actions[time_idx]], dtype=np.int32)

                    next_memory_bit0, next_memory_bit1 = self._update_policy_action_memory(
                        tscm,
                        replay_memory_bit0,
                        replay_memory_bit1,
                        action_t,
                    )
                    action_memory = np.stack([next_memory_bit0, next_memory_bit1], axis=1).astype(np.float32)

                    next_state = self.transition_state(
                        tscm,
                        replay_state,
                        replay_lag_buffer,
                        action_t,
                        action_memory,
                        mean_future_noise[replay_idx],
                        time_idx,
                    )

                    query_intervened_state_path[time_idx + 1] = next_state[0]

                    replay_lag_buffer = [replay_state] + replay_lag_buffer[:-1]
                    replay_state = next_state
                    replay_memory_bit0, replay_memory_bit1 = next_memory_bit0, next_memory_bit1

                intervention_actions = intervention_actions_1d.reshape(1, -1)

            query_factual_outcome_path = self._build_outcome_path(
                tscm,
                query_factual_state_path.reshape(1, MAX_SEQ_LEN + 1, d_state),
                query_actions.reshape(1, -1),
                mix=outcome_mix,
            )[0]

            if not np.isfinite(query_factual_outcome_path).all():
                continue

            if is_observational_query:
                query_intervened_outcome_path = query_factual_outcome_path[
                    :final_target_time + 1
                ].reshape(1, -1).copy()
            else:
                observed_outcome_prefix = query_factual_outcome_path[:obs_time + 1].reshape(1, -1).astype(np.float32)

                query_intervened_outcome_path = self._continue_outcome_from_history(
                    tscm,
                    query_intervened_state_path.reshape(1, final_target_time + 1, d_state),
                    intervention_actions,
                    observed_outcome_prefix=observed_outcome_prefix,
                    mix=outcome_mix,
                    add_future_noise=False,
                )

            oracle_outcome = query_intervened_outcome_path[0]

            if not np.isfinite(oracle_outcome).all() or abs(float(oracle_outcome[final_target_time])) > 1e4:
                continue

            current_time = int(self.rng.integers(obs_time, final_target_time))
            one_step_target_time = current_time + 1

            support_state_mean = support_state_path[:, :obs_time + 1, :].mean(axis=(0, 1))
            support_state_std = np.maximum(support_state_path[:, :obs_time + 1, :].std(axis=(0, 1)), 0.1)

            support_rollout_outcomes = support_outcome_path[:, 1:final_target_time + 1].reshape(-1)
            support_outcome_mean = float(support_rollout_outcomes.mean())
            support_outcome_std_raw = float(support_rollout_outcomes.std())

            if support_outcome_std_raw < 0.05:
                continue

            support_outcome_std = max(support_outcome_std_raw, 0.1)

            support_state_norm = (
                (support_state_path[:, :final_target_time, :] - support_state_mean)
                / support_state_std
            ).astype(np.float32)
            np.clip(support_state_norm, -3.0, 3.0, out=support_state_norm)

            support_y_history_norm = np.clip(
                (support_outcome_path[:, :final_target_time] - support_outcome_mean) / support_outcome_std,
                -10.0,
                10.0,
            ).astype(np.float32)

            support_x = np.full(
                (n_support, MAX_SEQ_LEN, d_state + D_OUTCOME),
                HIDDEN_SENTINEL,
                dtype=np.float32,
            )
            support_x[:, :final_target_time, :d_state] = support_state_norm
            support_x[:, :final_target_time, d_state] = support_y_history_norm

            if mask_future_support_covariates and (obs_time + 1) < final_target_time:
                support_x[:, obs_time + 1:final_target_time, :d_state] = HIDDEN_SENTINEL

            support_x[:, final_target_time:] = HIDDEN_SENTINEL

            query_teacher_outcome_path = query_factual_outcome_path[:MAX_SEQ_LEN].copy()

            if (obs_time + 1) < min(final_target_time + 1, MAX_SEQ_LEN):
                teacher_upto = min(final_target_time + 1, MAX_SEQ_LEN)
                query_teacher_outcome_path[obs_time + 1:teacher_upto] = oracle_outcome[
                    obs_time + 1:teacher_upto
                ]

            query_state_norm = (
                (query_factual_state_path[:MAX_SEQ_LEN, :] - support_state_mean)
                / support_state_std
            ).astype(np.float32)
            np.clip(query_state_norm, -3.0, 3.0, out=query_state_norm)

            query_y_history_norm = np.clip(
                (query_teacher_outcome_path - support_outcome_mean) / support_outcome_std,
                -10.0,
                10.0,
            ).astype(np.float32)

            query_x = np.concatenate(
                [query_state_norm, query_y_history_norm[..., None]],
                axis=-1,
            ).astype(np.float32)

            if (obs_time + 1) < MAX_SEQ_LEN:
                query_x[obs_time + 1:, :d_state] = HIDDEN_SENTINEL

            if one_step_target_time < MAX_SEQ_LEN:
                query_x[one_step_target_time:, :] = HIDDEN_SENTINEL

            support_anchor_times_1d = self._sample_support_anchor_times(
                obs_time,
                one_step_target_time,
            )
            support_anchor_time = np.tile(
                support_anchor_times_1d[None, :],
                (n_support, 1),
            ).astype(np.int32)

            support_anchor_y = np.zeros((n_support, N_SUPPORT_ANCHORS), dtype=np.float32)

            for anchor_idx in range(N_SUPPORT_ANCHORS):
                anchor_time = int(support_anchor_times_1d[anchor_idx])
                raw_anchor_y = support_outcome_path[:, anchor_time].copy()

                if anchor_idx == 0 and self.rng.random() < SUPPORT_LABEL_NOISE_PROB:
                    noise_scale = float(self.rng.choice([0.0, 0.05, 0.10], p=[0.45, 0.30, 0.25]))

                    if noise_scale > 0:
                        raw_anchor_y = raw_anchor_y + self.rng.normal(
                            0.0,
                            noise_scale * support_outcome_std,
                            size=n_support,
                        )

                support_anchor_y[:, anchor_idx] = np.clip(
                    (raw_anchor_y - support_outcome_mean) / support_outcome_std,
                    -10.0,
                    10.0,
                ).astype(np.float32)

            target_raw = float(oracle_outcome[one_step_target_time])
            target_y_norm = np.float32(
                np.clip(
                    (target_raw - support_outcome_mean) / support_outcome_std,
                    -10.0,
                    10.0,
                )
            )

            if not np.isfinite(support_x).all() or not np.isfinite(query_x).all():
                continue

            if not np.isfinite(support_anchor_y).all() or not np.isfinite(target_y_norm):
                continue

            return {
                "d_input": d_state + D_OUTCOME,
                "n_support": n_support,
                "current_time": current_time,

                "support_x": support_x.astype(np.float32),
                "support_actions": support_actions,
                "support_anchor_y": support_anchor_y,
                "support_anchor_time": support_anchor_time,

                "query_x": query_x.astype(np.float32),
                "query_actions": query_actions,

                "target_y_norm": target_y_norm,

                "support_static": support_static.astype(np.float32),
                "query_static": query_static.astype(np.float32),
            }
