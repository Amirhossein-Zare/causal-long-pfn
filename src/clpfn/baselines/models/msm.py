from __future__ import annotations

import logging
import time
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from clpfn.baselines.models.time_varying_model import TimeVaryingCausalModel, cfg_get
from clpfn.baselines.msm.config import N_ACTION_BITS
from clpfn.baselines.msm.data import (
    build_propensity_training_data,
    build_regression_data_for_tau,
    msm_feature,
)
from clpfn.evaluation.core import benchmark as common

logger = logging.getLogger(__name__)


def make_msm_args(hparams):
    return SimpleNamespace(
        model=SimpleNamespace(
            dim_treatments=N_ACTION_BITS,
            dim_vitals=0,
            dim_static_features=common.D_STATIC_MAX,
            dim_outcomes=1,
            lag_features=int(hparams["lag_features"]),
        ),
        dataset=SimpleNamespace(
            projection_horizon=common.PROJECTION_HORIZON,
            treatment_mode="multilabel",
        ),
        exp=SimpleNamespace(max_epochs=int(hparams.get("max_logistic_iter", 500))),
    )


class BinaryMultiOutputProbModel:
    """One logistic propensity model per binary treatment bit."""

    def __init__(self, n_bits=N_ACTION_BITS, logistic_C=1e6, max_iter=500):
        self.n_bits = int(n_bits)
        self.logistic_C = float(logistic_C)
        self.max_iter = int(max_iter)
        self.models = []

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int64)
        Y = np.clip(Y, 0, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Y.shape[1] != self.n_bits:
            raise ValueError(f"Expected {self.n_bits} treatment bits, got {Y.shape[1]}.")
        if X.shape[0] < 10:
            raise ValueError(f"Need at least 10 propensity rows, got {X.shape[0]}.")

        self.models = []
        for bit_idx in range(self.n_bits):
            y_bit = Y[:, bit_idx]
            if np.unique(y_bit).size < 2:
                raise ValueError(f"Treatment bit {bit_idx} has one class in propensity data.")
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=self.logistic_C,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=self.max_iter,
                ),
            )
            model.fit(X, y_bit)
            self.models.append(model)
        return self

    def predict_proba_one(self, X):
        if len(self.models) != self.n_bits:
            raise RuntimeError("Propensity model must be fitted before prediction.")
        X = np.asarray(X, dtype=np.float32)
        out = np.zeros((X.shape[0], self.n_bits), dtype=np.float64)
        for bit_idx, model in enumerate(self.models):
            lr = model.named_steps["logisticregression"]
            if 1 not in lr.classes_:
                raise RuntimeError(f"Treatment bit {bit_idx} model has no positive class.")
            proba = model.predict_proba(X)
            out[:, bit_idx] = proba[:, list(lr.classes_).index(1)]
        return np.clip(out, 1e-6, 1.0 - 1e-6)


def make_regressor(hparams):
    regressor = str(hparams.get("regressor", "ridge")).lower()
    if regressor == "linear":
        return LinearRegression()
    if regressor == "ridge":
        return Ridge(alpha=float(hparams.get("ridge_alpha", 1.0)))
    raise ValueError(f"Unknown MSM regressor: {regressor}")


class MSM(TimeVaryingCausalModel):
    """
    Marginal Structural Model base.

    Fitting and prediction operate on CLPFN raw bundle arrays.
    """

    model_type = None
    possible_model_types = {"msm_regressor", "propensity_treatment", "propensity_history"}
    tuning_criterion = None

    def __init__(
        self,
        args,
        dataset_collection=None,
        autoregressive=None,
        has_vitals=None,
        **kwargs,
    ):
        super().__init__(args, dataset_collection, autoregressive, has_vitals)
        self.lag_features = int(cfg_get(args.model, "lag_features", 0))


class MSMPropensityTreatment(MSM):
    """Numerator propensity model p(A_t | A_<t)."""

    model_type = "propensity_treatment"
    tuning_criterion = "bce"

    def __init__(self, args, dataset_collection=None, autoregressive=None, has_vitals=None, **kwargs):
        super().__init__(args, dataset_collection, autoregressive, has_vitals)
        self.input_size = self.dim_treatments
        self.output_size = self.dim_treatments
        self.model = BinaryMultiOutputProbModel(
            self.dim_treatments,
            logistic_C=cfg_get(args.model, "logistic_C", 1e6),
            max_iter=cfg_get(args.exp, "max_epochs", 500),
        )

    def fit_arrays(self, X, Y):
        self.model.fit(X, Y)
        return self

    def predict_proba_one(self, X):
        return self.model.predict_proba_one(X)


class MSMPropensityHistory(MSM):
    """Denominator propensity model p(A_t | history_t)."""

    model_type = "propensity_history"
    tuning_criterion = "bce"

    def __init__(self, args, dataset_collection=None, autoregressive=None, has_vitals=None, **kwargs):
        super().__init__(args, dataset_collection, autoregressive, has_vitals)
        self.input_size = None
        self.output_size = self.dim_treatments
        self.model = BinaryMultiOutputProbModel(
            self.dim_treatments,
            logistic_C=cfg_get(args.model, "logistic_C", 1e6),
            max_iter=cfg_get(args.exp, "max_epochs", 500),
        )

    def fit_arrays(self, X, Y):
        self.model.fit(X, Y)
        return self

    def predict_proba_one(self, X):
        return self.model.predict_proba_one(X)


class MSMRegressor(MSM):
    """MSM outcome regressor fitted on CLPFN support bundles."""

    model_type = "msm_regressor"
    tuning_criterion = "rmse"

    def __init__(
        self,
        args,
        propensity_treatment: MSMPropensityTreatment = None,
        propensity_history: MSMPropensityHistory = None,
        dataset_collection=None,
        autoregressive=None,
        has_vitals=None,
        hparams: dict | None = None,
        **kwargs,
    ):
        super().__init__(args, dataset_collection, autoregressive, has_vitals)
        self.input_size = None
        self.output_size = self.dim_outcome
        self.propensity_treatment = propensity_treatment
        self.propensity_history = propensity_history
        self.msm_regressor: dict[int, object] = {}
        self.train_info: dict[int, dict] = {}
        self.train_diag: dict = {}
        self.fit_hparams = dict(hparams or {})

    @classmethod
    def from_hparams(cls, hparams: dict):
        args = make_msm_args(hparams)
        prop_treatment = MSMPropensityTreatment(args)
        prop_history = MSMPropensityHistory(args)
        return cls(args, prop_treatment, prop_history, hparams=hparams)

    def compute_stabilized_weights(self, bundle, context_indices):
        n_ctx = int(bundle["covariates"].shape[0])
        time_dim = min(int(bundle["actions"].shape[1]), common.MAX_SEQ_LEN)
        sw_matrix = np.ones((n_ctx, time_dim), dtype=np.float32)

        data = build_propensity_training_data(bundle, self.fit_hparams, context_indices)
        if data is None:
            raise ValueError("No MSM propensity training rows are available.")
        x_num, x_den, y_bits, pairs = data
        if len(y_bits) < 20:
            raise ValueError(f"Need at least 20 MSM propensity rows, got {len(y_bits)}.")

        self.propensity_treatment.fit_arrays(x_num, y_bits)
        self.propensity_history.fit_arrays(x_den, y_bits)

        p_num_1 = self.propensity_treatment.predict_proba_one(x_num)
        p_den_1 = self.propensity_history.predict_proba_one(x_den)
        obs = np.asarray(y_bits, dtype=np.float64)
        p_num = np.prod(p_num_1 * obs + (1.0 - p_num_1) * (1.0 - obs), axis=-1)
        p_den = np.prod(p_den_1 * obs + (1.0 - p_den_1) * (1.0 - obs), axis=-1)
        ratios = p_num / np.maximum(p_den, 1e-6)
        ratios = np.nan_to_num(ratios, nan=1.0, posinf=1.0, neginf=1.0)
        ratios = np.clip(ratios, 1e-3, 1e3)

        for ratio, (row_idx, time_idx) in zip(ratios, pairs):
            if 0 <= int(row_idx) < n_ctx and 0 <= int(time_idx) < time_dim:
                sw_matrix[int(row_idx), int(time_idx)] = float(ratio)

        prop_info = {
            "propensity_status": "ok",
            "n_propensity": int(len(y_bits)),
            "mean_weight_raw": float(np.mean(ratios)),
        }
        return sw_matrix, prop_info

    def fit_bundle(self, bundle, context_indices):
        t0 = time.time()
        sw_matrix, prop_info = self.compute_stabilized_weights(bundle, context_indices)
        self.msm_regressor = {}
        self.train_info = {}

        for tau in range(1, common.TAU_MAX + 1):
            x_reg, y_reg, weights = build_regression_data_for_tau(
                bundle,
                sw_matrix,
                tau,
                self.fit_hparams,
                context_indices,
            )
            min_train = int(self.fit_hparams.get("min_train_per_tau", 5))
            if x_reg is None or len(y_reg) < min_train:
                raise ValueError(f"Need at least {min_train} MSM rows for tau={tau}.")

            regressor = make_regressor(self.fit_hparams)
            regressor.fit(x_reg, y_reg, sample_weight=weights)
            pred_train = regressor.predict(x_reg)
            self.msm_regressor[int(tau)] = regressor
            self.train_info[int(tau)] = {
                "n_train": int(len(y_reg)),
                "train_rmse_norm": float(np.sqrt(np.mean((pred_train - y_reg) ** 2))),
                "mean_weight": float(np.mean(weights)),
                "regressor_status": "ok",
            }

        self.train_diag = {
            "fit_time_sec": float(time.time() - t0),
            "propensity_status": str(prop_info["propensity_status"]),
            "n_propensity": int(prop_info["n_propensity"]),
            "mean_weight_raw": float(prop_info["mean_weight_raw"]),
            "train_loss": float(np.mean([v["train_rmse_norm"] for v in self.train_info.values()])),
        }
        return self

    def predict_single(self, bundle, row_id, t_obs, t_target):
        covariates = bundle["covariates"]
        outcomes = bundle["y_norm_clip"]
        actions = bundle["actions"]
        static = bundle["static"]
        row_id = int(row_id)
        t_obs = int(t_obs)
        t_target = int(t_target)
        tau = int(max(1, t_target - t_obs))

        if t_obs < self.lag_features:
            raise ValueError(f"MSM requires t_obs >= lag_features={self.lag_features}; got {t_obs}.")
        if tau not in self.msm_regressor:
            raise ValueError(f"No fitted MSM regressor for tau={tau}.")

        features = msm_feature(
            covariates[row_id],
            outcomes[row_id],
            actions[row_id],
            static[row_id],
            t=t_obs,
            tau=tau,
            lag_features=self.lag_features,
        ).reshape(1, -1)
        pred_norm = float(self.msm_regressor[tau].predict(features)[0])
        info = dict(self.train_info[tau])
        info["pred_status"] = "ok"
        return float(np.clip(pred_norm, -common.PRED_CLIP_REPORT, common.PRED_CLIP_REPORT)), info
