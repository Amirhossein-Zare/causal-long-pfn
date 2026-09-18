import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from clpfn.baselines.common.training import masked_sequence_loss, masked_weighted_mse_loss
from clpfn.baselines.models.time_varying_model import TimeVaryingCausalModel, cfg_get
from clpfn.baselines.models.utils_lstm import VariationalLSTM

logger = logging.getLogger(__name__)


class RMSN(TimeVaryingCausalModel):
    """Base class for the RMSN propensity, encoder, and decoder nets."""

    model_type = None
    possible_model_types = {
        "encoder",
        "decoder",
        "propensity_treatment",
        "propensity_history",
    }
    tuning_criterion = None

    def __init__(
        self,
        args,
        dataset_collection=None,
        autoregressive=None,
        has_vitals=None,
        bce_weights=None,
        **kwargs,
    ):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

    def _init_specific(self, sub_args, encoder_r_size=None):
        self.seq_hidden_units = sub_args.seq_hidden_units
        self.dropout_rate = sub_args.dropout_rate
        self.num_layer = sub_args.num_layer

        if self.seq_hidden_units is None or self.dropout_rate is None:
            raise ValueError(f"{self.model_type} mandatory hyperparameters are missing.")

        if self.model_type == "decoder":
            self.memory_adapter = nn.Linear(encoder_r_size, self.seq_hidden_units)

        self.lstm = VariationalLSTM(
            self.input_size,
            self.seq_hidden_units,
            self.num_layer,
            self.dropout_rate,
        )
        self.output_layer = nn.Linear(self.seq_hidden_units, self.output_size)

    def get_propensity_scores(self, dataset: Dataset) -> np.ndarray:
        if self.model_type in {"propensity_treatment", "propensity_history"}:
            data_loader = DataLoader(
                dataset,
                batch_size=self.hparams.dataset.val_batch_size,
                shuffle=False,
            )
            scores = []
            self.eval()
            with torch.no_grad():
                for batch in data_loader:
                    batch = self.move_batch_to_device(batch)
                    scores.append(self.predict_step(batch))
            return torch.cat(scores).numpy()

        raise NotImplementedError()


class _RMSNTreatmentPropensityMixin:
    def _treatment_loss(self, logits, current_treatments):
        loss = self.bce_loss(logits, current_treatments.float(), kind="predict")
        return loss

    def _treatment_probabilities(self, logits):
        mode = cfg_get(self.hparams.dataset, "treatment_mode")
        if mode == "multiclass":
            return torch.softmax(logits, dim=-1)
        if mode == "multilabel":
            return torch.sigmoid(logits)
        raise ValueError(f"Unknown RMSN treatment mode: {mode!r}")


class RMSNPropensityNetworkTreatment(_RMSNTreatmentPropensityMixin, RMSN):
    """Numerator treatment model P(A_t | A_<t)."""

    model_type = "propensity_treatment"
    tuning_criterion = "bce"

    def __init__(
        self,
        args,
        dataset_collection=None,
        autoregressive=None,
        has_vitals=None,
        bce_weights=None,
        **kwargs,
    ):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)
        self.input_size = self.dim_treatments
        self.output_size = self.dim_treatments
        logger.info("Input size of %s: %s", self.model_type, self.input_size)
        self._init_specific(args.model.propensity_treatment)
        self.save_hyperparameters(args)

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_multi:
            self.dataset_collection.process_data_multi()

    def forward(self, batch):
        x = self.lstm(batch["prev_treatments"], init_states=None)
        return self.output_layer(x)

    def training_step(self, batch, batch_ind=0):
        logits = self(batch)
        loss = self._treatment_loss(logits, batch["current_treatments"])
        loss = masked_sequence_loss(loss, batch["active_entries"])
        self.log(f"{self.model_type}_bce_loss", loss)
        return loss

    def predict_step(self, batch, batch_ind=0, dataset_idx=None):
        batch = self.move_batch_to_device(batch)
        return self._treatment_probabilities(self(batch)).cpu()


class RMSNPropensityNetworkHistory(_RMSNTreatmentPropensityMixin, RMSN):
    """Denominator treatment model P(A_t | H_t)."""

    model_type = "propensity_history"
    tuning_criterion = "bce"

    def __init__(
        self,
        args,
        dataset_collection=None,
        autoregressive=None,
        has_vitals=None,
        bce_weights=None,
        **kwargs,
    ):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)
        self.input_size = self.dim_treatments + self.dim_static_features
        self.input_size += self.dim_vitals if self.has_vitals else 0
        self.input_size += self.dim_outcome if self.autoregressive else 0
        self.output_size = self.dim_treatments
        logger.info("Input size of %s: %s", self.model_type, self.input_size)
        self._init_specific(args.model.propensity_history)
        self.save_hyperparameters(args)

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_multi:
            self.dataset_collection.process_data_multi()

    def forward(self, batch, detach_treatment=False):
        history = []
        if self.has_vitals:
            history.append(batch["vitals"])
        if self.autoregressive:
            history.append(batch["prev_outputs"])
        history = torch.cat(history, dim=-1)
        x = torch.cat((batch["prev_treatments"], history), dim=-1)
        x = torch.cat(
            (x, batch["static_features"].unsqueeze(1).expand(-1, x.size(1), -1)),
            dim=-1,
        )
        x = self.lstm(x, init_states=None)
        return self.output_layer(x)

    def training_step(self, batch, batch_ind=0):
        logits = self(batch)
        loss = self._treatment_loss(logits, batch["current_treatments"])
        loss = masked_sequence_loss(loss, batch["active_entries"])
        self.log(f"{self.model_type}_bce_loss", loss)
        return loss

    def predict_step(self, batch, batch_ind=0, dataset_idx=None):
        batch = self.move_batch_to_device(batch)
        return self._treatment_probabilities(self(batch)).cpu()


class RMSNEncoder(RMSN):
    """Propensity-weighted one-step outcome encoder."""

    model_type = "encoder"
    tuning_criterion = "rmse"

    def __init__(
        self,
        args,
        propensity_treatment=None,
        propensity_history=None,
        dataset_collection=None,
        autoregressive=None,
        has_vitals=None,
        bce_weights=None,
        **kwargs,
    ):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)
        self.input_size = self.dim_treatments + self.dim_static_features
        self.input_size += self.dim_vitals if self.has_vitals else 0
        self.input_size += self.dim_outcome if self.autoregressive else 0
        self.output_size = self.dim_outcome
        self.propensity_treatment = propensity_treatment
        self.propensity_history = propensity_history
        logger.info("Input size of %s: %s", self.model_type, self.input_size)
        self._init_specific(args.model.encoder)
        self.save_hyperparameters(args)

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_encoder:
            self.dataset_collection.process_data_encoder()
        if self.dataset_collection is not None and "stabilized_weights" not in self.dataset_collection.train_f.data:
            self.dataset_collection.process_propensity_train_f(self.propensity_treatment, self.propensity_history)

    def forward(self, batch, detach_treatment=False):
        history = []
        if self.has_vitals:
            history.append(batch["vitals"])
        if self.autoregressive:
            history.append(batch["prev_outputs"])
        history = torch.cat(history, dim=-1)
        x = torch.cat((history, batch["current_treatments"]), dim=-1)
        x = torch.cat(
            (x, batch["static_features"].unsqueeze(1).expand(-1, x.size(1), -1)),
            dim=-1,
        )
        r = self.lstm(x, init_states=None)
        outcome_pred = self.output_layer(r)
        return outcome_pred, r

    def training_step(self, batch, batch_ind=0):
        outcome_pred, _ = self(batch)
        loss = masked_weighted_mse_loss(
            outcome_pred,
            batch["outputs"],
            batch["active_entries"],
            batch["sw_tilde_enc"],
        )
        self.log(f"{self.model_type}_mse_loss", loss)
        return loss

    def predict_step(self, batch, batch_ind=0, dataset_idx=None):
        batch = self.move_batch_to_device(batch)
        outcome_pred, r = self(batch)
        return outcome_pred.cpu(), r.cpu()

    def get_representations(self, dataset: Dataset) -> np.ndarray:
        logger.info("Representations for %s.", dataset.subset_name)
        data_loader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        reps = []
        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch = self.move_batch_to_device(batch)
                _, r = self.predict_step(batch)
                reps.append(r.numpy())
        return np.concatenate(reps, axis=0)

    def get_predictions(self, dataset: Dataset) -> np.ndarray:
        logger.info("Predictions for %s.", dataset.subset_name)
        data_loader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        preds = []
        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch = self.move_batch_to_device(batch)
                outcome_pred, _ = self.predict_step(batch)
                preds.append(outcome_pred.numpy())
        return np.concatenate(preds, axis=0)


class RMSNDecoder(RMSN):
    """Autoregressive RMSN decoder initialized from the encoder state at t-1."""

    model_type = "decoder"
    tuning_criterion = "rmse"

    def __init__(
        self,
        args,
        encoder=None,
        dataset_collection=None,
        encoder_r_size=None,
        autoregressive=None,
        has_vitals=None,
        bce_weights=None,
        **kwargs,
    ):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)
        self.input_size = self.dim_treatments + self.dim_static_features + self.dim_outcome
        self.output_size = self.dim_outcome
        self.encoder = encoder
        encoder_r_size = self.encoder.seq_hidden_units if encoder is not None else encoder_r_size
        logger.info("Input size of %s: %s", self.model_type, self.input_size)
        self._init_specific(args.model.decoder, encoder_r_size=encoder_r_size)
        self.save_hyperparameters(args)

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_decoder:
            self.dataset_collection.process_data_decoder(self.encoder, save_encoder_r=True)

    def forward(self, batch, detach_treatment=False):
        x = torch.cat((batch["current_treatments"], batch["prev_outputs"]), dim=-1)
        x = torch.cat(
            (x, batch["static_features"].unsqueeze(1).expand(-1, x.size(1), -1)),
            dim=-1,
        )
        x = self.lstm(x, init_states=self.memory_adapter(batch["init_state"]))
        return self.output_layer(x)

    def training_step(self, batch, batch_ind=0):
        outcome_pred = self(batch)
        loss = masked_weighted_mse_loss(
            outcome_pred,
            batch["outputs"],
            batch["active_entries"],
            batch["sw_tilde_dec"],
        )
        self.log(f"{self.model_type}_mse_loss", loss)
        return loss

    def predict_step(self, batch, batch_ind=0, dataset_idx=None):
        batch = self.move_batch_to_device(batch)
        return self(batch).cpu()

    def get_predictions(self, dataset: Dataset) -> np.ndarray:
        logger.info("Predictions for %s.", dataset.subset_name)
        data_loader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        preds = []
        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch = self.move_batch_to_device(batch)
                preds.append(self.predict_step(batch).numpy())
        return np.concatenate(preds, axis=0)
