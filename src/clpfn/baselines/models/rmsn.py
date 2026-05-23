import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from clpfn.baselines.common.training import masked_sequence_loss, masked_weighted_mse_loss
from clpfn.baselines.models.time_varying_model import TimeVaryingCausalModel
from clpfn.baselines.models.utils_lstm import VariationalLSTM

logger = logging.getLogger(__name__)


class RMSN(TimeVaryingCausalModel):
    """
    RMSN base class.

    Model structure:
      - model_type is defined by subclasses
      - possible_model_types includes encoder/decoder/propensity networks
      - _init_specific builds VariationalLSTM + output_layer
      - decoder uses memory_adapter(init_state)
    """

    model_type = None
    possible_model_types = {"encoder", "decoder", "propensity_treatment", "propensity_history"}
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

    def get_propensity_scores(self, dataset: Dataset) -> np.array:
        if self.model_type in {"propensity_treatment", "propensity_history"}:
            data_loader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
            scores = []
            self.eval()
            with torch.no_grad():
                for batch in data_loader:
                    batch = self.move_batch_to_device(batch)
                    scores.append(self.predict_step(batch))
            propensity_scores = torch.cat(scores)
            return propensity_scores.numpy()

        raise NotImplementedError()


class RMSNPropensityNetworkTreatment(RMSN):
    """
    Propensity_treatment network.

    Input:
      prev_treatments

    Target:
      current_treatments
    """

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

        logger.info(f"Input size of {self.model_type}: {self.input_size}")

        self._init_specific(args.model.propensity_treatment)
        self.save_hyperparameters(args)

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_multi:
            self.dataset_collection.process_data_multi()

    def forward(self, batch):
        prev_treatments = batch["prev_treatments"]
        x = self.lstm(prev_treatments, init_states=None)
        treatment_pred = self.output_layer(x)
        return treatment_pred

    def training_step(self, batch, batch_ind=0):
        treatment_pred = self(batch)
        bce_loss = self.bce_loss(treatment_pred, batch["current_treatments"].float(), kind="predict")
        bce_loss = masked_sequence_loss(bce_loss, batch["active_entries"])
        self.log(f"{self.model_type}_bce_loss", bce_loss)
        return bce_loss

    def predict_step(self, batch, batch_ind=0, dataset_idx=None):
        batch = self.move_batch_to_device(batch)
        return torch.sigmoid(self(batch)).cpu()


class RMSNPropensityNetworkHistory(RMSN):
    """
    Propensity_history network.

    Input:
      prev_treatments, vitals, prev_outputs, static_features

    Target:
      current_treatments
    """

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

        logger.info(f"Input size of {self.model_type}: {self.input_size}")

        self._init_specific(args.model.propensity_history)
        self.save_hyperparameters(args)

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_multi:
            self.dataset_collection.process_data_multi()

    def forward(self, batch, detach_treatment=False):
        prev_treatments = batch["prev_treatments"]

        vitals_or_prev_outputs = []
        if self.has_vitals:
            vitals_or_prev_outputs.append(batch["vitals"])
        if self.autoregressive:
            vitals_or_prev_outputs.append(batch["prev_outputs"])

        vitals_or_prev_outputs = torch.cat(vitals_or_prev_outputs, dim=-1)

        static_features = batch["static_features"]

        x = torch.cat((prev_treatments, vitals_or_prev_outputs), dim=-1)
        x = torch.cat(
            (x, static_features.unsqueeze(1).expand(-1, x.size(1), -1)),
            dim=-1,
        )

        x = self.lstm(x, init_states=None)
        treatment_pred = self.output_layer(x)
        return treatment_pred

    def training_step(self, batch, batch_ind=0):
        treatment_pred = self(batch)
        bce_loss = self.bce_loss(treatment_pred, batch["current_treatments"].float(), kind="predict")
        bce_loss = masked_sequence_loss(bce_loss, batch["active_entries"])
        self.log(f"{self.model_type}_bce_loss", bce_loss)
        return bce_loss

    def predict_step(self, batch, batch_ind=0, dataset_idx=None):
        batch = self.move_batch_to_device(batch)
        return torch.sigmoid(self(batch)).cpu()


class RMSNEncoder(RMSN):
    """
    MSN encoder.

    Input construction:
        [vitals_t, prev_outputs_t, current_treatments_t, static]

    Output:
        outcome_pred_t, r_t
    """

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

        logger.info(f"Input size of {self.model_type}: {self.input_size}")

        self._init_specific(args.model.encoder)
        self.save_hyperparameters(args)

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_encoder:
            self.dataset_collection.process_data_encoder()
        if self.dataset_collection is not None and "stabilized_weights" not in self.dataset_collection.train_f.data:
            self.dataset_collection.process_propensity_train_f(self.propensity_treatment, self.propensity_history)

    def forward(self, batch, detach_treatment=False):
        vitals_or_prev_outputs = []

        if self.has_vitals:
            vitals_or_prev_outputs.append(batch["vitals"])

        if self.autoregressive:
            vitals_or_prev_outputs.append(batch["prev_outputs"])

        vitals_or_prev_outputs = torch.cat(vitals_or_prev_outputs, dim=-1)

        static_features = batch["static_features"]
        curr_treatments = batch["current_treatments"]

        x = torch.cat((vitals_or_prev_outputs, curr_treatments), dim=-1)
        x = torch.cat(
            (x, static_features.unsqueeze(1).expand(-1, x.size(1), -1)),
            dim=-1,
        )

        r = self.lstm(x, init_states=None)
        outcome_pred = self.output_layer(r)
        return outcome_pred, r

    def training_step(self, batch, batch_ind=0):
        outcome_pred, _ = self(batch)

        weighted_mse_loss = masked_weighted_mse_loss(
            outcome_pred,
            batch["outputs"],
            batch["active_entries"],
            batch["sw_tilde_enc"],
        )

        self.log(f"{self.model_type}_mse_loss", weighted_mse_loss)
        return weighted_mse_loss

    def predict_step(self, batch, batch_ind=0, dataset_idx=None):
        batch = self.move_batch_to_device(batch)
        outcome_pred, r = self(batch)
        return outcome_pred.cpu(), r.cpu()

    def get_representations(self, dataset: Dataset) -> np.array:
        logger.info(f"Representations for {dataset.subset_name}.")
        data_loader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        reps = []
        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch = self.move_batch_to_device(batch)
                _, r = self.predict_step(batch)
                reps.append(r.numpy())
        return np.concatenate(reps, axis=0)

    def get_predictions(self, dataset: Dataset) -> np.array:
        logger.info(f"Predictions for {dataset.subset_name}.")
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
    """
    RMSN decoder.

    Input construction:
        [current_treatments_t, prev_outputs_t, static]

    Decoder initialization:
        init_state -> memory_adapter -> VariationalLSTM init_states
    """

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

        logger.info(f"Input size of {self.model_type}: {self.input_size}")

        self._init_specific(args.model.decoder, encoder_r_size=encoder_r_size)
        self.save_hyperparameters(args)

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_decoder:
            self.dataset_collection.process_data_decoder(self.encoder, save_encoder_r=True)

    def forward(self, batch, detach_treatment=False):
        curr_treatments = batch["current_treatments"]
        prev_outputs = batch["prev_outputs"]
        static_features = batch["static_features"]
        init_states = batch["init_state"]

        x = torch.cat((curr_treatments, prev_outputs), dim=-1)
        x = torch.cat(
            (x, static_features.unsqueeze(1).expand(-1, x.size(1), -1)),
            dim=-1,
        )

        x = self.lstm(x, init_states=self.memory_adapter(init_states))
        outcome_pred = self.output_layer(x)
        return outcome_pred

    def training_step(self, batch, batch_ind=0):
        outcome_pred = self(batch)

        weighted_mse_loss = masked_weighted_mse_loss(
            outcome_pred,
            batch["outputs"],
            batch["active_entries"],
            batch["sw_tilde_dec"],
        )

        self.log(f"{self.model_type}_mse_loss", weighted_mse_loss)
        return weighted_mse_loss

    def predict_step(self, batch, batch_ind=0, dataset_idx=None):
        batch = self.move_batch_to_device(batch)
        return self(batch).cpu()

    def get_predictions(self, dataset: Dataset) -> np.array:
        logger.info(f"Predictions for {dataset.subset_name}.")
        data_loader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        preds = []
        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch = self.move_batch_to_device(batch)
                preds.append(self.predict_step(batch).numpy())
        return np.concatenate(preds, axis=0)
