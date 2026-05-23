import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from clpfn.baselines.models.time_varying_model import TimeVaryingCausalModel
from clpfn.baselines.models.utils import ROutcomeVitalsHead
from clpfn.baselines.models.utils_lstm import VariationalLSTM

logger = logging.getLogger(__name__)


class GNet(TimeVaryingCausalModel):
    """
    GNet method with one explicit benchmark extension:
      - vitals_loss_weight is tunable.

    Model structure:
      - model_type = g_net
      - autoregressive-only assertion
      - input construction:
            [current_treatment_t, vitals_t, prev_output_t, static]
      - VariationalLSTM representation network
      - ROutcomeVitalsHead conditional outcome/vitals head
      - training_step masked outcome/vitals MSE structure
    """

    model_type = "g_net"
    possible_model_types = {"g_net"}
    tuning_criterion = "rmse"

    def __init__(
        self,
        args,
        dataset_collection=None,
        autoregressive=None,
        has_vitals=None,
        projection_horizon=None,
        bce_weights=None,
        **kwargs,
    ):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

        if self.dataset_collection is not None:
            self.projection_horizon = self.dataset_collection.projection_horizon
        else:
            self.projection_horizon = projection_horizon

        assert self.autoregressive

        self.input_size = self.dim_treatments + self.dim_static_features + self.dim_outcome
        self.input_size += self.dim_vitals if self.has_vitals else 0

        self.output_size = self.dim_vitals + self.dim_outcome

        logger.info(f"Input size of {self.model_type}: {self.input_size}")

        self._init_specific(args.model.g_net)
        self.save_hyperparameters(args)

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_multi:
            self.dataset_collection.process_data_multi()

    def _init_specific(self, sub_args):
        self.dropout_rate = sub_args.dropout_rate
        self.seq_hidden_units = sub_args.seq_hidden_units
        self.r_size = sub_args.r_size
        self.num_layer = sub_args.num_layer
        self.comp_sizes = sub_args.comp_sizes
        self.num_comp = sub_args.num_comp
        self.fc_hidden_units = sub_args.fc_hidden_units
        self.mc_samples = sub_args.mc_samples

        if (
            self.seq_hidden_units is None
            or self.r_size is None
            or self.dropout_rate is None
            or self.fc_hidden_units is None
        ):
            raise ValueError("GNet mandatory hyperparameters are missing.")

        assert len(self.comp_sizes) == self.num_comp
        assert sum(self.comp_sizes) == self.output_size

        self.repr_net = VariationalLSTM(
            self.input_size,
            self.seq_hidden_units,
            self.num_layer,
            self.dropout_rate,
        )

        self.r_outcome_vitals_head = ROutcomeVitalsHead(
            self.seq_hidden_units,
            self.r_size,
            self.fc_hidden_units,
            self.dim_outcome,
            self.dim_vitals,
            self.num_comp,
            self.comp_sizes,
        )

    def forward(self, batch, sample=False):
        static_features = batch["static_features"]
        curr_treatments = batch["current_treatments"]
        vitals = batch["vitals"] if self.has_vitals else None
        prev_outputs = batch["prev_outputs"]

        r = self.build_r(curr_treatments, vitals, prev_outputs, static_features)
        vitals_outcome_pred = self.r_outcome_vitals_head.build_outcome_vitals(r)
        return vitals_outcome_pred

    def build_r(self, curr_treatments, vitals, prev_outputs, static_features):
        vitals_prev_outputs = []

        if self.has_vitals:
            vitals_prev_outputs.append(vitals)

        if self.autoregressive:
            vitals_prev_outputs.append(prev_outputs)

        vitals_prev_outputs = torch.cat(vitals_prev_outputs, dim=-1)

        x = torch.cat((curr_treatments, vitals_prev_outputs), dim=-1)
        x = torch.cat(
            (x, static_features.unsqueeze(1).expand(-1, x.size(1), -1)),
            dim=-1,
        )

        x = self.repr_net(x)
        r = self.r_outcome_vitals_head.build_r(x)
        return r

    def training_step(self, batch, batch_ind=0):
        outcome_next_vitals_pred = self(batch)

        outcome_pred = outcome_next_vitals_pred[:, :, :self.dim_outcome]
        next_vitals_pred = outcome_next_vitals_pred[:, :, self.dim_outcome:]

        outcome_mse_loss = F.mse_loss(
            outcome_pred,
            batch["outputs"],
            reduction="none",
        )

        if self.has_vitals:
            vitals_mse_loss = F.mse_loss(
                next_vitals_pred[:, :-1, :],
                batch["next_vitals"],
                reduction="none",
            )
        else:
            vitals_mse_loss = 0.0

        active = batch["active_entries"]
        mse_loss_outcome = (active * outcome_mse_loss).sum() / active.sum().clamp(min=1.0)

        if self.hparams.model.g_net.fit_vitals and self.has_vitals:
            vitals_active = active[:, 1:, :]
            mse_loss_vitals = (vitals_active * vitals_mse_loss).sum() / vitals_active.sum().clamp(min=1.0)
        else:
            mse_loss_vitals = 0.0

        vitals_loss_weight = float(getattr(self.hparams.model.g_net, "vitals_loss_weight", 1.0))
        mse_loss = mse_loss_outcome + vitals_loss_weight * mse_loss_vitals

        self.log(f"{self.model_type}_train_mse_loss_outcomes", mse_loss_outcome)
        self.log(f"{self.model_type}_train_mse_loss_vitals", mse_loss_vitals)
        self.log(f"{self.model_type}_train_mse_loss", mse_loss)

        return mse_loss

    def predict_step(self, batch, batch_ind=0, dataset_idx=None):
        batch = self.move_batch_to_device(batch)
        return self(batch).cpu()

    def on_fit_end(self) -> None:
        if (
            self.dataset_collection is not None
            and hasattr(self.dataset_collection, "train_f_holdout")
            and len(self.dataset_collection.train_f_holdout) > 0
        ):
            logger.info("Fitting residuals based on train_f_holdout.")
            self.eval()
            outcome_next_vitals_pred = self.get_predictions(
                self.dataset_collection.train_f_holdout,
                vitals=True,
            )

            outcomes_next_vitals = self.dataset_collection.train_f_holdout.data["outputs"]
            if self.has_vitals:
                outcome_next_vitals_pred = outcome_next_vitals_pred[:, :-1, :]
                outcomes_next_vitals = outcomes_next_vitals[:, :-1, :]

                vitals = self.dataset_collection.train_f_holdout.data["next_vitals"]
                outcomes_next_vitals = np.concatenate((outcomes_next_vitals, vitals), axis=-1)

            self.holdout_resid = outcomes_next_vitals - outcome_next_vitals_pred
            self.holdout_resid_len = self.dataset_collection.train_f_holdout.data["sequence_lengths"]
            if self.has_vitals:
                self.holdout_resid_len = self.holdout_resid_len - 1
        else:
            self.holdout_resid = None
            self.holdout_resid_len = None

    def get_predictions(self, dataset, vitals=False) -> np.array:
        if isinstance(dataset, list):
            return self.get_autoregressive_predictions(dataset)

        logger.info(f"Predictions for {dataset.subset_name}.")
        self.eval()
        loader = DataLoader(
            dataset,
            batch_size=getattr(self.hparams.dataset, "val_batch_size", 64),
            shuffle=False,
        )
        predictions = []
        with torch.no_grad():
            for batch in loader:
                batch = self.move_batch_to_device(batch)
                predictions.append(self.predict_step(batch).numpy())

        outcome_next_vitals_pred = np.concatenate(predictions, axis=0)
        if vitals:
            return outcome_next_vitals_pred[:, :, self.dim_outcome:]
        return outcome_next_vitals_pred[:, :, :self.dim_outcome]

    def get_autoregressive_predictions(self, datasets: list) -> np.array:
        logger.info("Autoregressive GNet predictions.")
        predictions = [self.get_predictions(dataset)[:, -1:, :] for dataset in datasets]
        return np.concatenate(predictions, axis=1)
