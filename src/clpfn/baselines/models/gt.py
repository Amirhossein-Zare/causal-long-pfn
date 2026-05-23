import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from clpfn.baselines.models.time_varying_model import TimeVaryingCausalModel
from clpfn.baselines.models.utils import OutcomeHead
from clpfn.baselines.models.utils_transformer import (
    AbsolutePositionalEncoding,
    RelativePositionalEncoding,
    TransformerMultiInputBlock,
)

logger = logging.getLogger(__name__)


class GT(TimeVaryingCausalModel):
    """
    G-Transformer method.

    Model structure:
      - model_type = "gt"
      - multi-input streams for treatments, outcomes, and vitals
      - TransformerMultiInputBlock stack
      - hr_output_transformation
      - G-computation heads
      - explicit projection_horizon, set to 0 by the CLPFN benchmark adapter for
        factual training plus autoregressive rollout evaluation
    """

    model_type = "gt"
    possible_model_types = {"gt"}

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

        if projection_horizon is not None:
            self.projection_horizon = projection_horizon
        elif getattr(args.model.gt, "projection_horizon", None) is not None:
            self.projection_horizon = args.model.gt.projection_horizon
        elif dataset_collection is not None:
            self.projection_horizon = args.dataset.projection_horizon
        else:
            raise ValueError("GT requires an explicit projection_horizon.")

        if getattr(args.dataset, "treatment_sequence", None) is None:
            raise ValueError("GT requires args.dataset.treatment_sequence.")
        if getattr(args.dataset, "projection_horizon", None) is None:
            raise ValueError("GT requires args.dataset.projection_horizon.")

        self.treatment_sequence = torch.tensor(args.dataset.treatment_sequence)[: self.projection_horizon + 1, :]

        if dataset_collection is not None:
            self.max_projection = args.dataset.projection_horizon
        else:
            self.max_projection = max(int(self.projection_horizon), int(args.dataset.projection_horizon))

        assert self.projection_horizon <= self.max_projection

        self.input_size = max(
            self.dim_treatments,
            self.dim_static_features,
            self.dim_vitals,
            self.dim_outcome,
        )

        logger.info(f"Max input size of {self.model_type}: {self.input_size}")

        self.basic_block_cls = TransformerMultiInputBlock
        self._init_specific(args)
        self.tuning_criterion = "rmse"
        self.save_hyperparameters(args)

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_multi:
            self.dataset_collection.process_data_multi()

    def _init_specific(self, args):
        sub_args = args.model.gt

        self.max_seq_length = sub_args.max_seq_length
        self.hr_size = sub_args.hr_size
        self.seq_hidden_units = sub_args.seq_hidden_units
        self.fc_hidden_units = sub_args.fc_hidden_units
        self.dropout_rate = sub_args.dropout_rate

        self.num_layer = sub_args.num_layer
        self.num_heads = sub_args.num_heads

        if (
            self.seq_hidden_units is None
            or self.hr_size is None
            or self.fc_hidden_units is None
            or self.dropout_rate is None
        ):
            raise ValueError("GT mandatory hyperparameters are missing.")

        self.head_size = sub_args.seq_hidden_units // sub_args.num_heads

        self.self_positional_encoding = None
        self.self_positional_encoding_k = None
        self.self_positional_encoding_v = None

        if sub_args.self_positional_encoding.absolute:
            self.self_positional_encoding = AbsolutePositionalEncoding(
                self.max_seq_length,
                self.seq_hidden_units,
                sub_args.self_positional_encoding.trainable,
            )
        else:
            self.self_positional_encoding_k = RelativePositionalEncoding(
                sub_args.self_positional_encoding.max_relative_position,
                self.head_size,
                sub_args.self_positional_encoding.trainable,
            )
            self.self_positional_encoding_v = RelativePositionalEncoding(
                sub_args.self_positional_encoding.max_relative_position,
                self.head_size,
                sub_args.self_positional_encoding.trainable,
            )

        self.treatments_input_transformation = nn.Linear(self.dim_treatments, self.seq_hidden_units)
        self.vitals_input_transformation = (
            nn.Linear(self.dim_vitals, self.seq_hidden_units) if self.has_vitals else None
        )
        self.outputs_input_transformation = nn.Linear(self.dim_outcome, self.seq_hidden_units)
        self.static_input_transformation = nn.Linear(self.dim_static_features, self.seq_hidden_units)

        self.n_inputs = 3 if self.has_vitals else 2

        self.transformer_blocks = nn.ModuleList(
            [
                self.basic_block_cls(
                    self.seq_hidden_units,
                    self.num_heads,
                    self.head_size,
                    self.seq_hidden_units * 4,
                    self.dropout_rate,
                    self.dropout_rate if sub_args.attn_dropout else 0.0,
                    self_positional_encoding_k=self.self_positional_encoding_k,
                    self_positional_encoding_v=self.self_positional_encoding_v,
                    n_inputs=self.n_inputs,
                    disable_cross_attention=sub_args.disable_cross_attention,
                    isolate_subnetwork=sub_args.isolate_subnetwork,
                )
                for _ in range(self.num_layer)
            ]
        )

        self.hr_output_transformation = nn.Linear(self.seq_hidden_units, self.hr_size)
        self.output_dropout = nn.Dropout(self.dropout_rate)

        self.G_comp_heads = nn.ModuleList(
            [
                OutcomeHead(
                    self.seq_hidden_units,
                    self.hr_size,
                    self.fc_hidden_units,
                    self.dim_treatments,
                    self.dim_outcome,
                )
                for _ in range(self.projection_horizon + 1)
            ]
        )

    def build_hr(self, prev_treatments, vitals, prev_outputs, static_features, active_entries):
        active_entries_treat_outcomes = torch.clone(active_entries)
        active_entries_vitals = torch.clone(active_entries)

        x_t = self.treatments_input_transformation(prev_treatments)
        x_o = self.outputs_input_transformation(prev_outputs)
        x_v = self.vitals_input_transformation(vitals) if self.has_vitals else None
        x_s = self.static_input_transformation(static_features.unsqueeze(1))

        for block in self.transformer_blocks:
            if self.self_positional_encoding is not None:
                x_t = x_t + self.self_positional_encoding(x_t)
                x_o = x_o + self.self_positional_encoding(x_o)
                x_v = x_v + self.self_positional_encoding(x_v) if self.has_vitals else None

            if self.has_vitals:
                x_t, x_o, x_v = block(
                    (x_t, x_o, x_v),
                    x_s,
                    active_entries_treat_outcomes,
                    active_entries_vitals,
                )
            else:
                x_t, x_o = block(
                    (x_t, x_o),
                    x_s,
                    active_entries_treat_outcomes,
                )

        if not self.has_vitals:
            x = (x_o + x_t) / 2
        else:
            x = (x_o + x_t + x_v) / 3

        output = self.output_dropout(x)
        hr = nn.ELU()(self.hr_output_transformation(output))
        return hr

    def forward(self, batch):
        prev_treatments = batch["prev_treatments"]
        vitals = batch["vitals"] if self.has_vitals else None
        prev_outputs = batch["prev_outputs"]
        static_features = batch["static_features"]
        curr_treatments = batch["current_treatments"]
        active_entries = batch["active_entries"].clone()

        batch_size = prev_treatments.size(0)
        time_dim = prev_treatments.size(1)
        device = prev_treatments.device

        if self.training:
            if self.projection_horizon == 0:
                hr = self.build_hr(prev_treatments, vitals, prev_outputs, static_features, active_entries)
                pred_factuals = self.G_comp_heads[0].build_outcome(hr, curr_treatments)
                pred_pseudos = None
                pseudo_outcomes = None
                return pred_factuals, pred_pseudos, pseudo_outcomes, active_entries

            pseudo_time = max(time_dim - self.projection_horizon - 1, 0)

            pseudo_outcomes_all_steps = torch.zeros(
                (batch_size, pseudo_time, self.projection_horizon + 1, self.dim_outcome),
                device=device,
            )
            pred_pseudos_all_steps = torch.zeros(
                (batch_size, pseudo_time, self.projection_horizon + 1, self.dim_outcome),
                device=device,
            )
            active_entries_all_steps = torch.zeros((batch_size, pseudo_time, 1), device=device)

            for t in range(1, time_dim - self.projection_horizon):
                current_active_entries = batch["active_entries"].clone()
                current_active_entries[:, int(t + self.projection_horizon):] = 0.0
                active_entries_all_steps[:, t - 1, :] = current_active_entries[:, t + self.projection_horizon - 1, :]

                with torch.no_grad():
                    idx = (
                        (torch.arange(0, time_dim, device=device) >= t - 1)
                        * (torch.arange(0, time_dim, device=device) < t + self.projection_horizon)
                    ).bool()

                    curr_treatments_cf = curr_treatments.clone()
                    treatment_seq = self.treatment_sequence.to(device)
                    curr_treatments_cf[:, idx, :] = treatment_seq[: idx.sum(), :]

                    prev_treatments_cf = torch.cat(
                        (prev_treatments[:, :1, :], curr_treatments_cf[:, :-1, :]),
                        dim=1,
                    )

                    hr_cf = self.build_hr(
                        prev_treatments_cf,
                        vitals,
                        prev_outputs,
                        static_features,
                        current_active_entries,
                    )

                    pseudo_outcomes = torch.zeros(
                        (batch_size, self.projection_horizon + 1, self.dim_outcome),
                        device=device,
                    )

                    for i in range(self.projection_horizon, 0, -1):
                        pseudo_outcome = self.G_comp_heads[i].build_outcome(
                            hr_cf,
                            curr_treatments_cf,
                        )[:, t + i - 1, :]
                        pseudo_outcomes[:, i - 1, :] = pseudo_outcome

                    pseudo_outcomes[:, -1, :] = batch["outputs"][:, t + self.projection_horizon - 1, :]
                    pseudo_outcomes_all_steps[:, t - 1, :, :] = pseudo_outcomes

                hr = self.build_hr(prev_treatments, vitals, prev_outputs, static_features, current_active_entries)

                pred_pseudos = torch.zeros(
                    (batch_size, self.projection_horizon + 1, self.dim_outcome),
                    device=device,
                )

                for i in range(self.projection_horizon, -1, -1):
                    pred_pseudo = self.G_comp_heads[i].build_outcome(hr, curr_treatments)[:, t + i - 1, :]
                    pred_pseudos[:, i, :] = pred_pseudo

                pred_pseudos_all_steps[:, t - 1, :, :] = pred_pseudos

            return None, pred_pseudos_all_steps, pseudo_outcomes_all_steps, active_entries_all_steps

        fixed_split = batch["sequence_lengths"] - self.max_projection if self.projection_horizon > 0 else batch["sequence_lengths"]

        for i in range(len(active_entries)):
            active_entries[i, int(fixed_split[i] + self.projection_horizon):] = 0.0

        hr = self.build_hr(prev_treatments, vitals, prev_outputs, static_features, active_entries)

        if self.projection_horizon > 0:
            pred_outcomes = self.G_comp_heads[0].build_outcome(hr, curr_treatments)
            index_pred = torch.arange(0, time_dim, device=device) == fixed_split[..., None] - 1
            pred_outcomes = pred_outcomes[index_pred]
        else:
            pred_outcomes = self.G_comp_heads[0].build_outcome(hr, curr_treatments)

        return pred_outcomes, hr

    def training_step(self, batch, batch_ind=0, optimizer_idx=None):
        for par in self.parameters():
            par.requires_grad = True

        pred_factuals, pred_pseudos, pseudo_outcomes, active_entries_all_steps = self(batch)

        if self.projection_horizon > 0:
            active_entries_all_steps = active_entries_all_steps.unsqueeze(-2)
            mse_gcomp = F.mse_loss(pred_pseudos, pseudo_outcomes, reduction="none")
            denom = (active_entries_all_steps.sum(dim=(0, 1)) * self.dim_outcome).clamp(min=1.0)
            mse_gcomp = (mse_gcomp * active_entries_all_steps).sum(dim=(0, 1)) / denom
            loss = mse_gcomp.mean()
        else:
            mse_factual = F.mse_loss(pred_factuals, batch["outputs"], reduction="none")
            loss = (mse_factual * batch["active_entries"]).sum() / (
                batch["active_entries"].sum().clamp(min=1.0) * self.dim_outcome
            )

        self.log(f"{self.model_type}_train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx=0, dataset_idx=None):
        batch = self.move_batch_to_device(batch)
        outcome_pred, hr = self(batch)
        return outcome_pred.cpu(), hr.cpu()

    def get_predictions(self, dataset) -> np.array:
        logger.info(f"Predictions for {dataset.subset_name}.")
        self.eval()
        loader = DataLoader(
            dataset,
            batch_size=getattr(self.hparams.dataset, "val_batch_size", 64),
            shuffle=False,
        )
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = self.move_batch_to_device(batch)
                outcome_pred, _ = self.predict_step(batch)
                preds.append(outcome_pred.numpy())
        return np.concatenate(preds, axis=0)

    def get_normalised_n_step_rmses(self, dataset):
        outcome_pred = self.get_predictions(dataset)
        outputs = dataset.data["outputs"][:, :outcome_pred.shape[1]]
        active_entries = dataset.data["active_entries"][:, :outcome_pred.shape[1]]
        mse = (active_entries * (outcome_pred - outputs) ** 2).sum(axis=(0, 2)) / active_entries.sum(axis=(0, 2))
        return np.sqrt(mse)

    def configure_optimizers(self):
        optimizer = self._get_optimizer(list(self.named_parameters()))
        return optimizer
