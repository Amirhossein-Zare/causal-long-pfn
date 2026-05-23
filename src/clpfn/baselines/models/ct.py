import logging
import numpy as np
import torch
from torch import nn

from clpfn.baselines.models.edct import EDCT
from clpfn.baselines.models.utils import BRTreatmentOutcomeHead
from clpfn.baselines.models.utils_transformer import TransformerMultiInputBlock

logger = logging.getLogger(__name__)


class CT(EDCT):
    """
    Causal Transformer method.

    Model structure:
      - model_type = "multi"
      - autoregressive assertion
      - separate treatment / outcome / vitals streams
      - TransformerMultiInputBlock stack
      - BRTreatmentOutcomeHead treatment/outcome heads
      - future vitals masking through future_past_split
      - autoregressive previous-outcome rollout contract
    """

    model_type = "multi"
    possible_model_types = {"multi"}

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

        self.input_size = max(
            self.dim_treatments,
            self.dim_static_features,
            self.dim_vitals,
            self.dim_outcome,
        )

        logger.info(f"Max input size of {self.model_type}: {self.input_size}")

        assert self.autoregressive

        self.basic_block_cls = TransformerMultiInputBlock
        self._init_specific(args.model.multi)
        self.save_hyperparameters(args)

    def _init_specific(self, sub_args):
        super(CT, self)._init_specific(sub_args)

        if (
            self.seq_hidden_units is None
            or self.br_size is None
            or self.fc_hidden_units is None
            or self.dropout_rate is None
        ):
            raise ValueError("CT mandatory hyperparameters are missing.")

        self.treatments_input_transformation = nn.Linear(self.dim_treatments, self.seq_hidden_units)

        self.vitals_input_transformation = (
            nn.Linear(self.dim_vitals, self.seq_hidden_units)
            if self.has_vitals
            else None
        )

        self.outputs_input_transformation = nn.Linear(self.dim_outcome, self.seq_hidden_units)
        self.static_input_transformation = nn.Linear(self.dim_static_features, self.seq_hidden_units)

        self.n_inputs = 3 if self.has_vitals else 2

        attn_dropout = (
            self.dropout_rate
            if getattr(sub_args, "attn_dropout", False)
            else 0.0
        )

        self.transformer_blocks = nn.ModuleList(
            [
                self.basic_block_cls(
                    self.seq_hidden_units,
                    self.num_heads,
                    self.head_size,
                    self.seq_hidden_units * 4,
                    self.dropout_rate,
                    attn_dropout,
                    self_positional_encoding_k=self.self_positional_encoding_k,
                    self_positional_encoding_v=self.self_positional_encoding_v,
                    n_inputs=self.n_inputs,
                    disable_cross_attention=getattr(sub_args, "disable_cross_attention", False),
                    isolate_subnetwork=getattr(sub_args, "isolate_subnetwork", ""),
                )
                for _ in range(self.num_layer)
            ]
        )

        self.br_treatment_outcome_head = BRTreatmentOutcomeHead(
            self.seq_hidden_units,
            self.br_size,
            self.fc_hidden_units,
            self.dim_treatments,
            self.dim_outcome,
            self.alpha,
            self.update_alpha,
            self.balancing,
        )

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_multi:
            self.dataset_collection.process_data_multi()
        if self.bce_weights is None and self.hparams.exp.bce_weight:
            self._calculate_bce_weights()

    def forward(self, batch, detach_treatment=False):
        batch = dict(batch)
        fixed_split = batch["future_past_split"] if "future_past_split" in batch else None

        if self.training and self.augment_with_masked_vitals and self.has_vitals:
            if fixed_split is not None:
                raise ValueError("Masked-vitals augmentation is only valid for factual training batches.")

            batch_size = int(batch["active_entries"].shape[0])
            seq_lengths = batch["active_entries"].sum(1).long().squeeze(-1)
            fixed_split = torch.empty(
                2 * batch_size,
                dtype=torch.long,
                device=batch["active_entries"].device,
            )

            for i, seq_len in enumerate(seq_lengths):
                seq_len_i = int(seq_len.item())
                fixed_split[i] = seq_len_i
                fixed_split[batch_size + i] = int(torch.randint(
                    low=0,
                    high=seq_len_i + 1,
                    size=(1,),
                    device=batch["active_entries"].device,
                ).item())

            for key, value in list(batch.items()):
                if torch.is_tensor(value) and value.shape[0] == batch_size:
                    batch[key] = torch.cat((value, value), dim=0)
            batch["future_past_split"] = fixed_split

        prev_treatments = batch["prev_treatments"]
        vitals = batch["vitals"] if self.has_vitals else None
        prev_outputs = batch["prev_outputs"]
        static_features = batch["static_features"]
        curr_treatments = batch["current_treatments"]
        active_entries = batch["active_entries"]

        br = self.build_br(
            prev_treatments,
            vitals,
            prev_outputs,
            static_features,
            active_entries,
            fixed_split,
        )

        treatment_pred = self.br_treatment_outcome_head.build_treatment(br, detach_treatment)
        outcome_pred = self.br_treatment_outcome_head.build_outcome(br, curr_treatments)

        return treatment_pred, outcome_pred, br

    def build_br(self, prev_treatments, vitals, prev_outputs, static_features, active_entries, fixed_split):
        active_entries_treat_outcomes = torch.clone(active_entries)
        active_entries_vitals = torch.clone(active_entries)

        if fixed_split is not None and self.has_vitals:
            vitals = vitals.clone()

            for i in range(len(active_entries)):
                split_i = int(fixed_split[i])
                active_entries_vitals[i, split_i:, :] = 0.0
                vitals[i, split_i:] = 0.0

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
            x = (x_o + x_t) / 2.0
        else:
            if fixed_split is not None:
                x = torch.empty_like(x_o)

                for i in range(len(active_entries)):
                    split_i = int(fixed_split[i])

                    x[i, :split_i] = (
                        x_o[i, :split_i]
                        + x_t[i, :split_i]
                        + x_v[i, :split_i]
                    ) / 3.0

                    x[i, split_i:] = (
                        x_o[i, split_i:]
                        + x_t[i, split_i:]
                    ) / 2.0
            else:
                x = (x_o + x_t + x_v) / 3.0

        output = self.output_dropout(x)
        br = self.br_treatment_outcome_head.build_br(output)
        return br

    def get_autoregressive_predictions(self, dataset) -> np.array:
        logger.info(f"Autoregressive Prediction for {dataset.subset_name}.")

        predicted_outputs = np.zeros(
            (len(dataset), self.hparams.dataset.projection_horizon, self.dim_outcome),
            dtype=np.float32,
        )

        for t in range(self.hparams.dataset.projection_horizon + 1):
            logger.info(f"t = {t + 1}")
            outputs_scaled = self.get_predictions(dataset)

            for i in range(len(dataset)):
                split = int(dataset.data["future_past_split"][i])
                if t < self.hparams.dataset.projection_horizon:
                    dataset.data["prev_outputs"][i, split + t, :] = outputs_scaled[i, split - 1 + t, :]
                if t > 0:
                    predicted_outputs[i, t - 1, :] = outputs_scaled[i, split - 1 + t, :]

        return predicted_outputs
