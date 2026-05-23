import logging

import torch

from clpfn.baselines.common.training import masked_mse_loss, masked_multiclass_ce_loss
from clpfn.baselines.models.time_varying_model import BRCausalModel, cfg_get
from clpfn.baselines.models.utils import BRTreatmentOutcomeHead
from clpfn.baselines.models.utils_lstm import VariationalLSTM

logger = logging.getLogger(__name__)


class CRN(BRCausalModel):
    """Reference-aligned Counterfactual Recurrent Network base class."""

    model_type = None
    possible_model_types = {"encoder", "decoder"}

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
        model_cfg = args.model
        self.alpha = float(cfg_get(model_cfg, "alpha", 1.0))
        self.update_alpha = bool(cfg_get(model_cfg, "update_alpha", False))
        self.balancing = str(cfg_get(model_cfg, "balancing", "grad_reverse"))
        self.treatment_loss_weight = float(cfg_get(model_cfg, "treatment_loss_weight", 1.0))

    def _init_specific(self, sub_args):
        self.br_size = int(sub_args.br_size)
        self.seq_hidden_units = int(sub_args.seq_hidden_units)
        self.fc_hidden_units = int(sub_args.fc_hidden_units)
        self.dropout_rate = float(sub_args.dropout_rate)
        self.num_layer = int(sub_args.num_layer)

        self.lstm = VariationalLSTM(
            self.input_size,
            self.seq_hidden_units,
            self.num_layer,
            self.dropout_rate,
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

    def build_br(self, prev_treatments, vitals_or_prev_outputs, static_features, init_states=None):
        x = torch.cat((prev_treatments, vitals_or_prev_outputs), dim=-1)
        x = torch.cat((x, static_features.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)
        x = self.lstm(x, init_states=init_states)
        return self.br_treatment_outcome_head.build_br(x)

    def training_step(self, batch, batch_ind=0):
        treatment_pred, outcome_pred, _ = self(batch)

        active = batch["active_entries"]
        outcome_loss = masked_mse_loss(outcome_pred, batch["outputs"], active)

        if "current_treatment_idx" in batch:
            treatment_idx = batch["current_treatment_idx"].long()
        else:
            treatment_idx = batch["current_treatments"].argmax(dim=-1).long()

        treatment_loss = masked_multiclass_ce_loss(treatment_pred, treatment_idx, active)

        loss = outcome_loss + self.treatment_loss_weight * treatment_loss
        self.log(f"{self.model_type}_train_loss", loss)
        self.log(f"{self.model_type}_train_mse_loss", outcome_loss)
        self.log(f"{self.model_type}_train_ce_loss", treatment_loss)
        return loss


class CRNEncoder(CRN):
    """CRN encoder: history -> balanced representation, treatment logits, outcome."""

    model_type = "encoder"

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
        logger.info("Input size of %s: %s", self.model_type, self.input_size)
        self._init_specific(args.model.encoder)
        self.save_hyperparameters(args)

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_encoder:
            self.dataset_collection.process_data_encoder()
        if self.bce_weights is None and self.hparams.exp.bce_weight:
            self._calculate_bce_weights()

    def forward(self, batch, detach_treatment=False):
        prev_treatments = batch["prev_treatments"]
        streams = []
        if self.has_vitals:
            streams.append(batch["vitals"])
        if self.autoregressive:
            streams.append(batch["prev_outputs"])
        vitals_or_prev_outputs = torch.cat(streams, dim=-1)
        br = self.build_br(prev_treatments, vitals_or_prev_outputs, batch["static_features"], init_states=None)
        treatment_pred = self.br_treatment_outcome_head.build_treatment(br, detach_treatment)
        outcome_pred = self.br_treatment_outcome_head.build_outcome(br, batch["current_treatments"])
        return treatment_pred, outcome_pred, br


class CRNDecoder(CRN):
    """CRN decoder initialized from an encoder balanced representation."""

    model_type = "decoder"

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
        self.encoder = encoder
        args.model.decoder.seq_hidden_units = self.encoder.br_size if encoder is not None else int(encoder_r_size)
        logger.info("Input size of %s: %s", self.model_type, self.input_size)
        self._init_specific(args.model.decoder)
        self.save_hyperparameters(args)

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_decoder:
            self.dataset_collection.process_data_decoder(self.encoder, save_encoder_r=True)
        if self.bce_weights is None and self.hparams.exp.bce_weight:
            self._calculate_bce_weights()

    def forward(self, batch, detach_treatment=False):
        br = self.build_br(
            batch["prev_treatments"],
            batch["prev_outputs"],
            batch["static_features"],
            init_states=batch["init_state"],
        )
        treatment_pred = self.br_treatment_outcome_head.build_treatment(br, detach_treatment)
        outcome_pred = self.br_treatment_outcome_head.build_outcome(br, batch["current_treatments"])
        return treatment_pred, outcome_pred, br
