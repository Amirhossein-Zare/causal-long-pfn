import logging
from collections.abc import Mapping
import torch
from torch import nn
import torch.nn.functional as F

from clpfn.baselines.models.time_varying_model import BRCausalModel
from clpfn.baselines.models.utils import BRTreatmentOutcomeHead
from clpfn.baselines.models.utils_transformer import (
    AbsolutePositionalEncoding,
    RelativePositionalEncoding,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
)

logger = logging.getLogger(__name__)


def cfg_get(obj, key, default=None):
    if obj is None:
        return default

    if isinstance(obj, Mapping):
        return obj.get(key, default)

    if hasattr(obj, key):
        return getattr(obj, key)

    return default


class EDCT(BRCausalModel):
    """
    Dependency-light EDCT subset used as the parent for CT.

    This preserves the CT-relevant common transformer attributes:
      - seq_hidden_units
      - br_size
      - fc_hidden_units
      - dropout_rate
      - num_layer
      - num_heads
      - head_size
      - balancing / alpha fields
      - optional self positional encoding objects
      - output_dropout
    """

    model_type = None
    possible_model_types = None
    tuning_criterion = "rmse"

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
        self.basic_block_cls = None

    def _init_specific(self, sub_args):
        self.max_seq_length = cfg_get(sub_args, "max_seq_length", 65)
        self.seq_hidden_units = cfg_get(sub_args, "seq_hidden_units")
        self.br_size = cfg_get(sub_args, "br_size")
        self.fc_hidden_units = cfg_get(sub_args, "fc_hidden_units")
        self.dropout_rate = cfg_get(sub_args, "dropout_rate")
        self.num_layer = cfg_get(sub_args, "num_layer")
        self.num_heads = cfg_get(sub_args, "num_heads")
        self.head_size = cfg_get(sub_args, "head_size", None)
        if self.head_size is None and self.seq_hidden_units is not None and self.num_heads:
            self.head_size = int(self.seq_hidden_units) // int(self.num_heads)

        self.alpha = cfg_get(sub_args, "alpha", 0.0)
        self.update_alpha = cfg_get(sub_args, "update_alpha", False)
        self.balancing = cfg_get(sub_args, "balancing", "grad_reverse")

        self.treatment_loss_weight = cfg_get(sub_args, "treatment_loss_weight", 1.0)
        self.augment_with_masked_vitals = cfg_get(sub_args, "augment_with_masked_vitals", False)

        if (
            self.seq_hidden_units is None
            or self.br_size is None
            or self.fc_hidden_units is None
            or self.dropout_rate is None
        ):
            raise ValueError(f"{self.model_type} mandatory hyperparameters are missing.")

        self.input_transformation = nn.Linear(self.input_size, self.seq_hidden_units) if self.input_size else None

        self.self_positional_encoding = None
        self.self_positional_encoding_k = None
        self.self_positional_encoding_v = None
        self.cross_positional_encoding = None
        self.cross_positional_encoding_k = None
        self.cross_positional_encoding_v = None

        self_positional_encoding = cfg_get(sub_args, "self_positional_encoding", "none")
        self_pe_absolute = cfg_get(self_positional_encoding, "absolute", None)
        self_pe_trainable = bool(cfg_get(self_positional_encoding, "trainable", False))
        self_pe_max_relative = int(cfg_get(self_positional_encoding, "max_relative_position", self.max_seq_length))

        if self_pe_absolute is None:
            self_pe_absolute = self_positional_encoding == "absolute"

        if self_pe_absolute:
            self.self_positional_encoding = AbsolutePositionalEncoding(
                max_len=int(self.max_seq_length),
                d_model=int(self.seq_hidden_units),
                trainable=self_pe_trainable or bool(cfg_get(sub_args, "trainable_positional_encoding", False)),
            )
        elif self_positional_encoding == "relative" or cfg_get(self_positional_encoding, "absolute", None) is not None:
            trainable_rel = self_pe_trainable or bool(cfg_get(sub_args, "trainable_positional_encoding", False))
            self.self_positional_encoding_k = RelativePositionalEncoding(
                max_relative_position=self_pe_max_relative,
                d_model=int(self.head_size),
                trainable=trainable_rel,
                cross_attn=False,
            )
            self.self_positional_encoding_v = RelativePositionalEncoding(
                max_relative_position=self_pe_max_relative,
                d_model=int(self.head_size),
                trainable=trainable_rel,
                cross_attn=False,
            )

        cross_positional_encoding = cfg_get(sub_args, "cross_positional_encoding", None)
        if cross_positional_encoding is not None:
            cross_pe_absolute = bool(cfg_get(cross_positional_encoding, "absolute", False))
            cross_pe_trainable = bool(cfg_get(cross_positional_encoding, "trainable", False))
            cross_pe_max_relative = int(
                cfg_get(cross_positional_encoding, "max_relative_position", self.max_seq_length)
            )
            if cross_pe_absolute:
                self.cross_positional_encoding = AbsolutePositionalEncoding(
                    max_len=int(self.max_seq_length),
                    d_model=int(self.seq_hidden_units),
                    trainable=cross_pe_trainable,
                )
            else:
                self.cross_positional_encoding_k = RelativePositionalEncoding(
                    max_relative_position=cross_pe_max_relative,
                    d_model=int(self.head_size),
                    trainable=cross_pe_trainable,
                    cross_attn=True,
                )
                self.cross_positional_encoding_v = RelativePositionalEncoding(
                    max_relative_position=cross_pe_max_relative,
                    d_model=int(self.head_size),
                    trainable=cross_pe_trainable,
                    cross_attn=True,
                )

        if self.basic_block_cls is not None:
            self.transformer_blocks = nn.ModuleList(
                [
                    self.basic_block_cls(
                        self.seq_hidden_units,
                        self.num_heads,
                        self.head_size,
                        self.seq_hidden_units * 4,
                        self.dropout_rate,
                        self.dropout_rate,
                        self_positional_encoding_k=self.self_positional_encoding_k,
                        self_positional_encoding_v=self.self_positional_encoding_v,
                        cross_positional_encoding_k=self.cross_positional_encoding_k,
                        cross_positional_encoding_v=self.cross_positional_encoding_v,
                    )
                    for _ in range(self.num_layer)
                ]
            )

        self.output_dropout = torch.nn.Dropout(float(self.dropout_rate))
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

    def build_br(
        self,
        prev_treatments,
        vitals_or_prev_outputs,
        static_features,
        active_entries,
        encoder_br=None,
        active_encoder_br=None,
    ):
        x = torch.cat((prev_treatments, vitals_or_prev_outputs), dim=-1)
        x = torch.cat((x, static_features.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)
        x = self.input_transformation(x)

        if active_encoder_br is None and encoder_br is None:
            for block in self.transformer_blocks:
                if self.self_positional_encoding is not None:
                    x = x + self.self_positional_encoding(x)
                x = block(x, active_entries)
        else:
            assert x.shape[-1] == encoder_br.shape[-1]
            for block in self.transformer_blocks:
                if self.cross_positional_encoding is not None:
                    encoder_br = encoder_br + self.cross_positional_encoding(encoder_br)
                if self.self_positional_encoding is not None:
                    x = x + self.self_positional_encoding(x)
                x = block(x, encoder_br, active_entries, active_encoder_br)

        output = self.output_dropout(x)
        br = self.br_treatment_outcome_head.build_br(output)
        return br

    def training_step(self, batch, batch_ind=0):
        treatment_pred, outcome_pred, _ = self(batch)

        outcome_mse_loss = F.mse_loss(
            outcome_pred,
            batch["outputs"],
            reduction="none",
        )

        mse_loss = (
            batch["active_entries"] * outcome_mse_loss
        ).sum() / batch["active_entries"].sum().clamp(min=1.0)

        treatment_loss = self.bce_loss(
            treatment_pred,
            batch["current_treatments"].float(),
            kind="predict",
        )

        treatment_loss = (
            batch["active_entries"].squeeze(-1) * treatment_loss
        ).sum() / batch["active_entries"].sum().clamp(min=1.0)

        loss = mse_loss + float(self.treatment_loss_weight) * treatment_loss

        self.log(f"{self.model_type}_mse_loss", mse_loss)
        self.log(f"{self.model_type}_treatment_loss", treatment_loss)
        self.log(f"{self.model_type}_loss", loss)

        return loss

    def predict_step(self, batch, batch_ind=0, dataset_idx=None):
        batch = self.move_batch_to_device(batch)
        treatment_pred, outcome_pred, br = self(batch)
        return treatment_pred.cpu(), outcome_pred.cpu(), br.cpu()

class EDCTEncoder(EDCT):
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
        logger.info(f"Input size of {self.model_type}: {self.input_size}")

        self.basic_block_cls = TransformerEncoderBlock
        self._init_specific(args.model.encoder)
        self.save_hyperparameters(args)

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_encoder:
            self.dataset_collection.process_data_encoder()
        if self.bce_weights is None and self.hparams.exp.bce_weight:
            self._calculate_bce_weights()

    def forward(self, batch, detach_treatment=False):
        prev_treatments = batch["prev_treatments"]
        vitals_or_prev_outputs = []
        if self.has_vitals:
            vitals_or_prev_outputs.append(batch["vitals"])
        if self.autoregressive:
            vitals_or_prev_outputs.append(batch["prev_outputs"])
        vitals_or_prev_outputs = torch.cat(vitals_or_prev_outputs, dim=-1)
        static_features = batch["static_features"]
        curr_treatments = batch["current_treatments"]
        active_entries = batch["active_entries"]

        br = self.build_br(prev_treatments, vitals_or_prev_outputs, static_features, active_entries)
        treatment_pred = self.br_treatment_outcome_head.build_treatment(br, detach_treatment)
        outcome_pred = self.br_treatment_outcome_head.build_outcome(br, curr_treatments)
        return treatment_pred, outcome_pred, br


class EDCTDecoder(EDCT):
    model_type = "decoder"

    def __init__(
        self,
        args,
        encoder: EDCTEncoder = None,
        dataset_collection=None,
        encoder_r_size=None,
        autoregressive=None,
        has_vitals=None,
        bce_weights=None,
        **kwargs,
    ):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)
        self.basic_block_cls = TransformerDecoderBlock

        self.input_size = self.dim_treatments + self.dim_static_features + self.dim_outcome
        logger.info(f"Input size of {self.model_type}: {self.input_size}")

        self.encoder = encoder
        args.model.decoder.seq_hidden_units = self.encoder.br_size if encoder is not None else encoder_r_size
        self._init_specific(args.model.decoder)
        self.save_hyperparameters(args)

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_decoder:
            self.dataset_collection.process_data_decoder(self.encoder, save_encoder_r=True)
        if self.bce_weights is None and self.hparams.exp.bce_weight:
            self._calculate_bce_weights()

    def forward(self, batch, detach_treatment=False):
        prev_treatments = batch["prev_treatments"]
        vitals_or_prev_outputs = batch["prev_outputs"]
        static_features = batch["static_features"]
        curr_treatments = batch["current_treatments"]
        encoder_br = batch["encoder_r"]
        active_entries = batch["active_entries"]
        active_encoder_br = batch["active_encoder_r"]

        br = self.build_br(
            prev_treatments,
            vitals_or_prev_outputs,
            static_features,
            active_entries,
            encoder_br,
            active_encoder_br,
        )
        treatment_pred = self.br_treatment_outcome_head.build_treatment(br, detach_treatment)
        outcome_pred = self.br_treatment_outcome_head.build_outcome(br, curr_treatments)
        return treatment_pred, outcome_pred, br
