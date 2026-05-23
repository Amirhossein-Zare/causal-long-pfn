import logging
from collections.abc import Mapping
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from clpfn.baselines.common.training import masked_mse_loss, masked_sequence_loss
from clpfn.baselines.models.utils import bce

logger = logging.getLogger(__name__)


def cfg_get(obj, key, default=None):
    if obj is None:
        return default

    if isinstance(obj, Mapping):
        return obj.get(key, default)

    if hasattr(obj, key):
        return getattr(obj, key)

    return default


def cfg_required(obj, key, owner):
    value = cfg_get(obj, key, None)
    if value is None:
        raise ValueError(f"{owner} requires '{key}'.")
    return value


class TimeVaryingCausalModel(nn.Module):
    """
    Base class for time-varying causal models.

    This provides the pieces used by GNet/RMSN/CRN ports:
      - hparams storage
      - model dimensions
      - optimizer construction
      - dataloader helpers
      - Lightning-style no-op log()

    Baselines are trained from scratch by adapter-level PyTorch loops on
    benchmark raw-pickle support data.
    """

    model_type = None
    possible_model_types = None
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
        super().__init__()

        self.dataset_collection = dataset_collection

        if dataset_collection is not None:
            self.autoregressive = self.dataset_collection.autoregressive
            self.has_vitals = self.dataset_collection.has_vitals
            self.bce_weights = None
        else:
            self.autoregressive = autoregressive
            self.has_vitals = has_vitals
            self.bce_weights = bce_weights

        model_cfg = args.model

        self.dim_treatments = cfg_required(model_cfg, "dim_treatments", "args.model")
        self.dim_vitals = cfg_required(model_cfg, "dim_vitals", "args.model")
        self.dim_static_features = cfg_required(model_cfg, "dim_static_features", "args.model")
        self.dim_outcome = cfg_required(model_cfg, "dim_outcomes", "args.model")

        self.input_size = None
        self.save_hyperparameters(args)

    def save_hyperparameters(self, args):
        self.hparams = args

    def log(self, *args, **kwargs):
        return None

    @property
    def device(self):
        return next(self.parameters()).device

    def move_batch_to_device(self, batch):
        if not isinstance(batch, Mapping):
            return batch
        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def _sub_args(self):
        return cfg_required(self.hparams.model, self.model_type, "args.model")

    def _get_optimizer(self, param_optimizer: list):
        no_decay = ["bias", "layer_norm"]
        sub_args = self._sub_args()
        optimizer_cfg = cfg_required(sub_args, "optimizer", f"args.model.{self.model_type}")

        weight_decay = cfg_required(optimizer_cfg, "weight_decay", f"args.model.{self.model_type}.optimizer")
        lr = cfg_required(optimizer_cfg, "learning_rate", f"args.model.{self.model_type}.optimizer")
        optimizer_cls = str(cfg_required(optimizer_cfg, "optimizer_cls", f"args.model.{self.model_type}.optimizer")).lower()

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        if optimizer_cls == "adamw":
            return optim.AdamW(optimizer_grouped_parameters, lr=lr)

        if optimizer_cls == "adam":
            return optim.Adam(optimizer_grouped_parameters, lr=lr)

        if optimizer_cls == "sgd":
            momentum = cfg_required(optimizer_cfg, "momentum", f"args.model.{self.model_type}.optimizer")
            return optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=momentum)

        raise NotImplementedError(f"Unknown optimizer_cls={optimizer_cls}")

    def _get_lr_schedulers(self, optimizer):
        if not isinstance(optimizer, list):
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            return [optimizer], [lr_scheduler]

        lr_schedulers = []
        for opt in optimizer:
            lr_schedulers.append(optim.lr_scheduler.ExponentialLR(opt, gamma=0.99))

        return optimizer, lr_schedulers

    def configure_optimizers(self):
        optimizer = self._get_optimizer(list(self.named_parameters()))
        sub_args = self._sub_args()
        optimizer_cfg = cfg_required(sub_args, "optimizer", f"args.model.{self.model_type}")
        use_scheduler = cfg_required(optimizer_cfg, "lr_scheduler", f"args.model.{self.model_type}.optimizer")

        if use_scheduler:
            return self._get_lr_schedulers(optimizer)

        return optimizer

    def bce_loss(self, treatment_pred, current_treatments, kind="predict"):
        mode = cfg_get(self.hparams.dataset, "treatment_mode")
        use_bce_weight = cfg_get(self.hparams.exp, "bce_weight", False)

        bce_weights = (
            torch.tensor(self.bce_weights).type_as(current_treatments)
            if use_bce_weight
            else None
        )

        if kind == "predict":
            return bce(treatment_pred, current_treatments, mode, bce_weights)

        if kind == "confuse":
            uniform_treatments = torch.ones_like(current_treatments)

            if mode == "multiclass":
                uniform_treatments *= 1 / current_treatments.shape[-1]
            elif mode == "multilabel":
                uniform_treatments *= 0.5
            else:
                raise NotImplementedError()

            return bce(treatment_pred, uniform_treatments, mode)

        raise NotImplementedError()

    def get_predictions(self, dataset: Dataset) -> np.array:
        raise NotImplementedError()

    def get_propensity_scores(self, dataset: Dataset) -> np.array:
        raise NotImplementedError()

    def get_representations(self, dataset: Dataset) -> np.array:
        raise NotImplementedError()

    def get_autoregressive_predictions(self, dataset: Dataset) -> np.array:
        raise NotImplementedError()

    def get_masked_bce(self, dataset: Dataset):
        logger.info(f"Masked BCE for {dataset.subset_name}.")
        treatment_pred = self.get_propensity_scores(dataset)
        current_treatments = dataset.data["current_treatments"]
        active_entries = dataset.data["active_entries"]
        bce_loss = bce(
            torch.tensor(treatment_pred),
            torch.tensor(current_treatments),
            cfg_get(self.hparams.dataset, "treatment_mode"),
        ).numpy()
        return (bce_loss * active_entries.squeeze(-1)).sum() / active_entries.sum()

    def get_normalised_masked_rmse(self, dataset: Dataset, one_step_counterfactual=False):
        logger.info(f"Normalised RMSE for {dataset.subset_name}.")
        outcome_pred = self.get_predictions(dataset)
        outputs = dataset.data["outputs"]
        active_entries = dataset.data["active_entries"]
        if one_step_counterfactual:
            active_entries = dataset.data.get("active_entries", active_entries)
        mse = (active_entries * (outcome_pred - outputs) ** 2).sum() / active_entries.sum()
        return float(np.sqrt(mse))

    def get_normalised_n_step_rmses(self, dataset: Dataset, datasets_mc: list[Dataset] = None):
        logger.info(f"Normalised n-step RMSE for {dataset.subset_name}.")
        outcome_pred = self.get_autoregressive_predictions(dataset if datasets_mc is None else datasets_mc)
        outputs = dataset.data["outputs"][:, :outcome_pred.shape[1]]
        active_entries = dataset.data["active_entries"][:, :outcome_pred.shape[1]]
        mse = (active_entries * (outcome_pred - outputs) ** 2).sum(axis=(0, 2)) / active_entries.sum(axis=(0, 2))
        return np.sqrt(mse)

class BRCausalModel(TimeVaryingCausalModel):
    """
    Balanced-representation causal sequence model base.
    """

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
        self.treatment_loss_weight = float(cfg_get(args.model, "treatment_loss_weight", 1.0))

    def _calculate_bce_weights(self) -> None:
        if self.dataset_collection is None:
            raise ValueError("dataset_collection is required to calculate BCE weights.")
        current_treatments = self.dataset_collection.train_f.data["current_treatments"]
        active_entries = self.dataset_collection.train_f.data["active_entries"]
        mode = cfg_get(self.hparams.dataset, "treatment_mode")
        if mode == "multilabel":
            counts = (current_treatments * active_entries).sum(axis=(0, 1))
            totals = active_entries.sum(axis=(0, 1))
            self.bce_weights = (totals - counts) / np.maximum(counts, 1.0)
        elif mode == "multiclass":
            labels = current_treatments.argmax(axis=-1)
            weights = []
            for treatment_idx in range(self.dim_treatments):
                positives = ((labels == treatment_idx) * active_entries.squeeze(-1)).sum()
                weights.append(active_entries.sum() / max(float(positives), 1.0))
            self.bce_weights = np.asarray(weights, dtype=np.float32)
        else:
            raise NotImplementedError()

    def training_step(self, batch, batch_ind, optimizer_idx=0):
        if getattr(self, "balancing", "grad_reverse") != "grad_reverse":
            raise NotImplementedError("CLPFN baseline ports use the G_transformer grad_reverse training path.")

        treatment_pred, outcome_pred, _ = self(batch)

        mse_loss = masked_mse_loss(outcome_pred, batch["outputs"], batch["active_entries"])

        treatment_loss = self.bce_loss(
            treatment_pred,
            batch["current_treatments"].float(),
            kind="predict",
        )
        treatment_loss = masked_sequence_loss(treatment_loss, batch["active_entries"])

        loss = mse_loss + self.treatment_loss_weight * treatment_loss

        self.log(f"{self.model_type}_train_loss", loss)
        self.log(f"{self.model_type}_train_bce_loss", treatment_loss)
        self.log(f"{self.model_type}_train_mse_loss", mse_loss)
        return loss

    def test_step(self, batch, batch_ind, **kwargs):
        batch = self.move_batch_to_device(batch)
        treatment_pred, outcome_pred, br = self(batch)
        return {
            "treatment_pred": treatment_pred.cpu(),
            "outcome_pred": outcome_pred.cpu(),
            "br": br.cpu(),
        }

    def predict_step(self, batch, batch_idx, dataset_idx=None):
        batch = self.move_batch_to_device(batch)
        treatment_pred, outcome_pred, br = self(batch)
        return treatment_pred.cpu(), outcome_pred.cpu(), br.cpu()

    def get_representations(self, dataset: Dataset) -> np.array:
        logger.info(f"Representations for {dataset.subset_name}.")
        self.eval()
        loader = DataLoader(dataset, batch_size=cfg_get(self.hparams.dataset, "val_batch_size", 64))
        reps = []
        with torch.no_grad():
            for batch in loader:
                batch = self.move_batch_to_device(batch)
                _, _, br = self.predict_step(batch, 0)
                reps.append(br.numpy())
        return np.concatenate(reps, axis=0)

    def get_predictions(self, dataset: Dataset) -> np.array:
        logger.info(f"Predictions for {dataset.subset_name}.")
        self.eval()
        loader = DataLoader(dataset, batch_size=cfg_get(self.hparams.dataset, "val_batch_size", 64))
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = self.move_batch_to_device(batch)
                _, outcome_pred, _ = self.predict_step(batch, 0)
                preds.append(outcome_pred.numpy())
        return np.concatenate(preds, axis=0)
