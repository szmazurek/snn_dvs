import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
import lightning.pytorch as pl
from torchmetrics import Accuracy, F1Score, Recall, Precision, AUROC
from spikingjelly.activation_based import (
    neuron,
    functional,
    surrogate,
)
from .models import slow_r50, Resnet18, Resnet18_spiking
from utils import (
    unsqueeze_dim_if_missing,
    perform_forward_pass_on_temporal_batch,
)
from typing import List, Dict, Any, Callable
from warnings import warn


class LightningModuleNonTemporal(pl.LightningModule):
    ACCEPTED_MODELS: Dict[str, Callable[..., nn.Module]] = {
        "resnet18": Resnet18,
        "resnet18_spiking": Resnet18_spiking,
    }
    ACCEPTED_NEURON_MODELS: Dict[str, neuron.BaseNode] = {
        "lif": neuron.LIFNode,
        "plif": neuron.ParametricLIFNode,
        "eif": neuron.EIFNode,
        "izhikevich": neuron.IzhikevichNode,
        "klif": neuron.KLIFNode,
        "liaf": neuron.LIAFNode,
    }
    ACCEPTED_SURROGATE_FUNCTIONS: Dict[
        str, surrogate.SurrogateFunctionBase
    ] = {
        "sigmoid": surrogate.Sigmoid,
        "atan": surrogate.ATan,
    }
    ACCEPTED_BACKENDS = {
        "cupy": "cupy",
        "torch": "torch",
    }
    ACCEPTED_STEP_MODES = {
        "single_step": "s",
        "multi_step": "m",
    }

    def __init__(
        self,
        model_name: str,
        pos_weight: float,
        dvs_mode: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        **kwargs,
    ):
        """
        Initializes the LightningModule.
        Args:
            model_name (str): Name of the model to use (slow_r50, resnet18, resnet18_spiking).
            pos_weight (float): Weight of the positive class.
            dvs_mode (bool): Whether to use DVS mode or not.
            lr (float): Learning rate.
            weight_decay (float): Weight decay.
            **kwargs: Keyword arguments for the Resnet18_spiking model.
        """
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.full([1], pos_weight)
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.model_name = model_name.lower()
        self.dvs_mode = dvs_mode
        # if kwargs are passed, parse them
        if kwargs:
            self._parse_kwargs(kwargs)
        # assign model
        self.model = self._assign_model()
        # For train metrics
        self.train_predictions: List[float] = []
        self.train_ground_truth: List[float] = []
        # For valid metrics
        self.val_predictions: List[float] = []
        self.val_ground_truth: List[float] = []
        # For test metrics
        self.test_predictions: List[float] = []
        self.test_ground_truth: List[float] = []
        # metrics
        self.metrics_calculators = {
            "accuracy": Accuracy("binary"),
            "f1": F1Score("binary"),
            "recall": Recall("binary"),
            "precision": Precision("binary"),
            "auroc": AUROC("binary"),
        }

    def _assign_model(
        self,
    ) -> nn.Module:
        module = self.ACCEPTED_MODELS[self.model_name]
        if self.model_name == "resnet18_spiking":
            self._ensure_spiking_attrs()
            model = module(
                neuron_model=self.neuron_model,
                surrogate_function=self.surrogate_function,
                dvs_mode=self.dvs_mode,
            )
            functional.set_step_mode(model, self.step_mode)
            functional.set_backend(model, self.backend)
            return model
        model = module(dvs_mode=self.dvs_mode)
        return model

    def _calculate_metrics(
        self, stage: str ,predictions: List[float], ground_truth: List[float]
    ) -> Dict[str, float]:
        """
        Calculates the metrics for the given predictions and ground truth.
        Args:
            stage (str): Stage of the model (train, val, test).
            predictions (List[float]): List of predictions.
            ground_truth (List[float]): List of ground truth.
        Returns:
            Dict[str, float]: Dictionary with the metrics.
        """
        assert stage in ["train", "val", "test"], (
            f"Stage {stage} not supported. "
            f"Choose one of ['train', 'val', 'test']"
        )
        metrics = {}
        for name, calculator in self.metrics_calculators.items():
            metrics[f"{stage}_{name}"] = calculator(
                torch.tensor(predictions), torch.tensor(ground_truth)
            ).item()
        return metrics

    def _parse_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """
        Parses the keyword arguments and sets the corresponding attributes.
        """
        for key, value in kwargs["kwargs"].items():
            value = value.lower()
            if key == "neuron_model":
                assert value in self.ACCEPTED_NEURON_MODELS.keys(), (
                    f"Neuron model {value} not supported. "
                    f"Choose one of {self.ACCEPTED_NEURON_MODELS.keys()}"
                )
                self.neuron_model = self.ACCEPTED_NEURON_MODELS[value]
            elif key == "surrogate_function":
                assert value in self.ACCEPTED_SURROGATE_FUNCTIONS.keys(), (
                    f"Surrogate function {value} not supported. "
                    f"Choose one of {self.ACCEPTED_SURROGATE_FUNCTIONS.keys()}"
                )
                self.surrogate_function = self.ACCEPTED_SURROGATE_FUNCTIONS[
                    value
                ]
            elif key == "backend":
                assert value in self.ACCEPTED_BACKENDS.keys(), (
                    f"Backend {value} not supported. "
                    f"Choose one of {self.ACCEPTED_BACKENDS.keys()}"
                )
                self.backend = self.ACCEPTED_BACKENDS[value]
            elif key == "step_mode":
                assert value in self.ACCEPTED_STEP_MODES.keys(), (
                    f"Step mode {value} not supported. "
                    f"Choose one of {self.ACCEPTED_STEP_MODES.keys()}"
                )
                self.step_mode = self.ACCEPTED_STEP_MODES[value]
            else:
                warn(
                    f"Unknown keyword argument {key} with value {value}",
                    UserWarning,
                )

    def _ensure_spiking_attrs(self) -> None:
        """
        Ensures that the necessary attributes for spiking models are set.
        """
        if not hasattr(self, "neuron_model"):
            warn(
                "No neuron model specified. Using LIFNode as default.",
                UserWarning,
            )
            self.neuron_model = neuron.LIFNode
        if not hasattr(self, "surrogate_function"):
            warn(
                "No surrogate function specified. Using Sigmoid as default.",
                UserWarning,
            )
            self.surrogate_function = surrogate.Sigmoid
        if not hasattr(self, "backend"):
            warn(
                "No backend specified. Using torch as default.",
                UserWarning,
            )
            self.backend = "torch"
        if not hasattr(self, "step_mode"):
            warn(
                """No step mode specified. Using single_step as default.
                This may crash for temporal models.""",
                UserWarning,
            )
            self.step_mode = "s"

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        """
        out = self.model(x).squeeze(1)
        if self.model_name == "resnet18_spiking":
            functional.reset_net(self.model)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        self.train_predictions.extend(y_hat.detach().cpu().tolist())
        self.train_ground_truth.extend(y.squeeze().tolist())
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.val_predictions.extend(y_hat.detach().cpu().tolist())
        self.val_ground_truth.extend(y.squeeze().tolist())
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,

        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        y_hat = self.forward(x)
        
        loss = self.loss(y_hat, y)
        self.test_predictions.extend(y_hat.detach().cpu().tolist())
        self.test_ground_truth.extend(y.squeeze().tolist())
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_end(self):
        metrics = self._calculate_metrics("train",
            self.train_predictions, self.train_ground_truth
        )
        self.log_dict(metrics, sync_dist=True,on_epoch=True,on_step=False,prog_bar=True, logger=True)
        self.train_predictions.clear()
        self.train_ground_truth.clear()

    def on_validation_epoch_end(self):
        metrics = self._calculate_metrics("val",
            self.val_predictions, self.val_ground_truth
        )
        self.log_dict(metrics,sync_dist=True,on_epoch=True,on_step=False, prog_bar=True, logger=True)
        self.val_predictions.clear()
        self.val_ground_truth.clear()

    def on_test_epoch_end(self):
        metrics = self._calculate_metrics("test",
            self.test_predictions, self.test_ground_truth
        ) 
        self.log_dict(metrics,sync_dist=True,on_epoch=True,on_step=False, prog_bar=True, logger=True)
        self.test_predictions.clear()
        self.test_ground_truth.clear()


class LightningModuleTemporalNets(LightningModuleNonTemporal):
    def __init__(
        self,
        model_name: str,
        pos_weight: float,
        dvs_mode: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        **kwargs,
    ):
        self.ACCEPTED_MODELS["slow_r50"] = slow_r50
        super().__init__(
            model_name, pos_weight, dvs_mode, lr, weight_decay, **kwargs
        )
        

    def forward_spiking(self, x: Tensor) -> Tensor:
        """
        Forward pass for spiking temporal model.
        """
        out = self.model(x.permute(1, 0, 2, 3, 4))
        out_averaged = out.squeeze(2).mean(0)
        out_processed = unsqueeze_dim_if_missing(out_averaged)
        functional.reset_net(self.model)
        return out_processed

    def forward_pseudotemporal_model(self, x: Tensor) -> Tensor:
        """
        Forward pass for adapting standard models for temporal processing.
        """
        out = (
            (perform_forward_pass_on_temporal_batch(self.model, x))
            .squeeze(2)
            .mean(1)
        )

        out_processed = unsqueeze_dim_if_missing(out)
        return out_processed

    def forward_slowr50(self, x: Tensor) -> Tensor:
        """
        Forward pass for slow_r50 models.
        """
        out = self.model(x.permute(0, 2, 1, 3, 4)).squeeze(1)
        out_processed = unsqueeze_dim_if_missing(out)
        return out_processed

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        """
        print(x.shape)
        if self.model_name == "resnet18_spiking":
            return self.forward_spiking(x)
        elif self.model_name == "slow_r50":
            return self.forward_slowr50(x)
        else:
            return self.forward_pseudotemporal_model(x)
