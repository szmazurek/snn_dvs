from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import os
import numpy as np


def unsqueeze_dim_if_missing(input_tensor: torch.Tensor) -> torch.Tensor:
    if input_tensor.dim() == 0:
        return input_tensor.unsqueeze(0)
    return input_tensor


def train_val_dataset(dataset, val_split=0.3):
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=val_split
    )
    _datasets = {
        "train": Subset(dataset, train_idx),
        "val": Subset(dataset, val_idx),
    }
    return _datasets


def check_labels(_label, _file_output):
    import numpy as np

    np.savetxt(_file_output, _label.flatten().cpu().numpy())


def f1_score(_pred, _target):
    pred = _pred.view(-1)
    target = _target.view(-1)
    tp = torch.sum((pred == 1) & (target == 1)).float()
    fp = torch.sum((pred == 1) & (target == 0)).float()
    fn = torch.sum((pred == 0) & (target == 1)).float()
    tn = torch.sum((pred == 0) & (target == 0)).float()
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    _f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return _f1.item()


def save_model(net, folder_path, file_name):
    checkpoint = {"net": net.state_dict()}
    torch.save(checkpoint, os.path.join(folder_path, file_name))


def perform_single_sample_time_dimension_forward_pass(
    model: nn.Module, sample: torch.Tensor
) -> torch.Tensor:
    """Perform forward pass over a temporal dimension in a single sample from a
    batch.
    Args:
        model (nn.Module): Model to perform forward pass on.
        sample (torch.Tensor): Sample from a batch of data, shape [T,C,H,W].
        T - Temporal dimension, C - Channels, H - Height, W - Width.
    Returns:
        result_tensor (torch.Tensor): Result of forward pass, shape [1,T,1].
    """
    result_tensor = torch.stack(
        [
            model(sample[t, :, :, :].unsqueeze(0))
            for t in range(sample.shape[0])
        ]
    )
    return result_tensor.permute(1, 0, 2)


def perform_forward_pass_on_full_batch(
    model: nn.Module, sample: torch.Tensor
) -> torch.Tensor:
    """Perform forward pass over batched data with temporal dimension
    using non-temporal model.
    Args:
        model (nn.Module): Model to perform forward pass on.
        sample (torch.Tensor): Batched data, shape [B,T,C,H,W].
    B - Batch size, T - Temporal dimension, C - Channels, H - Height,
    W - Width.
    Returns:
        result_tensor (torch.Tensor): Result of forward pass, shape [B,T,1].
    """
    result_tensor = torch.cat(
        [
            perform_single_sample_time_dimension_forward_pass(model, sample[i])
            for i in range(sample.shape[0])
        ]
    )
    return result_tensor


class EarlyStopping:
    """Credit to https://github.com/Bjarten/early-stopping-pytorch"""

    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model) -> None:
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        return None
