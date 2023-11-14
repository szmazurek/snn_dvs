from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import os


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
