from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import os
import numpy as np
import random
from sklearn.model_selection import StratifiedShuffleSplit


def set_random_seeds(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return None


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


def assign_label(tensor: torch.Tensor) -> float:
    """Detect if in the given tensor of numbers there occurs a situation at any place when the 1 is followed by 0.
    If yes, assigns label 100, otherwise sum of the numbers in tensor.
    """
    assert tensor.dim() == 1, "Tensor must be 1-dimensional"
    assert torch.all(
        tensor == 0 | (tensor == 1)
    ), "Tensor must contain only 0s or 1s"
    nonzero = tensor.nonzero()
    if nonzero.numel() == 0:
        return 0  # if negative sample, return 0
    if len(tensor) - 1 != nonzero[-1]:
        # return -1 # if ones occur, but there exists any 0 after them, return 1
        return -tensor.sum().item()
    recursive_difference_tensor = torch.diff(nonzero, dim=0)
    if torch.any(recursive_difference_tensor != 1):
        return (
            100  # if ones occur, but there exists any 0 between them, return 1
        )
    return tensor.sum().item()


def generate_extended_labels_from_sample(set_of_frame_filenames):
    per_frame_labels = torch.tensor(
        [
            int(os.path.basename(frame).split(".")[0].split("-")[1])
            for frame in set_of_frame_filenames
        ]
    )
    per_frame_label = assign_label(per_frame_labels)
    return per_frame_label


def train_val_test_split(full_temporal_dataset, val_size, test_size, seed):
    """Perform train-val-test split on a temporal dataset, taking into
    account stratification based on both main labels and "extended" labels.
    """
    normal_labels = torch.tensor(full_temporal_dataset.all_labels)
    all_sample_names = full_temporal_dataset.all_samples
    extended_labels = []
    for sample in all_sample_names:
        extended_labels.append(generate_extended_labels_from_sample(sample))
    # class weights
    neg_count, pos_count = torch.unique(normal_labels, return_counts=True)[1]
    pos_weight = neg_count / pos_count

    extended_labels = torch.tensor(extended_labels).unsqueeze(1)
    normal_labels = normal_labels.unsqueeze(1)
    data_indices = torch.arange(len(normal_labels))
    # performing split for train and test w.r.t both extended and normal labels
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    merged_labels = torch.cat([normal_labels, extended_labels], axis=1)
    train_idx, test_idx = next(splitter.split(data_indices, merged_labels))
    test_dataset = Subset(full_temporal_dataset, test_idx)
    # further split the training set into train and val subsets
    new_train_indices = data_indices[train_idx]
    new_merged_labels = merged_labels[train_idx]
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_size, random_state=seed
    )
    train_new_idx, val_idx = next(
        splitter.split(new_train_indices, new_merged_labels)
    )

    proper_train_idx = new_train_indices[train_new_idx]
    proper_val_idx = new_train_indices[val_idx]
    train_dataset = Subset(full_temporal_dataset, proper_train_idx)
    val_dataset = Subset(full_temporal_dataset, proper_val_idx)
    for ind in proper_train_idx:
        if ind in proper_val_idx:
            print("Leak into the val set")
        if ind in test_idx:
            print("Leak into test set")
        if ind not in data_indices:
            print("Sth is wrong")

    return train_dataset, val_dataset, test_dataset, pos_weight


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


def perform_forward_pass_on_temporal_batch(
    model: nn.Module, sample: torch.Tensor
) -> torch.Tensor:
    """Perform forward pass over batched data with temporal dimension
    using normal model.
    Args:
        model (nn.Module): Model to perform forward pass on.
        sample (torch.Tensor): Batched data, shape [B,T,C,H,W].
    B - Batch size, T - Temporal dimension, C - Channels, H - Height,
    W - Width.
    Returns:
        result_tensor (torch.Tensor): Result of forward pass, shape [T,B,1].
    """
    result_tensor = [
        model(sample[:, i, :, :, :]) for i in range(sample.shape[1])
    ]
    return torch.stack(result_tensor).permute(1, 0, 2)


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
