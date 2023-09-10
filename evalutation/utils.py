from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import torch
import os


def train_val_dataset(dataset, val_split=0.3):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    _datasets = {'train': Subset(dataset, train_idx), 'val': Subset(dataset, val_idx)}
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
    checkpoint = {'net': net.state_dict()}
    torch.save(checkpoint, os.path.join(folder_path, file_name))
