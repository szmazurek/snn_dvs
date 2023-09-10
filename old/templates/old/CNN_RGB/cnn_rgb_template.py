from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from typing import Tuple
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
import os
import glob
import random

class CSNN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.conv_fc = nn.Sequential(
            layer.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            nn.ReLU(),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            nn.ReLU(),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            nn.ReLU(),
            layer.MaxPool2d(2, 2))

        self.lin_fc = nn.Sequential(
            layer.Flatten(),
            layer.Linear(4 * 50*18, channels * 4 * 4, bias=False),
            nn.ReLU(),

            layer.Linear(channels * 4 * 4, 2, bias=False),
            nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        x_seq = self.conv_fc(x)
        x_seq = self.lin_fc(x_seq)
        return x_seq


class PredictionRGBDataset(Dataset):
    def __init__(self, targ_dir: str) -> None:

        self.all_folders = [os.path.join(targ_dir, directory) for directory in os.listdir(targ_dir)]
        self.all_files = [glob.glob(os.path.join( folders, "*.jpg")) for folders in self.all_folders]
        self.all_files = [e for sub in self.all_files for e in sub]
        self.transform = transforms.Compose([
                transforms.Resize((400, 150)),
                transforms.RandomHorizontalFlip(),
                transforms.PILToTensor()])
        self.flip_random=0

    def load_image(self, image) -> Image.Image:
        img = Image.open(image)

        return self.transform(img)

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        file_path = self.all_files[index]
        image = self.load_image(file_path)
        label = int(os.path.splitext(os.path.basename(file_path))[0].split("-")[1])
        return image, label


def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    _datasets = {'train': Subset(dataset, train_idx), 'val': Subset(dataset, val_idx)}
    return _datasets


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


if __name__ == "__main__":

    wandb.init(
        project="CNN_RGB",
        entity="snn_team"
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = PredictionRGBDataset(r"/home/plgkrzysjed1/datasets/dataset_prediction_rgbH")
    dataset = train_val_dataset(dataset)

    train_data_loader = DataLoader(dataset["train"], batch_size=30, shuffle=True, num_workers=16)
    test_data_loader = DataLoader(dataset["val"], batch_size=30, shuffle=True, num_workers=16)

    net = CSNN(4)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.to(device)
    epochs = 80
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for epoch in range(0, epochs):
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        train_f1 = 0
        i = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()
            label = label.to(device)
            label_onehot = F.one_hot(label, 2).float()
            img =img.to(device).float()

            out_fr = net(img)  # [B, 2]
            pred = torch.argmax(out_fr, dim=1)

            l = nn.BCELoss()
            s = nn.Softmax(dim=1)

            loss = l(s(out_fr), label_onehot.float())

            loss.backward()
            optimizer.step()

            train_acc += (pred == label.int()).float().sum().item()
            train_samples += label.numel()
            train_loss += loss.item() * label.numel()

            train_f1 += f1_score(pred, label.int())
            i += 1

        train_f1 /= i
        train_loss /= train_samples
        train_acc /= train_samples
        print("Train", epoch, train_acc, train_f1)


        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        test_f1 = 0
        i = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                label = label.to(device)
                # label_shifted = torch.roll(label, shifts=-30, dims=1)
                label_onehot = F.one_hot(label, 2).float()

                # img = img.permute(1, 0, 2, 3, 4)  # [B, T, C, H, W] --> [T, B, C, H, W]
                img = img.to(device).float()

                out_fr = net(img)  # [T, B, 2]
                # out_fr = out_fr.unsqueeze(0).permute(2, 1, 0, 3)  # [1, T, B, 2] --> [B, T, 1, 2]
                pred = torch.argmax(out_fr, dim=1)

                l = nn.BCELoss()
                s = nn.Softmax(dim=1)

                loss = l(s(out_fr), label_onehot.float())

                test_acc += (pred == label.int()).float().sum().item()
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()

                test_f1 += f1_score(pred, label.int())
                i += 1
        test_f1 /= i
        test_loss /= test_samples
        test_acc /= test_samples
        print("Test", epoch, test_acc, test_f1)
        wandb.log({"train_acc": train_acc, "train_loss": train_loss, "train_f1": train_f1,
                   "test_acc": test_acc, "test_loss": test_loss, "test_f1": test_f1})

    wandb.finish()

