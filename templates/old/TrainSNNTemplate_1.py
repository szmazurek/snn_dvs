from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from typing import Tuple
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
import os
import time


class SNN(nn.Module):
    def __init__(self, T: int, channels: int):
        super().__init__()
        self.T = T
        self.conv_enkoder = nn.Sequential(
            layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2))

        self.conv_fc = nn.Sequential(
            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2))

        self.lin_fc = nn.Sequential(
            layer.Flatten(),
            layer.Linear(channels * 9 * 25, channels * 4 * 4, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),

            layer.Linear(channels * 4 * 4, 2, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),)

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        x_seq = self.conv_enkoder(x_seq)
        x_seq = self.conv_fc(x_seq)
        x_seq = self.lin_fc(x_seq)
        fr = x_seq.mean(0)
        return fr

    def spiking_encoder(self):
        return self.conv_fc[0:3]


class SmallSampleDvsSeq(Dataset):

    def __init__(self, targ_dir: str) -> None:

        all_folder = [os.path.join(targ_dir, directory) for directory in os.listdir(targ_dir)]
        self.paths = []
        for folder in all_folder:
            self.paths.extend(list(pathlib.Path(folder).glob("*.png")))

        self.transform = transforms.Compose([
                transforms.Resize((600, 1600)),
                transforms.PILToTensor()])

    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self.load_image(index)
        class_name = int(str(self.paths[index])[-5])
        return self.transform(img), class_name


def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    _datasets = {'train': Subset(dataset, train_idx), 'val': Subset(dataset, val_idx)}
    return _datasets


if __name__ == "__main__":

    T = 2  # number copies per image
    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset =SmallSampleDvsSeq(r"D:\ProjectsPython\SpikingJelly\small-sample\image")
    dataset= train_val_dataset(dataset)
    train_data_loader = DataLoader(dataset["train"], batch_size=5, shuffle=True, num_workers=7)
    test_data_loader = DataLoader(dataset["val"], batch_size=5, shuffle=True, num_workers=7)
    net = SNN(T=T, channels=2)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.to(device)
    epochs = 20
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for epoch in range(0, epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()

            img = img.to(device).float()
            label = label.to(device)
            label_onehot = F.one_hot(label, 2).float()

            out_fr = net(img)
            loss = F.mse_loss(out_fr, label_onehot)

            loss.backward()
            optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label.int()).float().sum().item()
            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples
        print(train_acc)

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()


        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(device).float()
                label = label.to(device)
                label_onehot = F.one_hot(label, 2).float()

                out_fr = net(img)
                loss = F.mse_loss(out_fr, label_onehot)


                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)
        print(test_acc)
