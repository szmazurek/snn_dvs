from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
import os
import glob


class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        image_size = 150 * 400
        self.lin_fc = nn.Sequential(
            layer.Flatten(),
            layer.Linear(image_size, 400, bias=False),
            layer.Dropout(0.6),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan()),
            layer.Linear(400, 100, bias=False),
            layer.Dropout(0.3),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan()),
            layer.Linear(100, 2, bias=False),
            layer.Dropout(0.1),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan()))

        functional.set_step_mode(self, step_mode='s')

    def forward(self, x: torch.Tensor):
        x_seq = self.lin_fc(x)
        return x_seq

    def spiking_encoder(self):
        return self.conv_fc[0:3]


class PredictionDvsDataset(Dataset):
    def __init__(self, targ_dir: str) -> None:

        self.all_folders = [os.path.join(targ_dir, directory) for directory in os.listdir(targ_dir)]
        self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((600//4, 1600//4)),
                transforms.PILToTensor()])

    def load_image(self, image) -> Image.Image:
        img = Image.open(image)
        return self.transform(img)

    def __len__(self) -> int:
        return len(self.all_folders)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        folder_path = self.all_folders[index]
        file_list = glob.glob(os.path.join(folder_path, "*.png"))
        file_list_sorted = sorted(file_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[0]))
        images_tensor = torch.cat([self.load_image(image).unsqueeze(0) for image in file_list_sorted])

        label_list = [int(str(image)[-5]) for image in file_list_sorted]
        label_tensor = torch.tensor(label_list).unsqueeze(1)

        return images_tensor, label_tensor


def train_val_dataset(dataset, val_split=0.25):
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


if __name__ == "__main__":

    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PredictionDvsDataset(r"D:\datasets\big-sample\predictionDataset\dataset_prediction")
    dataset = train_val_dataset(dataset)

    train_data_loader = DataLoader(dataset["train"], batch_size=2, shuffle=True, num_workers=8)
    test_data_loader = DataLoader(dataset["val"], batch_size=2, shuffle=True, num_workers=8)

    net = SNN()

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.to(device)
    epochs = 40
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for epoch in range(0, epochs):
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        f1 = 0
        i = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()
            img = img.to(device).float()
            #check_labels(label, 'label.txt')

            label = label.to(device)
            label_shifted = torch.roll(label, shifts=-30, dims=1)
            label_onehot_shifted = F.one_hot(label_shifted, 2).float()

            img = img.permute(1, 0, 2, 3, 4)  # [B, T, C, H, W] --> [T, B, C, H, W]
            frames = img.shape[0]  # num frames

            out_fr = None
            for T in img:
                T = T.to(device).float()
                out = net(T).unsqueeze(0)
                if out_fr is None:
                    out_fr = out
                else:
                    out_fr = torch.cat((out_fr, out), dim=0)
            out_fr = out_fr.unsqueeze(0).permute(2, 1, 0, 3)
            loss = F.mse_loss(out_fr, label_onehot_shifted)

            loss.backward()
            optimizer.step()

            pred = torch.argmax(out_fr, dim=3)
            train_acc += (pred == label_shifted.int()).float().sum().item()
            train_samples += label.numel()
            train_loss += loss.item() * label.numel()

            f1 += f1_score(pred, label_shifted.int())
            i += 1
            functional.reset_net(net)

        f1 /= i
        train_loss /= train_samples
        train_acc /= train_samples
        print("Train", epoch, train_acc, f1)

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('train_f1', f1, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        f1 = 0
        i = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(device).float()

                label = label.to(device)
                label_shifted = torch.roll(label, shifts=-30, dims=1)
                label_onehot_shifted = F.one_hot(label_shifted, 2).float()

                img = img.permute(1, 0, 2, 3, 4)  # [B, T, C, H, W] --> [T, B, C, H, W]
                frames = img.shape[0]  # num frames

                out_fr = None
                for T in img:
                    T = T.to(device).float()
                    out = net(T).unsqueeze(0)
                    if out_fr is None:
                        out_fr = out
                    else:
                        out_fr = torch.cat((out_fr, out), dim=0)
                out_fr = out_fr.unsqueeze(0).permute(2, 1, 0, 3)
                loss = F.mse_loss(out_fr, label_onehot_shifted)

                pred = torch.argmax(out_fr, dim=3)
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (pred == label_shifted.int()).float().sum().item()

                f1 += f1_score(pred, label_shifted.int())
                i += 1
                functional.reset_net(net)
        f1 /= i
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)
        writer.add_scalar('test_f1', f1, epoch)
        print("Test", epoch, test_acc, f1)
