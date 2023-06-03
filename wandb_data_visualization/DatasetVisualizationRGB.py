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
import wandb
import os
import glob


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


class CarlaRGBDataset(Dataset):
    def __init__(self, targ_dir: str) -> None:

        self.all_folders = [os.path.join(targ_dir, directory) for directory in os.listdir(targ_dir)]
        self.transform = transforms.Compose([
                transforms.Resize((150, 400)),
                transforms.PILToTensor()])

    def load_image(self, image) -> Image.Image:
        img = Image.open(image)
        return self.transform(img)

    def __len__(self) -> int:
        return len(self.all_folders)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        folder_path = self.all_folders[index]

        file_list = glob.glob(os.path.join(folder_path, "*.jpg"))
        file_list_sorted = sorted(file_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[0]))
        images_tensor = torch.cat([self.load_image(image).unsqueeze(0) for image in file_list_sorted])

        label_list = [int(str(image)[-5]) for image in file_list_sorted]
        label_tensor = torch.tensor(label_list).unsqueeze(1)
        name_sample = os.path.basename(folder_path)
        return images_tensor, label_tensor, name_sample


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


def main():

    wandb.init(
         project="Dataset_Overview_RGB",
         entity="snn_team"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = CarlaRGBDataset(r"/home/plgkrzysjed1/datasets/dataset_prediction_rgb_hq")

    test_data_loader_rgb = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)
    checkpoint = torch.load("/home/plgkrzysjed1/workbench/data/prediction_version2/DataAnalysis/rgb/checkpoint.pth")
    net = CSNN(4)
    net.load_state_dict(checkpoint['net'])
    net.to(device)
    sample = []

    net.eval()
    with torch.no_grad():
        for img, label, name in test_data_loader_rgb:
            label = label.to(device)
            img = img.to(device).float()
            img = img.squeeze(0)  # [900,1,W,H] T->B
            label = label.squeeze(0)  # [900,1] T->B
            out_fr = net(img)  # output [T, 2]
            pred = torch.argmax(out_fr, dim=1)
            test_f1 = f1_score(pred, label)
            label = label.squeeze(1)
            train_acc = (pred == label.int()).float().sum().item()
            for i, (frame, l, p) in enumerate(zip(img, label, pred)):

                mask_image = wandb.Image(frame, caption=f"Frame number: {i} | Label: {str(l.item())} | Prediction: {str(p.item())}")

                wandb.log({f"{name[0]}": mask_image})

            sample.append([name[0], test_f1, train_acc/900])

    my_table = wandb.Table(columns=["name_sample", "f1_score", "acc"], data=sample)
    wandb.log({"Base table": my_table})
    wandb.finish()


if __name__ == "__main__":
    main()
