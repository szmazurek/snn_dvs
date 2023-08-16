import wandb
import os
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from utils import train_val_dataset, f1_score, save_model
from spikingjelly.activation_based import functional
from PIL import Image
from typing import Tuple
from torchvision.models import efficientnet_v2_s
from spikingjelly.activation_based import ann2snn
from models import SNN_1 as SNN


class DvsDataset(Dataset):
    def __init__(self, targ_dir: str) -> None:
        self.all_folders = [os.path.join(targ_dir, directory) for directory in os.listdir(targ_dir)]
        self.random_flip = 0
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(self.random_flip),
            transforms.Resize((150, 400), interpolation=Image.NEAREST),
            transforms.PILToTensor()])

    def load_image(self, image) -> Image.Image:
        return self.transform(Image.open(image))

    def __len__(self) -> int:
        return len(self.all_folders)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        self.random_flip = random.getrandbits(1)
        folder_path = self.all_folders[index]
        file_list = glob.glob(os.path.join(folder_path, "*.png"))
        file_list_sorted = sorted(file_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[0]))
        images_tensor = torch.cat([self.load_image(image).unsqueeze(0) for image in file_list_sorted])

        label_list = [int(str(image)[-5]) for image in file_list_sorted]
        label_tensor = torch.tensor(label_list).unsqueeze(1)

        return images_tensor, label_tensor


def main():
    wandb.init(
        project="SNN_1_DVS_Default",
        entity="snn_team"
    )

    epochs = 500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_file_save = "checkpoint_DVS_SNN1.pth"
    dataset = DvsDataset(r"/home/plgkrzysjed1/datasets/dataset_dvs")
    checkpoint_folder_path = r"/home/plgkrzysjed1/workbench/data/SpikingScripts/evalutation/save"
    dataset = train_val_dataset(dataset)

    train_data_loader = DataLoader(dataset["train"], batch_size=1, shuffle=True, num_workers=12)
    test_data_loader = DataLoader(dataset["val"], batch_size=1, shuffle=False, num_workers=12)

    #  Converter
    # coverter= ann2snn.Converter(train_data_loader)
    # eff_snn= coverter(efficientnet_v2_s)
    net = SNN()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.to(device)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    max_f1 = 0.0
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
            label_shifted = torch.roll(label, shifts=0, dims=1)
            label_onehot_shifted = F.one_hot(label_shifted, 2).float()

            img = img.permute(1, 0, 2, 3, 4)  # [B, T, C, H, W] --> [T, B, C, H, W]
            img =img.to(device).float()

            out_fr = net(img)  # [T, B, 2]
            out_fr = out_fr.unsqueeze(0).permute(2, 1, 0, 3)  # [1, T, B, 2] --> [B, T, 1, 2]
            pred = torch.argmax(out_fr, dim=3)

            l = nn.MSELoss()

            loss = l(out_fr, label_onehot_shifted.float())

            loss.backward()
            optimizer.step()

            train_acc += (pred == label_shifted.int()).float().sum().item()
            train_samples += label.numel()
            train_loss += loss.item() * label.numel()

            train_f1 += f1_score(pred, label_shifted.int())
            i += 1
            functional.reset_net(net)

        train_f1 /= i
        train_loss /= train_samples
        train_acc /= train_samples
        print("Train", epoch, train_acc, train_f1)

        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        test_f1 = 0
        i = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                # img = img.to(device).float()

                label = label.to(device)
                label_shifted = torch.roll(label, shifts=0, dims=1)
                label_onehot_shifted = F.one_hot(label_shifted, 2).float()

                img = img.permute(1, 0, 2, 3, 4)  # [B, T, C, H, W] --> [T, B, C, H, W]
                img = img.to(device).float()

                out_fr = net(img)  # [T, B, 2]
                out_fr = out_fr.unsqueeze(0).permute(2, 1, 0, 3)  # [1, T, B, 2] --> [B, T, 1, 2]
                pred = torch.argmax(out_fr, dim=3)

                l = nn.MSELoss()

                loss = l(out_fr, label_onehot_shifted.float())

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (pred == label_shifted.int()).float().sum().item()

                test_f1 += f1_score(pred, label_shifted.int())
                i += 1
                functional.reset_net(net)
        test_f1 /= i
        test_loss /= test_samples
        test_acc /= test_samples
        print("Test", epoch, test_acc, test_f1)
        wandb.log({"train_acc": train_acc, "train_loss": train_loss, "train_f1": train_f1,
                   "test_acc": test_acc, "test_loss": test_loss, "test_f1": test_f1})
        if float(max_f1) < float(test_f1):
            max_f1 = test_f1
            save_model(net, checkpoint_folder_path, checkpoint_file_save)

    wandb.log({"max_f1": max_f1})


if __name__ == "__main__":
    main()
