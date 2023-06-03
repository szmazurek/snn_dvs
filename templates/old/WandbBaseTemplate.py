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

sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize',
        'name': 'max_f1'
    },
    'parameters': {
        'epochs': {"value": 35},
        'lr': {"values": [0.9, 0.1, 0.01, 0.001, 0.0001]},
        'snn': {"distribution": "categorical", "values": ["PLIF", "LIF"]},
        'activation': {"distribution": "categorical", "values": ["ATAN", "SIGMOID"]},
        'neurons': {"values": [6400, 3200, 1600, 800, 400, 200, 100, 50]},
        'neurons2': {"values": [3200, 1600, 800, 400, 200, 100, 50, 25]}}
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project='SNN_Project_Team', entity="snn_team")


class SNN(nn.Module):
    def __init__(self, snn, neurons, activation, neurons2):
        super().__init__()

        if snn == "PLIF":
            ne = neuron.ParametricLIFNode
        elif snn == "LIF":
            ne = neuron.LIFNode
        else:
            print("Error no choose")
            ne = 0

        if activation == "ATAN":
            ac = surrogate.ATan()
        elif activation == "SIGMOID":
            ac = surrogate.Sigmoid()
        elif activation == "LRELU":
            ac = surrogate.LeakyKReLU()
        else:
            print("Error no choose")
            ac = 0

        image_size = 150 * 400
        lin_fc = nn.Sequential(
            layer.Flatten(),
            layer.Linear(image_size, neurons, bias=False),
            ne(surrogate_function=ac),

            layer.Linear(neurons, neurons2, bias=False),
            ne(surrogate_function=ac),

            layer.Linear(neurons2, 2, bias=False),
            ne(surrogate_function=ac))

        self.lin_fc = lin_fc

        functional.set_step_mode(self, step_mode='m')
        functional.set_backend(self, backend='cupy')

    def forward(self, x: torch.Tensor):

        x_seq = self.lin_fc(x)
        return x_seq


class PredictionDvsDataset(Dataset):
    def __init__(self, targ_dir: str) -> None:

        self.all_folders = [os.path.join(targ_dir, directory) for directory in os.listdir(targ_dir)]
        self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((600//4, 1600//4), interpolation=Image.NEAREST),
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


def main():
    wandb.init()
    lr = wandb.config.lr
    snn = wandb.config.snn
    neurons = wandb.config.neurons
    activation = wandb.config.activation
    epochs = wandb.config.epochs
    neurons2 = wandb.config.neurons2


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = PredictionDvsDataset(r"/home/plgkrzysjed1/datasets/dataset_prediction")
    dataset = train_val_dataset(dataset)

    train_data_loader = DataLoader(dataset["train"], batch_size=1, shuffle=True, num_workers=12, pin_memory=True)
    test_data_loader = DataLoader(dataset["val"], batch_size=1, shuffle=True, num_workers=12, pin_memory=True)

    net = SNN(snn=snn, neurons=neurons, activation=activation, neurons2=neurons2)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
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
            # img = img.to(device).float()
            # check_labels(label, 'label.txt')

            label = label.to(device)
            label_shifted = torch.roll(label, shifts=-30, dims=1)
            label_onehot_shifted = F.one_hot(label_shifted, 2).float()

            img = img.permute(1, 0, 2, 3, 4)  # [B, T, C, H, W] --> [T, B, C, H, W]
            img =img.to(device).float()

            out_fr = net(img)  # [T, B, 2]
            out_fr = out_fr.unsqueeze(0).permute(2, 1, 0, 3)  # [1, T, B, 2] --> [B, T, 1, 2]
            pred = torch.argmax(out_fr, dim=3)

            l = nn.BCELoss()
            s = nn.Softmax(dim=3)

            loss = l(s(out_fr), label_onehot_shifted.float())

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
                label_shifted = torch.roll(label, shifts=-30, dims=1)
                label_onehot_shifted = F.one_hot(label_shifted, 2).float()

                img = img.permute(1, 0, 2, 3, 4)  # [B, T, C, H, W] --> [T, B, C, H, W]
                img = img.to(device).float()

                out_fr = net(img)  # [T, B, 2]
                out_fr = out_fr.unsqueeze(0).permute(2, 1, 0, 3)  # [1, T, B, 2] --> [B, T, 1, 2]
                pred = torch.argmax(out_fr, dim=3)

                l = nn.BCELoss()
                s = nn.Softmax(dim=3)
                loss = l(s(out_fr), label_onehot_shifted.float())

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
        if epoch == 10 or epoch == 15:
            print(epoch)
            if float(max_f1) == 0.0:
                print("Fast end due to no learn")
                break
    wandb.log({"max_f1": max_f1})


if __name__ == "__main__":
    wandb.agent(sweep_id, function=main, count=500)

# if __name__ == "__main__":
#     wandb.agent(addidtional_agent, function=main, count=500) # add id from webside