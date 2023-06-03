from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from typing import Tuple
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer
import wandb
import os
import glob


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

        file_list = glob.glob(os.path.join(folder_path, "*.png"))
        file_list_sorted = sorted(file_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[0]))
        images_tensor = torch.cat([self.load_image(image).unsqueeze(0) for image in file_list_sorted])

        label_list = [int(str(image)[-5]) for image in file_list_sorted]
        label_tensor = torch.tensor(label_list).unsqueeze(1)
        name_sample = os.path.basename(folder_path)
        return images_tensor, label_tensor, name_sample


class PredictionDvsDataset(Dataset):
    def __init__(self, targ_dir: str) -> None:

        self.all_folders = [os.path.join(targ_dir, directory) for directory in os.listdir(targ_dir)]
        self.transform = transforms.Compose([
                transforms.Resize((600//4, 1600//4), interpolation=transforms.InterpolationMode.NEAREST),
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
        name_sample = os.path.basename(folder_path)
        return images_tensor, label_tensor, name_sample


def train_val_dataset(dataset, val_split=1):
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
    wandb.init(
        project="Dataset_Overview_DVS",
        entity="snn_team"
    )
    snn = "LIF"
    neurons = 50
    activation = "SIGMOID"
    neurons2 = 200

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datasetrgb = CarlaRGBDataset(r"/home/plgkrzysjed1/datasets/dataset_prediction_rgb")
    dataset = PredictionDvsDataset(r"/home/plgkrzysjed1/datasets/dataset_prediction")

    test_data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)
    test_data_loader_rgb = DataLoader(datasetrgb, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)
    checkpoint = torch.load("/home/plgkrzysjed1/workbench/data/prediction_version2/DataAnalysis/checkpoint.pth")
    net = SNN(snn=snn, neurons=neurons, activation=activation, neurons2=neurons2)
    net.load_state_dict(checkpoint['net'])
    net.to(device)
    sample = []


    net.eval()
    with torch.no_grad():
        for (img, label, name), (img_rgb, label_rgb, name_rgb) in zip(test_data_loader,test_data_loader_rgb):
            label = label.to(device)
            label_shifted = torch.roll(label, shifts=-30, dims=1)

            img = img.permute(1, 0, 2, 3, 4)  # [B, T, C, H, W] --> [T, B, C, H, W]
            img = img.to(device).float()
            img_rgb = img_rgb.permute(1, 0, 2, 3, 4)
            img_rgb = img_rgb.to(device).float()

            out_fr = net(img)  # output [T, B, 2]
            out_fr = out_fr.unsqueeze(0).permute(2, 1, 0, 3)  # [1, T, B, 2] --> [B, T, 1, 2]
            pred = torch.argmax(out_fr, dim=3)

            test_f1 = f1_score(pred, label_shifted.int())

            label_list = []
            pred_list = []
            functional.reset_net(net)
            # img [T, B, C, H, W] --> [T, C, H, W]
            for i, (ima,ima_rgb, l, lshift, lpred) in enumerate(
                    zip(img.squeeze(1), img_rgb.squeeze(1), label.squeeze(0).cpu(), label_shifted.squeeze(0).cpu(), pred.squeeze(0).cpu())):
                mask_image = wandb.Image(ima, caption=f"Frame number: {i} | Label: {str(l.item())} | Label_shifted: "
                                                      f"{str(lshift.item())} | Prediction: {str(lpred.item())}")
                mask_image_rgb = wandb.Image(ima_rgb, caption=f"Frame number: {i} | Label: {str(l.item())} | Label_shifted:"
                                                              f" {str(lshift.item())} | Prediction: {str(lpred.item())}")
                label_list.append(int(lshift.item()))
                pred_list.append(int(lpred.item()))
                wandb.log({f"{name[0]}": [mask_image,mask_image_rgb ]})

            sample.append([name[0], test_f1])
    my_table = wandb.Table(columns=["name_sample", "f1_score"], data=sample)
    wandb.log({"Base table": my_table})
    wandb.finish()


if __name__ == "__main__":
    main()
