import wandb
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from utils import train_val_dataset, f1_score, save_model
from PIL import Image
from typing import Tuple
import glob
from models import CNN_1 as CNN
from torchvision.models import efficientnet_v2_s

class RGBDataset(Dataset):
    def __init__(self, targ_dir: str) -> None:

        self.all_folders = [os.path.join(targ_dir, directory) for directory in os.listdir(targ_dir)]
        self.all_files = [glob.glob(os.path.join( folders, "*.jpg")) for folders in self.all_folders]
        self.all_files = [e for sub in self.all_files for e in sub]
        #print(self.all_files)
        self.transform = transforms.Compose([
                transforms.Resize((150, 400)),
                transforms.RandomHorizontalFlip(),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    def load_image(self, image) -> torch.Tensor:
        img = Image.open(image)
        return self.transform(img)

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        file_path = self.all_files[index]
        image = self.load_image(file_path)

        label = int(os.path.splitext(os.path.basename(file_path))[0].split("-")[1])
        return image, label


if __name__ == "__main__":

            wandb.init(
                project="CNN_1_RGB_Default",
                entity="snn_team"
            )
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            dataset = RGBDataset(r"/home/plgkrzysjed1/datasets/dataset_rgb")
            checkpoint_folder_path = r"/home/plgkrzysjed1/workbench/data/Spike_Scripts/evalutation/save"
            checkpoint_file_save = "checkpoint_RGB_CNN1.pth"
          #  checkpoint = torch.load(
          #      "/home/plgkrzysjed1/workbench/data/Spike_Scripts/evalutation/save/checkpoint_RGB_CNN2_small.pth")

            dataset = train_val_dataset(dataset)
            train_data_loader = DataLoader(dataset["train"], batch_size=5000, shuffle=True, num_workers=2)
            test_data_loader = DataLoader(dataset["val"], batch_size=5000, shuffle=False, num_workers=2)

            net = CNN()

            #net = efficientnet_v2_s(pretrained = True)
            #net.classifier.add_module("ac1", nn.Sigmoid())
            #net.classifier.add_module("fc", nn.Linear(1000,2,bias=False))
            #net.classifier.add_module("ac2", nn.Sigmoid())

            #net.load_state_dict(checkpoint['net'])
            optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
            net.to(device)
            epochs = 1000
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
            max_f1 = 0.0
            for epoch in range(0, epochs):
                print(epoch)
                net.train()
                train_loss = 0
                train_acc = 0
                train_samples = 0
                train_f1 = 0
                label_list = torch.Tensor().to(device)
                pred_list = torch.Tensor().to(device)
                for img, label in train_data_loader:
                    optimizer.zero_grad()
                    label = label.to(device)
                    label_onehot = F.one_hot(label, 2).float()
                    img = img.to(device).float()

                    out_fr = net(img)
                    pred = torch.argmax(out_fr, dim=1)
                    pred_list = torch.cat((pred_list, pred), dim=0)
                    label_list = torch.cat((label_list, label), dim=0)
                    l = nn.MSELoss()

                    loss = l(out_fr, label_onehot.float())
                    train_loss += loss.item() * label.numel()
                    train_samples += label.numel()
                    loss.backward()
                    optimizer.step()

                train_acc = (torch.Tensor(pred_list) == torch.Tensor(label_list)).sum().item()
                train_f1 = f1_score(torch.Tensor(pred_list), torch.Tensor(label_list))

                train_loss /= train_samples
                train_acc /= train_samples
                print("Train", epoch, train_acc, train_f1)
                
                lr_scheduler.step()

                net.eval()
                test_loss = 0
                test_acc = 0
                test_samples = 0
                test_f1 = 0
                label_list = torch.Tensor().to(device)
                pred_list = torch.Tensor().to(device)
                with torch.no_grad():
                    for img, label in test_data_loader:
                        label = label.to(device)
                        label_onehot_shifted = F.one_hot(label, 2).float()
                        img = img.to(device).float()

                        out_fr = net(img)
                        pred = torch.argmax(out_fr, dim=1)
                        pred_list = torch.cat((pred_list, pred), dim=0)
                        label_list = torch.cat((label_list, label), dim=0)

                        l = nn.MSELoss()
                        loss = l(out_fr, label_onehot_shifted.float())
                        test_samples += label.numel()
                        test_loss += loss.item() * label.numel()

                test_acc = (torch.Tensor(pred_list) == torch.Tensor(label_list)).sum().item()
                test_f1 = f1_score(torch.Tensor(pred_list), torch.Tensor(label_list))
                test_loss /= test_samples
                test_acc /= test_samples
                print("Test", epoch, test_acc, test_f1)
                wandb.log({"train_acc": train_acc, "train_loss": train_loss, "train_f1": train_f1,
                           "test_acc": test_acc, "test_loss": test_loss, "test_f1": test_f1})
                if float(max_f1) < float(test_f1):
                    max_f1 = test_f1
                    save_model(net, checkpoint_folder_path, checkpoint_file_save)
            wandb.finish()

