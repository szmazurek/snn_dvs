import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import train_val_dataset, f1_score, save_model
from models import CNN_1 as CNN
from data_loaders import RGBDataset


def main():
    wandb.init(
        project="Project_name",
        entity="snn_team"
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_folder_path = r"checkpoint_folder_path"
    checkpoint_file_save = "checkpoint.pth"

    dataset = RGBDataset(r"dataset_rgb_path")
    dataset = train_val_dataset(dataset)
    train_data_loader = DataLoader(dataset["train"], batch_size=5000, shuffle=True, num_workers=2)
    test_data_loader = DataLoader(dataset["val"], batch_size=5000, shuffle=False, num_workers=2)

    net = CNN()

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.01)
    net.to(device)
    epochs = 1000
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    max_f1 = 0.0
    for epoch in range(0, epochs):
        print(epoch)
        net.train()
        train_loss = 0
        train_samples = 0
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
        test_samples = 0
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


if __name__ == "__main__":
    main()
