import wandb
import torch
import os
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import train_val_dataset
from sklearn.model_selection import StratifiedKFold
from models import Resnet18
from data_loaders import RGBDataset
from torchmetrics import Accuracy, F1Score, AUROC
from torch.utils.data import Subset

api_key_file = open("./wandb_api_key.txt", "r")
API_KEY = api_key_file.read()
api_key_file.close()
os.environ["WANDB_API_KEY"] = API_KEY

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def main():
    # wandb.init(
    #     project="pedestrian_surrogate",
    #     entity="mazurek",
    #     name="rgb_weather_resnet_default_good_res_seed_0",
    # )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_folder_path = r"checkpoint_folder_path"
    checkpoint_file_save = "checkpoint.pth"

    full_dataset = RGBDataset(r"datasets/dataset_weather_rgb")
    labs = torch.tensor(full_dataset.all_labels)
    neg_count, pos_count = torch.unique(labs, return_counts=True)[1]
    pos_weight = neg_count / pos_count
    labels = full_dataset.all_labels
    data_indices = np.arange(len(labels))
    skf = StratifiedKFold(n_splits=10)
    for fold, (train_idx, test_idx) in enumerate(skf.split(data_indices, labels)):
        print(f"Fold {fold+1}")
        train_dataset = Subset(full_dataset, train_idx)
        test_dataset = Subset(full_dataset, test_idx)
        train_val_ds = train_val_dataset(train_dataset, val_split=0.15)
        print(f"Dataset sizes")
        print(f"Train {len(train_val_ds['train'])}")
        print(f"Val {len(train_val_ds['val'])}")
        print(f"Test {len(test_dataset)}")

        train_data_loader = DataLoader(
            train_val_ds["train"],
            batch_size=1024,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=3,
        )
        val_data_loader = DataLoader(
            train_val_ds["val"],
            batch_size=512,
            shuffle=False,
            num_workers=4,
            prefetch_factor=1,
            pin_memory=True,
        )
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=4,
            prefetch_factor=1,
            pin_memory=True,
        )

        net = Resnet18()
        net.to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.1)

        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        max_f1 = 0.0

        pos_weight_tensor = torch.full([1], pos_weight)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(device)
        accuracy_metric = Accuracy("binary").to(device)
        f1_metric = F1Score("binary").to(device)
        auroc_metric = AUROC("binary").to(device)
        epochs = 12
        wandb.init(
            project="pedestrian_surrogate",
            entity="mazurek",
            name=f"rgb_weather_resnet_default_kfold_{fold+1}",
        )

        for epoch in range(0, epochs):
            net.train()
            train_loss = 0
            label_list = torch.Tensor().to(device)
            pred_list = torch.Tensor().to(device)
            for i, (img, label) in enumerate(train_data_loader):
                optimizer.zero_grad()
                label = label.to(device)
                img = img.to(device).float()

                out_fr = net(img).squeeze()
                pred_list = torch.cat((pred_list, out_fr.detach()), dim=0)
                label_list = torch.cat((label_list, label), dim=0)

                loss = loss_fn(out_fr, label.float())
                train_loss += loss.detach().item()
                loss.backward()
                optimizer.step()
                if i % 10 == 0:
                    print(f"Processed {i} batches")
            train_acc = accuracy_metric(pred_list, label_list)
            train_f1 = f1_metric(pred_list, label_list)
            train_auroc = auroc_metric(pred_list, label_list)

            print(f"Train loss {train_loss/(i+1)}")
            print(
                f"Train epoch {epoch}, acc {train_acc}, f1 {train_f1}, auroc {train_auroc}"
            )

            net.eval()
            val_loss = 0
            label_list = torch.Tensor().to(device)
            pred_list = torch.Tensor().to(device)
            with torch.no_grad():
                for n, (img, label) in enumerate(val_data_loader):
                    label = label.to(device)
                    img = img.to(device).float()

                    out_fr = net(img).squeeze()
                    pred_list = torch.cat((pred_list, out_fr.detach()), dim=0)
                    label_list = torch.cat((label_list, label), dim=0)
                    loss = loss_fn(out_fr, label.float())
                    val_loss += loss.detach().item()
            val_acc = accuracy_metric(pred_list, label_list)
            val_f1 = f1_metric(pred_list, label_list)
            val_auroc = auroc_metric(pred_list, label_list)
            print(f"Val loss {val_loss/(n+1)}")
            print(f"Val epoch {epoch}, acc {val_acc}, f1 {val_f1}, auroc {val_auroc}")
            wandb.log(
                {
                    "train_acc": train_acc,
                    "train_loss": train_loss / (n + i),
                    "train_f1": train_f1,
                    "train_auroc": train_auroc,
                    "val_acc": val_acc,
                    "val_loss": val_loss / (n + 1),
                    "val_f1": val_f1,
                    "val_auroc": val_auroc,
                }
            )
        net.eval()
        test_loss = 0
        label_list = torch.Tensor().to(device)
        pred_list = torch.Tensor().to(device)
        with torch.no_grad():
            for n, (img, label) in enumerate(test_data_loader):
                label = label.to(device)
                img = img.to(device).float()

                out_fr = net(img).squeeze()
                pred_list = torch.cat((pred_list, out_fr.detach()), dim=0)
                label_list = torch.cat((label_list, label), dim=0)
                loss = loss_fn(out_fr, label.float())
                test_loss += loss.detach().item()
        test_acc = accuracy_metric(pred_list, label_list)
        test_f1 = f1_metric(pred_list, label_list)
        test_auroc = auroc_metric(pred_list, label_list)
        print(f"Test loss {test_loss/(n+1)}")
        print(f"Test epoch {epoch}, acc {test_acc}, f1 {test_f1}, auroc {test_auroc}")
        wandb.log(
            {
                "test_acc": test_acc,
                "test_loss": test_loss / (n + 1),
                "test_f1": test_f1,
                "test_auroc": test_auroc,
            }
        )
        # if float(max_f1) < float(test_f1):
        #     max_f1 = test_f1
        #     save_model(net, checkpoint_folder_path, checkpoint_file_save)
        wandb.finish()


if __name__ == "__main__":
    main()
