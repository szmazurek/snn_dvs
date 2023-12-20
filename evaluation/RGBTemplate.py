import wandb
import torch
import os
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from utils import train_val_dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from models import Resnet18, Resnet18_DVS_rgb, Resnet18_DVS
from data_loaders import RGBDataset, DVSDatasetAsRGB
from spikingjelly.activation_based import functional
from torchmetrics import Accuracy, F1Score, AUROC, Recall, Specificity
from torch.utils.data import Subset
from utils import EarlyStopping

api_key_file = open("./wandb_api_key.txt", "r")
API_KEY = api_key_file.read()
api_key_file.close()
os.environ["WANDB_API_KEY"] = API_KEY


def set_random_seeds(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return None


def main_kfold(args):
    # wandb.init(
    #     project="pedestrian_surrogate",
    #     entity="mazurek",
    #     name="rgb_weather_resnet_default_good_res_seed_0",
    # )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_folder_path = args.checkpoint_folder_path
    checkpoint_file_save = args.checkpoint_file_save

    full_dataset = RGBDataset(args.dataset_path)
    labs = torch.tensor(full_dataset.all_labels)
    neg_count, pos_count = torch.unique(labs, return_counts=True)[1]
    pos_weight = neg_count / pos_count
    labels = full_dataset.all_labels
    data_indices = np.arange(len(labels))
    skf = StratifiedKFold(
        n_splits=args.n_folds, shuffle=False, random_state=args.seed
    )
    for fold, (train_idx, test_idx) in enumerate(
        skf.split(data_indices, labels)
    ):
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
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=3,
        )
        val_data_loader = DataLoader(
            train_val_ds["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            prefetch_factor=1,
            pin_memory=True,
        )
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            prefetch_factor=1,
            pin_memory=True,
        )

        net = Resnet18(args.dvs_mode)
        net.to(device)
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        max_f1 = 0.0

        pos_weight_tensor = torch.full([1], pos_weight)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(device)
        accuracy_metric = Accuracy("binary").to(device)
        f1_metric = F1Score("binary").to(device)
        auroc_metric = AUROC("binary").to(device)
        epochs = args.epochs
        if args.wandb:
            wandb.init(
                project=args.wandb_project,
                entity="mazurek",
                group=args.wandb_group,
                name=f"{args.wandb_exp_name}_{fold+1}",
            )

        for epoch in range(0, epochs):
            net.train()
            train_loss = 0
            label_list = torch.Tensor().to(device)
            pred_list = torch.Tensor().to(device)
            for i, (img, label) in enumerate(train_data_loader):
                print(img.shape)
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
            print(
                f"Val epoch {epoch}, acc {val_acc}, f1 {val_f1}, auroc {val_auroc}"
            )
            if args.wandb:
                wandb.log(
                    {
                        "train_acc": train_acc,
                        "train_loss": train_loss / (i + 1),
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
        print(
            f"Test epoch {epoch}, acc {test_acc}, f1 {test_f1}, auroc {test_auroc}"
        )
        if args.wandb:
            wandb.log(
                {
                    "test_acc": test_acc,
                    "test_loss": test_loss / (n + 1),
                    "test_f1": test_f1,
                    "test_auroc": test_auroc,
                }
            )
            wandb.finish()
        # if float(max_f1) < float(test_f1):
        #     max_f1 = test_f1
        #     save_model(net, checkpoint_folder_path, checkpoint_file_save)


def snn_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_folder_path = args.checkpoint_folder_path
    checkpoint_file_save = args.checkpoint_file_save
    if args.dvs_mode:
        full_dataset = DVSDatasetAsRGB(
            args.dataset_path, target_size=(args.img_height, args.img_width)
        )
        net = Resnet18_DVS()

    else:
        full_dataset = RGBDataset(
            args.dataset_path, target_size=(args.img_height, args.img_width)
        )
        net = Resnet18_DVS_rgb()
    labs = torch.tensor(full_dataset.all_labels)

    neg_count, pos_count = torch.unique(labs, return_counts=True)[1]
    pos_weight = neg_count / pos_count
    labels = full_dataset.all_labels
    data_indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        data_indices,
        test_size=args.test_size,
        stratify=labels,
        random_state=args.seed,
    )
    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)
    train_val_ds = train_val_dataset(train_dataset, val_split=args.val_size)
    print(f"Dataset sizes")
    print(f"Train {len(train_val_ds['train'])}")
    print(f"Val {len(train_val_ds['val'])}")
    print(f"Test {len(test_dataset)}")

    train_data_loader = DataLoader(
        train_val_ds["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=3,
    )
    val_data_loader = DataLoader(
        train_val_ds["val"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        prefetch_factor=1,
        pin_memory=True,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        prefetch_factor=1,
        pin_memory=True,
    )

    net.to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    pos_weight_tensor = torch.full([1], pos_weight)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(device)
    accuracy_metric = Accuracy("binary").to(device)
    f1_metric = F1Score("binary").to(device)
    auroc_metric = AUROC("binary").to(device)
    epochs = args.epochs
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity="mazurek",
            group=args.wandb_group,
            name=f"{args.wandb_exp_name}",
        )
    early_stopping = EarlyStopping(
        patience=8, delta=0.01, path=f"temp_chkpt/{wandb.run.id}.pt"
    )
    functional.set_step_mode(net, "s")
    for epoch in range(0, epochs):
        net.train()
        train_loss = 0
        label_list_train = torch.Tensor().to(device)
        pred_list_train = torch.Tensor().to(device)
        for i, (img_train, label_train) in enumerate(train_data_loader):
            optimizer.zero_grad()
            label_train = label_train.to(device)
            img_train = img_train.to(device).float()
            out_fr_train = net(img_train).squeeze()
            pred_list_train = torch.cat(
                (pred_list_train, out_fr_train.detach()), dim=0
            )
            label_list_train = torch.cat(
                (label_list_train, label_train), dim=0
            )

            loss_train = loss_fn(out_fr_train, label_train.float())
            train_loss += loss_train.detach().item()
            loss_train.backward()
            optimizer.step()
            functional.reset_net(net)

        train_acc = accuracy_metric(pred_list_train, label_list_train)
        train_f1 = f1_metric(pred_list_train, label_list_train)
        train_auroc = auroc_metric(pred_list_train, label_list_train)
        label_list_train = torch.Tensor().to(device)
        pred_list_train = torch.Tensor().to(device)
        print(f"Train loss {train_loss/(i+1)}")
        print(
            f"Train epoch {epoch}, acc {train_acc}, f1 {train_f1}, auroc {train_auroc}"
        )

        net.eval()
        val_loss = 0
        label_list_val = torch.Tensor().to(device)
        pred_list_val = torch.Tensor().to(device)
        with torch.no_grad():
            for n, (img_val, label_val) in enumerate(val_data_loader):
                label_val = label_val.to(device)
                img_val = img_val.to(device).float()
                out_fr_val = net(img_val).squeeze()
                pred_list_val = torch.cat(
                    (pred_list_val, out_fr_val.detach()), dim=0
                )
                label_list_val = torch.cat((label_list_val, label_val), dim=0)
                loss_val = loss_fn(out_fr_val, label_val.float())
                val_loss += loss_val.detach().item()
                functional.reset_net(net)

        val_acc = accuracy_metric(pred_list_val, label_list_val)
        val_f1 = f1_metric(pred_list_val, label_list_val)
        val_auroc = auroc_metric(pred_list_val, label_list_val)
        label_list_val = torch.Tensor().to(device)
        pred_list_val = torch.Tensor().to(device)
        print(f"Val loss {val_loss/(n+1)}")
        print(
            f"Val epoch {epoch}, acc {val_acc}, f1 {val_f1}, auroc {val_auroc}"
        )
        if args.wandb:
            wandb.log(
                {
                    "train_acc": train_acc,
                    "train_loss": train_loss / (i + 1),
                    "train_f1": train_f1,
                    "train_auroc": train_auroc,
                    "val_acc": val_acc,
                    "val_loss": val_loss / (n + 1),
                    "val_f1": val_f1,
                    "val_auroc": val_auroc,
                }
            )
        early_stopping(val_loss / (n + 1), net)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    best_model_state = torch.load(f"temp_chkpt/{wandb.run.id}.pt")
    net.load_state_dict(best_model_state)
    net.eval()
    test_loss = 0
    label_list_test = torch.Tensor().to(device)
    pred_list_test = torch.Tensor().to(device)
    with torch.no_grad():
        for n, (img_test, label_test) in enumerate(test_data_loader):
            label_test = label_test.to(device)
            img_test = img_test.to(device).float()
            out_fr_test = net(img_test).squeeze()
            pred_list_test = torch.cat(
                (pred_list_test, out_fr_test.detach()), dim=0
            )
            label_list_test = torch.cat((label_list_test, label_test), dim=0)
            loss_test = loss_fn(out_fr_test, label_test.float())
            test_loss += loss_test.detach().item()
            functional.reset_net(net)
    test_acc = accuracy_metric(pred_list_test, label_list_test)
    test_f1 = f1_metric(pred_list_test, label_list_test)
    test_auroc = auroc_metric(pred_list_test, label_list_test)
    print(f"Test loss {test_loss/(n+1)}")
    print(f"Test acc {test_acc}, f1 {test_f1}, auroc {test_auroc}")
    # print("Checkpoint cleanup...")
    # os.remove(f"temp_chkpt/{wandb.run.id}.pt")
    if args.wandb:
        wandb.log(
            {
                "test_acc": test_acc,
                "test_loss": test_loss / (n + 1),
                "test_f1": test_f1,
                "test_auroc": test_auroc,
            }
        )
        wandb.finish()


def normal_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_folder_path = args.checkpoint_folder_path
    checkpoint_file_save = args.checkpoint_file_save
    if args.dvs_mode:
        full_dataset = DVSDatasetAsRGB(
            args.dataset_path, target_size=(args.img_height, args.img_width)
        )
    else:
        full_dataset = RGBDataset(
            args.dataset_path, target_size=(args.img_height, args.img_width)
        )
    labs = torch.tensor(full_dataset.all_labels)

    neg_count, pos_count = torch.unique(labs, return_counts=True)[1]
    pos_weight = neg_count / pos_count
    labels = full_dataset.all_labels
    data_indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        data_indices,
        test_size=args.test_size,
        stratify=labels,
        random_state=args.seed,
    )
    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)
    train_val_ds = train_val_dataset(train_dataset, val_split=args.val_size)
    print(f"Dataset sizes")
    print(f"Train {len(train_val_ds['train'])}")
    print(f"Val {len(train_val_ds['val'])}")
    print(f"Test {len(test_dataset)}")

    train_data_loader = DataLoader(
        train_val_ds["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=3,
    )
    val_data_loader = DataLoader(
        train_val_ds["val"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        prefetch_factor=1,
        pin_memory=True,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        prefetch_factor=1,
        pin_memory=True,
    )
    net = Resnet18(args.dvs_mode)
    net.to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    pos_weight_tensor = torch.full([1], pos_weight)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(device)
    accuracy_metric = Accuracy("binary").to(device)
    f1_metric = F1Score("binary").to(device)
    auroc_metric = AUROC("binary").to(device)
    epochs = args.epochs
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity="mazurek",
            group=args.wandb_group,
            name=f"{args.wandb_exp_name}",
        )
    early_stopping = EarlyStopping(
        patience=8, delta=0.01, path=f"temp_chkpt/{wandb.run.id}.pt"
    )

    for epoch in range(0, epochs):
        net.train()
        train_loss = 0
        label_list_train = torch.Tensor().to(device)
        pred_list_train = torch.Tensor().to(device)
        for i, (img_train, label_train) in enumerate(train_data_loader):
            optimizer.zero_grad()
            label_train = label_train.to(device)
            img_train = img_train.to(device).float()
            out_fr_train = net(img_train).squeeze()
            pred_list_train = torch.cat(
                (pred_list_train, out_fr_train.detach()), dim=0
            )
            label_list_train = torch.cat(
                (label_list_train, label_train), dim=0
            )

            loss_train = loss_fn(out_fr_train, label_train.float())
            train_loss += loss_train.detach().item()
            loss_train.backward()
            optimizer.step()
        train_acc = accuracy_metric(pred_list_train, label_list_train)
        train_f1 = f1_metric(pred_list_train, label_list_train)
        train_auroc = auroc_metric(pred_list_train, label_list_train)
        label_list_train = torch.Tensor().to(device)
        pred_list_train = torch.Tensor().to(device)
        print(f"Train loss {train_loss/(i+1)}")
        print(
            f"Train epoch {epoch}, acc {train_acc}, f1 {train_f1}, auroc {train_auroc}"
        )

        net.eval()
        val_loss = 0
        label_list_val = torch.Tensor().to(device)
        pred_list_val = torch.Tensor().to(device)
        with torch.no_grad():
            for n, (img_val, label_val) in enumerate(val_data_loader):
                label_val = label_val.to(device)
                img_val = img_val.to(device).float()
                out_fr_val = net(img_val).squeeze()
                pred_list_val = torch.cat(
                    (pred_list_val, out_fr_val.detach()), dim=0
                )
                label_list_val = torch.cat((label_list_val, label_val), dim=0)
                loss_val = loss_fn(out_fr_val, label_val.float())
                val_loss += loss_val.detach().item()

        val_acc = accuracy_metric(pred_list_val, label_list_val)
        val_f1 = f1_metric(pred_list_val, label_list_val)
        val_auroc = auroc_metric(pred_list_val, label_list_val)
        label_list_val = torch.Tensor().to(device)
        pred_list_val = torch.Tensor().to(device)
        print(f"Val loss {val_loss/(n+1)}")
        print(
            f"Val epoch {epoch}, acc {val_acc}, f1 {val_f1}, auroc {val_auroc}"
        )
        if args.wandb:
            wandb.log(
                {
                    "train_acc": train_acc,
                    "train_loss": train_loss / (i + 1),
                    "train_f1": train_f1,
                    "train_auroc": train_auroc,
                    "val_acc": val_acc,
                    "val_loss": val_loss / (n + 1),
                    "val_f1": val_f1,
                    "val_auroc": val_auroc,
                }
            )
        early_stopping(val_loss / (n + 1), net)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    best_model_state = torch.load(f"temp_chkpt/{wandb.run.id}.pt")
    net.load_state_dict(best_model_state)
    net.eval()
    test_loss = 0
    label_list_test = torch.Tensor().to(device)
    pred_list_test = torch.Tensor().to(device)
    with torch.no_grad():
        for n, (img_test, label_test) in enumerate(test_data_loader):
            label_test = label_test.to(device)
            img_test = img_test.to(device).float()
            out_fr_test = net(img_test).squeeze()
            pred_list_test = torch.cat(
                (pred_list_test, out_fr_test.detach()), dim=0
            )
            label_list_test = torch.cat((label_list_test, label_test), dim=0)
            loss_test = loss_fn(out_fr_test, label_test.float())
            test_loss += loss_test.detach().item()

    test_acc = accuracy_metric(pred_list_test, label_list_test)
    test_f1 = f1_metric(pred_list_test, label_list_test)
    test_auroc = auroc_metric(pred_list_test, label_list_test)
    print(f"Test loss {test_loss/(n+1)}")
    print(f"Test acc {test_acc}, f1 {test_f1}, auroc {test_auroc}")
    # print("Checkpoint cleanup...")
    # os.remove(f"temp_chkpt/{wandb.run.id}.pt")
    if args.wandb:
        wandb.log(
            {
                "test_acc": test_acc,
                "test_loss": test_loss / (n + 1),
                "test_f1": test_f1,
                "test_auroc": test_auroc,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="dataset_path")
    parser.add_argument("--kfold", action="store_true", default=False)
    parser.add_argument("--snn_mode", action="store_true", default=False)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dvs_mode", action="store_true", default=False)
    parser.add_argument(
        "--checkpoint_folder_path", type=str, default="checkpoint_path"
    )
    parser.add_argument("--img_height", type=int, default=600)
    parser.add_argument("--img_width", type=int, default=1600)
    parser.add_argument(
        "--checkpoint_file_save", type=str, default="checkpoint.pth"
    )
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="project_name")
    parser.add_argument("--wandb_group", type=str, default="group_name")
    parser.add_argument("--wandb_exp_name", type=str, default="exp_name")
    args = parser.parse_args()
    set_random_seeds(args.seed)
    if args.kfold:
        main_kfold(args)
    elif args.snn_mode:
        snn_loop(args)
    else:
        normal_training(args)
