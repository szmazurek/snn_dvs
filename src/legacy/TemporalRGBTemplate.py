import wandb
import torch
import os
import torch.nn as nn
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from utils import (
    train_val_test_split,
    unsqueeze_dim_if_missing,
    perform_forward_pass_on_temporal_batch,
    set_random_seeds,
)
from models import Resnet18
from legacy.data_loaders import (
    RGBDatasetTemporal,
    DVSDatasetTemporalforNonTemporalNet,
)
from torchmetrics import Accuracy, F1Score, AUROC
from utils import EarlyStopping

api_key_file = open("./wandb_api_key.txt", "r")
API_KEY = api_key_file.read()
api_key_file.close()
os.environ["WANDB_API_KEY"] = API_KEY


def temporal_rgb_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_folder_path = args.checkpoint_folder_path
    checkpoint_file_save = args.checkpoint_file_save
    if args.dvs_mode:
        full_dataset = DVSDatasetTemporalforNonTemporalNet(
            target_dir=args.dataset_path,
            target_size=(args.img_height, args.img_width),
            sample_len=args.sample_timestep,
            overlap=args.sample_overlap,
        )
    else:
        full_dataset = RGBDatasetTemporal(
            target_dir=args.dataset_path,
            target_size=(args.img_height, args.img_width),
            sample_len=args.sample_timestep,
            overlap=args.sample_overlap,
        )
    (
        train_dataset,
        val_dataset,
        test_dataset,
        pos_weight,
    ) = train_val_test_split(
        full_dataset, args.val_size, args.test_size, args.seed
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        prefetch_factor=5,
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=6,
        prefetch_factor=5,
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
    if args.resume_training_model_path is not None:
        state_dict = torch.load(args.resume_training_model_path)
        net.load_state_dict(state_dict)
        print(f"Restoring model from {args.resume_training_model_path}")
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
        run_id = wandb.run.id
    early_stopping = EarlyStopping(
        patience=8, delta=0.01, path=f"temp_chkpt/{run_id}.pt"
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

            out_fr_train = (
                (perform_forward_pass_on_temporal_batch(net, img_train))
                .squeeze(2)
                .mean(1)
            )

            out_fr_train = unsqueeze_dim_if_missing(out_fr_train)
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
            if i % 10 == 0 and i > 0:
                print(f"Completed batch {i}")

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
                out_fr_val = (
                    perform_forward_pass_on_temporal_batch(net, img_val)
                    .squeeze(2)
                    .mean(1)
                )
                out_fr_val = unsqueeze_dim_if_missing(out_fr_val)
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
    best_model_state = torch.load(f"temp_chkpt/{run_id}.pt")
    net.load_state_dict(best_model_state)
    net.eval()
    test_loss = 0
    label_list_test = torch.Tensor().to(device)
    pred_list_test = torch.Tensor().to(device)
    with torch.no_grad():
        for n, (img_test, label_test) in enumerate(test_data_loader):
            label_test = label_test.to(device)
            img_test = img_test.to(device).float()
            out_fr_test = (
                perform_forward_pass_on_temporal_batch(net, img_test)
                .squeeze(2)
                .mean(1)
            )
            out_fr_test = unsqueeze_dim_if_missing(out_fr_test)
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
    if args.save_final_model:
        dataset_save_path = "saved_datasets"
        model_save_path = "saved_models"
        # save both the eval subset of the dataset and the trained model
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path, exist_ok=True)
        if not os.path.exists(dataset_save_path):
            os.makedirs(dataset_save_path, exist_ok=True)
        torch.save(test_dataset, f"{dataset_save_path}/{run_id}.pt")
        torch.save(
            net.state_dict(),
            f"{model_save_path}/{run_id}.pt",
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="dataset_path")
    parser.add_argument("--kfold", action="store_true", default=False)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--sample_timestep", type=int, default=4)
    parser.add_argument("--sample_overlap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img_height", type=int, default=600)
    parser.add_argument("--img_width", type=int, default=1600)
    parser.add_argument("--dvs_mode", action="store_true", default=False)
    parser.add_argument(
        "--save_final_model", action="store_true", default=False
    )
    parser.add_argument(
        "--checkpoint_folder_path", type=str, default="checkpoint_path"
    )
    parser.add_argument(
        "--checkpoint_file_save", type=str, default="checkpoint.pth"
    )
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="project_name")
    parser.add_argument("--wandb_group", type=str, default="group_name")
    parser.add_argument("--wandb_exp_name", type=str, default="exp_name")
    parser.add_argument(
        "--resume_training_model_path",
        type=str,
        default=None,
        help="If set, loads the model state dict from specified path and continues training",
    )
    args = parser.parse_args()
    set_random_seeds(args.seed)
    temporal_rgb_training(args)