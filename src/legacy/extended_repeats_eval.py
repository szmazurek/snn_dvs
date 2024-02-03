import os
import json
import torch
from argparse import ArgumentParser
from legacy.data_loaders import (
    RGBDatasetRepeated,
    DVSDatasetRepeated,
)
from models import Resnet18_DVS
from utils import (
    set_random_seeds,
    train_val_test_split_single_labels,
)
from spikingjelly.activation_based.functional import set_step_mode, reset_net
from torchmetrics import AUROC, F1Score


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ds_path = args.ds_path
    model_path = args.model_path
    save_dir = args.save_dir

    save_name = os.path.basename(model_path.split(".")[0]) + ".json"
    batch_size = args.batch_size
    seed = args.seed
    sample_lengths_to_evaluate = args.sample_lengths_to_evaluate
    results_dict = {}
    set_random_seeds(seed)

    for sample_len in sample_lengths_to_evaluate:
        ds_class = DVSDatasetRepeated if args.dvs_mode else RGBDatasetRepeated
        ds_repeated = ds_class(
            ds_path,
            sample_len=sample_len,
            target_size=(256, 450),
        )

        (
            train_ds,
            val_ds,
            test_ds,
            pos_weight,
        ) = train_val_test_split_single_labels(
            ds_repeated, 0.15, 0.15, seed=seed
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            prefetch_factor=3,
        )
        net = Resnet18_DVS().to(device)
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict=state_dict)
        net.eval()
        set_step_mode(net, "m")
        label_list_test = torch.Tensor().to(device)
        pred_list_test = torch.Tensor().to(device)
        f1_score = F1Score("binary").to(device)
        auroc = AUROC("binary").to(device)
        dataloader_len = len(test_dataloader)
        with torch.no_grad():
            for i, (img, label) in enumerate(test_dataloader):
                print(f"Batch {i+1} of {dataloader_len}")
                img = img.permute(1, 0, 2, 3, 4).float().to(device)
                label = label.to(device)
                out = net(img).squeeze().mean(0)
                pred_list_test = torch.cat([pred_list_test, out], dim=0)
                label_list_test = torch.cat([label_list_test, label], dim=0)
                reset_net(net)
            f1_value = f1_score(pred_list_test, label_list_test).cpu().item()
            auroc_value = auroc(pred_list_test, label_list_test).cpu().item()
            print(f"Sample length: {sample_len}")
            print(f"F1 score: {f1_value}")
            print(f"AUROC score: {auroc_value}")
            results_dict[sample_len] = {"f1": f1_value, "auroc": auroc_value}
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, save_name), "w") as f:
        json.dump(results_dict, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--ds_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dvs_mode", action="store_true")
    parser.add_argument(
        "--sample_lengths_to_evaluate",
        nargs="+",
        type=int,
        default=[3, 5, 10, 15, 20, 25, 30],
    )
    args = parser.parse_args()
    main(args)
