import torch
from spikingjelly.activation_based import functional
from models import Resnet18_DVS, Resnet18_DVS_rgb
from utils import (
    unsqueeze_dim_if_missing,
)
from torchmetrics import AUROC, Accuracy, F1Score
from matplotlib import pyplot as plt
import os
import json
import logging
from argparse import ArgumentParser

# Configure the logging
logging.basicConfig(
    filename="logs/log_snn.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
    force=True,
)


def assign_label(tensor: torch.Tensor) -> int:
    """Detect if in the given tensor of numbers there occurs a situation at any place when the 1 is followed by 0.
    If yes, assigns label 100, otherwise sum of the numbers in tensor.
    """
    assert tensor.dim() == 1, "Tensor must be 1-dimensional"
    assert torch.all(
        tensor == 0 | (tensor == 1)
    ), "Tensor must contain only 0s or 1s"
    nonzero = tensor.nonzero()
    if nonzero.numel() == 0:
        return 0  # if negative sample, return 0
    if len(tensor) - 1 != nonzero[-1]:
        # return -1 # if ones occur, but there exists any 0 after them, return 1
        return -tensor.sum().item()
    recursive_difference_tensor = torch.diff(nonzero, dim=0)
    if torch.any(recursive_difference_tensor != 1):
        return (
            100  # if ones occur, but there exists any 0 between them, return 1
        )
    return tensor.sum().item()


def generate_extended_labels_from_sample(set_of_frame_filenames):
    per_frame_labels = torch.tensor(
        [
            int(os.path.basename(frame).split(".")[0].split("-")[1])
            for frame in set_of_frame_filenames
        ]
    )
    per_frame_label = assign_label(per_frame_labels)
    # per_frame_label = sum(per_frame_labels)
    return per_frame_label


def eval_threshold(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_results_save_path = os.path.join(args.save_dir, args.exp_name)
    if not os.path.exists(exp_results_save_path):
        os.makedirs(exp_results_save_path)
    subset = torch.load(
        f"./saved_datasets/{args.exp_name}.pt",
        map_location=device,
    )

    state_dict = torch.load(
        f"./saved_models/{args.exp_name}.pt",
        map_location=device,
    )
    model = Resnet18_DVS()
    model.load_state_dict(state_dict=state_dict)
    model = model.to(device)

    model.eval()
    functional.set_step_mode(model, "m")

    loader = torch.utils.data.DataLoader(
        subset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    auroc = AUROC("binary").to(device)
    threshold_results = {}
    accuracies_list = []
    f1_list = []
    auroc_list = []
    for threshold in args.thresholds_to_evaluate:
        pred_list_test = torch.Tensor().to(device)
        label_list_test = torch.Tensor().to(device)
        f1 = F1Score("binary", threshold=threshold).to(device)
        acc = Accuracy("binary", threshold=threshold).to(device)
        with torch.no_grad():
            for n, (img_test, label_test) in enumerate(loader):
                img_test = img_test.permute(1, 0, 2, 3, 4).float().to(device)
                label_test = label_test.to(device)
                logits = model(img_test).squeeze(2).mean(0)
                logits = unsqueeze_dim_if_missing(logits)
                pred_list_test = torch.cat([pred_list_test, logits], dim=0)
                label_list_test = torch.cat(
                    [label_list_test, label_test], dim=0
                )
                functional.reset_net(model)

            auroc_res = auroc(pred_list_test, label_list_test).item()
            f1_res = f1(pred_list_test, label_list_test).item()
            acc_res = acc(pred_list_test, label_list_test).item()
            accuracies_list.append(acc_res)
            f1_list.append(f1_res)
            auroc_list.append(auroc_res)
            threshold_results[threshold] = {
                "auroc": auroc_res,
                "f1": f1_res,
                "acc": acc_res,
            }

    with open(
        os.path.join(exp_results_save_path, "threshold_results.json"), "w"
    ) as f:
        json.dump(threshold_results, f)

    plt.plot(args.thresholds_to_evaluate, accuracies_list)
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.savefig(
        os.path.join(exp_results_save_path, "threshold_accuracy.pdf"), dpi=400
    )
    plt.clf()
    plt.plot(args.thresholds_to_evaluate, f1_list)
    plt.xlabel("Threshold")
    plt.ylabel("F1")
    plt.savefig(
        os.path.join(exp_results_save_path, "threshold_f1.pdf"), dpi=400
    )
    plt.clf()
    plt.plot(args.thresholds_to_evaluate, auroc_list)
    plt.xlabel("Threshold")
    plt.ylabel("AUROC")
    plt.savefig(
        os.path.join(exp_results_save_path, "threshold_auroc.pdf"), dpi=400
    )


def eval_horizon(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_results_save_path = os.path.join(args.save_dir, args.exp_name)
    if not os.path.exists(exp_results_save_path):
        os.makedirs(exp_results_save_path)
    subset = torch.load(
        f"./{args.root_data_dir}/datasets/{args.exp_name}.pt",
        map_location=device,
    )

    state_dict = torch.load(
        f"./{args.root_data_dir}/models/{args.exp_name}.pt",
        map_location=device,
    )
    if args.dvs_mode:
        model = Resnet18_DVS()
    else:
        model = Resnet18_DVS_rgb()
    model.load_state_dict(state_dict=state_dict)
    model = model.to(device)
    model.eval()

    functional.set_step_mode(model, "m")

    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )
    extended_labels = []

    all_frame_filenames = subset.dataset.all_samples
    subset_indices = subset.indices
    subset_filenames = [all_frame_filenames[i] for i in subset_indices]
    for frame_filename in subset_filenames:
        extended_labels.append(
            generate_extended_labels_from_sample(frame_filename)
        )

    dataloader_len = len(loader)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none").to(device)
    loss_values = torch.Tensor().to(device)
    correct_preds_dict = {}
    total_preds_for_extended_label_dict = {}
    with torch.no_grad():
        for n, (img_test, label_test) in enumerate(loader):
            extended_labels_batch = extended_labels[
                n * args.batch_size : (n + 1) * args.batch_size
            ]
            img_test = img_test.permute(1, 0, 2, 3, 4).float().to(device)
            label_test = label_test.to(device)
            preds = model(img_test).squeeze(2).mean(0)
            loss = loss_fn(preds, label_test.float())
            loss_values = torch.cat((loss_values, loss), dim=0)
            logits = torch.sigmoid(preds).cpu()
            functional.reset_net(model)
            pred = torch.where(
                logits > args.predictive_horizon_threshold, 1, 0
            )
            for i, extended_label in enumerate(extended_labels_batch):
                if (
                    extended_label
                    not in total_preds_for_extended_label_dict.keys()
                ):
                    total_preds_for_extended_label_dict[extended_label] = 1
                else:
                    total_preds_for_extended_label_dict[extended_label] += 1
                if pred[i] == label_test[i]:
                    if extended_label not in correct_preds_dict.keys():
                        correct_preds_dict[extended_label] = 1
                    else:
                        correct_preds_dict[extended_label] += 1
            logging.info(f"Processed {n+1} batches out of {dataloader_len}")

    with open(
        os.path.join(
            exp_results_save_path,
            f"horizon_{args.predictive_horizon_threshold}_losses.json",
        ),
        "w",
    ) as f:
        json.dump(loss_values.tolist(), f)
    with open(
        os.path.join(
            exp_results_save_path,
            f"horizon_{args.predictive_horizon_threshold}_correct_preds_dict.json",
        ),
        "w",
    ) as f:
        json.dump(correct_preds_dict, f)
    with open(
        os.path.join(
            exp_results_save_path,
            f"horizon_{args.predictive_horizon_threshold}_total_preds_for_extended_label_dict.json",
        ),
        "w",
    ) as f:
        json.dump(total_preds_for_extended_label_dict, f)
    logging.info("results saved")
    for key in correct_preds_dict.keys():
        correct_preds_dict[key] /= total_preds_for_extended_label_dict[key]
    sorted_label_accuracies = dict(sorted(correct_preds_dict.items()))

    labels, acc = zip(*sorted_label_accuracies.items())

    plt.bar(labels, acc)
    plt.xlabel("Extended label")
    # plt.ylim([0, 1.1])
    plt.ylabel("Accuracy")
    plt.savefig(
        os.path.join(
            exp_results_save_path,
            f"horizon_{args.predictive_horizon_threshold}_accuracy.pdf",
        ),
        dpi=400,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--root_data_dir", type=str)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--eval_horizon", action="store_true")
    parser.add_argument("--eval_threshold", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dvs_mode", action="store_true")
    parser.add_argument(
        "--thresholds_to_evaluate",
        nargs="+",
        type=float,
        default=[0.5],
    )
    parser.add_argument(
        "--predictive_horizon_threshold", type=float, default=0.5
    )
    args = parser.parse_args()
    if args.eval_threshold:
        eval_threshold(args)
    elif args.eval_horizon:
        eval_horizon(args)
