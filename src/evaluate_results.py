import os
import tqdm
import json
import torch
import wandb
from torch.utils.data import DataLoader, Dataset

import lightning.pytorch as pl
from models.lightning_module import (
    LightningModuleNonTemporal,
    LightningModuleTemporalNets,
)

from data_utils.datasets import (
    BaseDataset,
    SingleSampleDataset,
    RepeatedSampleDataset,
    TemporalSampleDataset,
    PredictionDataset,
    PredictionDatasetSingleStep,
    PredictionDatasetSingleStepRepeated,
)
from argparse import ArgumentParser
from typing import Dict, List, Type, Tuple

AVAILABLE_DATASETS: Dict[str, Type[BaseDataset]] = {
    "single_sample": SingleSampleDataset,
    "repeated": RepeatedSampleDataset,
    "temporal": TemporalSampleDataset,
}

AVAILABLE_DATASETS_PREDICTION: Dict[str, Type[PredictionDataset]] = {
    "prediction_single_sample": PredictionDatasetSingleStep,
    "prediction_repeated": PredictionDatasetSingleStepRepeated,
    "prediction_temporal": PredictionDataset,
}

AVAILABLE_MODELS: Dict[str, Type[pl.LightningModule]] = {
    "single_sample": LightningModuleNonTemporal,
    "temporal": LightningModuleTemporalNets,
}

torch.set_float32_matmul_precision("medium")


def load_dataset_object(dataset_save_dir: str, run_id: str, device="cuda") -> Dataset:
    """
    Load a saved dataset object from a file.
    Args:
        dataset_save_dir (str): Directory where the dataset object is saved.
        run_id (str): ID of the run.
        device (str): Device to load the dataset object to.
    """
    saved_dataset_path = os.path.join(dataset_save_dir, f"{run_id}.pt")
    saved_dataset = torch.load(saved_dataset_path, map_location=torch.device(device))
    return saved_dataset


def load_wandb_group_runs(project_name: str, group_name: str) -> Dict[str, str]:
    """
    Load the run IDs of a group of runs from W&B.
    Args:
        project_name (str): Name of the project in W&B.
        group_name (str): Name of the group in W&B.
    Returns:
        Dict[str, str]: Dictionary with the run names as keys and the run IDs as values.
    """
    api = wandb.Api()
    runs = api.runs(f"mazurek/{project_name}", {"group": group_name})
    return {run.name: run.summary.get("run_id") for run in runs}


def extract_model_config_from_run(run_name: str) -> Tuple[str, bool, bool, int]:
    TEMPORAL_KEYWORD = ["temporal", "pseudotemporal"]
    MODEL_NAMES = ["resnet18", "slow_r50", "sew_resnet18_spiking"]
    """
    Extract the model configuration from a W&B run name.
    Args:
        run_name (str): Name of the W&B run."""
    temporal = False
    dvs_mode = False
    if any([keyword in run_name for keyword in TEMPORAL_KEYWORD]):
        temporal = True
        n_samples_idx = -2
        if "horizon" in run_name:
            n_samples_idx = -4

        n_samples = int(run_name.split("_")[n_samples_idx])
        print(f"n_samples: {n_samples}")
    if "DVS" in run_name:
        dvs_mode = True
    if "single_sample" in run_name:
        n_samples = 1
    if "repeated" in run_name:
        temporal = True
        parts = run_name.split("_")
        index = parts.index("repeated")
        if index + 1 < len(parts) and parts[index + 1].isdigit():
            n_samples = int(parts[index + 1])

    for name in reversed(MODEL_NAMES):
        if name in run_name:
            model_name = name
            break
    return model_name, temporal, dvs_mode, n_samples


def get_checkpoint_path(checkpoint_dir: str, run_id: str):
    """
    Get the path to the checkpoint file of a run.
    Args:
        checkpoint_dir (str): Directory where the checkpoint file is located.
        run_id (str): ID of the run.
    Returns:
        str: Path to the checkpoint file.
    """
    run_directory = os.path.join(checkpoint_dir, run_id)
    checkpoint_file = os.listdir(run_directory)
    if len(checkpoint_file) > 1:
        raise ValueError(f"More than one checkpoint file found for {run_id}.")
    return os.path.join(run_directory, checkpoint_file[0])


def load_model_from_run(
    run_name: str, run_id: str, checkpoint_dir: str, device: str = "cuda"
) -> pl.LightningModule:
    """
    Load a model from a W&B run.
    Args:
        run_name (str): Name of the W&B run.
        run_id (str): ID of the W&B run.
        checkpoint_dir (str): Directory where the checkpoint file is located.
        device (str): Device to load the model to.
    Returns:
        pl.LightningModule: Model from the W&B run.
    """
    model_name, temporal, dvs_mode, n_samples = extract_model_config_from_run(run_name)
    checkpoint_path = get_checkpoint_path(checkpoint_dir, run_id)

    spiking_model_kwargs = {
        "neuron_model": "plif",
        "surrogate_function": "sigmoid",
        "step_mode": "multi_step" if temporal else "single_step",
        "n_samples": n_samples,
    }
    model_kwargs = {
        "model_name": model_name,
        "dvs_mode": dvs_mode,
        "pos_weight": 1.0,
        "kwargs": spiking_model_kwargs,
    }
    if temporal:
        model = AVAILABLE_MODELS["temporal"].load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=torch.device(device),
            **model_kwargs,
        )
    else:

        model = AVAILABLE_MODELS["single_sample"].load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=torch.device(device),
            **model_kwargs,
        )

    return model


def sort_losses_paired_with_indexes(
    losses: List[float],
    preds: List[float],
    labels: List[float],
    dataset: torch.utils.data.Dataset,
) -> Tuple[List[float], List[str], List[float], List[float]]:
    """From the list of lossses, create indexes of the data points, sort them together and return losses with string paths to the data points
    extracted from dataset.
    Args:
        losses (List[float]): List of losses.
        preds (List[float]): List of predictions.
        labels (List[float]): List of labels.
        dataset (torch.utils.data.Dataset): Dataset used to create the losses.
    Returns:
        Tuple[List[float], List[str], List[float], List[float]]: Tuple of sorted losses, sorted paths,
    sorted predictions and sorted labels.
    """
    indexes = list(range(len(losses)))
    indexes.sort(key=lambda x: losses[x], reverse=True)
    sorted_losses = [losses[i] for i in indexes]
    sorted_paths = [dataset.all_clips[i] for i in indexes]
    sorted_preds = [preds[i] for i in indexes]
    sorted_labels = [labels[i] for i in indexes]
    return sorted_losses, sorted_paths, sorted_preds, sorted_labels


def main(args):
    # load the runs from W&B
    runs_dict = load_wandb_group_runs(args.project_name, args.group_name)
    # iterate over the runs
    pl.seed_everything(args.seed)
    bcle_loss = torch.nn.BCEWithLogitsLoss(reduction="none").to(args.device)
    for run_name, run_id in runs_dict.items():
        # load the model

        model = load_model_from_run(run_name, run_id, args.checkpoint_dir, args.device)
        # load the dataset
        dataset = load_dataset_object(args.dataset_save_dir, run_id, args.device)
        # create the dataloader
        dataloader = DataLoader(
            dataset,
            num_workers=8,
            prefetch_factor=2,
            batch_size=args.batch_size,
            shuffle=False,
        )
        model.eval()
        loss_list = []
        pred_list = []
        label_list = []
        with torch.no_grad():
            for x, y in tqdm.tqdm(dataloader, total=len(dataloader)):
                x = x.float().to(args.device)
                y = y.float().to(args.device)
                y_hat = model(x)
                loss_per_example = bcle_loss(y_hat, y).flatten().cpu().tolist()
                y_hat_sigmoid = torch.round(torch.sigmoid(y_hat)).cpu().tolist()
                print(y_hat_sigmoid)
                print(y.flatten().cpu().tolist())
                print(loss_per_example)
                pred_list.extend(y_hat_sigmoid)
                label_list.extend(y.flatten().cpu().tolist())
                loss_list.extend(loss_per_example)
        sorted_losses, sorted_paths, sorted_preds, sorted_labels = (
            sort_losses_paired_with_indexes(
                loss_list,
                pred_list,
                label_list,
                dataset,
            )
        )
        results = {
            "sorted_losses": sorted_losses,
            "sorted_paths": sorted_paths,
            "sorted_preds": sorted_preds,
            "sorted_labels": sorted_labels,
        }
        save_path = os.path.join(
            args.results_save_dir, args.project_name, args.group_name, run_name
        )
        os.makedirs(save_path, exist_ok=True)
        with open(
            os.path.join(save_path, "sorted_losses_predictions.json"), "w"
        ) as file:
            json.dump(results, file)
        print(f"Results saved to {save_path}")


if __name__ == "__main__":
    with open("./wandb_api_key.txt", "r") as file:
        os.environ["WANDB_API_KEY"] = file.read().strip()

    parser = ArgumentParser()
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--group_name", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--dataset_save_dir", type=str, required=True)
    parser.add_argument("--results_save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
