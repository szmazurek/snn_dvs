import os
import yaml
import wandb
import torch
import uuid
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader

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
    PredictionDatasetSingleStepRepeated
)
from typing import Dict, List, Type

AVAILABLE_DATASETS: Dict[str, Type[BaseDataset]] = {
    "single_sample": SingleSampleDataset,
    "repeated": RepeatedSampleDataset,
    "temporal": TemporalSampleDataset,
}

AVAILABLE_DATASETS_PREDICTION : Dict[str, Type[PredictionDataset]] = {
    "prediction_single_sample": PredictionDatasetSingleStep,
    "prediction_repeated": PredictionDatasetSingleStepRepeated,
    "prediction_temporal": PredictionDataset
}

AVAILABLE_MODELS: Dict[str, Type[pl.LightningModule]] = {
    "single_sample": LightningModuleNonTemporal,
    "temporal": LightningModuleTemporalNets,
}


torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_float32_matmul_precision("medium")

SCRIPT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def calculate_pos_weight(labels: List[int]) -> float:
    neg_count, pos_count = torch.unique(
        torch.tensor(labels), return_counts=True
    )[1]
    return neg_count / pos_count


def construct_datasets(
    parameters: dict,
    train_videos: List[str],
    val_videos: List[str],
    test_videos: List[str],
):
    """Construct the train, val and test datasets."""
    dataset_type = parameters["dataset"]["type"]
    img_width = parameters["dataset"]["img_width"]
    img_height = parameters["dataset"]["img_height"]
    dvs_mode = parameters["dataset"]["dvs_mode"]
    target_size = (img_width, img_height)
    if  "prediction" in dataset_type.lower():
        dataset_class = AVAILABLE_DATASETS_PREDICTION[dataset_type]
    else:
        dataset_class = AVAILABLE_DATASETS[dataset_type] # type: ignore
    if dataset_type == "single_sample":
        train_dataset = dataset_class(
            folder_list=train_videos,
            dvs_mode=dvs_mode,
            target_size=target_size,
        )
        val_dataset = dataset_class(
            folder_list=val_videos,
            dvs_mode=dvs_mode,
            target_size=target_size,
        )
        test_dataset = dataset_class(
            folder_list=test_videos,
            dvs_mode=dvs_mode,
            target_size=target_size,
        )
    elif dataset_type == "repeated":
        train_dataset = dataset_class(
            folder_list=train_videos,
            dvs_mode=dvs_mode,
            target_size=target_size,
            repeats=parameters["dataset"]["repeats"],
        ) 
        val_dataset = dataset_class(
            folder_list=val_videos,
            dvs_mode=dvs_mode,
            target_size=target_size,
            repeats=parameters["dataset"]["repeats"],
        )
        test_dataset = dataset_class(
            folder_list=test_videos,
            dvs_mode=dvs_mode,
            target_size=target_size,
            repeats=parameters["dataset"]["repeats"],
        )
    elif dataset_type == "temporal":
        train_dataset = dataset_class(
            folder_list=train_videos,
            dvs_mode=dvs_mode,
            target_size=target_size,
            sample_len=parameters["dataset"]["timestep"],
            overlap=parameters["dataset"]["overlap"],
        )
        val_dataset = dataset_class(
            folder_list=val_videos,
            dvs_mode=dvs_mode,
            target_size=target_size,
            sample_len=parameters["dataset"]["timestep"],
            overlap=parameters["dataset"]["overlap"],
        )
        test_dataset = dataset_class(
            folder_list=test_videos,
            dvs_mode=dvs_mode,
            target_size=target_size,
            sample_len=parameters["dataset"]["timestep"],
            #overlap=parameters["dataset"]["overlap"],
            overlap=0
        )
    elif dataset_type == "prediction_temporal": 
        train_dataset = dataset_class(
            folder_list=train_videos,
            dvs_mode=dvs_mode,
            target_size=target_size,
            sample_len=parameters["dataset"]["timestep"],
            overlap=parameters["dataset"]["overlap"],
            n_frames_predictive_horizon=parameters["dataset"]["n_frames_predictive_horizon"])
        val_dataset = dataset_class(
            folder_list=val_videos,
            dvs_mode=dvs_mode,
            target_size=target_size,
            sample_len=parameters["dataset"]["timestep"],
            overlap=parameters["dataset"]["overlap"],
            n_frames_predictive_horizon=parameters["dataset"]["n_frames_predictive_horizon"])
        test_dataset = dataset_class(
            folder_list=test_videos,
            dvs_mode=dvs_mode,
            target_size=target_size,
            sample_len=parameters["dataset"]["timestep"],
            overlap=parameters["dataset"]["overlap"],
            n_frames_predictive_horizon=parameters["dataset"]["n_frames_predictive_horizon"])
    elif dataset_type == "prediction_single_sample":
        train_dataset = dataset_class(
            folder_list=train_videos,
            dvs_mode=dvs_mode,
            target_size=target_size,
            n_frames_predictive_horizon=parameters["dataset"]["n_frames_predictive_horizon"])
        val_dataset = dataset_class(
            folder_list=val_videos,
            dvs_mode=dvs_mode,
            target_size=target_size,
            n_frames_predictive_horizon=parameters["dataset"]["n_frames_predictive_horizon"])
        test_dataset = dataset_class(
            folder_list=test_videos,
            dvs_mode=dvs_mode,
            target_size=target_size,
            n_frames_predictive_horizon=parameters["dataset"]["n_frames_predictive_horizon"])
    elif dataset_type == "prediction_repeated":
        train_dataset = dataset_class(
            folder_list=train_videos,
            dvs_mode=dvs_mode,
            target_size=target_size,
            repeats=parameters["dataset"]["repeats"],
            n_frames_predictive_horizon=parameters["dataset"]["n_frames_predictive_horizon"])
        val_dataset = dataset_class(
            folder_list=val_videos,
            dvs_mode=dvs_mode,
            target_size=target_size,
            repeats=parameters["dataset"]["repeats"],
            n_frames_predictive_horizon=parameters["dataset"]["n_frames_predictive_horizon"])
        test_dataset = dataset_class(
            folder_list=test_videos,
            dvs_mode=dvs_mode,
            target_size=target_size,
            repeats=parameters["dataset"]["repeats"],
            n_frames_predictive_horizon=parameters["dataset"]["n_frames_predictive_horizon"])

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    return train_dataset, val_dataset, test_dataset


def train_normal_loop(parameters: dict):
    """Train the model using the normal training loop."""
    # general params
    EPOCHS = parameters["epochs"]
    BATCH_SIZE = parameters["batch_size"]
    LR = parameters["lr"]
    WEIGHT_DECAY = parameters["weight_decay"]
    EARLY_STOPPING_PATIENCE = parameters["early_stopping_patience"]
    CHECKPOINT_PATH = parameters["checkpoint_path"]
    DATASET_SAVE_DIR = parameters["dataset_save_dir"] if "dataset_save_dir" in parameters.keys() else None
    RESTORE_CHECKPOINT_PATH = parameters["checkpoint_restore_path"] if "checkpoint_restore_path" in parameters.keys() else None
    # model params
    MODEL_NAME = parameters["model"]["name"]
    MODEL_TYPE = parameters["model"]["type"]
    if "spiking" in MODEL_NAME.lower():
        SPIKING_PARAMS = parameters["model"]["spiking_params"]
    else:
        SPIKING_PARAMS = None
    # dataset params
    DATA_ROOT_PATH = parameters["dataset"]["root_path"]
    if "weather" in os.path.basename(DATA_ROOT_PATH).lower():
        DATASET_SUBSET = "weather"
    else:
        DATASET_SUBSET = "full_ds"
    DATASET_TYPE = parameters["dataset"]["type"]
    if "repeated" in DATASET_TYPE.lower():
        REPEATS = parameters["dataset"]["repeats"]
    elif "temporal" in DATASET_TYPE.lower():
        TIMESTEP = parameters["dataset"]["timestep"]
        OVERLAP = parameters["dataset"]["overlap"]
    if "prediction" in DATASET_TYPE.lower() :
        N_FRAMES_PREDICTIVE_HORIZON = parameters["dataset"]["n_frames_predictive_horizon"]
    DVS_MODE = parameters["dataset"]["dvs_mode"]
    MODALITY = "DVS" if DVS_MODE else "RGB"
    VAL_SIZE = parameters["dataset"]["val_size"]
    TEST_SIZE = parameters["dataset"]["test_size"]
    IMG_SIZE = (
        parameters["dataset"]["img_width"],
        parameters["dataset"]["img_height"],
    )
    # wandb params
    WANDB_ENTITY = parameters["wandb"]["entity"]
    WANDB_PROJECT = parameters["wandb"]["project"]
    WANDB_GROUP = parameters["wandb"]["group"]

    if DATASET_TYPE == "repeated":
        WANDB_NAME = (
            f"{MODEL_NAME}_{DATASET_TYPE}_{REPEATS}_{DATASET_SUBSET}_{MODALITY}_{IMG_SIZE[0]}x{IMG_SIZE[1]}"
        )
    elif DATASET_TYPE == "temporal":
        WANDB_NAME = (
            f"{MODEL_NAME}_{DATASET_TYPE}_{MODALITY}_{DATASET_SUBSET}_{IMG_SIZE[0]}x{IMG_SIZE[1]}_{TIMESTEP}_{OVERLAP}"
        )
    elif DATASET_TYPE == "prediction_temporal":
        WANDB_NAME = (
            f"{MODEL_NAME}_{DATASET_TYPE}_{MODALITY}_{DATASET_SUBSET}_{IMG_SIZE[0]}x{IMG_SIZE[1]}_{TIMESTEP}_{OVERLAP}_horizon_{N_FRAMES_PREDICTIVE_HORIZON}"
        )
    elif DATASET_TYPE == "prediction_repeated":
        WANDB_NAME = (
            f"{MODEL_NAME}_{DATASET_TYPE}_{REPEATS}_{MODALITY}_{DATASET_SUBSET}_{IMG_SIZE[0]}x{IMG_SIZE[1]}_horizon_{N_FRAMES_PREDICTIVE_HORIZON}"
        )
    elif DATASET_TYPE == "prediction_single_sample":
        WANDB_NAME = (
            f"{MODEL_NAME}_{DATASET_TYPE}_{MODALITY}_{DATASET_SUBSET}_{IMG_SIZE[0]}x{IMG_SIZE[1]}_horizon_{N_FRAMES_PREDICTIVE_HORIZON}"
        )
    else:
        WANDB_NAME = f"{MODEL_NAME}_{DATASET_TYPE}_{MODALITY}_{DATASET_SUBSET}_{IMG_SIZE[0]}x{IMG_SIZE[1]}"
    video_list = [
        os.path.join(DATA_ROOT_PATH, video)
        for video in os.listdir(DATA_ROOT_PATH)
    ]
    # videos do not have labels, no stratification
    train_videos, test_videos = train_test_split(
        video_list, test_size=TEST_SIZE, random_state=parameters["seed"]
    )
    train_videos, val_videos = train_test_split(
        train_videos, test_size=VAL_SIZE, random_state=parameters["seed"]
    )
    train_dataset, val_dataset, test_dataset = construct_datasets(
        parameters, train_videos, val_videos, test_videos
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Testo videos: {test_videos}")
    pos_weights = calculate_pos_weight(train_dataset.all_labels)
    print(pos_weights)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        prefetch_factor=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=6,
        prefetch_factor=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=6,
        prefetch_factor=4,
        pin_memory=True,
    )
    run_id = str(uuid.uuid4())
    if int(os.environ["SLURM_PROCID"]) == 0:
        wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                group=WANDB_GROUP,
                name=WANDB_NAME,
                config=parameters
            )
        wandb.log({"run_id": run_id})
   
    wandb_logger = pl.loggers.WandbLogger(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name=WANDB_NAME,
    )
   
    checkpoint_path_run = os.path.join(CHECKPOINT_PATH, run_id)

    # callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_path_run,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        mode="min",
        verbose=True,
        min_delta=0.001,
    )

   
    trainer = pl.Trainer(
        devices="auto",
        max_epochs=EPOCHS,
        accelerator="auto",
        precision="bf16-mixed",
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        sync_batchnorm=True,
        fast_dev_run=False,
        deterministic=False,
        num_nodes=1
    )
    # pl.seed_everything(parameters["seed"],workers=True)  # reset seed for reproducibility
    model = AVAILABLE_MODELS[MODEL_TYPE](
        model_name=MODEL_NAME,
        dvs_mode=DVS_MODE,
        pos_weight=pos_weights,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        kwargs=SPIKING_PARAMS if SPIKING_PARAMS else {},
    
    )
    trainer.fit(model, train_loader, val_loader,ckpt_path=RESTORE_CHECKPOINT_PATH)
    # configure new trainer for testing
    if int(os.environ["SLURM_PROCID"]) == 0:
        wandb_logger = pl.loggers.WandbLogger(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            group=WANDB_GROUP,
            name=WANDB_NAME,
        )
        trainer = pl.Trainer(
            devices=1,
            accelerator="auto",
            precision="bf16-mixed",
            logger=wandb_logger,
            num_nodes=1
        )
        
            
        checkpoint_root_path = os.path.join(SCRIPT_ROOT_DIR, checkpoint_path_run)
        best_checkpoint = os.listdir(checkpoint_root_path)[0]
        best_checkpoint_path = os.path.join(checkpoint_root_path, best_checkpoint)
        trainer.test(model=model,dataloaders=test_loader, ckpt_path=best_checkpoint_path)
        wandb.finish()
        if DATASET_SAVE_DIR is not None:
            # saving test_dataset
            if not os.path.exists(DATASET_SAVE_DIR):
                os.makedirs(DATASET_SAVE_DIR)
            dataset_save_path = os.path.join(DATASET_SAVE_DIR, run_id + ".pt")
            torch.save(test_dataset,dataset_save_path)
            


def train_lightning_kfold(parameters):
    """Train the model using the normal training loop."""
    pl.seed_everything(parameters["seed"])
    # general params
    EPOCHS = parameters["epochs"]
    BATCH_SIZE = parameters["batch_size"]
    LR = parameters["lr"]
    WEIGHT_DECAY = parameters["weight_decay"]
    EARLY_STOPPING_PATIENCE = parameters["early_stopping_patience"]
    CHECKPOINT_PATH = parameters["checkpoint_path"]
    DATASET_SAVE_DIR = parameters["dataset_save_dir"]
    # model params
    MODEL_NAME = parameters["model"]["name"]
    MODEL_TYPE = parameters["model"]["type"]
    if "spiking" in MODEL_NAME.lower():
        SPIKING_PARAMS = parameters["model"]["spiking_params"]
    else:
        SPIKING_PARAMS = None
    # dataset params
    DATA_ROOT_PATH = parameters["dataset"]["root_path"]
    DVS_MODE = parameters["dataset"]["dvs_mode"]
    MODALITY = "DVS" if DVS_MODE else "RGB"
    VAL_SIZE = parameters["dataset"]["val_size"]
    N_FOLDS = parameters["dataset"]["n_folds"]
    IMG_SIZE = (
        parameters["dataset"]["img_width"],
        parameters["dataset"]["img_height"],
    )
    # wandb params
    WANDB_ENTITY = parameters["wandb"]["entity"]
    WANDB_PROJECT = parameters["wandb"]["project"]
    WANDB_GROUP = parameters["wandb"]["group"]
    WANDB_JOB_TYPE = (
        f"{MODEL_NAME}_{MODEL_TYPE}_{MODALITY}_{IMG_SIZE[0]}x{IMG_SIZE[1]}"
    )

    video_list = [
        os.path.join(DATA_ROOT_PATH, video)
        for video in os.listdir(DATA_ROOT_PATH)
    ]
    kfold_splitter = KFold(
        n_splits=N_FOLDS, shuffle=True, random_state=parameters["seed"]
    )
    for fold, (train_index, test_index) in enumerate(
        kfold_splitter.split(video_list)
    ):
        WANDB_NAME = f"Fold {fold+1}"
        train_videos = [video_list[i] for i in train_index]
        test_videos = [video_list[i] for i in test_index]
        train_videos, val_videos = train_test_split(
            train_videos, test_size=VAL_SIZE, random_state=parameters["seed"]
        )
        train_dataset, val_dataset, test_dataset = construct_datasets(
            parameters, train_videos, val_videos, test_videos
        )
        pos_weights = calculate_pos_weight(train_dataset.all_labels)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=6,
            prefetch_factor=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=6,
            prefetch_factor=4,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=6,
            prefetch_factor=4,
            pin_memory=True,
        )

        # callbacks
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=CHECKPOINT_PATH,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=False,
        )
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            mode="min",
            verbose=True,
            min_delta=0.001,
        )

        wandb_logger = pl.loggers.WandbLogger(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            group=WANDB_GROUP,
            name=WANDB_NAME,
        )
        wandb_run_id = wandb_logger._id
        trainer = pl.Trainer(
            devices="auto",
            max_epochs=EPOCHS,
            accelerator="auto",
            precision=16,
            callbacks=[checkpoint_callback, early_stopping_callback],
            logger=wandb_logger,
            log_every_n_steps=1,
        )
        pl.seed_everything(
            parameters["seed"]
        )  # reset seed for reproducibility
        model = AVAILABLE_MODELS[MODEL_TYPE](
            model_name=MODEL_NAME,
            dvs_mode=DVS_MODE,
            pos_weight=pos_weights,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            kwargs=SPIKING_PARAMS if SPIKING_PARAMS else {},
        )
        trainer.fit(model, train_loader, val_loader)
        trainer.test(dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":
    
    with open("./config.yml", "r") as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)
    pl.seed_everything(parameters["seed"],workers=True)
    if parameters["dataset"]["evaluation_mode"] == "kfold":
        train_lightning_kfold(parameters)
    else:
        train_normal_loop(parameters)
