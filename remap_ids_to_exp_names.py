import os
import wandb
import shutil
from argparse import ArgumentParser

api_key_file = open("./wandb_api_key.txt", "r")
API_KEY = api_key_file.read()
api_key_file.close()
os.environ["WANDB_API_KEY"] = API_KEY


def remap_ids_to_exp_names(target_dir, wandb_group_name, new_save_dir):
    """Remap the saved files with exp_id as a name to a real name of the
    experiment.
    Args:
        target_dir (str): path to the directory with saved files
        wandb_group_name (str): name of the wandb group with the experiments
        new_save_dir (str): path to the directory where the new files should be saved
    """

    if not os.path.exists(new_save_dir):
        os.makedirs(new_save_dir)

    wandb_api = wandb.Api()
    runs = wandb_api.runs(
        "mazurek/dvs_final_exp", filters={"group": wandb_group_name}
    )
    ids_to_names_map = {run.id: run.name for run in runs}
    filenames_in_folder = [
        filename.split(".")[0]
        for filename in os.listdir(target_dir)
        if filename.endswith(".pt")
    ]
    for filename in filenames_in_folder:
        if filename in ids_to_names_map.keys():
            shutil.copy(
                os.path.join(target_dir, filename + ".pt"),
                os.path.join(new_save_dir, ids_to_names_map[filename] + ".pt"),
            )
            print(f"Copied {filename} to {ids_to_names_map[filename]}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Path to the directory with saved files",
    )
    parser.add_argument(
        "--wandb_group_name",
        type=str,
        required=True,
        help="Name of the wandb group with the experiments",
    )
    parser.add_argument(
        "--new_save_dir",
        type=str,
        required=True,
        help="Path to the directory where the new files should be saved",
    )
    args = parser.parse_args()
    remap_ids_to_exp_names(
        args.target_dir, args.wandb_group_name, args.new_save_dir
    )
