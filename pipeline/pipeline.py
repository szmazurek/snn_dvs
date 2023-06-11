import os
import shutil
from os.path import isfile, join
from os import listdir
import wandb
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from apng import APNG


class Pipeline:

    def __init__(self, path_clips: str, nn_class_model_rgb: nn.Module, nn_class_model_dvs: nn.Module, dataloader,
                 project_name_wandb: str):

        self.path_wandb_folder = os.path.join(os.path.abspath(os.getcwd()), "wandb")
        self.path_clips = path_clips
        self.path_clips_rgb = os.path.join(self.path_clips, "rgb")
        self.path_clips_dvs = os.path.join(self.path_clips, "dvs")

        self.nn_class_model_rgb = nn_class_model_rgb
        self.nn_class_model_dvs = nn_class_model_dvs

        self.project_name_wandb = project_name_wandb
        self.dataloader = dataloader

    def start(self):

        self._unpack_rgb()
        self._unpack_dvs()
        self._run(self.nn_class_model_rgb, self.path_clips_rgb, "Table_Rgb")
        self._run(self.nn_class_model_dvs, self.path_clips_dvs, "Table_Dvs")
        self._del_folder(self.path_wandb_folder)

    def delete_unpacked_dataset(self):
        self._del_folder(self.path_clips_rgb)
        self._del_folder(self.path_clips_dvs)

    def _run(self, nn_class_model, path_clips, wandb_table_name="Table"):
        wandb.init(
            project=self.project_name_wandb,
            entity="snn_team"
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = self.dataloader(path_clips)

        data_loader = DataLoader(dataset, shuffle=False, num_workers=12)
        nn_class_model.to(device)
        sample = []

        nn_class_model.eval()
        with torch.no_grad():
            for img, _, name in data_loader:
                img = img.to(device).float()
                img = img.squeeze(0)  # [1, 900, 1, W, H] -> [900, 1, W, H]
                out_fr = nn_class_model(img)

                if out_fr.dim() == 3:  # [900, 1, 2] -> [900, 2]
                    out_fr = out_fr.squeeze(1)
                pred = torch.argmax(out_fr, dim=1)
                loss = torch.abs(torch.sub(out_fr[:, 0], out_fr[:, 1]))
                loss_avg = loss.sum() / len(loss)

                for i, (frame, l, p) in enumerate(zip(img, loss, pred)):
                    mask_image = wandb.Image(frame,
                                             caption=f"Frame number: {i} | Diff: {str(l.item())} | Prediction: {str(p.item())}")

                    wandb.log({f"{name[0]}": mask_image})

                sample.append([name[0], loss_avg])

        my_table = wandb.Table(columns=["name_sample", "Diff_Avg"], data=sample)
        wandb.log({wandb_table_name: my_table})
        wandb.finish()

    def _unpack_rgb(self, image_size=(400, 150)):

        if not os.path.exists(self.path_clips_rgb):
            # os.mkdir(self.path_clips_rgb)
            os.makedirs(self.path_clips_rgb, exist_ok=True)
        else:
            print(f"In '{self.path_clips}' exist 'rgb' folder. Skipping unpack_rgb images.")
            return 0

        files_raw = [f for f in listdir(self.path_clips) if isfile(join(self.path_clips, f)) and "mp4" in f]
        for file in files_raw:
            base_name, ext = os.path.splitext(file)
            base_name_shortened = base_name[:-2]

            new_folder_name = os.path.join(self.path_clips_rgb, base_name_shortened)
            if not os.path.exists(new_folder_name):
                os.mkdir(new_folder_name)

            cap = cv2.VideoCapture(os.path.join(self.path_clips, file))

            if not cap.isOpened():
                print("Error opening video file")
            frame_id = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    resized_frame = cv2.resize(frame, image_size)
                    cv2.imwrite(
                        os.path.join(self.path_clips_rgb, new_folder_name, f'{frame_id}.jpg'), resized_frame)
                    frame_id += 1
                else:
                    break
        print("Successfully unpack_rgb ended")

    def _unpack_dvs(self):

        if not os.path.exists(self.path_clips_dvs):
            os.mkdir(self.path_clips_dvs)
        else:
            print(f"In '{self.path_clips}' exist 'dvs' folder. Skipping unpack_dvs images.")
            return 0

        files_raw = [f for f in listdir(self.path_clips) if isfile(join(self.path_clips, f)) and "apng" in f]

        for file in files_raw:
            base_name, ext = os.path.splitext(file)
            base_name_shortened = base_name[:-2]
            format_type = base_name[-1]

            if format_type == "1":
                continue

            new_folder_name = os.path.join(self.path_clips_dvs, base_name_shortened)
            if not os.path.exists(new_folder_name):
                os.mkdir(new_folder_name)

            im = APNG.open(os.path.join(self.path_clips, file))
            frame_id = 0
            for png, control in im.frames:
                png.save(
                    os.path.join(self.path_clips_dvs, new_folder_name, f'{frame_id}.jpg'))
                frame_id += 1

        print("Successfully unpack_dvs ended")

    def _del_folder(self, path_folder):
        try:
            print(f"Deleting {path_folder}")
            shutil.rmtree(path_folder)
        except FileNotFoundError:
            print(f"Cant delete folder {path_folder}, it doesnt exist")
        except Exception:
            print("Unexpected Exception")
