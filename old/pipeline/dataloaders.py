from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple
import torch
import os
import glob


class CarlaDataset(Dataset):
    def __init__(self, targ_dir: str) -> None:

        self.all_folders = [os.path.join(targ_dir, directory) for directory in os.listdir(targ_dir)]
        self.transform = transforms.Compose([
                transforms.Resize((150, 400)),
                transforms.PILToTensor()])

    def load_image(self, image) -> Image.Image:
        img = Image.open(image)
        return self.transform(img)

    def __len__(self) -> int:
        return len(self.all_folders)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        folder_path = self.all_folders[index]

        file_list = glob.glob(os.path.join(folder_path, "*.jpg"))
        file_list_sorted = sorted(file_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[0]))
        images_tensor = torch.cat([self.load_image(image).unsqueeze(0) for image in file_list_sorted])

        label_list = [int(str(image)[-5]) for image in file_list_sorted]
        label_tensor = torch.tensor(label_list).unsqueeze(1)
        name_sample = os.path.basename(folder_path)
        return images_tensor, label_tensor, name_sample


