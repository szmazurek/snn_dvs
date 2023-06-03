import glob
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple
import torch
import os
import random


class CarlaPredictionDvsDataset(Dataset):
    def __init__(self, targ_dir: str) -> None:
        self.all_folders = [os.path.join(targ_dir, directory) for directory in os.listdir(targ_dir)]
        self.random_flip = 0
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(self.random_flip),
            transforms.Resize((600 // 2, 1600 // 2), interpolation=Image.NEAREST),
            transforms.PILToTensor()])

    def load_image(self, image) -> Image.Image:
        return self.transform(Image.open(image))

    def __len__(self) -> int:
        return len(self.all_folders)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        self.random_flip = random.getrandbits(1)
        folder_path = self.all_folders[index]
        file_list = glob.glob(os.path.join(folder_path, "*.png"))
        file_list_sorted = sorted(file_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[0]))
        images_tensor = torch.cat([self.load_image(image).unsqueeze(0) for image in file_list_sorted])

        label_list = [int(str(image)[-5]) for image in file_list_sorted]
        label_tensor = torch.tensor(label_list).unsqueeze(1)

        return images_tensor, label_tensor
