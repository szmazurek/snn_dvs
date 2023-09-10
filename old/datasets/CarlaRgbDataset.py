from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple
import torch
import os
import glob


class CarlaRGBDataset(Dataset):
    def __init__(self, targ_dir: str) -> None:

        self.all_folders = [os.path.join(targ_dir, directory) for directory in os.listdir(targ_dir)]
        self.all_files = [glob.glob(os.path.join( folders, "*.jpg")) for folders in self.all_folders]
        self.all_files = [e for sub in self.all_files for e in sub]
        self.transform = transforms.Compose([
                transforms.Resize((150 // 4, 1600 // 4)),
                transforms.RandomHorizontalFlip(),
                transforms.PILToTensor()])

    def load_image(self, image) -> torch.Tensor:
        img = Image.open(image)
        return self.transform(img)

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        file_path = self.all_files[index]
        image = self.load_image(file_path)

        label = int(os.path.splitext(os.path.basename(file_path))[0].split("-")[1])
        return image, label
