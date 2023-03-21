from torch.utils.data import Dataset
from torch.utils.data import Subset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from typing import Tuple
import pathlib
import torch
import os
from torch.utils.data import DataLoader

class SmallSampleDvs(Dataset):

    def __init__(self, targ_dir: str) -> None:

        all_folder = [os.path.join(targ_dir, directory) for directory in os.listdir(targ_dir)]
        self.paths = []
        for folder in all_folder:
            self.paths.extend(list(pathlib.Path(folder).glob("*.png")))

        self.transform = transforms.Compose([
                transforms.Resize((600, 1600)),
                transforms.PILToTensor()])

    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self.load_image(index)
        class_name = int(str(self.paths[index])[-5])
        return self.transform(img), class_name


def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    _datasets = {'train': Subset(dataset, train_idx), 'val': Subset(dataset, val_idx)}
    return _datasets


# if __name__ == "__main__":
#     dataset = SmallSampleDvsSeq(r"D:\ProjectsPython\SpikingJelly\small-sample\imageDvs")
#     dataset = train_val_dataset(dataset)
#     train_data_loader = DataLoader(dataset['train'], batch_size=1, shuffle=True, num_workers=1)
