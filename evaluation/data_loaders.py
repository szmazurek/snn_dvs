import os
import glob
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple


class RGBDataset(Dataset):
    def __init__(self, target_dir: str) -> None:
        self.all_folders = [
            os.path.join(target_dir, directory) for directory in os.listdir(target_dir)
        ]
        all_files_png = [
            glob.glob(os.path.join(folders, "*.png")) for folders in self.all_folders
        ]
        all_files_jpg = [
            glob.glob(os.path.join(folders, "*.jpg")) for folders in self.all_folders
        ]
        self.all_files = all_files_png + all_files_jpg
        self.all_files = [e for sub in self.all_files for e in sub]

        self.all_labels = [
            int(os.path.splitext(os.path.basename(file_path))[0].split("-")[1])
            for file_path in self.all_files
        ]
        self.transform = transforms.Compose(
            [
                transforms.Resize((150, 400)),
                transforms.RandomHorizontalFlip(),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load_image(self, image) -> torch.Tensor:
        img = Image.open(image)
        return self.transform(img)

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        file_path = self.all_files[index]
        image = self.load_image(file_path)
        label = self.all_labels[index]

        return image, label


class DVSDatasetAsRGB(Dataset):
    """A class that loads everything the same as RGB one, just with
    with different resize method and not normalizing to ImageNet images
    """

    def __init__(self, target_dir: str) -> None:
        self.all_folders = [
            os.path.join(target_dir, directory) for directory in os.listdir(target_dir)
        ]
        all_files_png = [
            glob.glob(os.path.join(folders, "*.png")) for folders in self.all_folders
        ]
        all_files_jpg = [
            glob.glob(os.path.join(folders, "*.jpg")) for folders in self.all_folders
        ]
        self.all_files = all_files_png + all_files_jpg
        self.all_files = [e for sub in self.all_files for e in sub]

        self.all_labels = [
            int(os.path.splitext(os.path.basename(file_path))[0].split("-")[1])
            for file_path in self.all_files
        ]
        self.transform = transforms.Compose(
            [
                transforms.Resize((150, 400), interpolation=Image.NEAREST),
                transforms.RandomHorizontalFlip(),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
            ]
        )

    def load_image(self, image) -> torch.Tensor:
        img = Image.open(image)
        return self.transform(img)

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        file_path = self.all_files[index]
        image = self.load_image(file_path)
        label = self.all_labels[index]

        return image, label


class DVSDatasetRepeated(Dataset):
    """Abstraction representing a DVS based dataset, where eery sample
    is artificially transformed into timeseries via repeating one
    frame time_dim times.
    """

    def __init__(self, target_dir: str, time_dim: int = 32) -> None:
        self.all_folders = [
            os.path.join(target_dir, directory) for directory in os.listdir(target_dir)
        ]
        all_files_png = [
            glob.glob(os.path.join(folders, "*.png")) for folders in self.all_folders
        ]
        all_files_jpg = [
            glob.glob(os.path.join(folders, "*.jpg")) for folders in self.all_folders
        ]
        self.all_files = all_files_png + all_files_jpg
        self.all_files = [e for sub in self.all_files for e in sub]

        self.all_labels = [
            int(os.path.splitext(os.path.basename(file_path))[0].split("-")[1])
            for file_path in self.all_files
        ]
        self.time_dim = time_dim
        self.transform = transforms.Compose(
            [
                transforms.Resize((150, 400), interpolation=Image.NEAREST),
                transforms.RandomHorizontalFlip(),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
            ]
        )

    def load_image(self, image) -> torch.Tensor:
        img = Image.open(image)
        return self.transform(img)

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        file_path = self.all_files[index]
        image = self.load_image(file_path).repeat(self.time_dim, 1, 1, 1)
        label = torch.tensor(self.all_labels[index])

        return image, label


class DVSDatasetProper(Dataset):
    """Abstraction representing the dataset for DVS data, where every
    sample consists of n_frames from a given video. Length and overlap
    of a sample can be manipulated.
    """

    def __init__(self, target_dir: str, sample_len: int = 4, overlap : int =0) -> None:
        self.all_folders = [
            os.path.join(target_dir, directory) for directory in os.listdir(target_dir)
        ]
        all_files_png = [
            glob.glob(os.path.join(folders, "*.png")) for folders in self.all_folders
        ]
        all_files_jpg = [
            glob.glob(os.path.join(folders, "*.jpg")) for folders in self.all_folders
        ]
        self.all_files = all_files_png + all_files_jpg
        self.all_files = [e for sub in self.all_files for e in sub]

        self.all_labels = [
            int(os.path.splitext(os.path.basename(file_path))[0].split("-")[1])
            for file_path in self.all_files
        ]
        self.time_dim = overlap
        self.transform = transforms.Compose(
            [
                transforms.Resize((150, 400), interpolation=Image.NEAREST),
                transforms.RandomHorizontalFlip(),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
            ]
        )

    def load_image(self, image) -> torch.Tensor:
        img = Image.open(image)
        return self.transform(img)

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        file_path = self.all_files[index]
        image = self.load_image(file_path).repeat(self.time_dim, 1, 1, 1)
        label = torch.tensor(self.all_labels[index])

        return image, label


class DvsDataset(Dataset):
    def __init__(self, target_dir: str, decimation: int = 1) -> None:
        self.decimation = decimation
        self.all_folders = [
            os.path.join(target_dir, directory) for directory in os.listdir(target_dir)
        ]
        self.random_flip = 0
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0),
                transforms.Resize((150, 400), interpolation=Image.NEAREST),
                transforms.PILToTensor(),
            ]
        )
        self.transform_flip = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(1),
                transforms.Resize((150, 400), interpolation=Image.NEAREST),
                transforms.PILToTensor(),
            ]
        )

    def load_image(self, image) -> Image.Image:
        if self.random_flip:
            return self.transform(Image.open(image))
        else:
            return self.transform_flip(Image.open(image))

    def __len__(self) -> int:
        return len(self.all_folders)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        self.random_flip = random.getrandbits(1)
        folder_path = self.all_folders[index]
        file_list = glob.glob(os.path.join(folder_path, "*.png"))
        file_list_sorted = sorted(
            file_list,
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[0]),
        )
        images_tensor = torch.cat(
            [self.load_image(image).unsqueeze(0) for image in file_list_sorted]
        )

        label_list = [int(str(image)[-5]) for image in file_list_sorted]
        label_tensor = torch.tensor(label_list).unsqueeze(1)
        # Decimation
        images_tensor = images_tensor[:: self.decimation]
        label_tensor = label_tensor[:: self.decimation]
        return images_tensor, label_tensor
