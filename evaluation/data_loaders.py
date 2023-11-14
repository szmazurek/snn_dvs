import os
import glob
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, List
import re
from operator import itemgetter
from itertools import groupby


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
    Args:
        target_dir (str): directory with subfolders containing video
    frames.
        sample_len (int, optional): length of a sample. Defaults to 4.
        overlap (int, optional): overlap between samples. Defaults to 0.
        per_frame_label_mode (bool, optional): if True, every frame in
    a sample will have a label. Defaults to False.
    """

    def __init__(
        self,
        target_dir: str,
        sample_len: int = 4,
        overlap: int = 0,
        per_frame_label_mode: bool = False,
    ) -> None:
        self.all_folders = [
            os.path.join(target_dir, directory) for directory in os.listdir(target_dir)
        ]
        all_files_png = [
            glob.glob(os.path.join(folders, "*.png")) for folders in self.all_folders
        ]
        all_files_jpg = [
            glob.glob(os.path.join(folders, "*.jpg")) for folders in self.all_folders
        ]
        all_files = all_files_png + all_files_jpg
        all_files = [e for sub in all_files for e in sub]

        all_files_sorted = self.sort_frames(all_files)
        widnowed_samples = self.create_windowed_samples(
            all_files_sorted, sample_len, overlap
        )
        self.all_samples = widnowed_samples

        per_frame_labels = [
            self.create_window_labels(sample) for sample in widnowed_samples
        ]
        # labels_count kept for the purposes of class balancing in
        # training with BCEWithLogitsLoss
        if not per_frame_label_mode:
            per_window_labels = [
                int(max(window_labels)) for window_labels in per_frame_labels
            ]
            self.all_labels = per_window_labels
            self.labels_count = per_frame_labels
        else:
            self.all_labels = per_frame_labels
            self.labels_count = [
                int(os.path.splitext(os.path.basename(file_path))[0].split("-")[-1])
                for file_path in all_files_sorted
            ]
        self.time_dim = sample_len
        self.overlap = overlap
        self.transform = transforms.Compose(
            [
                transforms.Resize((150, 400), interpolation=Image.NEAREST),
                transforms.RandomHorizontalFlip(),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
            ]
        )

    @staticmethod
    def create_window_labels(window: List[str]) -> List[int]:
        """Creates a list of labels for a given windowed sample."""
        labels_per_frame = [
            int(os.path.splitext(os.path.basename(file_path))[0].split("-")[-1])
            for file_path in window
        ]
        return labels_per_frame

    @staticmethod
    def create_windowed_samples(
        filenames: List[str], sample_len: int, overlap: int
    ) -> List[List[str]]:
        """Creates a list of lists, where each sublist contains
        filenames of frames that belong to a single sample.
        """
        windowed_samples = []

        for i in range(0, len(filenames) - sample_len + 1, sample_len - overlap):
            candidate_window = filenames[i : i + sample_len]
            videos_in_window = set(
                [os.path.dirname(filename) for filename in candidate_window]
            )
            if len(videos_in_window) == 1:
                windowed_samples.append(candidate_window)

        return windowed_samples

    @staticmethod
    def sort_frames(filenames: List[str]) -> List[str]:
        parsed_filenames = []
        for filename in filenames:
            video_id = os.path.dirname(filename)
            frame_number = int(
                re.search(r"(\d+)-", os.path.basename(filename)).group(1)
            )
            parsed_filenames.append((video_id, frame_number, filename))

        # Sort by video ID and frame number
        parsed_filenames.sort(key=itemgetter(0, 1))

        # Group by video ID
        grouped_filenames = []
        for video_id, group in groupby(parsed_filenames, key=itemgetter(0)):
            grouped_filenames.extend(list(group))

        # Flatten the list and extract filenames
        sorted_filenames = [filename for _, _, filename in grouped_filenames]

        return sorted_filenames

    # Now grouped_filenames is a list of lists, where each sublist contains
    def load_image(self, image) -> torch.Tensor:
        img = Image.open(image)
        return self.transform(img)

    def __len__(self) -> int:
        return len(self.all_samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        file_path = self.all_samples[index]
        window = torch.cat([self.load_image(image) for image in file_path]).unsqueeze(1)
        label = torch.tensor(self.all_labels[index])
        return window, label


# class DvsDatasetOld(Dataset):
#     def __init__(self, target_dir: str, decimation: int = 1) -> None:
#         self.decimation = decimation
#         self.all_folders = [
#             os.path.join(target_dir, directory) for directory in os.listdir(target_dir)
#         ]
#         self.random_flip = 0
#         self.transform = transforms.Compose(
#             [
#                 transforms.RandomHorizontalFlip(0),
#                 transforms.Resize((150, 400), interpolation=Image.NEAREST),
#                 transforms.PILToTensor(),
#             ]
#         )
#         self.transform_flip = transforms.Compose(
#             [
#                 transforms.RandomHorizontalFlip(1),
#                 transforms.Resize((150, 400), interpolation=Image.NEAREST),
#                 transforms.PILToTensor(),
#             ]
#         )

#     def load_image(self, image) -> Image.Image:
#         if self.random_flip:
#             return self.transform(Image.open(image))
#         else:
#             return self.transform_flip(Image.open(image))

#     def __len__(self) -> int:
#         return len(self.all_folders)

#     def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
#         self.random_flip = random.getrandbits(1)
#         folder_path = self.all_folders[index]
#         file_list = glob.glob(os.path.join(folder_path, "*.png"))
#         file_list_sorted = sorted(
#             file_list,
#             key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[0]),
#         )
#         images_tensor = torch.cat(
#             [self.load_image(image).unsqueeze(0) for image in file_list_sorted]
#         )

#         label_list = [int(str(image)[-5]) for image in file_list_sorted]
#         label_tensor = torch.tensor(label_list).unsqueeze(1)
#         # Decimation
#         images_tensor = images_tensor[:: self.decimation]
#         label_tensor = label_tensor[:: self.decimation]
#         return images_tensor, label_tensor
