import os
import glob
import torch
import re

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Tuple, List, Union
from operator import itemgetter
from itertools import groupby
from abc import abstractmethod


class BaseDataset(Dataset):
    def __init__(
        self,
        folder_list: List[str],
        target_size: Tuple[int, int] = (600, 1600),
        dvs_mode: bool = False,
        repeats: Union[int, None] = None,
        sample_len: Union[int, None] = None,
        overlap: Union[int, None] = None,
        per_frame_label_mode: Union[bool, None] = None,
    ) -> None:
        """Base class for all datasets. It takes a list of folders
        representing separate videos and returns a single sample from a video.
        Args:
            folder_list (List[str]): list of folders with videos
            target_size (Tuple[int, int], optional): size of the image after
        resizing. Defaults to (600, 1600).
            dvs_mode (bool, optional): flag indicating if DVS data will be used.
        Used for choosing the proper transforms. Defaults to False.
        """

        self.target_size = target_size
        self.dvs_mode = dvs_mode
        all_files_png = [
            glob.glob(os.path.join(folders, "*.png"))
            for folders in folder_list
        ]
        all_files_jpg = [
            glob.glob(os.path.join(folders, "*.jpg"))
            for folders in folder_list
        ]
        all_files = all_files_png + all_files_jpg

        self.all_files = [e for sub in all_files for e in sub]
        self.transform = self._get_transforms()

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def load_image(self, image: str) -> torch.Tensor:
        """Loads image from a given path and applies transforms."""
        img = Image.open(image)
        return self.transform(img)

    def _get_transforms(self) -> transforms.Compose:
        if self.dvs_mode:
            return transforms.Compose(
                [
                    transforms.Resize(
                        self.target_size, interpolation=Image.NEAREST
                    ),
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize(self.target_size),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class SingleSampleDataset(BaseDataset):
    """A class that loads a single sample from a video."""

    def __init__(
        self,
        folder_list: List[str],
        target_size: Tuple[int, int] = (600, 1600),
        dvs_mode: bool = False,
    ) -> None:
        super().__init__(folder_list, target_size, dvs_mode)
        self.all_labels = [
            int(os.path.splitext(os.path.basename(file_path))[0].split("-")[1])
            for file_path in self.all_files
        ]

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.all_files[index]
        image = self.load_image(file_path)
        label = torch.tensor(self.all_labels[index])
        return image, label


class TemporalSampleDataset(BaseDataset):
    """A class that loads clips consisting of multiple frames from a video."""

    def __init__(
        self,
        folder_list: List[str],
        target_size: Tuple[int, int] = (600, 1600),
        sample_len: int = 4,
        overlap: int = 0,
        per_frame_label_mode: bool = False,
        dvs_mode: bool = False,
    ) -> None:
        super().__init__(folder_list, target_size, dvs_mode)
        self.sample_len = sample_len
        self.overlap = overlap
        self.per_frame_label_mode = per_frame_label_mode
        # sort frames and create windowed samples
        all_files_sorted = self.sort_frames(self.all_files)
        self.all_cilp_samples = self.create_windowed_samples(
            all_files_sorted, sample_len, overlap
        )
        self.all_labels = self.create_labels()

    def create_labels(self):
        def create_window_labels(window: List[str]) -> List[int]:
            """Creates a list of labels for a given windowed sample.
            Args:
                window (List[str]): list of filenames
            Returns:
                List[int]: list of labels
            """
            labels_per_frame = [
                int(
                    os.path.splitext(os.path.basename(file_path))[0].split(
                        "-"
                    )[-1]
                )
                for file_path in window
            ]
            return labels_per_frame

        per_frame_labels = [
            create_window_labels(sample) for sample in self.all_cilp_samples
        ]
        if not self.per_frame_label_mode:
            per_window_labels = [
                int(max(window_labels)) for window_labels in per_frame_labels
            ]
            return per_window_labels
        else:
            print("per_frame_label_mode is not implemented yet")
            raise NotImplementedError

    @staticmethod
    def create_windowed_samples(
        filenames: List[str], sample_len: int, overlap: int
    ) -> List[List[str]]:
        """Creates a list of lists, where each sublist contains
        filenames of frames that belong to a single sample.
        Args:
            filenames (List[str]): list of filenames
            sample_len (int): length of a sample
            overlap (int): overlap between samples (in frames)
        Returns:
            List[List[str]]: list of lists of filenames
        """
        windowed_samples = []

        for i in range(
            0, len(filenames) - sample_len + 1, sample_len - overlap
        ):
            candidate_window = filenames[i : i + sample_len]
            videos_in_window = set(
                [os.path.dirname(filename) for filename in candidate_window]
            )
            if len(videos_in_window) == 1:
                windowed_samples.append(candidate_window)

        return windowed_samples

    @staticmethod
    def sort_frames(filenames: List[str]) -> List[str]:
        """
        Sorts the filenames so that frame order is preserved.
        Args:
            filenames (List[str]): list of filenames
        Returns:
            List[str]: sorted list of filenames
        """
        parsed_filenames = []
        for filename in filenames:
            video_id = os.path.dirname(filename)
            match = re.search(r"(\d+)-", os.path.basename(filename))
            if match is None:
                raise ValueError(
                    f"Filename {filename} does not match the pattern"
                )
            frame_number = int(match.group(1))

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

    def __len__(self) -> int:
        return len(self.all_cilp_samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.all_cilp_samples[index]
        window = torch.stack(
            [self.load_image(image) for image in file_path]
        )
        label = torch.tensor(self.all_labels[index])
        return window, label


class RepeatedSampleDataset(SingleSampleDataset):
    """A class that loads a single sample from a video and repeats it
    multiple times.
    """

    def __init__(
        self,
        folder_list: List[str],
        target_size: Tuple[int, int] = (600, 1600),
        dvs_mode: bool = False,
        repeats: int = 1,
    ) -> None:
        super().__init__(folder_list, target_size, dvs_mode)
        self.repeats = repeats

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.all_files[index]
        image = self.load_image(file_path).repeat(self.repeats, 1, 1, 1)
        label = torch.tensor(self.all_labels[index])
        return image, label


class PredictionDataset(Dataset):
    """ A class that first extracts n_frames right BEFORE the event in the video,
    and then the same number of frames from videos where the event did not occur.
    Then it creates frames of length given by timestep and overlap parameters.
    """
    def __init__(self,
                 folder_list: List[str],
                target_size: Tuple[int, int] = (600, 1600),
                sample_len: int = 4,
                overlap: int = 0,
                n_frames_predictive_horizon: int = 10,
                dvs_mode: bool = False,
                repeats : int = 1
                ) -> None:
        self.folder_list = folder_list
        self.target_size = target_size
        self.sample_len = sample_len
        self.overlap = overlap
        self.n_frames_predictive_horizon = n_frames_predictive_horizon
        self.dvs_mode = dvs_mode
        self.all_clips :List[List[str]]= []
        self.all_labels : List[int]= []
        self.folder_without_events : List[str]= []
        self.transform = self._get_transforms()
        self._perform_full_sample_extraction()

    def __len__(self) -> int:
        return len(self.all_clips)

    def _perform_full_sample_extraction(self) -> None:
        for folder in self.folder_list:
            self._extract_frames_from_video_with_event(folder)
        n_negative_samples_per_video = len(self.all_clips) // len(self.folder_without_events)
        for folder in self.folder_without_events:
            self._extract_clips_from_video_without_event(folder, n_negative_samples_per_video)
        return None

    def _extract_frames_from_video_with_event(self, folder: str) -> None:
        all_files = self._extract_all_files_from_folder(folder)
        all_files = self.sort_frames(all_files)
        all_labels = [
            int(os.path.splitext(os.path.basename(file_path))[0].split("-")[1])
            for file_path in all_files
        ]
        try:
            event_frame = all_labels.index(1)
            all_files_before_event = all_files[max(0, event_frame - self.n_frames_predictive_horizon):event_frame]
            windows, labels = self._extract_clips_and_labels_from_video(all_files_before_event, True)
            self.all_clips.extend(windows)
            self.all_labels.extend(labels)

        except ValueError:
            self.folder_without_events.append(folder)
        return None
    
    def _extract_clips_from_video_without_event(self, folder: str, n_clips : int) -> None:
        all_files = self._extract_all_files_from_folder(folder)
        all_files = self.sort_frames(all_files)
        extracted_clips : List[List[str]] = []
        labels : List[int] = []

        while len(extracted_clips) < n_clips:
            start_frame = np.random.randint(0, len(all_files) - self.sample_len)
            candidate_window = all_files[start_frame : start_frame + self.sample_len]
            videos_in_window = set(
                [os.path.dirname(filename) for filename in candidate_window]
            )
            if len(videos_in_window) == 1:
                extracted_clips.append(candidate_window)
                labels.append(0)
        self.all_clips.extend(extracted_clips)
        self.all_labels.extend(labels)
        return None
 
    def _extract_clips_and_labels_from_video(self, filenames: List[str], positive: bool) -> Tuple[List[List[str]], List[int]]:
        windowed_samples = []
        for i in range(
            0, len(filenames) - self.sample_len + 1, self.sample_len - self.overlap
        ):
            candidate_window = filenames[i : i + self.sample_len]
            videos_in_window = set(
                [os.path.dirname(filename) for filename in candidate_window]
            )
            if len(videos_in_window) == 1:
                windowed_samples.append(candidate_window)
        if positive:
            labels = [1 for _ in range(len(windowed_samples))]
        else:
            labels = [0 for _ in range(len(windowed_samples))]
        return windowed_samples, labels
    @staticmethod
    def _extract_all_files_from_folder( folder: str) -> List[str]:
        all_files_png = glob.glob(os.path.join(folder, "*.png"))
        all_files_jpg = glob.glob(os.path.join(folder, "*.jpg"))
        all_files = all_files_png + all_files_jpg
        return all_files

    @staticmethod
    def sort_frames(filenames: List[str]) -> List[str]:
        """
        Sorts the filenames so that frame order is preserved.
        Args:
            filenames (List[str]): list of filenames
        Returns:
            List[str]: sorted list of filenames
        """
        parsed_filenames = []
        for filename in filenames:
            video_id = os.path.dirname(filename)
            match = re.search(r"(\d+)-", os.path.basename(filename))
            if match is None:
                raise ValueError(
                    f"Filename {filename} does not match the pattern"
                )
            frame_number = int(match.group(1))

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
    def _get_transforms(self) -> transforms.Compose:
        if self.dvs_mode:
            return transforms.Compose(
                [
                    transforms.Resize(
                        self.target_size, interpolation=Image.NEAREST
                    ),
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize(self.target_size),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load_image(self, image: str) -> torch.Tensor:
        """Loads image from a given path and applies transforms."""
        img = Image.open(image)
        return self.transform(img)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.all_clips[index]
        window = torch.stack(
            [self.load_image(image) for image in file_path]
        )
        label = torch.tensor(self.all_labels[index])
        return window, label
        

class PredictionDatasetSingleStep(PredictionDataset):
    """ A class that first extracts n_frames right BEFORE the event in the video,
    and then the same number of frames from videos where the event did not occur.
    Then the extracted frames are treated as separate samples.
    """
    def __init__(self,
                 folder_list: List[str],
                target_size: Tuple[int, int] = (600, 1600),
                n_frames_predictive_horizon: int = 10,
                dvs_mode: bool = False) -> None:
        self.folder_list = folder_list
        self.target_size = target_size
        self.n_frames_predictive_horizon = n_frames_predictive_horizon
        self.dvs_mode = dvs_mode
        self.all_clips :List[str]= [] # type: ignore
        self.all_labels : List[int]= []
        self.folder_without_events : List[str]= []
        self.transform = self._get_transforms()
        self._perform_full_sample_extraction()

    def _perform_full_sample_extraction(self) -> None:
        for folder in self.folder_list:
            self._extract_frames_from_video_with_event(folder)
        n_negative_samples_per_video = len(self.all_clips) // len(self.folder_without_events)
        for folder in self.folder_without_events:
            self._extract_clips_from_video_without_event(folder, n_negative_samples_per_video)
        return None

    def _extract_frames_from_video_with_event(self, folder: str) -> None:
        all_files = self._extract_all_files_from_folder(folder)
        all_files = self.sort_frames(all_files)
        all_labels = [
            int(os.path.splitext(os.path.basename(file_path))[0].split("-")[1])
            for file_path in all_files
        ]
        try:
            event_frame = all_labels.index(1)
            all_files_before_event = all_files[max(0, event_frame - self.n_frames_predictive_horizon):event_frame]
            
            self.all_clips.extend(all_files_before_event)
            self.all_labels.extend([1 for _ in range(len(all_files_before_event))])
        except ValueError:
            self.folder_without_events.append(folder)
        return None
    def _extract_clips_from_video_without_event(self, folder: str, n_clips : int) -> None:
        all_files = self._extract_all_files_from_folder(folder)
        all_files = self.sort_frames(all_files)

        random_n_frames = np.random.choice(all_files, n_clips, replace = False)
        self.all_clips.extend(random_n_frames)
        self.all_labels.extend([0 for _ in range(n_clips)])
    
    def __len__(self) -> int:
        return len(self.all_clips)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.all_clips[index]
        image = self.load_image(file_path)
        label = torch.tensor(self.all_labels[index])
        return image, label
    

class PredictionDatasetSingleStepRepeated(PredictionDatasetSingleStep):
    """ A class that first extracts n_frames right BEFORE the event in the video,
    and then the same number of frames from videos where the event did not occur.
    Then the extracted frames are treated as separate samples and repeated multiple times.
    """
    def __init__(self,
                 folder_list: List[str],
                target_size: Tuple[int, int] = (600, 1600),
                n_frames_predictive_horizon: int = 10,
                dvs_mode: bool = False,
                repeats: int = 1) -> None:
        super().__init__(folder_list, target_size, n_frames_predictive_horizon, dvs_mode)
        self.repeats = repeats
        
    def __len__(self) -> int:
        return len(self.all_clips)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.all_clips[index]
        image = self.load_image(file_path).repeat(self.repeats, 1, 1, 1)
        label = torch.tensor(self.all_labels[index])
        return image, label