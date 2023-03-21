import os
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms


class NCars_Parsed(Dataset):
    def __init__(self, path, mode, num_frame=5):
        self.num_frame = num_frame
        self.data_dir = os.path.join(path, mode)
        self.classes_dir = [os.path.join(self.data_dir, class_name) for class_name in os.listdir(self.data_dir)]
        self.transform = transforms.Compose([
            transforms.Resize((120, 100))])

    def __getitem__(self, index):
        class_dir = self.classes_dir[index]
        data_file = os.path.join(str(class_dir), "events.txt")
        label_file = os.path.join(str(class_dir), "is_car.txt")

        data = np.loadtxt(data_file)
        data = self.convert_in_to_frames(data)
        data = torch.from_numpy(data)
        data = self.transform(data).unsqueeze(0).permute(1, 0, 2, 3)

        label = int(np.loadtxt(label_file))

        return data, label

    def __len__(self):
        return len(self.classes_dir)

    def convert_in_to_frames(self, data: np.array):
        num_sample, _ = data.shape
        max_x, max_y = int(np.max(data[:, 0]) + 1), int(np.max(data[:, 1]) + 1)

        frame = np.zeros((max_y, max_x), dtype=np.int32)
        video = []
        spike = 0
        for ids, sample in enumerate(data):
            x, y, q = int(sample[0]), int(sample[1]), sample[3]  # x, y, q

            if q == 0:
                np.subtract.at(frame, (y, x), 1)
            else:
                np.add.at(frame, (y, x), 1)
            spike += 1

            if ids+1 == num_sample:
                video.append(frame.copy())
                frame.fill(0)
                spike = 0

            elif spike == (num_sample // self.num_frame) and self.num_frame-1 != len(video):
                video.append(frame.copy())
                frame.fill(0)
                spike = 0
        return np.array(video)


# if __name__ == "__main__":
#     train_dataset = ClassificationDataset(r'D:\datasets\N-Cars_parsed', "train", 30)
#     train_data_loader = DataLoader(train_dataset, batch_size=200, num_workers=6, shuffle=True)