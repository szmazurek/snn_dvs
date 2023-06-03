import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.activation_based import neuron, functional, surrogate, layer


class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        image_size = 120*100
        self.lin_fc = nn.Sequential(
            layer.Flatten(),
            layer.Linear(image_size, 400, bias=False),
            layer.Dropout(0.5),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan()),

            layer.Linear(400, 100, bias=False),
            layer.Dropout(0.2),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan()),

            layer.Linear(100, 25, bias=False),
            layer.Dropout(0.1),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan()),

            layer.Linear(25, 2, bias=False),
            layer.Dropout(0.05),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan()))

        functional.set_step_mode(self, step_mode='s')

    def forward(self, x: torch.Tensor):
        x_seq = self.lin_fc(x)
        return x_seq

    def spiking_encoder(self):
        return self.conv_fc[0:3]


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


if __name__ == "__main__":
    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_dataset = NCars_Parsed(r'D:\datasets\N-Cars_parsed', "train", 30)
    valid_dataset = NCars_Parsed(r'D:\datasets\N-Cars_parsed', "val", 30)
    max_test_acc = -1
    train_data_loader = DataLoader(train_dataset, batch_size=200, num_workers=6, shuffle=True)
    test_data_loader = DataLoader(valid_dataset, batch_size=200, num_workers=6)
    net = SNN()
    net.to(device)
    checkpoint_folder_path = r"D:\ProjectsPython\SpikingJelly\My_scipts\NcarMyScipts\LinearNetwork\net_checkpoints"

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    epochs = 30
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for epoch in range(0, epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()
            label = label.to(device)
            label_onehot = F.one_hot(label, 2).float()

            img = img.permute(1, 0, 2, 3, 4)  # [B, T, C, H, W] --> [T, B, C, H, W]
            frames = img.shape[0]  # num frames

            out_fr = 0
            for T in img:
                T = T.to(device).float()
                out_fr += net(T)
            out_fr /= frames
            loss = F.mse_loss(out_fr, label_onehot)

            loss.backward()
            optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label.int()).float().sum().item()
            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples
        print("Train", epoch, train_acc)

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                label = label.to(device)
                label_onehot = F.one_hot(label, 2).float()

                out_fr = 0
                img = img.permute(1, 0, 2, 3, 4)  # [B, T, C, H, W] --> [T, B, C, H, W]
                frames = img.shape[0]  # num frames
                for T in img:
                    T = T.to(device).float()
                    out_fr += net(T)
                out_fr /= frames
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            checkpoint = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'max_test_acc': max_test_acc
            }
            torch.save(checkpoint, os.path.join(checkpoint_folder_path, 'checkpoint.pth'))

        print("Test", epoch, test_acc)


