import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from spikingjelly.visualizing import plot_2d_bar_in_3d, plot_2d_heatmap



class CSNN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.conv_enkoder = nn.Sequential(
            layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2))

        self.conv_fc = nn.Sequential(
            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2))

        self.lin_fc = nn.Sequential(
            layer.Flatten(),
            layer.Linear(channels * 42, channels * 4 * 4, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),

            layer.Linear(channels * 4 * 4, 2, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),)

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(0)
        x_seq = self.conv_enkoder(x)
        x_seq = self.conv_fc(x_seq)
        x_seq = self.lin_fc(x_seq)
        x_seq = x_seq.squeeze(0)

        return x_seq

    def spiking_encoder(self):
        return self.conv_fc[0:3]



class ClassificationDataset(Dataset):
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


def _plot_video(video,_frame_num=0):
    print(video.shape)
    video = np.array(video[0])

    for frame in range(video.shape[0]):
        plt.figure(1)
        plt.clf()
        plt.axis(False)
        plt.imshow(video[frame, :, :], cmap="gray")
        plt.title('Frame ' + str(_frame_num))
        plt.pause(0.5)
    plt.show()


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = ClassificationDataset(r'D:\datasets\N-Cars_parsed', "vis", 40)
    test_data_loader = DataLoader(test_dataset, batch_size=1, num_workers=8)

    net = CSNN(channels=2)
    net.to(device)

    checkpoint = torch.load(r"path", map_location='cpu')
    net.load_state_dict(checkpoint['net'])

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    optimizer.load_state_dict(checkpoint['optimizer'])

    net.to(device)
    net.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    i = 0
    with torch.no_grad():
        for img, label in test_data_loader:
            i += 1
            frames = img.shape[0]  # num frames
            out_fr = 0
            v_list = []

            img = img.permute(1, 0, 2, 3, 4)  # [B, T, C, H, W] --> [T, B, C, H, W]
            frame_num = 0
            for T in img:
                T = T.to(device).float()

                out_fr += net(T)

                if_node_outputs = [moduleIF.v for moduleIF in net.lin_fc if
                                   isinstance(moduleIF, neuron.IFNode)]
                v_list.append(if_node_outputs[0].cpu())
                frame_num += 1

            functional.reset_net(net)

            v_list = torch.cat(v_list)
            fig = plot_2d_bar_in_3d(v_list, title=f'spiking rates of output layer{i}', xlabel='neuron index',
                                    ylabel='simulating step', zlabel='voltage')
            plt.show()
            plt.close(fig)

            v_list = v_list.cpu().numpy()
            v_list = np.asarray(v_list)

            fig = plot_2d_heatmap(array=v_list, title='Membrane Potentials', xlabel='Simulating Step',
                                  ylabel='Neuron Index', x_max=5)
            plt.show()
            plt.close(fig)
