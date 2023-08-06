import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer



class SNN_1(nn.Module):
    def __init__(self):
        super().__init__()

        image_size = 150 * 400
        lin_fc = nn.Sequential(
            layer.Flatten(),
            layer.Linear(image_size, 200, bias=False),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),

            layer.Linear(200, 50, bias=False),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),

            layer.Linear(50, 2, bias=False),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()))

        self.lin_fc = lin_fc

        functional.set_step_mode(self, step_mode='m')
        functional.set_backend(self, backend='cupy')

    def forward(self, x: torch.Tensor):

        x_seq = self.lin_fc(x)
        return x_seq


class CNN_1(nn.Module):
    def __init__(self):
        super().__init__()

        image_size = 150 * 400 * 3
        lin_fc = nn.Sequential(
            layer.Flatten(),
            layer.Linear(image_size, 200, bias=False),
            nn.ReLU(),

            layer.Linear(200, 50, bias=False),
            nn.ReLU(),

            layer.Linear(50, 2, bias=False),
            nn.ReLU())

        self.lin_fc = lin_fc
    def forward(self, x: torch.Tensor):

        x_seq = self.lin_fc(x)
        return x_seq


class SNN_2(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.conv_fc = nn.Sequential(
            layer.Conv2d(1, 6, kernel_size=3, padding=1, bias=False),
            layer.ThresholdDependentBatchNorm2d(6, 6),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(6, 6, kernel_size=3, padding=1, bias=False),
            layer.ThresholdDependentBatchNorm2d(6, 6),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(6, 6, kernel_size=3, padding=1, bias=False),
            layer.ThresholdDependentBatchNorm2d(6, 6),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
            layer.MaxPool2d(2, 2))

        self.lin_fc = nn.Sequential(
            layer.Flatten(),
            layer.Linear(4 * 50 * 6, 100, bias=False),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
            layer.Linear(100, 2, bias=False),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()))

        functional.set_step_mode(self, step_mode='m')
        functional.set_backend(self, backend='cupy')

    def forward(self, x: torch.Tensor):
        x_seq = self.conv_fc(x)
        x_seq = self.lin_fc(x_seq)
        return x_seq


class CNN_2(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.conv_fc = nn.Sequential(
            layer.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            nn.ReLU(),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            nn.ReLU(),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            nn.ReLU(),
            layer.MaxPool2d(2, 2))

        self.lin_fc = nn.Sequential(
            layer.Flatten(),
            layer.Linear(4 * 50 * 18, channels * 4 * 4, bias=False),
            nn.ReLU(),

            layer.Linear(channels * 4 * 4, 2, bias=False),
            nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        x_seq = self.conv_fc(x)
        x_seq = self.lin_fc(x_seq)
        return x_seq

