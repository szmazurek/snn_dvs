from spikingjelly.activation_based.model import spiking_resnet
import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate, neuron, functional, layer
from torchvision.models import resnet18, ResNet18_Weights


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
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
        )

        self.lin_fc = lin_fc

        functional.set_step_mode(self, step_mode="m")
        functional.set_backend(self, backend="cupy")

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
            nn.ReLU(),
        )

        self.lin_fc = lin_fc

    def forward(self, x: torch.Tensor):
        x_seq = self.lin_fc(x)
        return x_seq


class SNN_2(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_fc = nn.Sequential(
            layer.Conv2d(1, 6, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(6),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(6, 6, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(6),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(6, 6, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(6),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
            layer.MaxPool2d(2, 2),
        )

        self.lin_fc = nn.Sequential(
            layer.Flatten(),
            layer.Linear(5400, 100, bias=False),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
            layer.Linear(100, 2, bias=False),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
        )

        functional.set_step_mode(self, step_mode="m")
        functional.set_backend(self, backend="cupy")

    def forward(self, x: torch.Tensor):
        x_seq = self.conv_fc(x)
        x_seq = self.lin_fc(x_seq)
        return x_seq


class CNN_2(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_fc = nn.Sequential(
            layer.Conv2d(3, 6, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(6),
            nn.ReLU(),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(6, 6, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(6),
            nn.ReLU(),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(6, 6, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(6),
            nn.ReLU(),
            layer.MaxPool2d(2, 2),
        )

        self.lin_fc = nn.Sequential(
            layer.Flatten(),
            layer.Linear(5400, 100, bias=False),
            nn.ReLU(),
            layer.Linear(100, 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        x_seq = self.conv_fc(x)
        x_seq = self.lin_fc(x_seq)
        return x_seq


def Resnet18_DVS():
    net = spiking_resnet.spiking_resnet18(
        pretrained=True,
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.Sigmoid(),
        detach_reset=True,
    )
    net.conv1 = layer.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    net.fc = layer.Linear(512, 1)
    return net


def Resnet18_DVS_rgb():
    net = spiking_resnet.spiking_resnet18(
        pretrained=True,
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.Sigmoid(),
        detach_reset=True,
    )
    net.fc = layer.Linear(512, 1)
    return net


def Resnet18(dvs_mode=False):
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(512, 1)
    if dvs_mode:
        net.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
    return net
