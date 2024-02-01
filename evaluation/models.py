from spikingjelly.activation_based.model import spiking_resnet
import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate, neuron, functional, layer
from torchvision.models import resnet18, ResNet18_Weights
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.padding = int(
            (kernel_size - 1) / 2
        )  # Padding to keep the same size

        self.Wxi = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whi = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxf = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whf = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxc = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whc = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxo = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Who = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1]))
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1]))
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1]))
        else:
            assert shape[0] == self.Wci.size()[2], "Input Height Mismatched!"
            assert shape[1] == self.Wci.size()[3], "Input Width Mismatched!"
        return (
            Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])),
            Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])),
        )


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(
        self,
        input_channels,
        hidden_channels,
        kernel_size,
        n_layers=1,
        timestep=6,
        frame_size=(128, 128),
        device=torch.device("cpu"),
    ):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.frame_size = frame_size

        self.relu = nn.ReLU()
        self.device = device
        for i in range(n_layers):
            name = "cell{}".format(i)

            cell = ConvLSTMCell(
                self.input_channels[i],
                self.hidden_channels[i],
                self.kernel_size,
            )
            setattr(self, name, cell)
            # pooling
            name_pooling = "pooling{}".format(i)
            height = int(self.frame_size[0] / (2 ** (i + 1)))
            width = int(self.frame_size[1] / (2 ** (i + 1)))
            pooling = nn.AdaptiveMaxPool2d((height, width))
            setattr(self, name_pooling, pooling)

        final_channels = hidden_channels[-1]
        final_height = int(
            frame_size[0] / (n_layers * 2),
        )
        final_width = int(
            frame_size[1] / (n_layers * 2),
        )
        neurons_first_fc = (
            timestep * final_channels * final_height * final_width
        )
        self.linear_1 = nn.Linear(neurons_first_fc, 1024)
        self.linear_2 = nn.Linear(1024, 1)

    def forward(self, input):
        b_size, timesteps, _, height, width = input.size()
        for i in range(self.n_layers):
            layer = getattr(self, "cell{}".format(i))
            pooling = getattr(self, "pooling{}".format(i))

            height_pooled = int(self.frame_size[0] / (2 ** (i + 1)))
            width_pooled = int(self.frame_size[1] / (2 ** (i + 1)))
            if i != 0:
                o_prev = o
                height = int(self.frame_size[0] / (2**i))
                width = int(self.frame_size[1] / (2**i))
            o = torch.zeros(
                b_size,
                timesteps,
                self.hidden_channels[i],
                height_pooled,
                width_pooled,
            ).to(self.device)
            h, c = layer.init_hidden(
                b_size, self.hidden_channels[i], (height, width)
            )
            layer = layer.to(self.device)
            h = h.to(self.device)
            c = c.to(self.device)

            for timestep in range(timesteps):
                if i == 0:
                    timestep_frame = input[:, timestep, :, :, :]
                else:
                    timestep_frame = o_prev[:, timestep, :, :, :]

                h, c = layer(timestep_frame, h, c)
                o[:, timestep, :, :, :] = pooling(h)
        o = o.flatten(start_dim=1)
        o = self.linear_1(o)
        o = self.relu(o)
        o = self.linear_2(o)
        return o


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


class ResNet18VoteLayer(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.feature_extractor = Resnet18_DVS()
        self.voting = nn.AdaptiveAvgPool1d(num_classes)

    def forward(self, x: torch.Tensor):
        # x.shape = [N, voter_num * C]
        # ret.shape = [N, C]
        features = self.feature_extractor(x).squeeze(2).permute(1, 0)
        return self.voting(features)


def Resnet18_DVS_voting():
    net = ResNet18VoteLayer(1)
    return net


def Resnet18_DVS(neuron_model: neuron.BaseNode = neuron.LIFNode):
    net = spiking_resnet.spiking_resnet18(
        pretrained=True,
        spiking_neuron=neuron_model,
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


def slow_r50(dvs_mode=False):
    net = torch.hub.load(
        "facebookresearch/pytorchvideo", "slow_r50", pretrained=True
    )
    net.blocks[-1].proj = torch.nn.Linear(2048, 1)
    if dvs_mode:
        net.blocks[0].conv = torch.nn.Conv3d(
            1,
            64,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=False,
        )
    return net
