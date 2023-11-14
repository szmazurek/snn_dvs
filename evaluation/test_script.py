from spikingjelly.activation_based.model import spiking_resnet
import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate, neuron, functional, layer
from torchvision.models import resnet18, ResNet18_Weights
from data_loaders import DVSDatasetProper


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


dataset = DVSDatasetProper("datasets/dataset_weather_dvs", per_frame_label_mode=True)
img, label = dataset[0]
img = img.unsqueeze(0).permute(1, 0, 2, 3, 4)
print(img.shape)
print(label.shape)
net = Resnet18_DVS()
loss_fn = nn.BCEWithLogitsLoss()

with torch.no_grad():
    functional.set_step_mode(net, "m")
    y_seq = net(img).squeeze()
    print(f"y_seq.shape={y_seq.shape}")
    loss = loss_fn(y_seq, label.float())
    print(loss)
    functional.reset_net(net)


# full_dataset = DVSDatasetCorrected("datasets/dataset_weather_dvs")
# sample = full_dataset[0][0].unsqueeze(1)
# print(sample.shape)
# net = Resnet18_DVS()

# with torch.no_grad():
#     functional.set_step_mode(net, "m")
#     y_seq = net(sample)
#     print(f"y_seq.shape={y_seq.shape}")
#     functional.reset_net(net)


# with torch.no_grad():
#     T = 4
#     N = 12
#     x_seq = torch.rand([T, N, 1, 224, 224])
#     print(x_seq.shape)
#     functional.set_step_mode(net, "m")
#     y_seq = net(x_seq)
#     print(f"y_seq.shape={y_seq.shape}")
#     functional.reset_net(net)
