from spikingjelly.activation_based.model import spiking_resnet
from spikingjelly.activation_based.model.spiking_vgg import spiking_vgg11_bn
from spikingjelly.activation_based.model.sew_resnet import sew_resnet18, sew_resnext50_32x4d
import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate, neuron, functional, layer
from torchvision.models import resnet18, ResNet18_Weights
from torch.autograd import Variable
from spikingjelly import activation_based
from typing import Optional

def replace_bn_with_tebn(network: nn.Module, n_timesteps : int = 1):
    """ Replaces all BatchNorm2d layers in the network with TemporalEffectiveBatchNorm2d layers in the SewResnet model.
    Infers the number of features from the BatchNorm2d layer and replaces it with a TemporalEffectiveBatchNorm2d layer with the same number of features.
    The step mode is also inferred from the BatchNorm2d layer and used to initialize the TemporalEffectiveBatchNorm2d layer.
    Args:
        network (nn.Module): The network to replace the BatchNorm2d layers in.
        n_timesteps (int, optional): The size of temporal dimension in the input samples. Defaults to 1.
    """
    for name, module in network._modules.items():
        if isinstance(module,layer.BatchNorm2d):
            n_features = module.__dict__["num_features"]
            step_mode = module.__dict__["_step_mode"]
            new_tebn_layer = layer.TemporalEffectiveBatchNorm2d(num_features=n_features, T=n_timesteps, step_mode=step_mode)
            network._modules[name] = new_tebn_layer
        elif isinstance(module, nn.Sequential):
            replace_bn_with_tebn(module, n_timesteps)
        elif isinstance(module, activation_based.model.sew_resnet.BasicBlock):
            replace_bn_with_tebn(module, n_timesteps)




# class ResNet18VoteLayer(nn.Module):
#     def __init__(self, num_classes: int):
#         super().__init__()
#         self.feature_extractor = Resnet18_DVS()
#         self.voting = nn.AdaptiveAvgPool1d(num_classes)

#     def forward(self, x: torch.Tensor):
#         # x.shape = [N, voter_num * C]
#         # ret.shape = [N, C]
#         features = self.feature_extractor(x).squeeze(2).permute(1, 0)
#         return self.voting(features)


# def Resnet18_DVS_voting():
#     net = ResNet18VoteLayer(1)
#     return net

def VGG11_spiking(
    dvs_mode=False,
    neuron_model: neuron.BaseNode = neuron.LIFNode,
    surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid,
) -> nn.Module:
    net = spiking_vgg11_bn(
        pretrained=True,
        spiking_neuron=neuron_model,
        surrogate_function=surrogate_function(),
        detach_reset=True,
    )
    if dvs_mode:
        net.features[0] = layer.Conv2d(
            1,
            64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
    net.classifier[6] = layer.Linear(4096, 1)
    return net

def SewResnet18(
        dvs_mode=False,
        neuron_model: neuron.BaseNode = neuron.LIFNode,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid,
        convert_bn_to_tebn: bool = False,
        n_samples : int = 10
) -> nn.Module:
    net = sew_resnet18(
        pretrained=False,
        spiking_neuron=neuron_model,
        cnf="IAND",
        surrogate_function=surrogate_function(),
        # detach_reset=True,
    )
    if dvs_mode:
        net.conv1 = layer.Conv2d(
            1,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
    if convert_bn_to_tebn:
        replace_bn_with_tebn(net, n_samples)
    net.fc = layer.Linear(512, 1)
    return net


def Resnet18_spiking(
    dvs_mode=False,
    neuron_model: neuron.BaseNode = neuron.LIFNode,
    surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid,
    n_samples : int = 10
) -> nn.Module:
    net = spiking_resnet.spiking_resnet18(
        pretrained=False,
        spiking_neuron=neuron_model,
        surrogate_function=surrogate_function(),
        detach_reset=False,
    )
    if dvs_mode:
        net.conv1 = layer.Conv2d(
            1,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
    net.fc = layer.Linear(512, 1)
    return net


def Resnet18(dvs_mode=False) -> nn.Module:
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


def slow_r50(dvs_mode=False) -> nn.Module:
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


if __name__ == "__main__":
    net = VGG11_spiking()
    print(net)