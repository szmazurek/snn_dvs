import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, functional, surrogate, layer


class NCarsNet_1(nn.Module):
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
