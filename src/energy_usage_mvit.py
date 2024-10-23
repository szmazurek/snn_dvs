import syops
import syops.engine
from models.models import mvitv2_s
from spikingjelly.activation_based.functional import set_step_mode
from spikingjelly.activation_based import neuron, surrogate, layer
import torch
import os


Eac = 0.9e-12
Emac = 4.6e-12


with torch.cuda.device(0):
    net = mvitv2_s(n_samples=9, dvs_mode=True)
    print(net)
    input_shape = (30, 1, 256, 450)
    input_tensor = torch.rand(input_shape)
    ops, params = syops.get_model_complexity_info(
        net,
        input_shape,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=True,
        output_precision=4,
    )
    print(ops)
    energy_acs = ops[1] * Eac
    energy_macs = ops[2] * Emac
    total_mj = (energy_acs + energy_macs) * 1e3
    print(f"Computational complexity OPs: {ops[0]}")
    print(f"Computational complexity ACs: {ops[1]}")
    print(f"Computational complexity MACs: {ops[2]}")
    print(f"Number of parameters: {params}")

    print(f"Energy consumption for ACs: {energy_acs} J")
    print(f"Energy consumption for MACs: {energy_macs} J")
    print(f"Total energy consumption: {total_mj} mJ")
