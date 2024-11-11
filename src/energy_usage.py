import syops
import syops.engine
from models.models import SewResnet18, Resnet18, slow_r50, Resnet18_spiking
from spikingjelly.activation_based.functional import set_step_mode
from spikingjelly.activation_based import neuron, surrogate, layer
import torch
import os

Eac = 0.9e-12
Emac = 4.6e-12

## THESE ARE MAPPINGS FOR EXPERIMENT NAMES TO THEIR RESPECTIVE SAVE IDS
## PLEASE CHANGE THESE MAPPINGS ACCORDING TO YOUR EXPERIMENT IDS
# exp_ids_map_sew_resnet = {
#     "dvs_weather_30_sew": "98deba80-b6b4-4e8f-a901-d9eea3daf93e",
#     "dvs_weather_9_sew": "0ed638af-491e-4a39-ae5c-bd7b615a78cd",
#     "rgb_weather_30_sew": "449a2183-60bf-4afb-9120-ebd237063140",
#     "rgb_weather_9_sew": "97b6db9d-775f-4829-ae94-e5bc9572510b",
#     "rgb_full_9_sew": "cb2de899-6834-49b9-9f37-3257986befef",
#     "rgb_full_30_sew": "1e5cef9a-fc9f-413b-9757-e7c72bbcf6f3",
#     "dvs_full_30_sew": "84e475b3-18a2-4098-946c-6c3032d255b1",
#     "dvs_full_9_sew": "e5e7e8f4-8fb5-4c66-a8e4-bf222654261b",
# }

exp_ids_map_sew_resnet = {
    "rgb_sew_10": "e5a0a5c8-ae26-4d76-b045-9a3721e0d2bb"
}
# exp_ids_map_sew_resnet = {
#     "rgb_sew_10": "1416d20f-72d8-4957-9097-daad366d5781"
# }

def get_dataloader(exp_name):
    dataset_root = "results_new_runs/datasets"
    dataset_path = os.path.join(dataset_root, exp_ids_map_sew_resnet[exp_name] + ".pt")
    dataset = torch.load(dataset_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        prefetch_factor=4,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )
    return dataloader


def get_state_dict(exp_name):
    exp_save_dir = os.path.join(
        "checkpoints_lightning", exp_ids_map_sew_resnet[exp_name]
    )
    model_weights_path = os.path.join(exp_save_dir, os.listdir(exp_save_dir)[0])
    lightning_checkpoint = torch.load(model_weights_path)
    state_dict = {
        k.replace("model.", ""): v
        for k, v in lightning_checkpoint["state_dict"].items()
        if k.startswith("model.")
    }
    return state_dict


#
with torch.cuda.device(0):

    # net = Resnet18_spiking(dvs_mode=False, neuron_model=neuron.IFNode,
    # 		surrogate_function=surrogate.ATan)
    # set_step_mode(net, 'm')
    # timestep = 30
    # input_shape = (timestep,1,3, 224, 224)
    timestep = 9
    exp_evaluated = "rgb_sew_10"
    state_dict = get_state_dict(exp_evaluated)
    net = SewResnet18(
        dvs_mode=False,
        convert_bn_to_tebn=False,
        n_samples=timestep,
        neuron_model=neuron.ParametricLIFNode,
    ).to("cuda")
    #print(state_dict)
    net.load_state_dict(state_dict)
    print(net)

    dataloader = get_dataloader(exp_evaluated)

    set_step_mode(net, "m")
    input_shape = (timestep, 1, 1, 450, 256)

    # net = Resnet18(dvs_mode=False).to('cuda')
    # # # net.forward = pt_resnet_forward
    # input_shape = (1,9,3, 450, 256)
    # random_tensor = torch.rand(input_shape).to('cuda')
    # net(random_tensor)
    # net = slow_r50(dvs_mode=False).to('cuda')
    # input_shape = (1,3,9, 450, 256)

    ops, params = syops.get_model_complexity_info(
        net,
        input_shape,
        dataloader=dataloader,
        as_strings=True,
        print_per_layer_stat=False,
        verbose=True,
        output_precision=4,
    )
    print(ops)
    print(ops[1])
    print(type(ops[1]))
    print(type(ops[2]))
    #energy_acs = ops[1] * Eac
    #energy_macs = ops[2] * Emac
    #total_mj = (energy_acs + energy_macs) * 1e3

    # conversion from string to flat
    energy_acs = float(ops[1].split()[0]) * 1e9 * Eac  # Convert to float and multiply by 10^9
    energy_macs = float(ops[2].split()[0]) * 1e9 * Eac  # Convert to float and multiply by 10^9
    total_mj = (energy_acs + energy_macs) * 1e3
    print(f"Computational complexity OPs: {ops[0]}")
    print(f"Computational complexity ACs: {ops[1]}")
    print(f"Computational complexity MACs: {ops[2]}")
    print(f"Number of parameters: {params}")

    print(f"Energy consumption for ACs: {energy_acs} J")
    print(f"Energy consumption for MACs: {energy_macs} J")
    print(f"Total energy consumption: {total_mj} mJ")