from spikingjelly.activation_based.functional import reset_net


from models.models import SewResnet18, Resnet18, slow_r50, Resnet18_spiking
from spikingjelly.activation_based.functional import set_step_mode
from spikingjelly.activation_based import neuron, surrogate, layer
import torch
import os

def spike_count(model, input_shape, timestep, device='cuda'):
    model.to(device)
    sample_input = torch.randn(input_shape).to(device)
    total_spikes = 0
    
    for t in range(timestep):
        out = model(sample_input)
        print(out.sum())
        print(out.sum().item())
        print(type(out))
          # Forward pass for each timestep
        total_spikes += out.sum().item()  # Sum all spikes across neurons
        reset_net(model)  # Reset neurons for the next timestep
    
    print(f"Total Spike Count: {total_spikes}")
    return total_spikes

def spike_count2(model, dataloader, timestep, device='cuda'):
    model.to(device)
    total_spikes = 0
    
    for batch_idx, (inputs, _) in enumerate(dataloader):
        inputs = inputs.to(device)  # Move batch to device (GPU)
        batch_spikes = 0

        # Reshape inputs to 4D: (batch_size * timestep, channels, height, width)
        inputs = inputs.view(-1, inputs.size(2), inputs.size(3), inputs.size(4))

        for t in range(timestep):
            out = model(inputs)
            #print(out.sum())
            #print(out.sum().item())
            #print(type(out))
            batch_spikes += out.sum().item()  # Sum all spikes across neurons
        
        total_spikes += batch_spikes
        reset_net(model)  # Reset neurons for the next batch
        
        print(f"Batch {batch_idx + 1} Spike Count: {batch_spikes}")

    print(f"Total Spike Count: {total_spikes}")
    return total_spikes




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
def get_dataloader(exp_name):
    dataset_root = "results_new_runs/datasets"
    dataset_path = os.path.join(dataset_root, exp_ids_map_sew_resnet[exp_name] + ".pt")
    dataset = torch.load(dataset_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        prefetch_factor=4,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )
    return dataloader

exp_ids_map_sew_resnet = {
    "rgb_sew_10": "e5a0a5c8-ae26-4d76-b045-9a3721e0d2bb"
}


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
    net.load_state_dict(state_dict)

    dataloader = get_dataloader(exp_evaluated)

    #set_step_mode(net, "m")
    input_shape = (timestep, 3, 450, 256)


spike_count2(net, dataloader, timestep)