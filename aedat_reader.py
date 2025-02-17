import torch
from torch.utils.data import DataLoader
from spikingjelly.datasets import pad_sequence_collate, padded_sequence_mask
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture


# Set the root directory for the dataset
root_dir = 'datasets/DVS128Gesture'
# Load event dataset
event_set = DVS128Gesture(root_dir, train=True, data_type='event')
event, label = event_set[0]
# Print the keys and their corresponding values in the event data
for k in event.keys():
    print(k, event[k])
print(1)