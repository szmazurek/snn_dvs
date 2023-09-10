from pipeline import Pipeline
from dataloaders import CarlaDataset
from nets import CSNN, SNN
import torch

if __name__ == "__main__":
    checkpoint_dvs = torch.load("/home/plgkrzysjed1/workbench/data/prediction_version2/DataAnalysis/dvs/checkpoint.pth")
    checkpoint_rgb = torch.load("/home/plgkrzysjed1/workbench/data/prediction_version2/DataAnalysis/rgb/checkpoint.pth")

    net_rgb = CSNN()
    net_dvs = SNN()
    net_rgb.load_state_dict(checkpoint_rgb['net'])
    net_dvs.load_state_dict(checkpoint_dvs['net'])
    print("Networks loaded")
    a = Pipeline("/home/plgkrzysjed1/datasets/wet_weather/clips", net_rgb, net_dvs, CarlaDataset, "Pipeline_Test")
    a.start()
