import wandb
import os
import glob
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from typing import Tuple


class DvsDataset(Dataset):
    def __init__(self, targ_dir: str) -> None:
        self.all_folders = [os.path.join(targ_dir, directory) for directory in os.listdir(targ_dir)]
        self.random_flip = 0
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(self.random_flip),
            transforms.Resize((150, 400), interpolation=Image.NEAREST),
            transforms.PILToTensor()])

    def load_image(self, image) -> Image.Image:
        return self.transform(Image.open(image))

    def __len__(self) -> int:
        return len(self.all_folders)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        self.random_flip = random.getrandbits(1)
        folder_path = self.all_folders[index]
        file_list = glob.glob(os.path.join(folder_path, "*.png"))
        file_list_sorted = sorted(file_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[0]))
        images_tensor = torch.cat([self.load_image(image).unsqueeze(0) for image in file_list_sorted])

        label_list = [int(str(image)[-5]) for image in file_list_sorted]
        label_tensor = torch.tensor(label_list).unsqueeze(1)
        name_sample = os.path.basename(folder_path)
        print(self.random_flip)
        print(name_sample)
        return images_tensor, label_tensor, name_sample


def main():

    wandb.init(
        project="Dataset_Overview",
        entity="snn_team"
    )
    dataset = DvsDataset(r"/home/plgkrzysjed1/datasets/dataset_dvs")

    test_data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)


    for (img, label, name) in test_data_loader:

        img= img.squeeze()
        label = label.squeeze()

        for i, (ima, l) in enumerate(zip(img,  label)):
            mask_image = wandb.Image(ima.float(), caption=f"Frame number: {i} | Label: {str(l.item())}")
            wandb.log({f"{name[0]}": mask_image})
    wandb.finish()


if __name__ == "__main__":
    main()
