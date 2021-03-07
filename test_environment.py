import sys
import os
from src.data.datasets import get_dataset
from src.data.constants import *
from src.models.constants import *
from src.models.train_model import train_model, get_transforms#, collate_fn

from torch.utils.data import DataLoader
from torchvision import transforms
import torch


REQUIRED_PYTHON = "python3"


def main():
    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError("Unrecognized python interpreter: {}".format(
            REQUIRED_PYTHON))

    if system_major != required_major:
        raise TypeError(
            "This project requires Python {}. Found: Python {}".format(
                required_major, sys.version))
    else:
        print(">>> Development environment passes all tests!")


if __name__ == '__main__':
    main()

    raw_data_folder = f"{os.getcwd()}/data/raw"

    dataset = get_dataset(name=CIFAR10, root=raw_data_folder, transforms=transforms.ToTensor())

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)#, collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    file_name = 'model'

    train_model(epochs=10, data_loader=data_loader,
                device=device, file_name=file_name,
                model_name=RESNET34)
