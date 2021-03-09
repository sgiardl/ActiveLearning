import sys
import os
from src.data.datasets import get_dataset
from src.data.constants import *
from src.models.constants import *
from src.models.train_model import train_model, get_transforms
from src.models.expert import Expert

from torch.utils.data import DataLoader


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

    dataset = get_dataset(name=CIFAR10, root=f"{os.getcwd()}/data/raw", transforms=get_transforms())
    expert = Expert(dataset, 200, None)
    data_loader = DataLoader(dataset, batch_size=200, shuffle=False, num_workers=1, sampler=expert.sampler)
    train_model(epochs=2, data_loader=data_loader, file_name='model', model_name=RESNET34)
