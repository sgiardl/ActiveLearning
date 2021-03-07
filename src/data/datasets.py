import torchvision.datasets as datasets
from src.data.constants import *
DATASETS = [CIFAR10, EMNIST]

def get_dataset(name, root, transforms=None, download=True):
    """
    Downloads dataset from torchvision

    :param name: Name of the dataset must be in DATASETS list
    :param root: Path where the data shall be loaded/downloaded
    :param download: Bool, if true, downloads the dataset and saves it in root directory.
                           If dataset is already downloaded, it is loaded from root directory.
    :return: PyTorch Dataset
    """

    if name not in DATASETS:
        raise Exception(f"The name provided must be in {DATASETS}")

    elif name == CIFAR10:
        return datasets.CIFAR10(root=root,
                                transform=transforms, download=download)

    else:
        return datasets.EMNIST(root=root, split='mnist',
                               transform=transforms, download=download)