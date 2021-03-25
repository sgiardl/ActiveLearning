import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit
from src.data.constants import *
from typing import Union, Sequence, Callable
DATASETS = [CIFAR10, EMNIST]


def get_dataset(name: str, root: str, transforms: Union[Sequence[Callable], Callable] = None,
                download: bool = True, train: bool = True) -> datasets:
    """
    Downloads dataset from torchvision

    :param name: Name of the dataset must be in DATASETS list
    :param root: Path where the data shall be loaded/downloaded
    :param transforms list of transforms or Compose of transforms
    :param download: Bool, if true, downloads the dataset and saves it in root directory.
                           If dataset is already downloaded, it is loaded from root directory.
    :param train: Bool, if true, creates dataset from training set, otherwise creates from test set.
    :return: PyTorch Dataset
    """

    if name not in DATASETS:
        raise Exception(f"The name provided must be in {DATASETS}")

    elif name == CIFAR10:
        return datasets.CIFAR10(root=root,
                                transform=transforms, download=download, train=train)

    else:
        return datasets.EMNIST(root=root, split='byclass',
                               transform=transforms, download=download, train=train)


def split_dataset(dataset: datasets, split_size: float) -> (Subset, Subset):
    '''
    Splits a dataset into two subsets.

    :param dataset: torchvision.datasets, input PyTorch dataset object to split
    :param split_size: float, proportion of dataset to include in subset_2.
                       The complement will be included in subset_1.
    :return: tuple, subset_1 and subset_2
    '''

    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=split_size)

    y = dataset.targets
    X = np.zeros(len(y))

    for index_1, index_2 in strat_split.split(X, y):
        break

    subset_1 = Subset(dataset, index_1)
    subset_2 = Subset(dataset, index_2)

    return subset_1, subset_2
