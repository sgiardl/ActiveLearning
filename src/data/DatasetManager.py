import numpy as np
import os
import torchvision.datasets as datasets
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit
from src.data.constants import *
from typing import Union, Sequence
DATASETS = [CIFAR10, EMNIST]
from typing import Callable
import torchvision.transforms as transforms
from src.models.expert import Expert

from torch.utils.data import DataLoader

class DatasetManager():
    def __init__(self, dataset_name: str,
                 valid_size: float,
                 batch_size: int,
                 shuffle: bool = False,
                 num_workers: int = 1):
        dataset_train = self.get_dataset(name=dataset_name, root=f"{os.getcwd()}/data/raw",
                                    transforms=self.get_transforms(), train=True)

        self.dataset_test = self.get_dataset(name=dataset_name, root=f"{os.getcwd()}/data/raw",
                                   transforms=self.get_transforms(), train=False)

        self.dataset_train, self.dataset_valid = self.split_dataset(dataset_train, valid_size)

        self.expert = Expert(self.dataset_train, 2, None)

        self.data_loader_train = DataLoader(self.dataset_train, batch_size=batch_size, shuffle=shuffle,
                                            num_workers=num_workers, sampler=self.expert.sampler)

        self.data_loader_valid = DataLoader(self.dataset_valid, batch_size=batch_size, shuffle=shuffle,
                                            num_workers=num_workers)

    def get_dataset(self, name: str, root: str, transforms: Union[Sequence[Callable], Callable] = None,
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


    def split_dataset(self, dataset: datasets, split_size: float) -> (Subset, Subset):
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

    def get_transforms(self) -> Callable:
        """
        Get transforms to apply to the data.

        :return: Composed transforms
                 Type : torchvision.transforms.transforms.Compose
        """

        transforms_list = [
            # Convert images to RGB (3-channels) since models used expect 3 channels as input.
            # Needed for 1 channel datasets, such as EMNIST.
            transforms.Lambda(lambda image: image.convert('RGB')),

            # Convert images to pytorch tensors
            transforms.ToTensor()
        ]

        # Return composed transforms
        return transforms.Compose(transforms_list)