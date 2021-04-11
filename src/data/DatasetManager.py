"""
File:
    data/DatasetManager.py

Authors:
    - Abir Riahi
    - Nicolas Raymond
    - Simon Giard-Leroux

Description:
    Defines the DatasetManager class.
"""

import numpy as np
import os
from copy import deepcopy
import torchvision.datasets as datasets
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision.transforms as transforms

from src.data.constants import *

DATASETS = [CIFAR10, EMNIST]


class DatasetManager:
    """
    Dataset Manager class, handles the creation of the training, validation and testing datasets.
    """
    def __init__(self, dataset_name: str,
                 valid_size_1: float,
                 valid_size_2: float,
                 data_aug: bool = False) -> None:
        """
        :param dataset_name: string, name of the dataset to load
        :param valid_size_1: float, size of validation subset 1 as a fraction of the training set
        :param valid_size_2: float, size of validation subset 2 as a fraction of the training set
        :param data_aug: bool, if true, data augmentation will be used, if false it will not be used
        """
        dataset_train = self.get_dataset(name=dataset_name, root=f"{os.getcwd()}/data/raw",
                                         composed_transforms=self.get_base_transforms(), train=True)

        self.dataset_test = self.get_dataset(name=dataset_name, root=f"{os.getcwd()}/data/raw",
                                             composed_transforms=self.get_base_transforms(), train=False)

        self.dataset_train, self.dataset_valid_1, self.dataset_valid_2 = self.split_dataset(dataset_train,
                                                                                            valid_size_1,
                                                                                            valid_size_2)
        if data_aug:
            self.dataset_train.dataset = deepcopy(self.dataset_train.dataset)
            self.dataset_train.dataset.transform = self.get_augment_transforms()

    @staticmethod
    def get_dataset(name: str, root: str, composed_transforms: transforms.Compose = None,
                    download: bool = True, train: bool = True) -> datasets:
        """
        Downloads dataset from torchvision

        :param name: Name of the dataset must be in DATASETS list
        :param root: Path where the data shall be loaded/downloaded
        :param composed_transforms list of transforms or Compose of transforms
        :param download: Bool, if true, downloads the dataset and saves it in root directory.
                               If dataset is already downloaded, it is loaded from root directory.
        :param train: Bool, if true, creates dataset from training set, otherwise creates from test set.
        :return: PyTorch Dataset
        """
        if name not in DATASETS:
            raise Exception(f"The name provided must be in {DATASETS}")

        elif name == CIFAR10:
            return datasets.CIFAR10(root=root,
                                    transform=composed_transforms, download=download, train=train)

        else:
            return datasets.EMNIST(root=root, split='balanced',
                                   transform=composed_transforms, download=download, train=train)

    @staticmethod
    def split_dataset(dataset: datasets, split_size_1: float, split_size_2: float) -> (Subset, Subset, Subset):
        """
        Splits a dataset into three subsets.

        :param dataset: torchvision.datasets, input PyTorch dataset object to split
        :param split_size_1: float, proportion of dataset to include in subset_2.
                             The complement will be included in subset_1 & 3.
        :param split_size_2: float, proportion of dataset to include in subset_3.
                             The complement will be included in subset_1 & 2.
        :return: tuple, subset_1, subset_2 and subset_3
        """
        # Declare generator object for indices of stratified shuffle split
        strat_split = StratifiedShuffleSplit(n_splits=1, test_size=split_size_1)

        # Iterate through generator object once and break
        for index_1, index_2 in strat_split.split(np.zeros(len(dataset.targets)), dataset.targets):
            break

        # Declare generator object for indices of stratified shuffle split
        strat_split = StratifiedShuffleSplit(n_splits=1, test_size=split_size_2)

        # Iterate through generator object once and break
        for index_1, index_3 in strat_split.split(np.zeros(len(index_1)), [dataset.targets[i] for i in index_1]):
            break

        # Return subsets 1, 2 and 3
        return Subset(dataset, index_1), Subset(dataset, index_2), Subset(dataset, index_3)

    @staticmethod
    def get_base_transforms() -> transforms.Compose:
        """
        Get base transforms to apply to the data (with no data augmentation).

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

        return transforms.Compose(transforms_list)

    @staticmethod
    def get_augment_transforms() -> transforms.Compose:
        """
        Get transforms to apply to the train data if there is data augmentation.

        :return: Composed transforms
                 Type : torchvision.transforms.transforms.Compose
        """
        transforms_list = [
            # Convert images to RGB (3-channels) since models used expect 3 channels as input.
            # Needed for 1 channel datasets, such as EMNIST.
            transforms.Lambda(lambda image: image.convert('RGB')),

            # Add few augmentation transforms
            transforms.RandomRotation(15),
            transforms.ColorJitter(contrast=0.1,
                                   hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),

            # Convert images to pytorch tensors
            transforms.ToTensor()
        ]

        return transforms.Compose(transforms_list)
