import numpy as np
import os
import torchvision.datasets as datasets
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision.transforms as transforms
from src.models.expert import Expert
from torch.utils.data import DataLoader
from src.data.constants import *
DATASETS = [CIFAR10, EMNIST]


class DatasetManager:
    def __init__(self, dataset_name: str,
                 valid_size: float,
                 batch_size: int,
                 shuffle: bool = False,
                 num_workers: int = 1) -> None:
        """
        Dataset Manager class, handles the creation of the training, validation and testing datasets and
        data loaders.

        :param dataset_name: string, name of the dataset to load
        :param valid_size: float, size of validation subset as a fraction of the training set
        :param batch_size: int, batch size for forward pass
        :param shuffle: bool, to shuffle the data loaders
        :param num_workers: int, number of multiprocessing workers,
                            should be smaller or equal to the number of cpu threads
        """
        dataset_train = self.get_dataset(name=dataset_name, root=f"{os.getcwd()}/data/raw",
                                         composed_transforms=self.get_transforms(), train=True)

        self.dataset_test = self.get_dataset(name=dataset_name, root=f"{os.getcwd()}/data/raw",
                                             composed_transforms=self.get_transforms(), train=False)

        self.dataset_train, self.dataset_valid = self.split_dataset(dataset_train, valid_size)

        self.expert = Expert(self.dataset_train, 2, None)

        self.data_loader_train = DataLoader(self.dataset_train, batch_size=batch_size, shuffle=shuffle,
                                            num_workers=num_workers, sampler=self.expert.sampler)

        self.data_loader_valid = DataLoader(self.dataset_valid, batch_size=batch_size, shuffle=shuffle,
                                            num_workers=num_workers)

        self.data_loader_test = DataLoader(self.dataset_test, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers)

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
            return datasets.EMNIST(root=root, split='byclass',
                                   transform=composed_transforms, download=download, train=train)

    @staticmethod
    def split_dataset(dataset: datasets, split_size: float) -> (Subset, Subset):
        """
        Splits a dataset into two subsets.

        :param dataset: torchvision.datasets, input PyTorch dataset object to split
        :param split_size: float, proportion of dataset to include in subset_2.
                           The complement will be included in subset_1.
        :return: tuple, subset_1 and subset_2
        """
        # Declare generator object for indices of stratified shuffle split
        strat_split = StratifiedShuffleSplit(n_splits=1, test_size=split_size)

        # Iterate through generator object once and break
        for index_1, index_2 in strat_split.split(np.zeros(len(dataset.targets)), dataset.targets):
            break

        # Return subsets 1 and 2
        return Subset(dataset, index_1), Subset(dataset, index_2)

    @staticmethod
    def get_transforms() -> transforms.Compose:
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

        return transforms.Compose(transforms_list)
