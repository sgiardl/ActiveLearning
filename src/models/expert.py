"""
This file stores the Expert class.
It simulates an expert who labels item from the dataset.
The indices are then fed to a Pytorch SubsetRandomSampler used by the training DataLoader
"""
from numpy import array
from numpy.random import choice
from torch import tensor, nonzero


class Expert:

    def __init__(self, dataset, n, prioritisation_criterion):

        """
        Select randomly n items from each class of the training dataset.
        These will be the first labeled items from our expert.

        :param dataset: PyTorch dataset
        :param n: Number of item to label per class at start
        :param prioritisation_criterion: Function that our expert uses to prioritise next images to label
        """

        self.criterion = prioritisation_criterion
        self.idx2class = {v: k for k, v in dataset.class_to_idx.items()}

        # We retrieve class distribution from the dataset
        class_dist = self.get_class_distribution(dataset)
        for k, v in class_dist.items():
            if v < n:
                raise Exception(f"Class {k} has less the {n} items")

        # We "annotate" n images of each class
        self.labeled = []
        self.initialize_labels(dataset, n)

    def get_class_distribution(self, dataset):

        """
        Count number of instances of each class in the dataset.
        Inspired from code at : https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a

        :param dataset: PyTorch dataset
        :return: dict
        """

        # We initialize a count of dataset class count
        count_dict = {k: 0 for k, v in dataset.class_to_idx.items()}

        # We count number of items in each class
        for item in dataset:
            count_dict[self.idx2class[item[1]]] += 1

        return count_dict

    def initialize_labels(self, dataset, n):
        """
        Selects randomly n indexes from each class of a dataset

        :param dataset: PyTorch dataset
        :param n: Number of items to label per class at start
        """
        # We save targets in a tensor
        targets = tensor(dataset.targets)

        # For each class we select n item randomly without replacement
        for k, _ in self.idx2class.items():
            class_idx = (nonzero(targets == k)).squeeze()
            self.labeled.extend(choice(class_idx, n, replace=False))

        # We turn the indexes list into a tensor
        self.labeled = tensor(self.labeled)

    def add_anotations(self, sofmax_outputs, n):
        """
        Add anotations based on prioritisation criterion used

        :param sofmax_outputs: softmax outputs of our model of the unlabeld data
        :param n: number of items to label
        """
        raise NotImplementedError
