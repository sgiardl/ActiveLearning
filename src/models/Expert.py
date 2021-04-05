"""
This file stores the Expert class.
It simulates an expert who labels item from the dataset.
The indices are then fed to a Pytorch SubsetRandomSampler used by the training DataLoader
"""

from numpy.random import choice
from typing import Callable, Union
from torch import tensor, nonzero
from torch.utils.data import SubsetRandomSampler, Dataset, Subset
import matplotlib.pyplot as plt
import torch
from torch.distributions import Categorical

PRIORITISATION_CRITERION = ['least_confident', 'margin_sampling', 'entropy_sampling']


class Expert:
    def __init__(self, dataset: Dataset, n: int, query_strategy: str):
        """
        Select randomly n items from each class of the training dataset.
        These will be the first labeled items from our expert.

        :param dataset: PyTorch dataset
        :param n: Number of item to label per class at start
        :param query_strategy: Name of the function that our expert uses to prioritise next images to label
        """

        # We initialize the criterion object
        self.initialize_query_strategy(self, query_strategy)

        if type(dataset) == Subset:
            self.idx_to_class = {v: k for k, v in dataset.dataset.class_to_idx.items()}
        else:
            self.idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

        # We retrieve class distribution from the dataset
        class_dist = self.get_class_distribution(dataset)
        for k, v in class_dist.items():
            assert v >= n, f"Class {k} has less than {n} items"

        # We "label" n images of each class
        self.labeled_idx = []
        self.initialize_labels(dataset, n)
        self.labeled_history = {k: [n] for k in self.idx_to_class.keys()}

        # We initialize the sampler object
        self.update_expert_sampler()

    @staticmethod
    def initialize_query_strategy(self, prioritisation_criterion: str) -> None:
        """
        This method initializes prioritisation criterion

        :param prioritisation_criterion: Name of the function that our expert uses to prioritise next images to label
        """
        if prioritisation_criterion not in PRIORITISATION_CRITERION:
            raise Exception("The prioritisation_criterion provided must be in {PRIORITISATION_CRITERION}")
        elif prioritisation_criterion == 'least_confident':
            self.criterion = self.least_confident_criterion
        elif prioritisation_criterion == 'margin_sampling':
            self.criterion = self.margin_sampling_criterion
        elif prioritisation_criterion == 'entropy_sampling':
            self.criterion = self.entropy_sampling_criterion
        # else:
        #     self.criterion = random
        # should be random by default

    @staticmethod
    def least_confident_criterion(softmax_outputs: tensor, n: int) -> tensor:
        """
        This method implements the "Least Confidence" strategy

        :param softmax_outputs: Softmax outputs of our model of the unlabeled data
        :param n: Number of items to label
        :return: tensor
        """
        softmax_outputs_max, _ = torch.max(1 - softmax_outputs, dim=1)
        _, max_uncertainty_indices = torch.sort(-softmax_outputs_max)
        prioritisation_softmax_indices = max_uncertainty_indices[0:n]
        return prioritisation_softmax_indices

    @staticmethod
    def margin_sampling_criterion(softmax_outputs: tensor, n: int) -> tensor:
        """
        This method implements the "Margin Sampling" strategy

        :param softmax_outputs: Softmax outputs of our model of the unlabeled data
        :param n: Number of items to label
        :return: tensor
        """
        sort_softmax_outputs, _ = torch.sort(-softmax_outputs)
        margin = - sort_softmax_outputs[:, 0] + sort_softmax_outputs[:, 1]
        _, min_margin_indices = torch.sort(margin)
        prioritisation_softmax_indices = min_margin_indices[0:n]
        return prioritisation_softmax_indices

    @staticmethod
    def entropy_sampling_criterion(softmax_outputs: tensor, n: int) -> tensor:
        """
        This method implements the "Entropy Sampling" strategy

        :param softmax_outputs: Softmax outputs of our model of the unlabeled data
        :param n: Number of items to label
        :return: tensor
        """
        softmax_outputs_entropy = Categorical(probs=softmax_outputs).entropy()
        _, max_entropy_indices = torch.sort(-softmax_outputs_entropy)
        prioritisation_softmax_indices = max_entropy_indices[0:n]
        return prioritisation_softmax_indices

    def get_class_distribution(self, dataset: Dataset) -> dict:
        """
        Count number of instances of each class in the dataset.
        Inspired from code at : https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a

        :param dataset: PyTorch dataset
        :return: dict
        """

        # We initialize a count of dataset class count
        if type(dataset) == Subset:
            count_dict = {k: 0 for k, v in dataset.dataset.class_to_idx.items()}
        else:
            count_dict = {k: 0 for k, v in dataset.class_to_idx.items()}

        # We count number of items in each class
        for item in dataset:
            count_dict[self.idx_to_class[item[1]]] += 1

        return count_dict

    def initialize_labels(self, dataset: Dataset, n: int) -> None:
        """
        Selects randomly n indexes from each class of a dataset

        :param dataset: PyTorch dataset
        :param n: Number of items to label per class at start
        """
        # We save targets in a tensor
        if type(dataset) == Subset:
            targets = tensor([dataset.dataset.targets[i] for i in dataset.indices])
        else:
            targets = tensor(dataset.targets)

        # For each class we select n items randomly without replacement
        for k, _ in self.idx_to_class.items():
            class_idx = (nonzero(targets == k)).squeeze()
            self.labeled_idx.extend(choice(class_idx, n, replace=False))

        # We turn the indexes list into a tensor
        self.labeled_idx = tensor(self.labeled_idx)

    def add_labels(self, unlabeled_data_idx, softmax_outputs, n: int, dataset: Dataset) -> None:
        """
        Add labels based on prioritisation criterion used

        :param unlabeled_data_idx: Dataloader with a single batch with all unlabeled images
        :param softmax_outputs: Softmax outputs of our model of the unlabeled data
        :param n: Number of items to label
        :param dataset: PyTorch dataset
        """
        # Evaluate prioritisation score of each image using the softmax_outputs
        # and appropriate prioritisation criterion method
        prioritisation_softmax_indices = self.criterion(softmax_outputs, n)
        prioritisation_indices = unlabeled_data_idx[prioritisation_softmax_indices]

        # Add the idx of the n most important images based on their prioritisation score
        self.labeled_idx = torch.cat((self.labeled_idx, prioritisation_indices), dim=0)

        # Update the labeled history.
        self.update_labels_history(n, dataset, prioritisation_indices)

        # Update the expert sampler
        self.update_expert_sampler()

    def update_labels_history(self, n: int, dataset: Dataset, prioritisation_indices) -> None:
        """
        Update labeled history
        :param n: Number of items to label
        :param dataset: PyTorch dataset
        :param prioritisation_indices: Dataloader with a single batch with all unlabeled images
        """

        for i in range(n):
            prioritisation_idx = prioritisation_indices[i]
            _, prioritisation_class = dataset.__getitem__(self.labeled_idx[prioritisation_idx])
            self.labeled_history[prioritisation_class][0] += 1

    def show_labels_history(self, show: bool = True, save_path: Union[str, None] = None,
                            fig_format: str = 'pdf') -> None:
        """
        Plot the growth of labeled items per class throughout the active learning iteration

        :param show: Boolean indicating we want to show the figure
        :param save_path: Path to save the image. The paths must include the file name. (None == unsaved)
        :param fig_format: Format used to save the figure
        """

        # We save the number of active learning iterations done
        x = range(len(self.labeled_history[0]))
        for k, history in self.labeled_history.items():
            plt.plot(x, history, label=self.idx_to_class[k])

        # We set x-axis steps
        plt.xticks(x)

        # We set axis labels and legend
        plt.ylabel('Number of labeled images')
        plt.xlabel('Active learning iterations')
        plt.legend()

        # We show the plot
        if show:
            plt.show()

        # We save it
        if save_path is not None:
            plt.savefig(f"{save_path}.{fig_format}")

    def update_expert_sampler(self) -> None:
        """
        Update the PyTorch sampler object to give to the dataloader for the training
        """
        self.sampler = SubsetRandomSampler(self.labeled_idx)
