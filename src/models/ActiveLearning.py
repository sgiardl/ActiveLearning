"""
File:
    models/ActiveLearning.py

Authors:
    - Abir Riahi
    - Nicolas Raymond
    - Simon Giard-Leroux

Description:
    Defines the ActiveLearner class.
"""

from src.models.Expert import Expert
from src.models.TrainValidTestManager import TrainValidTestManager
from src.data.DatasetManager import DatasetManager
from src.data.DataLoaderManager import DataLoaderManager
from torch.utils.data import Subset
from src.visualization.VisualizationManager import VisualizationManager
import json
import os
import time


class ActiveLearner:
    """
    Object in charge of pool-based active learning.
    """
    def __init__(self, model: str, dataset: str, n_start: int, n_new: int, epochs: int,
                 query_strategy: str, experiment_name: str, patience: int = 3,
                 batch_size: int = 50, shuffle: bool = False, num_workers: int = 1,
                 lr: float = 0.005, weight_decay: float = 0, pretrained: bool = False,
                 valid_size_1: float = 0.20, valid_size_2: float = 0.20, data_aug: bool = False) -> None:
        """
        :param model: Name of the model to train ("ResNet34" or "SqueezeNet11")
        :param dataset: Name of the dataset to learn on ("CIFAR10" or "EMNIST")
        :param n_start: The number of items that must be randomly labeled in each class by the Expert
        :param n_new: The number of new items that must be labeled within each active learning loop
        :param epochs: Number of training epochs in each active learning loop
        :param query_strategy: Query strategy of the expert, 4 choices :
                               'random_sampling', 'least_confident',
                               'margin_sampling', 'entropy_sampling'
        :param experiment_name: Name of the active learning experiment
        :param patience: Maximal number of consecutive rounds without improvement
        :param batch_size: Batch size of dataloaders storing train, valid and test set
        :param shuffle: Bool indicating if data are shuffled within loaders
        :param num_workers: Number of workers used by the dataloaders, used for multiprocessing
        :param lr: Learning rate of the model during training
        :param pretrained: Bool indicating if the model used must be pretrained on ImageNet
        :param valid_size_1: Portion of train set used as valid 1
        :param valid_size_2: Portion of train set used as valid 2
        :param data_aug: Bool indicating if we want data augmentation in the training set
        """

        # We create a temporary manager
        dataset_manager = DatasetManager(dataset, valid_size_1, valid_size_2, data_aug)

        # We first initialize an expert
        self.expert = Expert(dataset_manager.dataset_train, n_start, query_strategy)

        # We initialize DataLoaderManager and save DatasetManager
        self.loader_manager = DataLoaderManager(dataset_manager, self.expert, batch_size, shuffle, num_workers)
        self.dataset_manager = dataset_manager

        # We initialize TrainValidTestManager
        self.training_manager = TrainValidTestManager(self.loader_manager, None, model, lr, weight_decay, pretrained)

        # We initialize important attributes to lead active learning
        self.n_new = n_new
        self.epochs = epochs
        self.experiment_name = experiment_name + time.strftime("_%Y-%m-%d_%H-%M-%S")

        # We initialize attributes to keep track of active learning loops
        self.loop_progress = []
        self.best_accuracy = 0
        self.patience = patience
        self.patience_count = 0

        # We initialize a visualization manager
        self.visualization_manager = VisualizationManager()

        # We initialize a dictionary that will contains all data to save in a json file at the end
        self.history = {'Initialization': {'model': model, 'dataset': dataset, 'n_start': n_start, 'n_new': n_new,
                                           'epochs': epochs, 'valid_size_1': valid_size_1, 'valid_size_2': valid_size_2,
                                           'lr': lr, 'pretrained': pretrained, 'query_strategy:': query_strategy,
                                           'batch_size': batch_size, 'weight_decay': weight_decay, 'patience': patience,
                                           'shuffle': shuffle, 'data_aug': data_aug}}

    def update_labeled_items(self) -> None:
        """
        Updates DataLoaderManager and TrainValidTestManager
        """

        # Update of DataLoaderManager
        self.loader_manager.update(self.expert)

        # Update TrainValidTestManager
        self.training_manager.update_train_loader(self.loader_manager)

    def save_all_history(self) -> None:
        # We create a directory for the experiment
        os.mkdir(self.experiment_name)

        # We add training loss, valid loss, training accuracy and test accuracy to the dictionary
        self.history['Loops Progress'] = self.training_manager.results

        # We add the valid_2 accuracy over rounds
        self.history['Validation-2 Accuracy Coordinates'] = self.loop_progress

        # We save the final test score
        self.history['Final Test Score'] = self.training_manager.test_model(final_eval=True)

        # We save the stopping condition reached
        self.history['Premature Stop'] = True if self.patience_count == self.patience else False

        # We dump the dictionary into a json file
        json_obj = json.dumps(self.history, indent=4)
        with open(os.path.join(self.experiment_name, "records.json"), "w") as outfile:
            outfile.write(json_obj)

        # We save the plots
        self.visualization_manager.show_labels_history(self.expert, show=True,
                                                       save_path=os.path.join(self.experiment_name, "labels_prog"))

        self.visualization_manager.show_loss_acc_chart(self.training_manager.results,
                                                       save_path=os.path.join(self.experiment_name, "loss_acc_prog"))

    def __call__(self, n_rounds: int) -> None:
        """
        Executes the active learning loops

        :param n_rounds: Number of active learning rounds
        """

        # We save the number of rounds to the history
        self.history['Initialization']['n_rounds'] = n_rounds

        # We warn that the active learning is started
        print(f"Unlabeled items : {len(self.loader_manager.unlabeled_idx)}")
        print("Active Learning Started\n")

        i = 0

        while True:
            # We update a string that will be used for multiple prints
            loop_reference = f"Active Loop #{i}"

            # We train the model on labeled image in the training set
            print(f"{loop_reference} - Training...")
            self.training_manager.train_model(self.epochs)
            self.visualization_manager.show_loss_acc_chart(self.training_manager.results)

            # We evaluate our model on the current test set
            print(f"{loop_reference} - Validation...")
            accuracy = self.training_manager.test_model()
            self.loop_progress.append((i*self.n_new, accuracy))
            print(f"{loop_reference} - Validation-2 Accuracy {round(accuracy,4)}")

            # We evaluate the patience
            accuracy_diff = accuracy - self.best_accuracy
            print(f"{loop_reference} - Accuracy Difference With Best {round(accuracy_diff, 4)}")
            if accuracy_diff <= 0:
                self.patience_count += 1
            else:
                self.patience_count = 0

            # We look if stopping conditions are reached
            if i == n_rounds or self.patience_count == self.patience:
                print("Active learning stop - Stopping criteria reached")
                self.save_all_history()
                break

            # We update last accuracy attribute
            self.best_accuracy = max(accuracy, self.best_accuracy)

            # We make prediction on the unlabeled data
            print(f"{loop_reference} - Unlabeled Evaluation")
            unlabeled_dataset = Subset(self.dataset_manager.dataset_train.dataset, self.loader_manager.unlabeled_idx)
            unlabeled_softmax = self.training_manager.evaluate_unlabeled(unlabeled_dataset)

            # We request labels to our expert
            print(f"{loop_reference} - Labels Request")
            self.expert.add_labels(self.loader_manager.unlabeled_idx, unlabeled_softmax,
                                   self.n_new, self.dataset_manager.dataset_train)

            # We update internal attributes
            self.update_labeled_items()
            i += 1

            print(f"{loop_reference} - Unlabeled items : {len(self.loader_manager.unlabeled_idx)}\n")
