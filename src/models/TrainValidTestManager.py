"""
This file stores all functions related to model training.
"""
import os
import numpy as np
import torch
from torch.utils.data import Subset, Dataset, DataLoader
from torch import tensor
from torch.nn.functional import softmax
from src.data.DataLoaderManager import DataLoaderManager
import matplotlib.pyplot as plt
from .model import load_zoo_models
from tqdm import tqdm
from typing import Sequence, Union


class TrainValidTestManager:
    def __init__(self, data_loader_manager: DataLoaderManager,
                 file_name: str,
                 model_name: str,
                 learning_rate: float,
                 pretrained: bool = False):
        """
        Training, validation and testing manager.

        :param data_loader_manager: DatasetManager, class with each dataset and dataloader
        :param file_name: str, file name with which to save the trained model
        :param model_name: str, name of the model (RESNET34 or SQUEEZE_NET_1_1)
        :param learning_rate: float, learning rate for the Adam optimizer
        :param pretrained: bool, indicates if the model should be pretrained on ImageNet
        """
        # Flag to enable the inbuilt cudnn auto-tuner
        # to find the best algorithm to use for the hardware used
        torch.backends.cudnn.benchmark = True

        # Extract the training, validation and testing data loaders
        self.data_loader_train = data_loader_manager.data_loader_train
        self.data_loader_valid = data_loader_manager.data_loader_valid
        self.data_loader_test = data_loader_manager.data_loader_test

        # Save the file name
        self.file_name = file_name

        # Define device as the GPU if available, else use the CPU
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Number of classes in the data_loader dataset
        if type(self.data_loader_train.dataset) == Subset:
            num_classes = len(self.data_loader_train.dataset.dataset.class_to_idx)
        else:
            num_classes = len(self.data_loader_train.dataset.class_to_idx)

        # Get model and set last fully-connected layer with the right number of classes
        self.model = load_zoo_models(model_name, num_classes, pretrained=pretrained)

        # Define loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # Find which parameters to train (those with .requires_grad = True)
        params = [p for p in self.model.parameters() if p.requires_grad]

        # Define Adam optimizer
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)

        # Send the model to the device
        self.model.to(self.device)

        # Declare empty training loss and accuracy lists
        self.train_loss_list = []
        self.train_accuracy_list = []

        # Declare empty validation loss and accuracy lists
        self.valid_loss_list = []
        self.valid_accuracy_list = []

    def update_train_loader(self, data_loader_manager: DataLoaderManager) -> None:
        """
        Updates the train loader

        :param data_loader_manager: DataLoaderManager
        """
        self.data_loader_train = data_loader_manager.data_loader_train

    def train_model(self, epochs: int) -> None:
        """
        Trains the model and saves the trained model.

        :param epochs: int, number of epochs
        """
        # Specify that the model will be trained
        self.model.train()

        # Declare tqdm progress bar
        pbar = tqdm(total=len(self.data_loader_train), leave=False,
                    desc='Epoch 0', postfix='Training Loss: 0')

        # Main training loop, loop through each epoch
        for epoch in range(epochs):
            # Declare empty loss and accuracy lists for the current epoch
            loss_list_epoch = []
            accuracy_list_epoch = []

            # Loop through each mini-batch from the data loader
            for i, (images, labels) in enumerate(self.data_loader_train):
                # Send images and labels to the device
                images, labels = images.to(self.device), labels.to(self.device)

                # Reset all gradients to zero
                self.optimizer.zero_grad()

                # Perform a forward pass
                outputs = self.model.forward(images)

                # Calculate the loss, comparing outputs with the ground truth labels
                loss = self.criterion(outputs, labels)

                # Appending the current loss to the loss list and the current accuracy to the accuracy list
                loss_list_epoch.append(loss.item())
                accuracy_list_epoch.append(self.get_accuracy(outputs, labels))

                # Perform a backward pass (calculate gradient)
                loss.backward()

                # Perform a parameter update based on the current gradient
                self.optimizer.step()

                # Update progress bar
                pbar.set_description_str(f'Epoch {epoch}')
                pbar.set_postfix_str(f'Training Loss: {loss:.5f}')
                pbar.update()

            # Reset progress bar after epoch completion
            pbar.reset()

            # Save the training loss and accuracy in the object
            self.train_loss_list.append(np.mean(loss_list_epoch))
            self.train_accuracy_list.append(np.mean(accuracy_list_epoch))

            # Validate the model
            self.validate_model()

        # Display and format chart of loss and accuracy per epoch
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.plot(self.train_loss_list, marker='.', label='Training')
        ax1.plot(self.valid_loss_list, marker='.', label='Validation')
        ax1.legend(loc='upper right')
        ax1.set_title('Mean loss per epoch')
        ax1.set(xlabel='Epoch', ylabel='Mean loss')

        ax2.plot(self.train_accuracy_list, marker='.', label='Training')
        ax2.plot(self.valid_accuracy_list, marker='.', label='Validation')
        ax2.legend(loc='upper right')
        ax2.set_ylim([0, 1])
        ax2.set_title('Mean accuracy per epoch')
        ax2.set(xlabel='Epoch', ylabel='Mean accuracy')

        fig.tight_layout()
        fig.show()

        # If file_name is specified, save the trained model
        if self.file_name is not None:
            torch.save(self.model.state_dict(), f'{os.getcwd()}/models/{self.file_name}')

    def validate_model(self) -> None:
        """
        Method to validate the model saved in the self.model class attribute.

        :return: None
        """
        # Specify that the model will be evaluated
        self.model.eval()

        # Declare empty loss and accuracy lists
        loss_list = []
        accuracy_list = []

        # Deactivate the autograd engine
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.data_loader_valid):
                # Send images and labels to the device
                images, labels = images.to(self.device), labels.to(self.device)

                # Perform a forward pass
                outputs = self.model.forward(images)

                # Calculate the loss, comparing outputs with the ground truth labels
                loss = self.criterion(outputs, labels)

                # Appending the current loss to the loss list and current accuracy to the accuracy list
                loss_list.append(loss.item())
                accuracy_list.append(self.get_accuracy(outputs, labels))

        # Calculate mean loss and mean accuracy over all batches
        mean_loss = np.mean(loss_list)
        mean_accuracy = np.mean(accuracy_list)

        # Save mean loss and mean accuracy in the object
        self.valid_loss_list.append(mean_loss)
        self.valid_accuracy_list.append(mean_accuracy)

    def test_model(self) -> float:
        """
        Method to test the model saved in the self.model class attribute.

        :return: None
        """
        # Specify that the model will be evaluated
        self.model.eval()

        # Initialize empty accuracy list
        accuracy_list = []

        # Deactivate the autograd engine
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.data_loader_test):
                # Send images and labels to the device
                images, labels = images.to(self.device), labels.to(self.device)

                # Perform a forward pass
                outputs = self.model.forward(images)

                # Calculate the accuracy for the current batch
                accuracy_list.append(self.get_accuracy(outputs, labels))

        # Print mean test accuracy over all batches
        mean_accuracy = np.mean(accuracy_list)
        print(f'\nTest Accuracy: {mean_accuracy:.5f}')
        return mean_accuracy

    def evaluate_unlabeled(self, unlabeled_subset: Subset) -> tensor:
        """
        Returns softmax of the prediction
        :param unlabeled_subset: Subset containing unlabeled data
        :return: softmax outputs
        """
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=len(unlabeled_subset.dataset))

        # Specify that the model will be evaluated
        self.model.eval()

        # Deactivate the autograd engine
        with torch.no_grad():

            # Send images to the device
            images, _ = next(iter(unlabeled_loader))
            images = images.to(self.device)

            # Perform a forward pass
            outputs = self.model.forward(images)

        return softmax(outputs, dim=1)

    @staticmethod
    def get_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Method to calculate accuracy of predicted outputs vs ground truth labels.

        :param outputs: torch.Tensor, predicted outputs classes
        :param labels: torch.Tensor, ground truth labels classes
        :return: float, accuracy of the predicted outputs vs the ground truth labels
        """
        return (outputs.argmax(dim=1) == labels).sum().item() / labels.shape[0]
