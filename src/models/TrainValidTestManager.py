"""
This file stores all functions related to model training.
"""
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset
from src.data.DatasetManager import DatasetManager
import matplotlib.pyplot as plt

from .constants import *
from .model import load_zoo_models
from tqdm import tqdm


class TrainValidTestManager():
    def __init__(self, dataset_manager: DatasetManager,
                 file_name: str,
                 model_name: str,
                 learning_rate: float,
                 pretrained: bool = False):
        self.data_loader_train = dataset_manager.data_loader_train
        self.file_name = file_name

        # Define device as the GPU if available, else use the CPU
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Number of classes in the data_loader dataset
        if type(self.data_loader_train.dataset) == Subset:
            num_classes = len(self.data_loader_train.dataset.dataset.class_to_idx)
        else:
            num_classes = len(self.data_loader_train.dataset.class_to_idx)

        # Get model and set last fully-connected layer with the right
        # number of classes
        self.model = load_zoo_models(model_name, num_classes, pretrained=pretrained)

        # Define loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # Find which parameters to train (those with .requires_grad = True)
        params = [p for p in self.model.parameters() if p.requires_grad]

        # Define Adam optimizer
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)

        # Send the model to the device
        self.model.to(self.device)

        # Get length of data_loader once
        self.len_data_loader = len(self.data_loader_train)


    def train_model(self, epochs: int) -> None:
        """
        Trains the model and saves the trained model.

        :param epochs: int, number of epochs
        :param data_loader: DataLoader object
        :param file_name: str, file name with which to save the trained model
        :param model_name: str, name of the model (RESNET34 or SQUEEZE_NET_1_1)
        :param pretrained: bool, indicates if the model should be pretrained on ImageNet
        """

        # Flag to enable the inbuilt cudnn auto-tuner
        # to find the best algorithm to use for the hardware used
        cudnn.benchmark = True

        # Specify that the model will be trained
        self.model.train()

        # Declare tqdm progress bar
        pbar = tqdm(total=self.len_data_loader, leave=False,
                    desc='Epoch 0', postfix='Loss: 0')

        loss_list = []
        accuracy_list = []
        fig, (ax1, ax2) = plt.subplots(1, 2)

        # Main training loop, loop through each epoch
        for epoch in range(epochs):
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

                # Calculate the loss, comparing outputs with the
                # ground truth labels
                loss = self.criterion(outputs, labels)

                # Appending the current loss to the loss list
                loss_list_epoch.append(loss.item())
                accuracy_list_epoch.append(self.get_accuracy(outputs, labels))

                # Perform a backward pass (calculate gradient)
                loss.backward()

                # Perform a parameter update based on the git current gradient
                self.optimizer.step()

                # Update progress bar
                pbar.set_description_str(f'Epoch {epoch}')
                pbar.set_postfix_str(f'Loss: {loss:.5f}')
                pbar.update()

            # Reset progress bar after epoch completion
            pbar.reset()

            loss_list.append(np.mean(loss_list_epoch))
            accuracy_list.append(np.mean(accuracy_list_epoch))

        # Display and format chart of loss per iteration
        ax1.plot(loss_list, marker='.')
        ax1.set_title('Mean loss per epoch')
        ax1.set(xlabel='Epoch', ylabel='Mean loss')

        ax2.plot(accuracy_list, marker='.')
        ax2.set_title('Mean accuracy per epoch')
        ax2.set(xlabel='Epoch', ylabel='Mean accuracy')

        fig.tight_layout()
        fig.show()

        # If file_name is specified, save the trained model
        if self.file_name is not None:
            torch.save(self.model.state_dict(), f'{os.getcwd()}/models/{self.file_name}')

    # def validate_model(self):
    #

    def get_accuracy(self, outputs, labels):
        return (outputs.argmax(dim=1) == labels).sum().item() / labels.size(0)
