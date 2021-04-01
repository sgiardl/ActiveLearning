"""
This file stores all functions related to model training.
"""
import os

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import Subset

import matplotlib.pyplot as plt

from .constants import *
from .model import load_zoo_models
from tqdm import tqdm
from typing import Callable


def get_transforms() -> Callable:
    """
    Get transforms to apply to the data.

    Transform # 1 : convert image to RGB (3-channels),
                    useful in case the input data is 1-channel only

    Transform # 2 : convert image to pytorch tensors

    :return: Composed transforms
             Type : torchvision.transforms.transforms.Compose
    """

    # Initialize empty list
    transforms_list = []

    # Convert images to RGB (3-channels) since models used expect 3 channels as input.
    # Needed for 1 channel datasets, such as EMNIST.
    transforms_list.append(transforms.Lambda(lambda image: image.convert('RGB')))

    # Convert images to pytorch tensors
    transforms_list.append(transforms.ToTensor())

    # Return composed transforms
    return transforms.Compose(transforms_list)


def train_model(epochs: int, data_loader: torch.utils.data.DataLoader, file_name: str,
                model_name: str, pretrained: bool = False, **kwargs) -> None:
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

    # Number of classes in the data_loader dataset
    if type(data_loader.dataset) == Subset:
        num_classes = len(data_loader.dataset.dataset.class_to_idx)
    else:
        num_classes = len(data_loader.dataset.class_to_idx)

    # Get model and set last fully-connected layer with the right
    # number of classes
    model = load_zoo_models(model_name, num_classes, pretrained=pretrained)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Find which parameters to train (those with .requires_grad = True)
    params = [p for p in model.parameters() if p.requires_grad]

    # Define Adam optimizer
    optimizer = torch.optim.Adam(params, lr=kwargs.get('learning_rate', LEARNING_RATE))

    # Define device as the GPU if available, else use the CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Send the model to the device
    model.to(device)

    # Get length of data_loader once
    len_data_loader = len(data_loader)

    # Initialize loss = 0 and loss_list as a blank list
    loss_list = []

    # Specify that the model will be trained
    model.train()

    # Declare tqdm progress bar
    pbar = tqdm(total=len_data_loader, leave=False,
                desc='Epoch 0', postfix='Loss: 0')

    # Main training loop, loop through each epoch
    for epoch in range(epochs):
        # Loop through each mini-batch from the data loader
        for i, (images, labels) in enumerate(data_loader):
            # Send images and labels to the device
            images, labels = images.to(device), labels.to(device)

            # Reset all gradients to zero
            optimizer.zero_grad()

            # Perform a forward pass
            outputs = model.forward(images)

            # Get the maximum prediction & class associated with
            # the maximum prediction
            # max_pred, max_class = preds.topk(1, dim=1)

            # Calculate the loss, comparing outputs with the
            # ground truth labels
            loss = criterion(outputs, labels)

            # Appending the current loss to the loss list
            loss_list.append(loss)

            # Perform a backward pass (calculate gradient)
            loss.backward()

            # Perform a parameter update based on the git current gradient
            optimizer.step()

            # Update progress bar
            pbar.set_description_str(f'Epoch {epoch}')
            pbar.set_postfix_str(f'Loss: {loss:.5f}')
            pbar.update()

        # Reset progress bar after epoch completion
        pbar.reset()

    # Display and format chart of loss per iteration
    plt.plot(loss_list)
    plt.title('Loss per iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

    # If file_name is specified, save the trained model
    if file_name is not None:
        torch.save(model.state_dict(), f'{os.getcwd()}/models/{file_name}')
