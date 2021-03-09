"""
This file stores all functions related to model training.
"""
import os

import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from tqdm import tqdm

from src.models.constants import *
MODELS = [RESNET34, MOBILENETV2]


def load_zoo_models(name, num_classes, pretrained=False):
    """
    Loads model from torchvision

    :param name: Name of the model must be in MODELS list
    :param num_classes: Number of classes in the last fully connected layer (nn.Linear)
    :param pretrained: bool indicating if we want the pretrained version of the model
    :return: PyTorch Model
    """
    if name not in MODELS:
        raise Exception(f"The name provided must be in {MODELS}")

    elif name == RESNET34:
        return models.resnet34(pretrained, num_classes=num_classes)

    else:
        return models.mobilenet_v2(pretrained, num_classes=num_classes)

def get_transforms():
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

def train_model(epochs, data_loader, file_name, model_name):
    """
    Trains the model and saves the trained model.

    :param epochs: int, number of epochs
    :param data_loader: DataLoader object
    :param file_name: str, file name with which to save the trained model
    :param model_name: str, name of the model (RESNET34 or MOBILENETV2)
    """

    # Flag to enable the inbuilt cudnn auto-tuner
    # to find the best algorithm to use for the hardware used
    cudnn.benchmark = True

    # Number of classes in the data_loader dataset
    num_classes = len(data_loader.dataset.class_to_idx)

    # Get model and set last fully-connected layer with the right
    # number of classes
    model = load_zoo_models(model_name, num_classes)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Find which parameters to train (those with .requires_grad = True)
    params = [p for p in model.parameters() if p.requires_grad]

    # Define stochastic gradient descent optimizer
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM)

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

            # Perform a forward pass and calculate log(preds)
            log_preds = model.forward(images)

            # Calculate the preds with exp(log(preds))
            preds = torch.exp(log_preds)

            # Get the maximum prediction & class associated with
            # the maximum prediction
            max_pred, max_class = preds.topk(1, dim=1)

            # Calculate the loss, comparing log(preds) with the
            # ground truth labels
            loss = criterion(log_preds, labels)

            # Appending the current loss to the loss list
            loss_list.append(loss)

            # Perform a backward pass (calculate gradient)
            loss.backward()

            # Perform a parameter update based on the current gradient
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