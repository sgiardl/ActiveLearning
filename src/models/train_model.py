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
    cudnn.benchmark = True

    num_classes = len(data_loader.dataset.class_to_idx)

    model = load_zoo_models(model_name, num_classes)

    criterion = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    len_data_loader = len(data_loader)

    loss = 0
    loss_list = []

    for epoch in range(epochs):
        model.train()

        pbar = tqdm(desc=f'Epoch {epoch}',
                    total=len_data_loader,
                    leave=True,
                    postfix=f'Loss: {loss:.5f}')

        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            model.eval()

            log_preds = model.forward(images)
            preds = torch.exp(log_preds)
            max_pred, max_class = preds.topk(1, dim=1)
            loss = criterion(log_preds, labels)
            loss_list.append(loss)

            loss.backward()
            optimizer.step()

            pbar.set_postfix_str(f'Loss: {loss:.5f}')
            pbar.update()

        pbar.__del__()

    plt.plot(loss_list)
    plt.title('Loss per iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

    if file_name is not None:
        torch.save(model.state_dict(), f'{os.getcwd()}/models/{file_name}')