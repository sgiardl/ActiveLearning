"""
This file stores all functions related to model training.
"""
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from src.models.constants import *
MODELS = [RESNET34, MOBILENETV2]


def load_zoo_models(name, pretrained=False):
    """
    Loads model from torchvision

    :param name: Name of the model must be in MODELS list
    :param pretrained: bool indicating if we want the pretrained version of the model
    :return: PyTorch Model
    """
    if name not in MODELS:
        raise Exception(f"The name provided must be in {MODELS}")

    elif name == RESNET34:
        return models.resnet34(pretrained)

    else:
        return models.mobilenet_v2(pretrained)

def get_transforms():
    custom_transforms = []
    # Convert images to RGB (3-channels) since models used expect 3 channels as input.
    # Needed for 1 channel datasets, such as EMNIST.
    custom_transforms.append(transforms.Lambda(lambda image: image.convert('RGB')))
    custom_transforms.append(transforms.ToTensor())
    return transforms.Compose(custom_transforms)

def define_fc_layer(model, model_name, num_classes):
    for param in model.parameters():
        param.requires_grad = False

    if model_name == RESNET34:
        model.fc = nn.Sequential(nn.Linear(512, num_classes))

    elif model_name == MOBILENETV2:
        model.classifier = nn.Sequential(nn.Dropout(0.2),
                                         nn.Linear(model.last_channel, num_classes))

    return model

def train_model(epochs, data_loader, device, file_name, model_name):
    cudnn.benchmark = True

    num_classes = len(data_loader.dataset.class_to_idx)

    model = load_zoo_models(model_name)
    model = define_fc_layer(model, model_name, num_classes)

    criterion = torch.nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM)
    model.to(device)

    len_data_loader = len(data_loader)

    loss_list = []

    for epoch in range(epochs):
        model.train()

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

            if i % 100 == 0:
                print(f'[Epoch: {epoch}] Iteration: {i}/{len_data_loader}, Loss: {loss}')

    plt.plot(loss_list)
    plt.title('Loss per iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

    torch.save(model.state_dict(), f'{os.getcwd()}/models/{file_name}')