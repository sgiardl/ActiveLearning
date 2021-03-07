"""
This file stores all functions related to model training.
"""
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.transforms as transforms

from src.models.constants import *
MODELS = [RESNET34, MOBILENETV2]


def load_zoo_models(name, num_classes, pretrained=False):
    """
    Loads model from torchvision

    :param name: Name of the model must be in MODELS list
    :param pretrained: bool indicating if we want the pretrained version of the model
    :return: PyTorch Model
    """
    if name not in MODELS:
        raise Exception(f"The name provided must be in {MODELS}")

    elif name == RESNET34:
        return models.resnet34(pretrained)#, num_classes=num_classes)

    else:
        return models.mobilenet_v2(pretrained)#, num_classes=num_classes)

def get_transforms():
    custom_transforms = []
    custom_transforms.append(transforms.ToTensor())
    return transforms.Compose(custom_transforms)

# def collate_fn(batch):
#     return tuple(zip(*batch))

def train_model(epochs, data_loader, device, file_name, model_name):
    cudnn.benchmark = True

    num_classes = len(data_loader.dataset.class_to_idx)
    model = load_zoo_models(name=model_name, num_classes=num_classes)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(512, num_classes),
                             nn.LogSoftmax(dim=1))

    params = [p for p in model.parameters() if p.requires_grad]
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE)#model.fc.parameters()
    model.to(device)

    len_data_loader = len(data_loader)

    for epoch in range(epochs):
        model.train()

        for i, data in enumerate(data_loader):
            images, labels = data
#            images = torch.stack(images).to(device)
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()


            model.eval()
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'[Epoch: {epoch}] Iteration: {i}/{len_data_loader}, Loss: {loss}')

    torch.save(model.state_dict(), f'{os.getcwd()}/models/{file_name}')