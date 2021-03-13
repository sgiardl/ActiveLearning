"""
This file stores a class called PretrainedClassificationModel that adapt a pretrained model from
torchvision.models to a new classification tasks.
"""

import torchvision.models as models

import torch.nn as nn
from .constants import *
MODELS = [RESNET34, SQUEEZE_NET_1_1]


def load_zoo_models(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """
    Loads model from torchvision and changes last layer if pretrained = True and num_classes != 1000

    :param name: Name of the model must be in MODELS list
    :param num_classes: Number of classes in the last fully connected layer (nn.Linear)
    :param pretrained: bool indicating if we want the pretrained version of the model on ImageNet
    :return: PyTorch Model
    """
    assert name in MODELS, f"The name provided must be in {MODELS}"

    if name == RESNET34:
        if pretrained and num_classes != NUM_CLASSES_IMAGENET:
            m = models.resnet34(pretrained)
            m.fc = nn.Linear(512, num_classes)
        else:
            m = models.resnet34(pretrained, num_classes=num_classes)
    else:
        if pretrained and num_classes != NUM_CLASSES_IMAGENET:
            m = models.squeezenet1_1(pretrained)
            m.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        else:
            m = models.squeezenet1_1(pretrained, num_classes=num_classes)
    return m
