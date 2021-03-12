"""
This file stores a class called PretrainedClassificationModel that adapt a pretrained model from
torchvision.models to a new classification tasks.
"""

import torchvision.models as models

import torch.nn as nn
from .constants import *
MODELS = [RESNET34, SQUEEZE_NET_1_1]


def load_zoo_models(name, num_classes, pretrained=False):
    """
    Loads model from torchvision

    :param name: Name of the model must be in MODELS list
    :param num_classes: Number of classes in the last fully connected layer (nn.Linear)
    :param pretrained: bool indicating if we want the pretrained version of the model on ImageNet
    :return: PyTorch Model
    """
    if name not in MODELS:
        raise Exception(f"The name provided must be in {MODELS}")

    if name == RESNET34:
        if pretrained and num_classes != NUM_CLASSES_IMAGENET:
            return PretrainedClassifier(models.resnet34(pretrained), num_classes=num_classes)
        else:
            return models.resnet34(pretrained, num_classes=num_classes)

    else:
        if pretrained and num_classes != NUM_CLASSES_IMAGENET:
            return PretrainedClassifier(models.squeezenet1_1(pretrained), num_classes=num_classes)
        else:
            return models.squeezenet1_1(pretrained, num_classes=num_classes)


class PretrainedClassifier(nn.Module):

    def __init__(self, pretrained, num_classes):
        """
        Adds a linear layer and a ReLU to a pretrained model on ImageNet

        :param pretrained: Pretrained classifier
        :param num_classes: number of classes in the new classification task
        """
        super(PretrainedClassifier, self).__init__()
        self.pretrained = pretrained
        self.new_layer = nn.Sequential(nn.Linear(NUM_CLASSES_IMAGENET, num_classes),
                                       nn.ReLU())

    def forward(self, x):
        return self.new_layer(self.pretrained(x))
