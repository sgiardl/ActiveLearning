"""
This file stores all functions related to model training.
"""

import torchvision.models as models
from constants import *
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