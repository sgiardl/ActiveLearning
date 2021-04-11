import os
import sys
sys.path.insert(0, os.getcwd())
from src.models.constants import SQUEEZE_NET_1_1, RESNET18
from src.data.constants import CIFAR10, EMNIST
from Helpers import send_experiment_cmds
EXPERIMENT_NAME = 'query_strategy'

FIXED_SQUEEZENET_CMDS = ['-en', EXPERIMENT_NAME, '--model', SQUEEZE_NET_1_1, '--dataset', CIFAR10, '--n_start', '100',
                         '--n_new', '100', '--epochs', '10', '--batch_size', '50', '--learning_rate', '0.0001',
                         '--n_rounds', '20', '--patience', '3', '--data_aug', '--weight_decay', '0.005',
                         '--query_strategy']

FIXED_RESNET_CMDS = ['-en', EXPERIMENT_NAME, '--model', RESNET18, '--dataset', EMNIST, '--n_start', '50',
                     '--n_new', '1000', '--epochs', '10', '--batch_size', '50', '--learning_rate', '0.0001',
                     '--n_rounds', '10', '--patience', '3', '--data_aug', '--weight_decay', '0.005',
                     '--query_strategy']


def combination_generator(fixed_model_cmds):
    return [fixed_model_cmds + ['random_sampling'],
            fixed_model_cmds + ['least_confident'],
            fixed_model_cmds + ['margin_sampling'],
            fixed_model_cmds + ['entropy_sampling']]


if __name__ == '__main__':
    send_experiment_cmds(combination_generator, FIXED_SQUEEZENET_CMDS, FIXED_RESNET_CMDS)




