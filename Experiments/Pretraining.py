import os
import sys
sys.path.insert(0, os.getcwd())
import subprocess as sp
import time
from src.models.constants import SQUEEZE_NET_1_1, RESNET34
from src.data.constants import CIFAR10, EMNIST

FIXED_SQUEEZENET_PARAM = ['--model', SQUEEZE_NET_1_1, '--dataset', CIFAR10, '--n_start', '100',
                          '--n_new', '100', '--epochs', '20', '--query_strategy', 'least_confident',
                          '--batch_size', '50', '--lr', '0.0001', '--n_rounds', '20', '--patience', '3',
                          '--data_aug', '--weight_decay', '0.001']

FIXED_RESNET_PARAM = ['--model', RESNET34, '--dataset', EMNIST, '--n_start', '50',
                      '--n_new', '1000', '--epochs', '10', '--query_strategy', 'least_confident',
                      '--batch_size', '50', '--lr', '0.0001', '--n_rounds', '10', '--patience', '3',
                      '--data_aug', '--weight_decay', '0.001']

FIXED_CMDS = ['python3', 'train.py']


def generate_combinations(fixed_model_param):
    return [FIXED_CMDS + fixed_model_param,
            FIXED_CMDS + fixed_model_param + ['--pretrained']]


if __name__ == '__main__':

    # Resnet commands list
    resnet_cmds = generate_combinations(FIXED_RESNET_PARAM)

    # SqueezeNet commands list
    squeezenet_cmds = generate_combinations(FIXED_RESNET_PARAM)

    # List of all cmds to run
    cmds = squeezenet_cmds + resnet_cmds

    # We run experiments
    start = time.time()
    for cmd in cmds:
        print("\n", cmd, "\n")
        p = sp.Popen(cmd)
        p.wait()

    print("Time Taken (minutes): ", round((time.time() - start) / 60, 2))