"""
File:
    test_environment.py

Authors:
    - Abir Riahi
    - Nicolas Raymond
    - Simon Giard-Leroux

Description:
    Test environment to test the active learning loop.
"""

import sys
from src.data.constants import *
from src.models.constants import *
from src.models.ActiveLearning import ActiveLearner

REQUIRED_PYTHON = "python3"


def main():
    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError("Unrecognized python interpreter: {}".format(
            REQUIRED_PYTHON))

    if system_major != required_major:
        raise TypeError(
            "This project requires Python {}. Found: Python {}".format(
                required_major, sys.version))
    else:
        print(">>> Development environment passes all tests!")


if __name__ == '__main__':
    # Development tests
    main()

    # Active learning test
    active_learner = ActiveLearner(SQUEEZE_NET_1_1, CIFAR10, n_start=100, n_new=100, epochs=50,
                                   query_strategy='least_confident', experiment_name="test",
                                   batch_size=50, lr=0.0001, weight_decay=0,
                                   pretrained=False, data_aug=False)
    active_learner(n_rounds=3)
