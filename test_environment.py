import sys
from src.data.DatasetManager import DatasetManager
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
    dataset_manager = DatasetManager(CIFAR10, valid_size_1=0.15, valid_size_2=0.15)
    active_learner = ActiveLearner(SQUEEZE_NET_1_1, dataset_manager, n_start=500, n_new=100, epochs=10,
                                   query_strategy='least_confident', experiment_name="test",
                                   batch_size=50, lr=0.0001, weight_decay=0.01, pretrained=False)
    active_learner(n_rounds=10)

