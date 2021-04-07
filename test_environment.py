import sys
from src.data.DatasetManager import DatasetManager
from src.data.DataLoaderManager import DataLoaderManager
from src.data.constants import *
from src.models.constants import *
from src.models.TrainValidTestManager import TrainValidTestManager
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
    dataset_manager = DatasetManager(CIFAR10, valid_size=0.10)
    active_learner = ActiveLearner(RESNET34, dataset_manager, n_start=20, n_new=50, epochs=10,
                                   accuracy_goal=0.40, improvement_threshold=0.005, query_strategy='least_confident',
                                   saving_file_name="test", batch_size=50, lr=0.001, pretrained=False)
    active_learner()

    # show charts
