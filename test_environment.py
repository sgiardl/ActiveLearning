import sys
from src.data.DatasetManager import DatasetManager
from src.data.DataLoaderManager import DataLoaderManager
from src.data.constants import *
from src.models.constants import *
from src.models.TrainValidTestManager import TrainValidTestManager
from src.visualization.VisualizationManager import VisualizationManager

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
    main()

    dataset_manager = DatasetManager(CIFAR10, valid_size_1=0.1, valid_size_2=0.05)
    data_loader_manager = DataLoaderManager(dataset_manager, query_strategy='least_confident',
                                            batch_size=100, shuffle=False, num_workers=8)
    visualization_manager = VisualizationManager()
    query_strategies = ['least_confident', 'margin_sampling']

    for i in range(len(query_strategies)):
        data_loader_manager(query_strategy=query_strategies[i])
        train_valid_test_manager = TrainValidTestManager(data_loader_manager, file_name='model',
                                                         model_name=SQUEEZE_NET_1_1, learning_rate=0.0001,
                                                         pretrained=True)
        train_valid_test_manager.train_model(epochs=20)
        train_valid_test_manager.test_model()

        visualization_manager.show_loss_acc_chart(train_valid_test_manager)

        data_loader_manager.expert.show_labels_history()

    # show charts
