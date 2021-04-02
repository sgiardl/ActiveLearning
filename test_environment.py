import sys
from src.data.DatasetManager import DatasetManager
from src.data.constants import *
from src.models.constants import *
from src.models.TrainValidTestManager import TrainValidTestManager

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

<<<<<<< HEAD
    dataset_manager = DatasetManager(CIFAR10, valid_size=0.1, batch_size=100, shuffle=False, num_workers=8)
    train_valid_test_manager = TrainValidTestManager(dataset_manager, file_name='model',
                                                     model_name=SQUEEZE_NET_1_1, learning_rate=0.0001,
                                                     pretrained=True)
    train_valid_test_manager.train_model(epochs=20)
    train_valid_test_manager.test_model()
=======
    dataset = get_dataset(name=CIFAR10, root=f"{os.getcwd()}/data/raw", transforms=get_transforms())
    expert = Expert(dataset, 2, 'least_confident')
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=1, sampler=expert.sampler)
    train_model(epochs=100, data_loader=data_loader, file_name='model',
                model_name=SQUEEZE_NET_1_1, pretrained=True, learning_rate=0.0001)
>>>>>>> query_strategies
