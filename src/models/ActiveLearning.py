from src.models.Expert import Expert
from src.models.TrainValidTestManager import TrainValidTestManager
from src.data.DatasetManager import DatasetManager
from src.data.DataLoaderManager import DataLoaderManager
from torch.utils.data import Subset


class ActiveLearner:
    def __init__(self, model: str, dataset_manager: DatasetManager, n_start: int, n_new: int, epochs: int,
                 accuracy_goal: float, improvement_threshold: float, query_strategy: str, saving_file_name: str,
                 batch_size: int = 50, shuffle: bool = False, num_workers: int = 1,
                 lr: float = 0.005, pretrained: bool = False):

        """
        Object in charge of pool-based active learning

        :param model: Name of the model to train ("ResNet34" or "SqueezeNet11")
        :param dataset_manager: Object containing the train, valid and test datasets
        :param n_start: The number of items that must be randomly labeled in each class by the Expert
        :param n_new: The number of new items that must be labeled within each active learning loop
        :param epochs: Number of training epochs in each active learning loop
        :param accuracy_goal: Accuracy that we want to achieve before we stop active learning
        :param improvement_threshold: Accuracy improvement require to continue active learning loops
        :param query_strategy: Query strategy of the expert
        :param saving_file_name: Name of the file in which a checkpoint of the model will be saved after each training
        :param batch_size: Batch size of dataloaders storing train, valid and test set
        :param shuffle: Bool indicating if data are shuffled within loaders
        :param num_workers: Number of workers used by the dataloaders
        :param lr: Learning rate of the model during training
        :param pretrained: Bool indicating if the model used must be pretrained on ImageNet
        """
        assert improvement_threshold > 0, "Threshold must be greater than 0"

        # We first initialize an expert
        self.expert = Expert(dataset_manager.dataset_train, n_start, query_strategy)

        # We initialize DataLoaderManager and save DatasetManager
        self.loader_manager = DataLoaderManager(dataset_manager, self.expert, batch_size, shuffle, num_workers)
        self.dataset_manager = dataset_manager

        # We initialize TrainValidTestManager
        self.training_manager = TrainValidTestManager(self.loader_manager, saving_file_name, model, lr, pretrained)

        # We initialize important attributes to lead active learning
        self.n_new = n_new
        self.epochs = epochs
        self.goal = accuracy_goal
        self.threshold = improvement_threshold

        # We initialize attributes to keep track of active learning loops
        self.loop_progress = []
        self.last_accuracy = 0

    def update_labeled_items(self):
        """
        Updates DataLoaderManager and TrainValidTestManager
        """

        # Update of DataLoaderManager
        self.loader_manager.update(self.expert)

        # Update TrainValidTestManager
        self.training_manager.update_train_loader(self.loader_manager)

    def __call__(self):

        print(f"Unlabled items : {len(self.loader_manager.unlabeled_idx)}")
        print("Active Learning Started")
        i = 0
        while True:

            # We update a string that will be used for multiple prints
            loop_reference = f"Active Loop #{i}"

            # We train the model on labeled image in the training set
            print(f"{loop_reference} - Training...")
            self.training_manager.train_model(self.epochs)

            # We evaluate our model on the current test set
            print(f"\n{loop_reference} - Test...")
            accuracy = self.training_manager.test_model()
            self.loop_progress.append(accuracy)
            print(f"{loop_reference} - Test Accuracy {round(accuracy,4)}")

            # We look if stopping conditions are reached
            accuracy_diff = accuracy - self.last_accuracy
            print(f"{loop_reference} - Accuracy Difference {round(accuracy_diff,4)}")
            if accuracy_diff < self.threshold or accuracy >= self.goal:
                print("Active learning stop - Stopping criteria reached")
                self.expert.show_labels_history(show=True)
                break

            # We update last accuracy attribute
            self.last_accuracy = accuracy

            # We make prediction on the unlabeled data
            print(f"{loop_reference} - Unlabeled Evaluation")
            unlabeled_dataset = Subset(self.dataset_manager.dataset_train.dataset, self.loader_manager.unlabeled_idx)
            unlabeled_softmax = self.training_manager.evaluate_unlabeled(unlabeled_dataset)

            # We request labels to our expert
            print(f"{loop_reference} - Labels Request")
            self.expert.add_labels(self.loader_manager.unlabeled_idx, unlabeled_softmax,
                                   self.n_new, self.dataset_manager.dataset_train)

            # We update internal attributes
            self.update_labeled_items()

            print(f"{loop_reference} - Unlabeled items : {len(self.loader_manager.unlabeled_idx)}")


