from src.models.Expert import Expert
from src.data.DatasetManager import DatasetManager
from torch.utils.data import DataLoader


class DataLoaderManager:
    def __init__(self,
                 dataset_manager: DatasetManager,
                 expert: Expert,
                 batch_size: int,
                 shuffle: bool = False,
                 num_workers: int = 1) -> None:
        """
        Data Loader Manager class, handles the creation of the training, validation and testing data loaders.

        :param dataset_manager: DatasetManager, dataset manager containing the
                                training, validation and testing datasets
        :param expert: expert
        :param batch_size: int, batch size for forward pass
        :param shuffle: bool, to shuffle the data loaders
        :param num_workers: int, number of multiprocessing workers,
                            should be smaller or equal to the number of cpu threads
        """
        # We save important attributes for further dataloader updates
        self.dataset_manager = dataset_manager
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.dataset_train = dataset_manager.dataset_train
        self.train_length = len(self.dataset_train)

        # We keep track of unlabeled index
        self.unlabeled_idx = []
        self.__update_unlabeled_idx(expert)

        # We initialize dataloaders
        self.data_loader_train = DataLoader(self.dataset_train, batch_size=batch_size, shuffle=shuffle,
                                            num_workers=num_workers, sampler=expert.sampler)

        self.data_loader_valid_1 = DataLoader(dataset_manager.dataset_valid_1, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_workers)

        self.data_loader_valid_2 = DataLoader(dataset_manager.dataset_valid_2, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_workers)

        self.data_loader_test = DataLoader(dataset_manager.dataset_test, batch_size=batch_size,
                                           shuffle=shuffle, num_workers=num_workers)

    def __update_unlabeled_idx(self, expert: Expert) -> None:
        """
        Updates the unlabeled idx in the training dataset
        :param expert: Expert
        """
        self.unlabeled_idx = [i for i in range(self.train_length) if i not in expert.labeled_idx]

    def update(self, expert: Expert) -> None:

        # We update unlabeled indices
        self.__update_unlabeled_idx(expert)

        # We update the train loader
        self.data_loader_train = DataLoader(self.dataset_train, batch_size=self.batch_size,
                                            shuffle=self.shuffle, num_workers=self.num_workers,
                                            sampler=expert.sampler)
