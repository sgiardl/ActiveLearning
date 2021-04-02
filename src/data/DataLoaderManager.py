from src.models.Expert import Expert
from src.data.DatasetManager import DatasetManager
from torch.utils.data import DataLoader


class DataLoaderManager:
    def __init__(self,
                 dataset_manager: DatasetManager,
                 query_strategy: str,
                 batch_size: int,
                 shuffle: bool = False,
                 num_workers: int = 1):
        """

        :param dataset_manager:
        :param query_strategy:
        :param batch_size: int, batch size for forward pass
        :param shuffle: bool, to shuffle the data loaders
        :param num_workers: int, number of multiprocessing workers,
                            should be smaller or equal to the number of cpu threads
        """
        self.dataset_manager = dataset_manager
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.dataset_train = dataset_manager.dataset_train
        self.dataset_valid = dataset_manager.dataset_valid
        self.dataset_test = dataset_manager.dataset_test

        self.expert = Expert(self.dataset_train, 2, query_strategy)

        self.data_loader_train = DataLoader(self.dataset_train, batch_size=batch_size, shuffle=shuffle,
                                            num_workers=num_workers, sampler=self.expert.sampler)

        self.data_loader_valid = DataLoader(self.dataset_valid, batch_size=batch_size, shuffle=shuffle,
                                            num_workers=num_workers)

        self.data_loader_test = DataLoader(self.dataset_test, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers)

    def __call__(self, *args, **kwargs):
        self.expert = Expert(self.dataset_train, 2, kwargs.get('query_strategy'))

        self.data_loader_train = DataLoader(self.dataset_train, batch_size=self.batch_size,
                                            shuffle=self.shuffle, num_workers=self.num_workers,
                                            sampler=self.expert.sampler)
