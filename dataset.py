import os
import pandas as pd
import torch 
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class PivotDataset(Dataset):
    def __init__(self, pivots, number_of_nodes):
        self.pivots = pivots
        self.number_of_nodes = number_of_nodes

    def __len__(self):
        return len(self.pivots)

    def __getitem__(self, idx):
        key_i, value_i = self.pivots[idx]
        assert(isinstance(value_i, list))

        some_hot_encoding = np.zeros(self.number_of_nodes)
        some_hot_encoding[value_i] = 1

        return (np.array(key_i), some_hot_encoding)

    def get_splits(self):
        # determine sizes
        val_size = round(0.10 * self.__len__())
        test_size = round(0.10 * self.__len__())
        train_size = self.__len__() - val_size - test_size

        # calculate the split
        return torch.utils.data.random_split(self, [train_size, val_size, test_size])

    def get_test_splits(self):
        # determine sizes
        test_size = self.__len__()

        # calculate the split
        return torch.utils.data.random_split(self, [test_size])

def prepare_data(pivots, number_of_nodes, batch_size, mode = 'train'):
    # load the dataset
    dataset = PivotDataset(pivots, number_of_nodes)

    # calculate split
    if mode == 'train':
        train_set, val_set, test_set = dataset.get_splits()

        # prepare data loaders
        train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_set, batch_size=batch_size, shuffle=True)
        test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        return train_dl, val_dl, test_dl
    else:       
        test_dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return test_dl