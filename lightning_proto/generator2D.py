import functools
import glob
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from augmentation2D import augment
from standard_utils import flatten_list


class Paths():
    def __init__(self, data_dir, k_folds, k_fold_index):
        self.data_dir = data_dir
        self.k_folds = k_folds
        self.k_fold_index = k_fold_index
        self.parent_paths = None
        self.training_paths = None
        self.validating_paths = None
        self.testing_paths = None

    def print_split(self):
        print("\n-------------------------------------")
        print("Training paths:", len(self.training_paths[0]))
        print("Validating paths:", len(self.validating_paths[0]))
        print("Testing paths:", len(self.testing_paths[0]))
        print("\n")

    def kfold(self):
        assert self.k_fold_index >= 0
        assert self.k_fold_index < self.k_folds
        assert self.k_folds < len(self.parent_paths)

        partitions = np.array_split(self.parent_paths, self.k_folds + 1)
        fold_partitions = partitions[:self.k_folds]

        test_partitions = partitions[self.k_folds:]
        valid_partition = fold_partitions[self.k_fold_index:self.k_fold_index +
                                          1]
        train_partition = []
        for x in fold_partitions:
            for y in valid_partition:
                if x is not y:
                    train_partition.append(x)

        return train_partition, valid_partition, test_partitions

    def get_pairs(self, parent_paths):
        x_paths = [glob.glob(path + "/img/*") for path in parent_paths]
        y_paths = [glob.glob(path + "/mask/*") for path in parent_paths]

        x_paths = flatten_list(x_paths)
        y_paths = flatten_list(y_paths)

        x_paths.sort()
        y_paths.sort()

        return x_paths, y_paths

    def test_pairs(self, x_paths, y_paths):
        for x_path, y_path in zip(x_paths, y_paths):

            # Check both from same parent (i.e. patient)
            parent_x = Path(x_path).parent.parent
            parent_y = Path(y_path).parent.parent
            assert parent_x == parent_y

            # Check array names are identical (i.e. slice)
            sub_x = Path(x_path).parts[-1]
            sub_y = Path(y_path).parts[-1]
            assert sub_x == sub_y

    def test_exclusion(self, list_x, list_y):
        intersection = list(set(list_x).intersection(set(list_y)))
        assert len(intersection) == 0

    def setup(self):
        parent_paths = glob.glob(self.data_dir + "/*")
        parent_paths.sort()
        assert len(parent_paths) > 1

        self.parent_paths = parent_paths

        train_partition, valid_partition, test_partition = self.kfold()

        train_parents = flatten_list(train_partition)
        valid_parents = flatten_list(valid_partition)
        test_parents = flatten_list(test_partition)

        self.test_exclusion(train_parents, valid_parents)
        self.test_exclusion(train_parents, test_parents)
        self.test_exclusion(valid_parents, test_parents)

        self.training_paths = self.get_pairs(train_parents)
        self.validating_paths = self.get_pairs(valid_parents)
        self.testing_paths = self.get_pairs(test_parents)

        self.test_pairs(*self.training_paths)
        self.test_pairs(*self.validating_paths)
        self.test_pairs(*self.testing_paths)

        self.print_split()


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data_paths, augment, dims):
        self.x_paths, self.y_paths = data_paths
        self.augment = augment
        self.dims = dims
        self.assert_shape()

    def __len__(self):
        'Denotes the total number of samples'
        assert len(self.x_paths) == len(self.y_paths)
        return len(self.x_paths)

    def __getitem__(self, index):
        'Generates one sample of data'
        x_path = self.x_paths[index]
        y_path = self.y_paths[index]

        x, y = self.read_arrays(x_path, y_path)

        if self.augment is True:
            x, y = augment(x, y)

        x = torch.Tensor(x)
        y = torch.Tensor(y)

        return x, y

    # @functools.lru_cache(maxsize=None)
    def read_arrays(self, x_path, y_path):
        x = np.load(x_path).astype("float32")
        y = np.load(y_path).astype("float32")
        return x, y

    def assert_shape(self):
        for path in self.x_paths + self.y_paths:
            assert np.load(path).shape == self.dims


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir,
                 batch_size,
                 k_folds,
                 k_fold_index,
                 dims=(1, 512, 512),
                 num_workers=12):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.k_folds = k_folds
        self.k_fold_index = k_fold_index
        self.dims = dims
        self.num_workers = num_workers
        self.training_dataset = None
        self.validating_dataset = None
        self.testing_dataset = None

    def prepare_data(self):
        # prepare_data (how to download(), tokenize, etc…)
        # prepare_data is called from a single GPU
        return None

    def setup(self):
        # Setup is called from multiple GPUs
        data_paths = Paths(self.data_dir, self.k_folds, self.k_fold_index)
        data_paths.setup()

        training_paths = data_paths.training_paths
        validating_paths = data_paths.validating_paths
        testing_paths = data_paths.testing_paths

        self.training_dataset = Dataset(training_paths,
                                        augment=True,
                                        dims=self.dims)
        self.validating_dataset = Dataset(validating_paths,
                                          augment=False,
                                          dims=self.dims)
        self.testing_dataset = Dataset(testing_paths,
                                       augment=False,
                                       dims=self.dims)

    def train_dataloader(self):
        return DataLoader(self.training_dataset,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validating_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.testing_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)


if __name__ == "__main__":
    batch_size = 10
    data_dir = "/home/matthew/github/vacunet/tests/canine_imaging_dataset/"
    k_folds = 4
    k_fold_index = 0

    dm = DataModule(data_dir, batch_size, k_folds, k_fold_index)
    dm.prepare_data()
    dm.setup()