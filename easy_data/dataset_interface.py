
#import collections.Iterator
import numpy as np
from subset_iterators import *
import sys

#this is the interface that all datasets should fulfill
class DatasetInterface():

    #generate or load the given example
    def get_example(self, index):
        raise NotImplementedError

    #return the number of examples in the dataset
    def get_num_examples(self):
        raise NotImplementedError

    def iterator(self,
                 batch_size,
                 num_batches,
                 subset_iterator="RandomSubsetIterator"):

            return DatasetIterator(self,
                                   batch_size=batch_size,
                                   num_batches=num_batches,
                                   subset_iterator=subset_iterator)


class DatasetIterator():

    def __init__(self, dataset, batch_size, num_batches, subset_iterator="RandomSubsetIterator"):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches

        subset_iterator_class = getattr(sys.modules['easy_data.subset_iterators'], subset_iterator)
        self.subset_iterator = subset_iterator_class(batch_size, self.dataset.get_num_examples())

    def next(self):

        batch_x = []
        batch_y = []

        batch_indices = self.subset_iterator.get_batch_indices()

        for batch_index in batch_indices:

            x, y = self.dataset.get_example(batch_index)

            batch_x.append(x)
            batch_y.append(y)

        return np.array(batch_x), np.array(batch_y)