import random


#Subset iterators are used internally by the dataset iterator to return the set of indices for the next batch.
class SubsetIteratorInterface():

    def __init__(self, batch_size, num_examples):
        self.batch_size = batch_size
        self.num_examples = num_examples

        self.current_batch_index = 0

    def get_batch_indices(self):
        self._increment_current_batch_index()
        return self._next_batch_indices()

    def _increment_current_batch_index(self):
        if self.current_batch_index == self.batch_size:
            raise StopIteration
        else:
            self.current_batch_index += 1

    #this is implemented by the subclasses
    def _next_batch_indices(self):
        raise NotImplementedError


#returns a random set of indices without replacement
class RandomSubsetIterator(SubsetIteratorInterface):

    def _next_batch_indices(self):
        indices = xrange(self.num_examples)
        return random.sample(indices, self.batch_size)


#each batch returns a sequential set of examples
#ex:
#batch 0  = range(0,20)
#batch 1 = range(20, 40)
#...
class SequentialSubsetIterator(SubsetIteratorInterface):

    def _next_batch_indices(self):

        if not hasattr(self, 'current_index'):
            self.current_index = 0
        else:
            self.current_index += self.batch_size

        start = self.current_index
        end = self.current_index + self.batch_size

        if end > self.num_examples:
            raise StopIteration

        else:
            return range(start, end)