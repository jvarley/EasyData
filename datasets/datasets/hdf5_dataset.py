from datasets.dataset_interface import DatasetInterface
import h5py


class HDF5Dataset(DatasetInterface):

    def __init__(self,
                 hdf5_filepath='data_file.h5',
                 x_key='x',
                 y_key='y'):

        self.dset = h5py.File(hdf5_filepath, 'r')

        self.num_examples = self.dset['x'].shape[0]

        self.x_key = x_key
        self.y_key = y_key

    def get_num_examples(self):
        return self.num_examples

    def get_example(self, index):
        return self.dset[self.x_key][index], self.dset[self.y_key][index]
