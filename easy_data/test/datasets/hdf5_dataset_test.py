
import unittest
import os
import tempfile
import h5py

from easy_data.datasets.hdf5_dataset import HDF5Dataset


class TestDataset(unittest.TestCase):

    def setUp(self):

        self.test_dir = tempfile.mkdtemp()
        self.hdf5_filename = self.test_dir + "/hdf5_reconstruction_dataset_test.h5"

        dset = h5py.File(self.hdf5_filename)
        dset.create_dataset('x', (100, 28, 28, 1))
        dset.create_dataset('y', (100, 10))
        dset.close()

        self.dataset = HDF5Dataset(self.hdf5_filename, x_key='x', y_key='y')

    def test_iterator(self):

        num_batches = 3
        batch_size = 5

        iterator = self.dataset.iterator(batch_size=batch_size,
                                         num_batches=num_batches,
                                         subset_iterator_class_name="RandomSubsetIterator")

        batch_x, batch_y = iterator.next()

        self.assertEqual(batch_x.shape, (5, 28, 28, 1))
        self.assertEqual(batch_y.shape, (5, 10))

    def tearDown(self):
        #delete temp file
        os.remove(self.hdf5_filename)
        #delete temp dir
        os.removedirs(self.test_dir)


if __name__ == '__main__':
    unittest.main()
