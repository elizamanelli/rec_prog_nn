import logging
import numpy as np


class data_iterator(object):
    '''
    This class creates an iterator from the data given or returns an iterator specified for a certain dataset.
    '''

    def __init__(self):
        """
        Initializing the data iterator.
        """
        return

    def iter_data_epoch(self, data, target, n_batches, **kwargs):
        '''
        :param data: list input data
        :param target: list target data
        :param n_batches: Int given the number of samples per batch
        :param kwargs:
        :return: An iterator which always yields a full epoch of data in batches

        Iterating over the given data & target in NumBatches. This iterator returns all the feed in data at once but
        split in a list of batches.
        '''
        logging.info('Creating an epoch iterator')
        shuffle = kwargs.get('shuffle', True)
        data_size = data.shape[0]
        while True:
            indices_new_order = np.arange(data_size)
            if shuffle:
                np.random.shuffle(indices_new_order)
            # run for 1 epoch
            new_epoch = []
            for i in range(0, data_size, n_batches):
                batch_indices = indices_new_order[i:i+n_batches]
                new_epoch.append((data[batch_indices], target[batch_indices]))
            yield new_epoch

    def iter_data_batch(self, data, target, n_batches, **kwargs):
        '''
        :param data: list training data
        :param target: list target data
        :param n_batches: int number of samples per batch
        :param kwargs:
        :return: An iterator which always yields a batch of data.

        Iterating over the given data & target in NumBatches. This iterator returns batches of the fed in data but in a
        list format so it is usable for applications programmed for the epoch iterator.
        '''
        logging.info('Creating a batch iterator.')
        shuffle = kwargs.get('shuffle', True)
        data_size = data.shape[0]
        while True:
            indices_new_order = np.arange(data_size)
            if shuffle:
                np.random.shuffle(indices_new_order)
            # run for 1 epoch
            new_epoch = []
            for i in range(0, data_size, n_batches-1):
                batch_indices = indices_new_order[i:i+n_batches]
                yield [(data[batch_indices], target[batch_indices])]


class sequence_iterator(object):
    '''
    This class creates an iterator from the data given or returns an iterator specified for a certain dataset.
    Adapted for time continous data.
    '''

    def __init__(self):
        """
        Initializing the data iterator.
        """
        return

    def iter_data_epoch(self, data, target, n_batches, length, long=False, **kwargs):
        '''
        :param data: list input data
        :param target: list target data
        :param n_batches: Int given the number of samples per batch
        :param length: list of length of data
        :param long: Bool inidicating if a long version of the data shall be yielded
        :param kwargs:
        :return: An iterator which always yields a full epoch of data in batches

        Iterating over the given data & target in NumBatches. This iterator returns all the feed in data at once but
        split in a list of batches. Padding data to the max length of the batch with zeros and returning
        the length list of the batch as well.
        '''
        logging.info('Creating an epoch iterator')
        shuffle = kwargs.get('shuffle', True)
        data_size = data.shape[0]
        while True:
            indices_new_order = np.arange(data_size)
            if shuffle:
                np.random.shuffle(indices_new_order)
            # run for 1 epoch
            new_epoch = []
            for i in range(0, data_size, n_batches):
                batch_indices = indices_new_order[i:i+n_batches]
                batchlen = np.asarray(length)[batch_indices]
                if long:
                    maxlen = data.shape[1]+1
                else:
                    maxlen = np.max(batchlen)
                batchdata = data[batch_indices][:, :maxlen, :]
                if target.ndim > 1:
                    batchtarget = target[batch_indices, :maxlen]
                else:
                    batchtarget = np.repeat(target[batch_indices],  maxlen)
                    batchtarget = np.reshape(batchtarget, (len(batch_indices), maxlen))

                new_epoch.append((batchdata, batchtarget, batchlen))
            yield new_epoch

    def iter_data_batch(self, data, target, n_batches, **kwargs):
        '''
        :param data: list training data
        :param target: list target data
        :param n_batches: int number of samples per batch
        :param kwargs:
        :return: An iterator which always yields a batch of data.

        Iterating over the given data & target in NumBatches. This iterator returns batches of the fed in data but in a
        list format so it is usable for applications programmed for the epoch iterator.
        '''
        logging.info('Creating a batch iterator.')
        shuffle = kwargs.get('shuffle', True)
        data_size = data.shape[0]
        while True:
            indices_new_order = np.arange(data_size)
            if shuffle:
                np.random.shuffle(indices_new_order)
            for i in range(0, data_size, n_batches-1):
                batch_indices = indices_new_order[i:i+n_batches]
                yield [(data[batch_indices], target[batch_indices])]

