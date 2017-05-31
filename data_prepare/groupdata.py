import os
import h5py as h5
import numpy as np
from utils import print_


def batchnorm(x, axis=0):
    """renormalize independantly each column of input.
    input : x variable of size batch * 1 * n * d_out
    output : y = (x - E(x)) / sqrt(var(x)) of same size
    """

    # Compute empirical expectancy and substract
    ex = x.mean(axis)  # axis representing the event length
    x = x - ex

    # Compute empirical standard deviation and divide
    std = np.sqrt((x * x).mean(axis))
    x = x / std

    return x


class BatchGroup:
    """Creates and fills each group (representing a batch) and dataset"""

    def __init__(self, batchsize, weight_factors,
                 rawdatadir, datafile,
                 namenum, length, batch_idx, stdout):
        self.namenum = namenum
        self.length = length
        self.batchsizes = self._balance_batchs(batchsize)
        self.rawdatadir = rawdatadir
        self.datafile = datafile
        self.weight_factors = weight_factors  # mapping from event type to weight factor
        self.batch_idx = batch_idx  # index to next non existing batch
        self.stdout = stdout

        self.count_events = 0

    def randomize_events(self):
        self.namenum = np.random.permutation(self.namenum)

    def _balance_batchs(self, batchsize):
        size = len(self.namenum)
        if size < batchsize:
            return [size]
        nb_batch = int(float(size) / float(batchsize) + 0.5)  # rounding (positive)
        leftovers = size - nb_batch * batchsize
        res = []
        for i in range(nb_batch):
            extra = leftovers // nb_batch  # extra event for this batch
            leftovers -= extra
            nb_batch -= 1
            res.append(batchsize + extra)

        return res

    def next_batch(self, curr_batchsize):
        """initiates new batch"""

        # initialise batch variables
        self.event_idx = 0
        self.group_name = 'batch{}'.format(self.batch_idx)
        if self.batch_idx % 500 == 0:
            print_(self.group_name, self.stdout)

        # create group structure
        with h5.File(self.datafile, 'a') as fileout:
            group = fileout.create_group(self.group_name)
            group.create_dataset('E', shape=(curr_batchsize, self.length))
            group.create_dataset('EM', shape=(curr_batchsize, self.length))
            group.create_dataset('eta', shape=(curr_batchsize, self.length))
            group.create_dataset('phi', shape=(curr_batchsize, self.length))
            group.create_dataset('label', shape=(curr_batchsize,))
            group.create_dataset('weight', shape=(curr_batchsize,))

            # update counters
            self.batch_idx += 1
            self.count_events += curr_batchsize

            # load events
            for _ in range(curr_batchsize):
                self.next_event(fileout)

    def next_event(self, fileout):
        """reads and organize from next event"""

        filename, event_num = self.namenum.pop()
        label = 'GG' in filename

        # read from raw data file
        with h5.File(os.path.join(self.rawdatadir, filename), 'r') as filein:
            E = filein[event_num]['clusE'].value
            EM = filein[event_num]['clusEM'].value
            eta = filein[event_num]['clusEta'].value
            phi = filein[event_num]['clusPhi'].value
            weight = filein[event_num]['weight'].value

        # modifications on data
        E = batchnorm(E)
        EM = batchnorm(EM)
        weight = weight / self.weight_factors[filename]

        # write in new data file
        fileout[self.group_name + '/EM'][self.event_idx, :] = EM
        fileout[self.group_name + '/E'][self.event_idx, :] = E
        fileout[self.group_name + '/eta'][self.event_idx, :] = eta
        fileout[self.group_name + '/phi'][self.event_idx, :] = phi
        fileout[self.group_name + '/weight'][self.event_idx] = weight
        fileout[self.group_name + '/label'][self.event_idx] = label

        # update event index
        self.event_idx += 1

    def iter_data(self):
        """calls `self.next_batch` and `self.next_event` until all data has been
        processed"""

        for batchsize in self.batchsizes:
            self.next_batch(batchsize)

        return self.batch_idx, self.count_events


def group_batchs(len2namenum, batchsize, weight_factors,
                 rawdatadir, datadir, stdout=None):
    """reads from len2namenum dictionary and randomly groups events of
    same length into batchs"""

    datafile = os.path.join(datadir, 'data.h5')  # must be cleared
    with h5.File(datafile, 'w') as fileout:
        pass

    curr_batch_idx, total_count_events = 0, 0
    print_('organizing data from `{}` to `{}`.'.format(rawdatadir, datafile), stdout)

    for length in len2namenum.keys():
        organizer = BatchGroup(
            batchsize, weight_factors,
            rawdatadir, datafile,
            len2namenum[length], length, curr_batch_idx, stdout
        )
        curr_batch_idx, count_events = organizer.iter_data()
        total_count_events += count_events

    with h5.File(datafile, 'a') as fileout:
        fileout.create_dataset('nb_batch', data=curr_batch_idx)
        fileout.create_dataset('nb_events', data=total_count_events)
