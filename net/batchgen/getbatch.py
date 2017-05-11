from os.path import join
import h5py as h5
from numpy.random import permutation
import torch
from torch.autograd import Variable
import pickle


"""Defines a object that returns a batch of data when called."""


class GetBatch:
    """Handles the recovery of data and the conversion to an autograd
    Variable"""

    def __init__(self, param, random=None, datatype=None, thr=10):
        self.param = param
        self.is_test = param.datatype == 'test'
        self.random = not self.is_test if random is None else random
        self.datatype = self.param.datatype if datatype is None else datatype

        # Recover batch table to be randomized
        tablename = join(self.param.datadir, 'batchtable_' + self.datatype + '.pkl')
        table = pickle.load(open(tablename, 'rb'))
        self.filenames = table['filenames']  # list of files in data directory
        self.batchnames = table['batchnames']  # list of (i, batchname) where i is an index in `self.filenames`

        # Store arguments
        self.thr = thr
        self.is_cuda = param.cuda

        self.nbbatch = len(self.batchnames)
        assert self.nbbatch > 0

        self.batch_idx = -1  # index to current batch

        # Randomize order of batchs if needed
        if self.random:
            self.batchnames = permutation(self.batchnames)

        self._nextbatch()

    def _nextbatch(self):
        """Returns False if no more files are available"""

        self.batch_idx += 1

        if self.batch_idx >= self.nbbatch:
            return False  # no batch left

        # tests if this batch contains at least `thr` events
        file_idx, self.batchname = self.batchnames[self.batch_idx]
        file_idx = int(file_idx)
        self.currfile = self.filenames[file_idx]
        with h5.File(join(self.param.datadir, self.currfile), 'r') as currfile:
            if currfile[self.batchname]['batchsize'][()] >= self.thr:
                return True
            else:
                return self._nextbatch()

    def batch(self):
        """returns batch and a boolean : False if there is no batch left
        after this one"""

        # Read data
        with h5.File(join(self.param.datadir, self.currfile), 'r') as currfile:
            batch = currfile[self.batchname]
            energy = batch['energy'][()]  # read energy
            phi = batch['phi'][()]  # read phi
            eta = batch['eta'][()]  # read eta
            label = batch['label'][()]  # read label
            weight = batch['weight'][()]  # read weight
            batchsize = batch['batchsize'][()]

        # Maker pytorch Variables
        energy = Variable(torch.Tensor(energy))
        phi = Variable(torch.Tensor(phi))
        eta = Variable(torch.Tensor(eta))
        label = Variable(torch.Tensor(label))
        weight = torch.Tensor(weight)

        # Move on GPU if cuda
        if self.is_cuda:
            energy = energy.cuda()
            phi = phi.cuda()
            eta = eta.cuda()
            label = label.cuda()
            weight = weight.cuda()

        data = {'energy': energy, 'phi': phi, 'eta': eta,
                'label': label, 'batchsize': batchsize, 'weight': weight}

        # Update index for next batch
        batch_left = self._nextbatch()

        return (data, batch_left)
