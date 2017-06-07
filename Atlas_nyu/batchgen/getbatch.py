from os.path import join
import pickle
from numpy.random import permutation
from torch import Tensor
from torch.autograd import Variable
import pickle

"""Defines a object that returns a batch of data when called."""


class GetBatch:
    """Handles the recovery of data and the conversion to a Pytorch Variable"""

    def __init__(self, param, datatype=None):
        self.is_test = param.datatype == 'test'
        self.datatype = param.datatype if datatype is None else datatype
        self.is_cuda = param.cuda

        # Load data
        filename = join(param.datadir, 'antikt-kt-' + self.datatype + '-gcnn.pickle')
        self.data, self.labels = pickle.load(open(filename, 'rb'), encoding='latin1')

        self.batch_idx = len(self.labels)  # index to current batch

    def batch(self):
        """returns batch and a boolean : False if there is no batch left
        after this one"""

        # update index, check if last batch
        self.batch_idx -= 1
        batches_left = self.batch_idx > 0

        # select data
        label = self.labels[self.batch_idx]
        data = self.data[self.batch_idx]
        data_dict = {
            feature: Variable(Tensor(data[:, k])).view(1, -1)  # batchsize = 1
            for k, feature in enumerate(['p', 'eta', 'phi', 'E', 'pt', 'theta', 'px', 'py', 'pz'])
        }
        data_dict['label'] = Variable(Tensor([label]))
        data_dict['weight'] = Tensor([1.])  # EDIT : replace with real weight

        data_dict['E'] = data_dict['E'] / 100

        # CUDA
        if self.is_cuda:
            for feature in data_dict.keys():
                data_dict[feature] = data_dict[feature].cuda()



        return (data_dict, batches_left)
