from os.path import join
import pickle
from utils.files import makefile_if_not_there, print_

from Atlas_nyu.batchgen.getbatch import GetBatch
from Atlas_nyu.statistics import Statistics
from Atlas_nyu.batchgen.weightfunc import getweightfunc
from Atlas_nyu.choose_model import init_model_type

from graphic.roccurve import ROCCurve


class Model:
    def __init__(self, param, stdout=None):
        # Create net
        self.param = param
        self.net, parameters_used = init_model_type(param)
        self.lr = param.lr

        # Create statistics object
        nbnetparameters = sum(param.numel() for param in self.net.parameters())
        self.statistics = Statistics(param, nbnetparameters, parameters_used, stdout)

        # initialize is_cuda
        self.is_cuda = False

        # weights
        if self.param.weightfunc is not None:
            self.weightfunc = getweightfunc(param)
        else:
            self.weightfunc = None

        # counters
        self.nb_batch_seen = None

    def newparameters(self, param, stdout=None):
        self.param = param
        nbnetparameters = sum(param.numel() for param in self.net.parameters())
        self.statistics.newparameters(param, nbnetparameters, stdout)

    def cuda(self):
        self.net.cuda()
        self.is_cuda = True

    def cpu(self):
        self.net.cpu()
        self.is_cuda = False

    def do_one_batch(self, batchgen, mode, criterion=None, optimizer=None, roccurve=None):
        # get data
        data, is_not_last_batch = batchgen.batch()
        energy = data['energy']
        phi = data['phi']
        eta = data['eta']
        label = data['label']
        weight = data['weight']

        # zero parameters gradients
        if optimizer is not None:
            optimizer.zero_grad()

        # forward
        output = self.net(energy, phi, eta)
        if criterion is not None:
            if self.weightfunc is None:
                loss = criterion(output, label, weight=weight)
            else:
                loss = criterion(output, label, weight=self.weightfunc(self.param, weight, label))

        # backward + optimizer + update learning rate
        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.update(loss.data[0])

        # update stats
        # isotropic kernel : kernel_std = self.net.adjacency.std.abs().data[0]
        kernel_std = (self.net.adjacency.alpha / self.net.adjacency.beta).abs().data[0]
        if criterion is not None:
            self.statistics.update(
                mode, output.data, label.data, loss.data.mean(), kernel_std
            )
        else:
            self.statistics.update(
                'test', output.data, label.data, float('nan'), kernel_std
            )
        self.nb_batch_seen += 1

        # update ROC curve
        if roccurve is not None:
            roccurve.update(output, label, weight)

        return is_not_last_batch

    def train_epoch(self, optimizer, criterion):
        """One epoch of training"""
        # Generator
        batchgen = GetBatch(self.param, random=True, datatype='train')

        # initiate batch counter
        self.nb_batch_seen = 0

        # loop on batchs
        is_not_last_batch = True
        not_enough_batch = self.nb_batch_seen < self.param.nb_batch
        while is_not_last_batch and not_enough_batch:
            is_not_last_batch = self.do_one_batch(batchgen, 'train', criterion, optimizer)
            not_enough_batch = self.nb_batch_seen < self.param.nb_batch

            if self.nb_batch_seen % self.param.nb_save == 0:
                self.save_model()
        self.save_model()

    def plot_epoch(self, mode, nb_batch):
        """Tests model on `mode` dataset, plots ROC curve"""

        # Generator
        batchgen = GetBatch(self.param, random=True, datatype=mode)

        # initiate batch counter
        self.nb_batch_seen = 0

        # initialize ROC curve
        roccurve = ROCCurve()

        # loop on batchs
        is_not_last_batch = True
        not_enough_batch = self.nb_batch_seen < nb_batch
        while is_not_last_batch and not_enough_batch:
            is_not_last_batch = self.do_one_batch(batchgen, mode, roccurve=roccurve)
            not_enough_batch = self.nb_batch_seen < nb_batch

        # plot ROC curve in graph directory
        roccurve.plot_roc_curve(mode, self.param.graphdir, zooms=self.param.zoom)

        return self.nb_batch_seen

    def do_one_epoch(self, optimizer, criterion):

        # train if needed
        if self.param.mode == 'train':
            self.train_epoch(optimizer, criterion)

        # plot ROC Curves
        nb_plot = self.param.nb_batch
        nb_plot_test = self.plot_epoch('test', nb_plot)
        self.statistics.flush()
        self.plot_epoch('train', nb_plot_test)
        self.statistics.flush()

    def save_model(self):
        makefile_if_not_there(self.param.netdir, 'model')
        filepath = join(self.param.netdir, 'model')
        with open(filepath, 'wb') as fileout:
            pickle.Pickler(fileout).dump(self)
        print_('-' * 5 + 'Saved in `{}`'.format(filepath) + '-' * 5 + '\n', stdout=self.statistics.stdout)

    def printdescr(self, stdout=None):
        self.statistics.printdescr()
