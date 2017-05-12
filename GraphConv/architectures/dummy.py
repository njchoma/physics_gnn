import torch
import torch.nn as nn
import torch.nn.functional as F
# EDIT : choice of adjacency type should be done in metaparameter file
from GraphConv.kernel.learnkernel import DirectionnalGaussian as Adjacency
from GraphConv.graphlayer.graphconv import GraphConv as gconv
from GraphConv.functional.batchnorm import batchnorm


# ----------------------------------------------------------------------
# ------------------- Very simple 'graph linear' -----------------------
# ----------------------------------------------------------------------

class Dummy(nn.Module):

    def __init__(self, logistic_bias=0):
        super(Dummy, self).__init__()
        self.adjacency = Adjacency(thr=0.011, std=1)
        self.Linear = nn.Conv1d(2, 1, 1)
        self.logistic_bias = logistic_bias

    def forward(self, e, phi, eta):
        e = batchnorm(e, axis=1).unsqueeze(2)

        # applying the GNN
        adj = self.adjacency(phi, eta)

        f = torch.bmm(adj, e)
        e = torch.stack((e, f), dim=1).squeeze(3)
        e = self.Linear(e)
        e = e.sum(2).squeeze(2).squeeze(1)
        e = e + self.logistic_bias

        # sigmoid for logistic regression
        e = F.sigmoid(e)
        return e


# ----------------------------------------------------------------------
# ------------------- Very simple 'graph linear' -----------------------
# ----------------------------------------------------------------------

class DummyGconv(nn.Module):

    def __init__(self, logistic_bias=0):
        super(DummyGconv, self).__init__()
        self.adjacency = Adjacency()
        self.gconv = gconv(1, 1, 1)
        self.logistic_bias = logistic_bias

    def forward(self, e, phi, eta):
        e = batchnorm(e, axis=1)
        e = e.unsqueeze(1)

        # applying the GNN
        adj = self.adjacency(phi, eta)

        e = self.gconv(adj, e)
        e = e.mean(2).squeeze(2).squeeze(1)
        e = e + self.logistic_bias

        # sigmoid for logistic regression
        e = F.sigmoid(e)
        return e
