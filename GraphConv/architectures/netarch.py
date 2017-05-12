import torch
import torch.nn as nn
import torch.nn.functional as F
# EDIT : choice of adjacency type should be done in metaparameter file
from GraphConv.kernel.learnkernel import Gaussian as Adjacency
from GraphConv.graphlayer.graphconv import GraphConv as gconv
from GraphConv.functional.batchnorm import batchnorm


# ----------------------------------------------------------------------
# ------------------- Basic GNN with list input -----------------------
# ----------------------------------------------------------------------

class GNN(nn.Module):
    """implements a Graph Neural Networks to analyse the data from the Large Hadron
    Collider. The graph is built by defining weight depending on the euclidean
    distance between two energy bursts."""

    def __init__(self, dim, deg, logistic_bias=0):
        super(GNN, self).__init__()
        dim.insert(0, 1)
        assert(len(dim) == len(deg))
        self.nblayer = len(dim)
        self.dim = dim
        self.deg = deg
        self.dim.append(1)  # output dimension
        self.gconv = nn.ModuleList([gconv(dim[i], dim[i + 1], deg[i]) for i in range(self.nblayer)])
        self.adjacency = Adjacency()
        self.logistic_bias = logistic_bias

    def forward(self, e, phi, eta):
        e = batchnorm(e, axis=1)
        e = e.unsqueeze(1)

        # applying the GNN
        adj = self.adjacency(phi, eta)

        for i in range(self.nblayer - 1):
            e = self.gconv[i](adj, e)
            e = F.relu(e)
        e = self.gconv[-1](adj, e)

        # compute mean : structural information is already exploited
        e = e.mean(2).squeeze(2).squeeze(1)
        e = e + self.logistic_bias

        # sigmoid for logistic regression
        e = F.sigmoid(e)

        return e


# ----------------------------------------------------------------------
# ------------------- Basic GNN with list input -----------------------
# ---- The position parameters are used as channels for convolution ----
# ----------------------------------------------------------------------

class GNNSpatial(nn.Module):
    """implements a Graph Neural Networks to analyse the data from the Large Hadron
    Collider. The graph is built by defining weight depending on the euclidean
    distance between two energy bursts."""

    def __init__(self, dim, deg, logistic_bias=0):
        super(GNNSpatial, self).__init__()
        dim.insert(0, 3)
        assert(len(dim) == len(deg))
        self.nblayer = len(dim)
        self.dim = dim
        self.deg = deg
        self.dim.append(1)  # output dimension
        self.gconv = nn.ModuleList([gconv(dim[i], dim[i + 1], deg[i]) for i in range(self.nblayer)])
        self.adjacency = Adjacency()
        self.logistic_bias = logistic_bias

    def forward(self, e, phi, eta):
        e = batchnorm(e, axis=1)

        # applying the GNN
        adj = self.adjacency(phi, eta)

        # input is a concatenation of e and eta
        eta = batchnorm(eta, axis=1)
        phi = batchnorm(phi, axis=1)
        e = torch.stack((e, eta, phi), dim=1)

        for i in range(self.nblayer - 1):
            e = self.gconv[i](adj, e)
            e = F.relu(e)
        e = self.gconv[-1](adj, e)

        # compute mean : structural information is already exploited
        e = e.mean(2).squeeze(2).squeeze(1)
        e = e + self.logistic_bias

        # sigmoid for logistic regression
        e = F.sigmoid(e)

        return e
