import torch
import torch.nn as nn
import torch.nn.functional as F
from GraphConv.kernel.learnkernel import DirectionnalGaussian as Adjacency  # EDIT : choice of adjacency type should be done in metparameter file
from GraphConv.graphlayer.graphconv import GraphConv as gconv
from GraphConv.graphlayer.residualgraphconv import ResidualGraphConv as resgconv
from GraphConv.functional.batchnorm import batchnorm


# ----------------------------------------------------------------------
# -------------- GNN with residual graph convolutions ------------------
# ----------------------------------------------------------------------

class GNNRes(nn.Module):
    """implements a Graph Neural Networks to analyse the data from the Large Hadron
    Collider. The graph is built by defining weight depending on the euclidean
    distance between two energy bursts."""

    def __init__(self, dim, deg, usebatchnorm=False, logistic_bias=0):
        super(GNNRes, self).__init__()
        dim.insert(0, 1)
        assert(len(dim) == len(deg))
        self.nblayer = len(dim)
        self.bn = usebatchnorm
        self.dim = dim
        self.deg = deg
        self.resgconv = nn.ModuleList([resgconv(dim[i], dim[i + 1], deg[i], F.relu, bn=self.bn) for i in range(self.nblayer - 1)])
        self.finalgconv = gconv(dim[-1], 1, deg[-1])  # don't batchnorm before mean...
        self.logistic_bias = logistic_bias
        self.adjacency = Adjacency()

    def forward(self, e, phi, eta):
        e = batchnorm(e, axis=1)
        e = e.unsqueeze(1)

        # applying the GNN
        adj = self.adjacency(phi, eta)

        for i in range(self.nblayer - 1):
            e = self.resgconv[i](adj, e)
        e = self.finalgconv(adj, e)
        e = e + self.logistic_bias
        # compute mean : structural information is already exploited
        e = e.mean(2).squeeze(2).squeeze(1)

        # sigmoid for logistic regression
        e = F.sigmoid(e)

        return e


# ----------------------------------------------------------------------
# -------------- GNN with residual graph convolutions ------------------
# ------- input is a combination of energy and spatial parameter -------
# ----------------------------------------------------------------------

class GNNResSpatial(nn.Module):
    """implements a Graph Neural Networks to analyse the data from the Large Hadron
    Collider. The graph is built by defining weight depending on the euclidean
    distance between two energy bursts."""

    def __init__(self, dim, deg, usebatchnorm=False, logistic_bias=0, normalize=False):
        super(GNNResSpatial, self).__init__()

        self.normalize = normalize
        self.adjacency = Adjacency(self.normalize)

        if self.normalize:
            dim.insert(0, 3)
        else:
            dim.insert(0, 4)

        assert(len(dim) == len(deg))
        self.nblayer = len(dim)
        self.bn = usebatchnorm
        self.dim = dim
        self.deg = deg
        self.resgconv = nn.ModuleList([resgconv(dim[i], dim[i + 1], deg[i], F.relu, bn=self.bn) for i in range(self.nblayer - 1)])
        self.finalgconv = gconv(dim[-1], 1, deg[-1])
        self.logistic_bias = logistic_bias

    def forward(self, e, phi, eta):
        e = batchnorm(e, axis=1)

        # adjacency for GNN
        adj = self.adjacency(phi, eta)

        # input is a concatenation of e and eta
        eta = batchnorm(eta, axis=1)
        phi = batchnorm(phi, axis=1)
        features = [e, eta, phi]
        if self.normalize:  # the renormalization factors were returned with adj
            adj, factors = adj
            features.append(factors)
        e = torch.stack(features, dim=1)

        for i in range(self.nblayer - 1):
            e = self.resgconv[i](adj, e)
        e = self.finalgconv(adj, e)
        e = e + self.logistic_bias

        # compute mean : structural information is already exploited
        e = e.mean(2).squeeze(2).squeeze(1)

        # sigmoid for logistic regression
        e = F.sigmoid(e)

        return e
