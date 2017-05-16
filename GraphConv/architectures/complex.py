import torch
import torch.nn as nn
import torch.nn.functional as F
from GraphConv.kernel.learnkernel import DirectionnalGaussian as Adjacency  # EDIT : choice of adjacency type should be done in metaparameter file
from GraphConv.kernel.sparsekernel import DirectionnalGaussianKNN as AdjacencyKNN  # EDIT : choice of adjacency type should be done in metaparameter file
from GraphConv.graphlayer.residualgraphconv import ResidualGraphConv as resgconv
from GraphConv.functional.batchnorm import batchnorm


class RGCs_FCL(nn.Module):
    """implements a Graph Neural Networks to analyse the data from the Large Hadron
    Collider. The graph is built by defining weight depending on the euclidean
    distance between two energy bursts."""

    def __init__(
            self, dim, deg, usebatchnorm=False,
            normalize=False, knn=None, logistic_regression=None):
        super(RGCs_FCL, self).__init__()

        self.normalize = normalize

        if knn is None:
            self.adjacency = Adjacency(normalize=self.normalize)
        else:
            self.adjacency = AdjacencyKNN(knn, normalize=self.normalize)

        if self.normalize:
            dim.insert(0, 4)
        else:
            dim.insert(0, 3)

        self.nblayer = len(dim)
        self.bn = usebatchnorm
        self.dim = dim
        self.deg = deg
        self.resgconv = nn.ModuleList([resgconv(dim[i], dim[i + 1], deg[i], F.relu, bn=self.bn) for i in range(self.nblayer - 1)])
        self.fc = nn.Linear(dim[-1], 1)
        self.logistic_regression = logistic_regression

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
            factors = batchnorm(eta, axis=1)
            features.append(factors)
        e = torch.stack(features, dim=1)

        for i in range(self.nblayer - 1):
            e = self.resgconv[i](adj, e)
        e = e.mean(2).squeeze(2)
        e = self.fc(e).squeeze(1)

        # sigmoid for logistic regression
        if self.logistic_regression is not None:
            e = e + self.logistic_regression
            e = F.sigmoid(e)

        return e
