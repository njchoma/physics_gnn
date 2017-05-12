import torch
import torch.nn as nn
import torch.nn.functional as F
from GraphConv.kernel.learnkernel import DirectionnalGaussian as Adjacency  # EDIT : choice of adjacency type should be done in metaparameter file
from GraphConv.graphlayer.residualgraphconv import ResidualGraphConv as resgconv
from GraphConv.functional.batchnorm import batchnorm


class RGCs_FCL(nn.Module):
    """implements a Graph Neural Networks to analyse the data from the Large Hadron
    Collider. The graph is built by defining weight depending on the euclidean
    distance between two energy bursts."""

    def __init__(self, dim, deg, usebatchnorm=False, logistic_bias=0):
        super(RGCs_FCL, self).__init__()
        dim.insert(0, 3)
        self.nblayer = len(dim)
        self.bn = usebatchnorm
        self.dim = dim
        self.deg = deg
        self.resgconv = nn.ModuleList([resgconv(dim[i], dim[i + 1], deg[i], F.relu, bn=self.bn) for i in range(self.nblayer - 1)])
        self.fc = nn.Linear(dim[-1], 1)
        self.logistic_bias = logistic_bias
        self.adjacency = Adjacency()

    def forward(self, e, phi, eta):
        e = batchnorm(e, axis=1)

        # applying the GNN
        adj = self.adjacency(phi, eta)

        # input is a concatenation of e and eta
        eta = batchnorm(eta, axis=1)
        phi = batchnorm(phi, axis=1)
        e = torch.stack((e, eta, phi), dim=1)

        for i in range(self.nblayer - 1):
            e = self.resgconv[i](adj, e)
        e = e.mean(2).squeeze(2)
        e = self.fc(e).squeeze(1)

        # sigmoid for logistic regression
        e = e + self.logistic_bias
        e = F.sigmoid(e)

        return e
