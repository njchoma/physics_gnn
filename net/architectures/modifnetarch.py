import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# EDIT : choice of adjacency type should be done in metaparameter file
from net.kernel.learnkernel import Gaussian as Adjacency
from net.graphlayer.graphconv import GraphConv as gconv
from net.functional.batchnorm import batchnorm


# ----------------------------------------------------------------------
# ------- GNN with list input, then small modifications layers ---------
# ----------------------------------------------------------------------

class GNNModif(nn.Module):
    """implements a Graph Neural Networks to analyse the data from the Large Hadron
    Collider. The graph is built by defining weight depending on the euclidean
    distance between two energy bursts."""

    def __init__(self, dim, deg, modifdeg, logistic_bias=0):
        super(GNNModif, self).__init__()
        dim.insert(0, 1)
        assert(len(dim) == len(deg))
        self.nblayer = len(dim)
        self.dim = dim
        self.deg = deg
        self.modifdeg = modifdeg
        self.gconv = nn.ModuleList([gconv(dim[i], dim[i + 1], deg[i]) for i in range(self.nblayer - 1)])
        modif_dim = dim[-1]
        self.gconv_modif = nn.ModuleList([gconv(modif_dim, modif_dim, mdeg, bias=False) for mdeg in self.modifdeg])
        self.coeff_modif = nn.ModuleList([nn.Conv1d(modif_dim, modif_dim, 1) for mdeg in self.modifdeg])
        self.gconv_final = gconv(modif_dim, 1, deg[-1])
        self.logistic_bias = logistic_bias
        self.adjacency = Adjacency()

        # need maximum degree to iterate the right number of time
        self.maxdeg = np.argmax(deg)

    def forward(self, e, phi, eta):
        e = batchnorm(e, axis=1)
        e = e.unsqueeze(1)

        # applying the GNN
        adj = self.adjacency(phi, eta)

        for layer in self.gconv:  # apply first convolutions
            e = layer(adj, e)
            e = F.relu(e)

        for i, layer in enumerate(self.gconv_modif):  # apply modifications
            modif = self.coeff_modif[i](F.relu(layer(adj, e)))
            e = e + modif

        e = self.gconv_final(adj, e)
        e = e + self.logistic_bias

        # compute mean : structural information is already exploited
        e = e.mean(2).squeeze(2).squeeze(1)

        # sigmoid for logistic regression
        e = F.sigmoid(e)

        return e


# ----------------------------------------------------------------------
# ------- GNN with list input, then small modifications layers ---------
# ---- The position parameters are used as channels for convolution ----
# ----------------------------------------------------------------------

class GNNModifSpatial(nn.Module):
    """implements a Graph Neural Networks to analyse the data from the Large Hadron
    Collider. The graph is built by defining weight depending on the euclidean
    distance between two energy bursts."""

    def __init__(self, dim, deg, modifdeg, logistic_bias=0, usebatchnorm=False):
        super(GNNModifSpatial, self).__init__()
        dim.insert(0, 3)
        assert(len(dim) == len(deg))
        self.nblayer = len(deg)
        self.dim = dim
        self.deg = deg
        self.modifdeg = modifdeg
        self.bn = usebatchnorm

        self.gconv = nn.ModuleList([gconv(dim[i], dim[i + 1], deg[i]) for i in range(self.nblayer - 1)])
        modif_dim = dim[-1]
        self.gconv_modif = nn.ModuleList([gconv(modif_dim, modif_dim, mdeg, bias=False) for mdeg in self.modifdeg])
        self.coeff_modif = nn.ModuleList([nn.Conv1d(modif_dim, modif_dim, 1) for mdeg in self.modifdeg])
        self.gconv_final = gconv(modif_dim, 1, deg[-1])
        self.logistic_bias = logistic_bias

        self.adjacency = Adjacency()
        self.is_cuda = False

        # need maximum degree to iterate the right number of time
        self.maxdeg = np.argmax(deg)

    def forward(self, e, phi, eta):

        # renormalise energy
        e = batchnorm(e, axis=1)

        # initiate adjacency matrix
        adj = self.adjacency(phi, eta)

        # input is a concatenation of e and eta
        eta = batchnorm(eta, axis=1)
        phi = batchnorm(phi, axis=1)
        e = torch.stack((e, eta, phi), dim=1)

        # apply GNN
        for layer in self.gconv:  # apply first convolutions
            e = layer(adj, e)
            e = F.relu(e)
            if self.bn:
                e = batchnorm(e, axis=2)

        for i, layer in enumerate(self.gconv_modif):  # apply modifications
            modif = self.coeff_modif[i](F.relu(layer(adj, e)))
            e = e + modif
            if self.bn:
                e = batchnorm(e, axis=2)

        e = self.gconv_final(adj, e)
        e = e + self.logistic_bias
        # compute mean : structural information is already exploited
        e = e.mean(2).squeeze(2).squeeze(1)

        # sigmoid for logistic regression
        e = F.sigmoid(e)

        return e
