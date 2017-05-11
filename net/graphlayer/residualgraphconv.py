import torch
import torch.nn as nn
from net.functional.batchnorm import batchnorm
from net.graphlayer.graphconv import GraphConv


class ResidualGraphConv(nn.Module):
    """Implements a "residual layer" using graph convolution :
    inputs : deg - degree of the polynomial used on adjacency matrix
             nonlin - nonlinearity
             dim_in - number of channels in input vector
             dim_out - number of channels in output vector - dim_in by default
             bn - boolean : use batch normalisation - False by default

    outputs : neural network layer which from inputs adj and x returns
                y = gconv1(adj, x) + \rho(gconv2(adj, x))
              where \rho is the non linearity `nonlin`, and
              gconv1, gconv2 are graph convolutions created with parameters
              `deg`, `dim_in` and `dim_out`.
              if `bn` is True, a normalisation is performed on `y` :
              for each example and each channel, mean and var are set to
              0 and 1 throughout the graph node descriptors.
    """

    def __init__(self, dim_in, dim_out, deg, nonlin, bn=False, bias=True):
        super(ResidualGraphConv, self).__init__()
        if dim_out % 2 != 0:
            raise ValueError('ResidualGraphConv expected an even output dimension, got {}'.format(dim_out))
        self.gconv1 = GraphConv(dim_in, int(dim_out / 2), deg, bias)
        self.gconv2 = GraphConv(dim_in, int(dim_out / 2), deg, bias)
        self.nonlin = nonlin
        self.bn = bn

    def forward(self, adj, x):
        ylin = self.gconv1(adj, x)
        ynlin = self.nonlin(self.gconv2(adj, x))
        y = torch.cat((ylin, ynlin), 1)
        if self.bn:
            y = batchnorm(y)
        return y
