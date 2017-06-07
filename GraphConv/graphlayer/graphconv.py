import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.init
from math import sqrt


class GraphConv(nn.Module):
    """Applies a graph convolution over an input signal composed of several input planes.

    The output value of the layer with inputs adjacency matrix W of size (N, L, L) and
    a node descripors of size (N, C_in, L) - and output (N, C_out, L) -
    can be precisely described as:

    out(N_i, C_out_j) = bias(C_out_j)+\sum_{k=0}^{C_in-1} P_k(W).input()

    where the P_k are polynomial whose weights are the parameters of the layer
    W is the adjacency matrix of the graph - for computationnal reasons, it
    is assumed to be normalized - it can be asymmetric.

    `in_channels` is the number C_in of input channels
    `out_channels` is the number C_out of output channels
    `degree` is the degree of polynomials P_k
    `bias` gives the option not to add a bias
    """

    def __init__(self, in_channels, out_channels, degree, bias=True, chebyshev=True):
        super(GraphConv, self).__init__()
        self.degree = degree
        self.conv = nn.Conv2d(in_channels, out_channels, (1, degree + 1), bias=bias)
        self.chebyshev = chebyshev

    def _iter_adj(self, adj, x):
        """computes W^k.x for k between 0 and `degree`"""

        xiter = [x]
        y = x.transpose(1, 2).contiguous()
        for k in range(self.degree):
            y = adj.bmm(y)
            xiter.append(y.transpose(1, 2).contiguous())

        return xiter

    def _iter_adj_chebyshev(self, adj, x):

        xiter = [x]
        y = x.transpose(1, 2).contiguous()  # T_0(W).x, y is T_n(W).x
        z = adj.bmm(y)                      # T_1(W).x, z is T_n+1(W).x
        xiter.append(z.transpose(1, 2).contiguous())  # T1 = X
        for k in range(self.degree - 1):
            z, y = 2 * adj.bmm(z) - y, z  # T_n+2 = 2.X.T_n+1 - T_n
            xiter.append(z.transpose(1, 2).contiguous())
        return xiter

    def forward(self, adj, x):
        xiter = self._iter_adj_chebyshev(adj, x) if self.chebyshev else self._iter_adj(adj, x)
        xiter = torch.stack(xiter, dim=3)
        out = self.conv(xiter).squeeze(3)

        return out


class GraphOpConv(nn.Module):
    """Performs graph convolution.
    parameters : - in_fm : number of feature maps in input
                 - out_fm : number of feature maps in output
                 - nb_op : number of graph operators

    inputs : - ops : concatenated graph operators, those are being applied
                    to x by right side dot product : x.op
                    shape (batch, nb_node, J * nb_node)
             - emb_in : signal embedding. shape (batch, in_fm, nb_node)

    output : - emb_out : new embedding. shape (batch, out_fm, nb_node)
    """

    def __init__(self, in_fm, out_fm, nb_op):
        super(GraphOpConv, self).__init__()
        invsqrt2 = sqrt(2) / 2

        weight = torch.Tensor(1, out_fm, in_fm * nb_op)
        nn.init.uniform(weight, -invsqrt2, invsqrt2)
        self.register_parameter('weight', Parameter(weight))

        bias = torch.Tensor(1, out_fm, 1)
        nn.init.uniform(bias, -invsqrt2, invsqrt2)
        self.register_parameter('bias', Parameter(bias))

    def forward(self, ops, emb_in):
        batch_size = emb_in.size()[0]
        nb_node = emb_in.size()[2]
        spread = torch.bmm(emb_in, ops)

        # split effects from different operators, concatenate on feature maps
        spread = spread.split(nb_node, 2)
        spread = torch.cat(spread, 1)

        # apply weights and bias
        weight, bias = self._resized_params(batch_size, nb_node)
        emb_out = torch.bmm(weight, spread)
        emb_out += bias

        return emb_out

    def _resized_params(self, batch_size, nb_node):
        no_bs_weight_shape = self.weight.size()[1:]
        weight = self.weight.expand(batch_size, *no_bs_weight_shape)

        nb_fm = self.bias.size()[1]
        bias = self.bias.expand(batch_size, nb_fm, nb_node)

        return (weight, bias)

if __name__ == '__main__':
    v = 0.2
    gconv = GraphOpConv(3, 10, 2)
    adj = Variable(torch.rand(10, 4, 4)).view(10, 4, 4)
    adj = adj + adj.transpose(1, 2).contiguous()
    eye = Variable(torch.eye(4)).unsqueeze(0)
    ops = torch.cat((eye.expand_as(adj), adj), 2)

    x = Variable(torch.rand(10, 3, 4))
    gconv(ops, x)
