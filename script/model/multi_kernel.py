from math import sqrt
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.functional import relu, sigmoid
from torch.autograd import Variable
from model import multi_operators as ops
from model import graphconv as gc
from utils.tensor import spatialnorm, make_tensor_as, sqdist_, sqdist_periodic_, sym_min


def resized(param, batchsize):
    """Expands a parameter to fit the batchsize currently used"""

    no_bs_weight_shape = param.size()[1:]
    param = param.expand(batchsize, *no_bs_weight_shape)
    return param


class MultiQCDAware(nn.Module):
    """Same as QCDAware kernel, but with multiple kernels for each layer"""

    def __init__(self, edge_feature, periodic=False):
        super(MultiQCDAware, self).__init__()
        self.out_feature = edge_feature

        self._declare_parameter('alpha', (1, self.out_feature, 1), 0.5, 0.01)
        self._declare_parameter('beta', (1, self.out_feature, 1, 1), 0.7, 0.007)

        self.softmax = nn.Softmax()
        self.sqdist = sqdist_periodic_ if periodic else sqdist_

    def _declare_parameter(self, name, shape, avg, rand):
        param = torch.Tensor(*shape)
        nn.init.uniform(param, avg - rand, avg + rand)
        self.register_parameter(name, Parameter(param))

    def forward(self, emb):
        """Computes edge features as :
            W_ij = \rho(max(\theta12.H_i, \theta12.H_j)
                   + \theta34.(H_i + H_j) + \theta5.delta(i-j))
        """

        # preparation
        batchsize, _, nb_node = emb.size()
        out_tensor_size = (batchsize, self.out_feature, nb_node, nb_node)

        # d_ij^alpha
        sqdist = self.sqdist(emb)
        momentum = emb[:, 4, :]
        momentum = momentum.unsqueeze(1).expand(batchsize, self.out_feature, nb_node)
        alpha = self.alpha.expand_as(momentum)
        pow_momenta = (2 * alpha * momentum.log()).exp()
        min_momenta = sym_min(pow_momenta)
        sqdist = sqdist.unsqueeze(1).expand(out_tensor_size)
        d_ij_alpha = sqdist * min_momenta

        # identity
        if d_ij_alpha.is_cuda:
            eye = torch.cuda.FloatTensor(nb_node, nb_node)
        else:
            eye = torch.FloatTensor(nb_node, nb_node)
        eye = Variable(eye)
        nn.init.eye(eye)
        eye = eye.unsqueeze(0).unsqueeze(1).expand_as(d_ij_alpha)

        # exclude diagonal from dmin definition
        val = d_ij_alpha.data.max()
        tens_nodiag = d_ij_alpha + val * eye
        tens_min, _ = tens_nodiag.min(2)
        dmin, _ = tens_min.min(3)
        zero_div_protec = ((dmin == 0).detach().type_as(d_ij_alpha) * 1e-9).expand_as(d_ij_alpha)

        # center d_ij_alpha
        dmin_x = dmin.expand_as(d_ij_alpha)
        d_ij_center = (d_ij_alpha - dmin_x) / (dmin_x + zero_div_protec)

        # exponential amplitude beta
        beta = (self.beta ** 2).expand_as(d_ij_center)
        d_ij_norm = - beta * d_ij_center

        # w_ij : softmax on (-beta * d_ij_alpha / dmin)
        d_ij_norm = d_ij_norm - relu(float('+inf') * eye)  # removes diagonal
        w_ij = self._softmax(d_ij_norm)
        # w_ij = d_ij_norm.exp()

        return w_ij

    def _softmax(self, dij):
        """applies a softmax independantly on each dij[batch, fm, n, :]"""

        batch, edge_fm = dij.size()[0], dij.size()[1]

        dij = torch.unbind(dij, dim=0)
        dij = torch.cat(dij, dim=0)
        dij = torch.unbind(dij, dim=0)
        dij = torch.cat(dij, dim=0)

        dij = self.softmax(dij)

        dij = torch.chunk(dij, batch * edge_fm, dim=0)
        dij = torch.stack(dij, dim=0)
        dij = torch.chunk(dij, batch, dim=0)
        dij = torch.stack(dij, dim=0)

        return dij


class Node2Edge(nn.Module):
    """Doesn't actually define a kernel, but rather edge features that are
    parametrized operations applied to the node feature maps

    inputs: - emb : node features, can include global input concatenated as features.
                (batch, `node_feature`, N)

    output: - kernel : adjacency matrix defined as
               W_ij = \rho(max(\theta1.h_i + \theta2.x_i, \theta1.h_j + \theta2.x_j)
                   + \theta3.(h_i + h_j) + \theta4.(x_i + x_j) + \theta5.delta(i-j))
                of size (batch, `edge_feature`, N, N)
    """

    def __init__(self, node_feature, edge_feature):
        super(Node2Edge, self).__init__()
        self.in_feature = node_feature
        self.out_feature = edge_feature

        self.declare_parameter('theta12', (1, self.out_feature, self.in_feature))
        self.declare_parameter('theta34', (1, self.out_feature, self.in_feature))
        self.declare_parameter('theta5', (1, self.out_feature, 1, 1))

    def declare_parameter(self, name, shape):
        """Initiates parameter called `name` with shape `shape`"""

        invsqrt2 = sqrt(2) / 2
        param = torch.Tensor(*shape)
        nn.init.uniform(param, -invsqrt2, invsqrt2)
        self.register_parameter(name, Parameter(param))

    def forward(self, emb):
        """Computes edge features as :
            W_ij = \rho(max(\theta12.H_i, \theta12.H_j)
                   + \theta34.(H_i + H_j) + \theta5.delta(i-j))
        """

        # preparation
        batchsize, _, nb_node = emb.size()
        out_tensor_size = (batchsize, self.out_feature, nb_node, nb_node)

        # max(\theta1.h_i + \theta2.x_i, \theta1.h_j + \theta2.x_j)
        theta12 = resized(self.theta12, batchsize)
        theta12hx = torch.bmm(theta12, emb)
        theta12hx = theta12hx.unsqueeze(3).expand(*out_tensor_size)
        theta12hx_t = theta12hx.transpose(2, 3).contiguous()
        max_thx = torch.max(theta12hx, theta12hx_t)

        # \theta3.(h_i + h_j) + \theta4.(x_i + x_j)
        theta34 = resized(self.theta34, batchsize)
        theta34hx = torch.bmm(theta34, emb)
        theta34hx = theta34hx.unsqueeze(3).expand(*out_tensor_size)
        theta34hx_t = theta34hx.transpose(2, 3).contiguous()
        sum_thx = (theta34hx + theta34hx_t)

        # \theta5.delta(i-j) i.e. identity
        eye = make_tensor_as(emb, (nb_node, nb_node))
        nn.init.eye(eye)
        eye = Variable(eye).unsqueeze(0).unsqueeze(1).expand(*out_tensor_size)
        theta5 = self.theta5.expand(*out_tensor_size)
        eye_t = theta5 * eye

        adj = max_thx + max_thx + eye_t
        nonlin_adj = _softmax(adj)

        return nonlin_adj


class GatedNode2Edge(nn.Module):
    """Gated version of `Node2Edge` :
            kernel = W_1 * sigmoid(W_2)
    where both W_1 and W_2 are instances of `Node2Edge`.
    """

    def __init__(self, node_feature, edge_feature):
        super(GatedNode2Edge, self).__init__()
        self.ker1 = Node2Edge(node_feature, edge_feature)
        self.ker2 = Node2Edge(node_feature, edge_feature)

    def forward(self, emb):
        adj = self.ker1(emb)
        gate = sigmoid(self.ker2(emb))
        adj = adj * gate

        return adj

