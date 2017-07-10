import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import relu, sigmoid
from operators import join_operators
from gcnn import spatialNorm
import graphconv as gc
from torch.nn import Parameter
from math import sqrt

from kernel import FixedGaussian



class GCNN_EdgeFeature(nn.Module):
    """Same as gcnn.GCNN, but using a fully learnt kernel.
    The kernel is 'Node2Edge' defined below. Difference with gcnn.GCNN is
    the use of multiple kernels, event in one layer
    """

    def __init__(self, kernel, operators, input_fm, node_fm, edge_fm, nb_layer):
        super(GCNN_EdgeFeature, self).__init__()

        self.operators = operators
        self.nb_op = edge_fm * len(operators)  # ops on operators use adjacency

        self.fst_kernel = kernel(input_fm, edge_fm)
        self.fst_resgconv = gc.ResGOpConv(input_fm, node_fm, self.nb_op)

        self.kernels = nn.ModuleList(
            [kernel(input_fm + node_fm, edge_fm)
             for _ in range(nb_layer - 1)]
        )
        self.resgconvs = nn.ModuleList(
            [gc.ResGOpConv(input_fm + node_fm, node_fm, self.nb_op)
             for _ in range(nb_layer - 1)]
        )

        self.instance_norm = nn.InstanceNorm1d(1)
        self.fcl = nn.Linear(node_fm, 1)

    def forward(self, global_input):

        kernel = self.fst_kernel(global_input)
        ops = join_operators(kernel, self.operators)
        emb = self.fst_resgconv(ops, global_input)

        for i, resgconv in enumerate(self.resgconvs):
            emb, _, _ = spatialNorm(emb)
            emb = torch.cat((emb, global_input), dim=1)  # concat (h, x)
            kernel = self.kernels[i](emb)
            ops = join_operators(kernel, self.operators)
            emb = resgconv(ops, emb)

        emb = emb.mean(2).squeeze(2).unsqueeze(1)
        emb = self.instance_norm(emb).squeeze(1)

        # # logistic regression
        emb = self.fcl(emb).squeeze(1)
        emb = sigmoid(emb)

        if (emb != emb).data.sum() > 0:
            print('WARNING : NAN')
        return emb


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
        invsqrt2 = sqrt(2) / 2
        param = torch.Tensor(*shape)
        nn.init.uniform(param, -invsqrt2, invsqrt2)
        self.register_parameter(name, Parameter(param))
        self.safe = FixedGaussian(0.7)

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
        eye = _tensor_as(emb, (nb_node, nb_node))
        nn.init.eye(eye)
        eye = Variable(eye).unsqueeze(0).unsqueeze(1).expand(*out_tensor_size)
        theta5 = self.theta5.expand(*out_tensor_size)
        eye_t = theta5 * eye

        return relu(max_thx + max_thx + eye_t)

def resized(param, batchsize):
    no_bs_weight_shape = param.size()[1:]
    param = param.expand(batchsize, *no_bs_weight_shape)
    return param


def stack_feature_operators(ops):
    """reorganises a multitude of operators stacked on a feature dimension
    to match the size convention for graph convolutions
    """

    ops_list = torch.unbind(ops, 1)  # unbind feature dimension
    ops = torch.cat(ops_list, 2)  # concatenate on 2nd dimension
    return ops


def degree_multikernel(adjs):
    """Degree matrices, same as operators.degree, but with an edge feature
    dimension inserted in dim 1"""
    
    nb_node = adjs.size()[2]
    deg = adjs.sum(2)
    deg = deg.expand_as(adjs)

    tensor_type = torch.cuda.FloatTensor if adjs.is_cuda else torch.FloatTensor
    eye = tensor_type(nb_node, nb_node)
    nn.init.eye(eye)
    eye = eye.unsqueeze(0).unsqueeze(1).expand_as(adjs)
    if isinstance(deg, Variable):
        eye = Variable(eye)

    degree = deg * eye
    # reorganise dimensions
    degree = stack_feature_operators(degree)

    return degree


def adjacency_multikernel(adjs):
    """Ajacency natrices"""

    return stack_feature_operators(adjs)


def _tensor_as(tensor, shape):
    if tensor.is_cuda:
        return torch.cuda.FloatTensor(*shape)
    return torch.FloatTensor(*shape)
