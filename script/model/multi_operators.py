import torch
import torch.nn as nn
from torch.autograd import Variable


def stack_feature_operators(ops):
    """reorganises a multitude of operators stacked on a feature dimension
    to match the size convention for graph convolutions : 
    """

    ops_list = torch.unbind(ops, 1)
    ops = torch.cat(ops_list, 2)
    return ops


def degree(adjs):
    """Degree matrices, same as operators.degree, but with an edge feature
    dimension inserted in dim 1"""

    nb_node = adjs.size()[2]
    deg = adjs.sum(2)
    deg = deg.expand_as(adjs)

    tensor_type = torch.cuda.FloatTensor if adjs.is_cuda else torch.FloatTensor
    eye = tensor_type(nb_node, nb_node)
    nn.init.eye(eye)
    eye = eye.unsqueeze(0).unsqueeze(1).expand_as(adjs)
    eye = Variable(eye)
    deg = deg * eye

    return stack_feature_operators(deg)


def adjacency(adjs):
    """Ajacency natrices"""

    return stack_feature_operators(adjs)
