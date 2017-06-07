import torch
from torch.autograd import Variable


def copy_type(tensor1, tensor2):
    """Makes tensor1 a Variable and cuda depending
    on tensor2"""

    if isinstance(tensor2, Variable):
        tensor1 = Variable(tensor1)

    if tensor2.is_cuda:
        tensor1 = tensor1.cuda()

    return tensor1


def adjacency(adj):
    """adjacency matrix"""
    return adj


def identity(adj):
    """Identity matrix"""

    nb_node = adj.size()[2]
    operator = torch.eye(nb_node)
    operator = operator.unsqueeze(0).expand_as(adj)
    operator = copy_type(operator, adj)

    return operator


def average(adj):
    """Global average over the graph"""

    nb_node = adj.size()[2]
    operator = torch.ones(nb_node) / nb_node
    operator = operator.unsqueeze(0).expand_as(adj)
    operator = copy_type(operator, adj)

    return operator


def degree(adj):
    """Degree matrix"""

    deg = adj.sum(1)  # operators are used with right side dot product
    deg = deg.expand_as(adj)
    operator = identity(adj) * deg

    return operator
