import torch
import torch.nn as nn
from utils.tensor import variable_as, make_tensor_as


def adjacency(adj):
    """adjacency matrix.
    Counts as multiple operators if `adj` is a tuple of adjacencys
    """

    if isinstance(adj, tuple) or isinstance(adj, list):
        return torch.cat(adj, 2)
    return adj

def adjacency_transpose(adj):
    """transposed adjacency matrix.
    """
    return adj.transpose(1,2)

def identity(adj):
    """Identity matrix"""

    nb_node = adj.size()[2]
    if adj.is_cuda:
        eye = torch.cuda.FloatTensor(nb_node, nb_node)
    else:
        eye = torch.FloatTensor(nb_node, nb_node)
    torch.nn.init.eye(eye)
    operator = eye.unsqueeze(0).expand_as(adj)
    operator = variable_as(operator, adj)

    return operator


def average(adj):
    """DEPRECATED
    Global average over the graph
    """

    if isinstance(adj, tuple) or isinstance(adj, list):
        adj = adj[0]
    batch_size = adj.size()[0]
    nb_node = adj.size()[2]
    operator = make_tensor_as(adj, (batch_size, nb_node, nb_node))
    nn.init.constant(operator, 1. / nb_node)
    operator = variable_as(operator, adj)

    return operator


def degree(adj):
    """Degree matrix"""

    def _degree_one(adj):
        batch_size, nb_node, _ = adj.size()
        deg = adj.sum(1,keepdim=True)  # operators are used with right side dot product
        deg = deg.repeat(1, nb_node, 1)
        operator = identity(adj) * deg

        return operator

    if isinstance(adj, tuple) or isinstance(adj, list):
        operator = torch.cat(
            [_degree_one(one_adj) for one_adj in adj], 2
        )
        return operator
    operator = _degree_one(adj)
    return operator
