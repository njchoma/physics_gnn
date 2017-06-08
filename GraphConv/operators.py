import torch
from torch.autograd import Variable


def join_operators(adj, operator_iter):
    """Applies each operator in `operator_iter` to adjacency matrix `adj`,
    then change format to be compatible with `GraphOpConv`s.

    inputs : - adj : adjacency matrix (graph structure). shape (batch, n, n)
             - operator_iter : iterable containing graph operators.

    output : - ops : `GraphOpConv` compatible representation of operators from
                `operator_iter` applied to `adj`. shape (batch, n, n * nb_op)
    """

    ops = tuple(operator(adj) for operator in operator_iter)
    ops = torch.stack(ops, 2)

    return ops


def adjacency(adj):
    """adjacency matrix"""
    return adj


def identity(adj):
    """Identity matrix"""

    nb_node = adj.size()[2]
    operator = torch.eye(nb_node)
    operator = operator.unsqueeze(0).expand_as(adj)
    operator = _variable_as(operator, adj)
    operator = _cuda_as(operator, adj)

    return operator


def average(adj):
    """Global average over the graph"""

    nb_node = adj.size()[2]
    operator = torch.ones(nb_node) / nb_node
    operator = operator.unsqueeze(0).expand_as(adj)
    operator = _variable_as(operator, adj)
    operator = _cuda_as(operator, adj)

    return operator


def degree(adj):
    """Degree matrix"""

    deg = adj.sum(1)  # operators are used with right side dot product
    deg = deg.expand_as(adj)
    operator = identity(adj) * deg

    return operator


def _variable_as(tensor1, tensor2):
    """Makes tensor1 a Variable depending on tensor2"""

    if isinstance(tensor2, Variable):
        tensor1 = Variable(tensor1)

    return tensor1


def _cuda_as(tensor1, tensor2):
    """Makes tensor1 cuda depending on tensor2"""

    if tensor2.is_cuda:
        tensor1 = tensor1.cuda()

    return tensor1
