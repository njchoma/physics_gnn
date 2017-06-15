import torch
from torch.autograd import Variable
from utils import _variable_as, _cuda_as


def join_operators(adj, operator_iter):
    """Applies each operator in `operator_iter` to adjacency matrix `adj`,
    then change format to be compatible with `GraphOpConv`s.

    inputs : - adj : adjacency matrix (graph structure). shape (batch, n, n)
             - operator_iter : iterable containing graph operators.

    output : - ops : `GraphOpConv` compatible representation of operators from
                `operator_iter` applied to `adj`. shape (batch, n, n * nb_op)
    """

    if not operator_iter:  # empty list
        return None
    ops = tuple(operator(adj) for operator in operator_iter)
    ops = torch.cat(ops, 2)

    return ops


def count_operators(operator_iter, kernel):
    """Givent the operator list that would be given to `joint_operator`,
    returns the number `nb_op` that should be used to initiate a Graph Conv layer
    """

    def _operator_size(operator):
        fake_input = Variable(torch.ones(1, 3, 1))
        fake_output = operator(kernel(fake_input))
        return fake_output.numel()

    return sum(_operator_size(operator) for operator in operator_iter)


def adjacency(adj):
    """adjacency matrix.
    Counts as multiple operators if `adj` is a tuple of adjacencys"""

    if isinstance(adj, tuple) or isinstance(adj, list):
        return torch.cat(adj, 2)
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

    if isinstance(adj, tuple) or isinstance(adj, list):
        adj = adj[0]
    nb_node = adj.size()[2]
    operator = torch.ones(nb_node) / nb_node
    operator = operator.unsqueeze(0).expand_as(adj)
    operator = _variable_as(operator, adj)
    operator = _cuda_as(operator, adj)

    return operator


def degree(adj):
    """Degree matrix"""

    def _degree_one(adj):
        """when there is only one adjacency"""

        deg = adj.sum(1)  # operators are used with right side dot product
        deg = deg.expand_as(adj)
        operator = identity(adj) * deg

        return operator

    if isinstance(adj, tuple) or isinstance(adj, list):
        return torch.cat(
            [_degree_one(one_adj) for one_adj in adj], 2
        )
    return adj
