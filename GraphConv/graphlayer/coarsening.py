import numpy as np
import networkx as nx
import torch
from torch.autograd import Variable


def default_zero(edges, i, j):
    if (i is None) or (j is None):
        return 0
    try:
        return edges[i][j]['w']
    except KeyError:
        return 0


def transmit_none(obj, idx):
    if idx is None:
        return None
    return obj[idx]


def replace_none(v, replacement):
    if v is None:
        return replacement
    return v


def branch_pooling(adj, depth):
    """`adj` is assumed to be a (n, n) torch tensor. All n sized
    vector given after `adj` will be reordered accordingly"""

    n = adj.size()[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    edges = ((i, j, {'w': adj[i, j]}) for i in range(n) for j in range(i + 1, n) if adj[i, j] != 0)
    graph.add_edges_from(edges)

    cur_graph = list(range(n))  # contains the forest
    for layer in range(depth):
        graph_mem = graph.copy()
        new_graph = []
        visit_order = np.random.permutation(n)
        for node in visit_order:
            if node in graph.nodes():
                # find neighbour with maximum weight
                neigh = graph.neighbors(node)
                neigh_w = [graph.edge[node][neighboor]['w'] for neighboor in neigh]
                if len(neigh_w) > 0:  # `node` can be paired
                    idx = np.argmax(neigh_w)
                    maxneigh = neigh[idx]

                    # merge node and maxneigh
                    new_graph.append((node, maxneigh))
                    graph.remove_nodes_from([node, maxneigh])

                else:  # not match found to pair `node`
                    new_graph.append((node, None))
                    graph.remove_node(node)

        # update graphs
        n = len(new_graph)

        # graph has just been emptied...
        edges = [(i, j, {'w': sum([default_zero(graph_mem, x, y)
                                  for x in new_graph[i] for y in new_graph[j]])})
                 for i in range(n) for j in range(i + 1, n)]

        edges = [(i, j, v) for i, j, v in edges if v != 0]
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from(edges)
        cur_graph = [(transmit_none(cur_graph, i), transmit_none(cur_graph, j))
                     for i, j in new_graph]

    return cur_graph


def flatten(forest, depth, default):
    def _flatten(tree, depth, acc):
        if tree is None:  # fake node
            if depth == 0:
                acc.append(default)
            else:
                _flatten(None, depth - 1, acc)
                _flatten(None, depth - 1, acc)
        elif type(tree) is tuple:  # two branches
            _flatten(tree[0], depth - 1, acc)
            _flatten(tree[1], depth - 1, acc)
        else:  # leaf
            if depth != 0:
                raise ValueError('Wrong depth during flattening.')
            acc.append(tree)

    acc = []
    for tree in forest:
        _flatten(tree, depth, acc)
    return acc


def order_pooling(adj, depth, default=None):
    """computes one possible index order to coarsen the graph with 1D
    pooling."""
    forest = branch_pooling(adj, depth)
    order = flatten(forest, depth, default)
    add_fake = None in order
    if add_fake:
        fake = adj.size[0]
        order = [replace_none(v, fake) for v in order]
    order = torch.LongTensor(order)

    return order, add_fake


def nb_none(l):
    nbnone = len([i for i in l if i is None])
    return nbnone


# The following functions assume the inputs are torch Tensors.

def order_vector(order, vect, add_fake):
    """reorders `vect` according to `order`"""
    if add_fake:
        padd = Variable(torch.zeros(vect.size()[0], 1))
        vect = torch.cat((vect, padd), 1)

    return vect.index_select(0, order)


def order_matrix(order, vect, add_fake):
    """reorders `vect` according to `order`"""
    vect = vect.index_select(0, order)
    return vect.index_select(1, order)
