import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils.files import makedir_if_not_there
from os.path import join


def plot_kernel(x, y, adj, fileout=None, dirout='.'):
    """plots the network corresponding to adjacency matrix `adj` with node
    coordinates (`x`, `y`).abs

    inputs : - x, y : numpy vectors of size n
             - adj  : numpy matrix of size n*n
             - fileout : optionnal - saving file for the network plot
             - dirout  : optionnal - path to saving directory, default is '.'
    If no argument `fileout` is given, the network will be ploted but not saved.
    """

    G = nx.Graph()
    n = x.size
    plt.figure(figsize=(25, 25))

    # create nodes
    G.add_nodes_from(range(n))

    # create edges
    edges = [[i, j, {'w': adj[i, j]}] for i in range(n) for j in range(n) if adj[i, j] != 0]
    G.add_edges_from(edges)
    print('-' * 40 +
          '\n{} : {}\n'.format(n, float(len(edges)) / n) +
          '-' * 40)

    pos = {i: (x[i], y[i]) for i in range(n)}
    nx.draw_networkx(G, pos=pos)

    if fileout is not None:
        makedir_if_not_there(dirout)
        plt.savefig(join(dirout, fileout))
    raise EOFError


if __name__ == '__main__':
    n = 4
    x = np.random.rand(n)
    y = np.random.rand(n)
    adj = np.random.normal(size=(n, n))
    adj = adj + adj.T
    adj = adj * (adj > 0)

    print(adj)
    plot_kernel(x, y, adj)
