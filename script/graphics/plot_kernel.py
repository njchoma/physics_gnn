import pickle
import numpy as np
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import kernel as Kernel
import torch
from torch.autograd import Variable


def plot_graph(x, y, G, name):
    plt.figure(figsize=(25, 25))
    n = x.size
    pos = {i: (x[i], y[i]) for i in range(n)}
    nx.draw_networkx(G, pos=pos)

    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.savefig('kernel_graphics/' + name)
    plt.close()


def plot_kernel(emb_in, graphClass, sigma, thr=0):
    """plots the network corresponding to adjacency matrix `adj` with node
    coordinates contained in `emb_in`
    """

    G = nx.Graph()
    x = emb_in[0, 1, :].data.numpy()
    y = emb_in[0, 2, :].data.numpy()
    adj = graphClass(sigma)(emb_in)[0, :, :].data.numpy()
    adj_norm = graphClass(sigma)(emb_in)[0, :, :]
    adj_norm = adj_norm / adj_norm.sum(1).expand_as(adj_norm)
    adj_norm = adj_norm.data.numpy()
    n = x.size

    # create nodes
    G.add_nodes_from(range(n))

    # create edges
    edges = [[i, j, {'w': adj[i, j]}] for i in range(n) for j in range(n) if adj[i, j] > thr]
    G.add_edges_from(edges)
    print('{} : {} with thr {}'.format(n, float(len(edges)) / n, thr))

    name = 's{}_thr{}.png'.format(sigma, thr)
    plot_graph(x, y, G, name)

    eig_val, eig_vect = np.linalg.eig(adj_norm)
    x_proj, y_proj = eig_vect[:, 1], eig_vect[:, 2]
    name_proj = 's{}_thr{}_proj.png'.format(sigma, thr)
    plot_graph(x_proj, y_proj, G, name_proj)

    eig_val, eig_vect = np.linalg.eig(adj)
    x_proj, y_proj = eig_vect[:, -1], eig_vect[:, -2]
    print(eig_val[-1], eig_val[-2])
    name_proj = 's{}_thr{}_proj_nonorm.png'.format(sigma, thr)
    plot_graph(x_proj, y_proj, G, name_proj)


def _read_args():
    parser = argparse.ArgumentParser(description='simple arguments to train GCNN')
    add_arg = parser.add_argument

    add_arg('--sigma', dest='sigma', help='kernel stdev initial value', type=float)
    add_arg('--thr', dest='thr', help='threshold', type=float, nargs='+')

    args = parser.parse_args()
    return args


def main():
    trainfile = '/data/grochette/data_nyu/antikt-kt-train-gcnn.pickle'
    kernel = Kernel.FixedGaussian
    args = _read_args()
    print('sigma : {}'.format(args.sigma))
    print('thr : {}'.format(args.thr))
    X, y = pickle.load(open(trainfile, 'rb'), encoding='latin1')
    # emb_in = np.random.choice(X, 1)[0]
    emb_in = X[23]  # size 31
    emb_in = Variable(torch.Tensor(emb_in)).t().unsqueeze(0)
    # for thr in args.thr:
    #     plot_kernel(emb_in, kernel, args.sigma, thr)
    for sigma in [float(str(0.2 * (i + 1))[:3]) for i in range(15)]:
        plot_kernel(emb_in, kernel, sigma, 0.6)


if __name__ == '__main__':
    main()
