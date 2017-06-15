from os.path import join, exists
import pickle
import numpy as np
import matplotlib.pyplot as plt
import kernel as Kernel
import torch
from torch.autograd import Variable


"""
Representation of the kernels effect when sigma varies.
Let eig2, eig3 be the eigen vectors corresponding to the 2nd and 3rd greatest
eigen values of the adjacency. (eig2[i], eig3[i]) are plane coordinate for
node i, this is what is being ploted here. The full line represents this point's
movement when sigma changes. The dash line links the real plane coordinates
(eta, phi) to the eigen vector projection (this allows to anderstand which points
are affected how). the point at the end of both the full line and the dash line
is created with 'start' sigma.
"""

def _eig_vect_same_dir(new, old):
    if (new * old).sum() < 0:  # eigen vector was flipped
        return -new
    return new


def create_tracker(datafile, donormalize, diag, start, stop, nb_pts):
    kernel = Kernel.FixedGaussian

    data, _ = pickle.load(open(datafile, 'rb'), encoding='latin1')
    emb_in = data[23]
    emb_in = Variable(torch.Tensor(emb_in)).t().unsqueeze(0)
    sigma_range = np.linspace(start, stop, num=nb_pts)

    x = emb_in[0, 1, :].data.numpy()
    y = emb_in[0, 2, :].data.numpy()
    n = x.size
    tracker = []
    x_proj, y_proj = x, y


    for sigma in sigma_range:
        adj = kernel(sigma, diag=diag)(emb_in)[0, :, :]
        if donormalize:
            adj = adj / adj.sum(1).expand_as(adj)
        adj = adj.data.numpy()

        eig_val, eig_vect = np.linalg.eig(adj)
        x_proj, x_old = eig_vect[:, 1], x_proj
        y_proj, y_old = eig_vect[:, 2], y_proj
        x_proj = _eig_vect_same_dir(x_proj, x_old)
        y_proj = _eig_vect_same_dir(y_proj, y_old)

        tracker.append((sigma, x_proj, y_proj))
    obj = {'nb_node': n,
           'normalized': donormalize, 'diag': diag,
           'tracker': tracker, 'emb': (x, y)}
    with open('track_kernel.pkl', 'wb') as fileout:
        pickle.dump(obj, fileout)


def plot_tracker(plot_precision=(25, 25)):
    with open('track_kernel.pkl', 'rb') as filein:
        obj = pickle.load(filein)

    nb_node = obj['nb_node']
    normalized = obj['normalized']
    diag = obj['diag']
    tracker = obj['tracker']
    x, y = obj['emb']
    colors = np.random.rand(3, nb_node)

    x_list = [[scatter[1][node] for scatter in tracker] for node in range(nb_node)]
    y_list = [[scatter[2][node] for scatter in tracker] for node in range(nb_node)]

    plt.figure(figsize=plot_precision)
    for node in range(nb_node):
        plt.plot(x[node], y[node], 'o', c=colors[:, node])
        plt.plot((x[node], x_list[node][0]), (y[node], y_list[node][0]), '--', c=colors[:, node])
        plt.plot(x_list[node], y_list[node], c=colors[:, node])
    plt.title(
        'sigma from {} to {} '.format(tracker[0][0], tracker[-1][0])
        + ('with' if diag else 'without') + ' diag, '
        + ('' if normalized else 'not') + ' normalized'
        )
    plt.plot(0, 0, '+k')
    plt.savefig(
        'track_kernel'
        + ('' if diag else '_nodiag')
        + ('_norm' if normalized else '')
        + '.png'
    )
    plt.show()


def main(datadir, donormalize, diag):
    datafile = join(datadir, 'antikt-kt-train-gcnn.pickle')
    print('CREATING TRACKER')
    create_tracker(datafile, donormalize, diag, 0.4, 1.8, 100)
    print('TRACKER SAVED')
    plot_tracker((40, 40))

if __name__ == '__main__':
    remote_data = '/data/grochette/data_nyu'
    local_data = '/home/gaspar/Desktop/dataNYU'
    donormalize = False
    diag = True

    if exists(remote_data):
        print(remote_data)
        main(remote_data, donormalize, diag)
    else:  # local run
        print(local_data)
        main(local_data, donormalize, diag)