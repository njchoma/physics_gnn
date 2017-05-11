from math import pi
import torch
import torch.nn as nn
from torch.nn import Parameter
from net.graphlayer.coarsening import order_pooling, order_matrix, order_vector
from net.functional.softmax import softmax


def _order(adj, e, phi, eta):
    order = order_pooling(adj)
    adj = order_matrix(order, adj)
    e = order_vector(order, e)
    phi = order_vector(order, phi)
    eta = order_vector(order, eta)
    return (adj, e, phi, eta)


class Kernel(nn.Module):
    def __init__(self, thr=None, std=1.):
        super(Kernel, self).__init__()
        self.perimeter = 2 * pi
        self.thr = thr
        self.std = std

    def compute_kernel(self, e, phi, eta):
        raise NotImplementedError

    def forward(self, e, phi, eta):
        kernels = self.compute_kernel(e, phi, eta)
        kernels = torch.unbind(kernels, 0)
        e = torch.unbind(e, 0)
        phi = torch.unbind(phi, 0)
        eta = torch.unbind(eta, 0)
        res_list = [_order(adj, e[i], phi[i], eta[i]) for i, adj in enumerate(kernels)]
        adj = torch.stack([res[0] for res in res_list])
        e = torch.stack([res[1] for res in res_list])
        phi = torch.stack([res[2] for res in res_list])
        eta = torch.stack([res[3] for res in res_list])
        return (adj, e, phi, eta)

    def distances(self, phi, eta, alpha=1.):
        """computes squared distances with 2*pi cyclicity on phi"""

        p = torch.stack((eta, phi % self.perimeter), dim=2)
        dist1 = self._distances(p, alpha)

        p = torch.stack((eta, (phi + self.perimeter / 2) % self.perimeter), dim=2)
        dist2 = self._distances(p, alpha)

        # pointwise minimum between dist1 and dist2
        min_dist = (dist1 + dist2) / 2 - torch.abs(dist1 - dist2) / 2

        return min_dist

    def _distances(self, p, alpha):
        """squared euclidean distances"""
        p[:, :, 0] *= alpha
        sqnorm = (p * p).sum(2)
        dotprod = p.bmm(torch.transpose(p, 1, 2))
        sqnorm = sqnorm.expand_as(dotprod)

        return(sqnorm + torch.transpose(sqnorm, 1, 2) - 2 * dotprod)

    def pooling(adj, features, pool_func):
        """sums the row and columns of gathered nodes, applies `pool_func` (1D pooling)
        on the features."""



class Gaussian(Kernel):

    def __init__(self, thr=None, std=1., alpha=1.):
        super(Gaussian, self).__init__(thr, std)
        std = Parameter(torch.rand(1) * 0.2 - 0.1 + self.std)
        self.register_parameter('stdv', std)  # Uniform on [0.9, 1.1]
        alpha = Parameter(torch.rand(1) * 0.2 - 0.1 + alpha)
        self.register_parameter('alpha', alpha)  # Uniform on [0.9, 1.1]

    def forward(self, phi, eta):
        """takes the exponential of squared distances and renormalizes"""

        distances = self.distances(phi, eta, self.alpha)
        var = (self.stdv ** 2).expand_as(distances)
        adj = softmax(- distances / var, axis=2)

        return adj

