import torch
from torch.nn import Parameter
from GraphConv.functional.softmax import softmax
from GraphConv.kernel.distancekernel import Distance
import torch.functional as F


class Gaussian(Distance):

    def __init__(self, thr=None, std=1., normalize=False):
        super(Gaussian, self).__init__()
        # std = Parameter(std)
        std = Parameter(torch.rand(1) * 0.2 - 0.1 + std)
        self.register_parameter('std', std)  # Uniform on [0.9, 1.1]
        self.normalize = normalize
        self.thr = thr

    def forward(self, phi, eta):
        """takes the exponential of squared distances and renormalizes"""

        sqdist = self.distances(phi, eta)
        var = (self.std ** 2).expand_as(sqdist)
        if self.normalize:
            adj = softmax(-sqdist / var, axis=2)
        else:
            adj = (-sqdist / var).exp()

        if self.thr is not None:
            energy_per_row = adj.sum(2)
            thr = energy_per_row * self.thr  # threshold is a percentage of weight energy
            thr = thr.expand_as(adj)
            adj = F.Relu(adj - thr) + thr  # nullify all value below thr in a differtiable manner

        return adj


class DirectionnalGaussian(Distance):

    def __init__(self, thr=None, std=1.):
        super(DirectionnalGaussian, self).__init__(thr, std)
        alpha = Parameter(torch.rand(1) * 0.2 + 0.9)
        beta = Parameter(torch.rand(1) * 0.2 + 0.9)
        self.register_parameter('alpha', alpha)
        self.register_parameter('beta', beta)

    def forward(self, phi, eta):

        phi = phi * self.alpha
        eta = eta * self.beta
        distances = self.distances(phi, eta)
        adj = softmax(-distances, axis=2)

        return adj
