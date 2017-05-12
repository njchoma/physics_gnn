import torch
from torch.nn import Parameter
from GraphConv.functional.softmax import softmax
from GraphConv.kernel.distancekernel import Distance


class Gaussian(Distance):

    def __init__(self, *args, **kwargs):
        super(Gaussian, self).__init__(*args, **kwargs)
        std = Parameter((torch.rand(1) * 0.2 + 0.9) * self.std)
        self.register_parameter('std', std)  # Uniform on [0.9, 1.1]

    def kernel(self, phi, eta):
        """takes the exponential of squared distances and renormalizes"""

        sqdist = self.distances(phi, eta)
        var = (self.std ** 2).expand_as(sqdist)

        if self.normalize:
            adj = softmax(-sqdist / var, axis=2)
        else:
            adj = (-sqdist / var).exp()

        return adj


class DirectionnalGaussian(Distance):

    def __init__(self, *args, **kwargs):
        super(DirectionnalGaussian, self).__init__(*args, **kwargs)

        alpha = Parameter((torch.rand(1) * 0.2 + 0.9) * self.std)
        beta = Parameter((torch.rand(1) * 0.2 + 0.9) * self.std)
        self.register_parameter('alpha', alpha)
        self.register_parameter('beta', beta)

    def kernel(self, phi, eta):

        var_dir0 = (self.alpha ** 2).expand_as(phi)
        var_dir1 = (self.beta ** 2).expand_as(eta)

        phi = phi * var_dir0
        eta = eta * var_dir1
        distances = self.distances(phi, eta)
        adj = softmax(-distances, axis=2)

        return adj
