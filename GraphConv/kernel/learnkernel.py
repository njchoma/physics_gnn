import torch
from torch.nn import Parameter
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

        std_dir0 = self.alpha.expand_as(phi)
        std_dir1 = self.beta.expand_as(eta)

        phi = phi * std_dir0
        eta = eta * std_dir1

        sqdist = self.distances(phi, eta)
        adj = (-sqdist).exp()

        return adj
