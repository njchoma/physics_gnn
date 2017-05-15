import torch
from torch.nn import Parameter
from torch.autograd import Variable
from GraphConv.kernel.distancekernel import DistanceKNN


class DirectionnalGaussianKNN(DistanceKNN):
    def __init__(self, k, *args, **kwargs):
        super(DirectionnalGaussianKNN, self).__init__(k, *args, **kwargs)

        self.thr = None  # sparsification already done by K-NN
        alpha = Parameter((torch.rand(1) * 0.2 + 0.9) * self.std)
        beta = Parameter((torch.rand(1) * 0.2 + 0.9) * self.std)
        self.register_parameter('alpha', alpha)
        self.register_parameter('beta', beta)

    # EDIT : change this method when sparse variables are available
    def kernel(self, phi, eta):
        std_dir0 = self.alpha.expand_as(phi)
        std_dir1 = self.beta.expand_as(eta)

        phi = phi * std_dir0
        eta = eta * std_dir1

        sqdist = self.distances(phi, eta)
        adj = (-sqdist).exp()

        # sparsify with mask
        mask = self.knn(sqdist.data)
        mask = Variable(mask.to_dense())
        return mask * adj

