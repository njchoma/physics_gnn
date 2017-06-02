import torch
from torch.nn import Parameter
from torch.autograd import Variable
from GraphConv.kernel.distancekernel import Distance, DistanceKNN


class SparseDirectionnalGaussianKNN(DistanceKNN):
    def __init__(self, k, *args, **kwargs):
        super(SparseDirectionnalGaussianKNN, self).__init__(k, *args, **kwargs)

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
        mask = Variable(mask.to_dense()).cuda()
        return mask * adj


class GaussianKNN(Distance):
    def __init__(self, k, *args, **kwargs):
        super(GaussianKNN, self).__init__(*args, **kwargs)

        self.k = k
        self.thr = None  # sparsification already done by K-NN
        sigma = Parameter((torch.rand(1) * 0.2 + 0.9) * self.std)
        self.register_parameter('sigma', sigma)

    def knn(self, dist):
        """copy of DistanceKNN.knn without sparse future"""

        value, _ = dist.sort(2)
        k = min(self.k, dist.size()[2])
        kthvalue = value[:, :, k].unsqueeze(2).expand_as(dist)
        mask = dist <= kthvalue

        return mask

    def kernel(self, phi, eta):

        sqdist = self.distances(phi, eta)
        adj = (-sqdist * self.sigma.expand_as()).exp()

        # sparsify with mask
        mask = Variable(self.knn(sqdist.data)).type_as(adj)
        adj = adj * mask
        return adj


class DirectionnalGaussianKNN(Distance):
    def __init__(self, k, *args, **kwargs):
        super(DirectionnalGaussianKNN, self).__init__(*args, **kwargs)

        self.k = k
        self.thr = None  # sparsification already done by K-NN
        alpha = Parameter((torch.rand(1) * 0.2 + 0.9) * self.std)
        beta = Parameter((torch.rand(1) * 0.2 + 0.9) * self.std)
        self.register_parameter('alpha', alpha)
        self.register_parameter('beta', beta)

    def knn(self, dist):
        """copy of DistanceKNN.knn without sparse future"""

        value, _ = dist.sort(2)
        k = min(self.k, dist.size()[2])
        kthvalue = value[:, :, k].unsqueeze(2).expand_as(dist)
        mask = dist <= kthvalue

        return mask

    def kernel(self, phi, eta):
        std_dir0 = self.alpha.expand_as(phi)
        std_dir1 = self.beta.expand_as(eta)

        phi = phi * std_dir0
        eta = eta * std_dir1

        sqdist = self.distances(phi, eta)
        adj = (-sqdist).exp()

        # sparsify with mask
        mask = Variable(self.knn(sqdist.data)).type_as(adj)
        adj = adj * mask
        return adj
