import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
from torch.nn.functional import relu
from utils import _variable_as, _cuda_as
from math import sqrt

def _sqdist(emb):
    coord = emb[:, 1:3, :]
    sqnorm = (coord * coord).sum(1)
    dotprod = coord.transpose(1, 2).bmm(coord)
    sqnorm = sqnorm.expand_as(dotprod)
    sqdist = sqnorm + sqnorm.transpose(1, 2) - 2 * dotprod

    return sqdist


def _gaussian(sqdist, sigma):
    var = sigma ** 2
    adj = (-sqdist * var).exp()

    return adj


def _delete_diag(adj):
    diag_mask = torch.eye(adj.size()[1]).type(torch.ByteTensor)
    diag_mask = _variable_as(diag_mask, adj)
    diag_mask = _cuda_as(diag_mask, adj)
    diag_mask = diag_mask.unsqueeze(0).expand_as(adj)
    adj.masked_fill_(diag_mask, 0)

    return adj


def _stochastich(adj):
    deg = adj.sum(2)
    adj /= deg.expand_as(adj)

    return adj


def _mmin(tensor):
    batch = tensor.size()[0]
    nb_node = tensor.size()[1]

    tens0 = tensor.unsqueeze(2).expand(batch, nb_node, nb_node)
    tens1 = tens0.transpose(1, 2).contiguous()

    res = torch.stack((tens0, tens1), 1)
    res, _ = res.min(1)
    res = res.squeeze(1)
    return res


def _hook_reduce_grad(divider):
    return lambda grad: grad / divider


class FixedGaussian(nn.Module):
    """Gaussian kernel with fixed sigma"""
    def __init__(self, sigma, diag=True, norm=False):
        super(FixedGaussian, self).__init__()
        self.sigma = sigma
        self.diag = diag
        self.norm = norm

    def forward(self, emb):
        """takes the exponential of squared distances"""

        adj = _gaussian(_sqdist(emb), self.sigma)
        if not self.diag:
            adj = _delete_diag(adj)
        if self.norm:
            adj = _stochastich(adj)

        return adj


class FixedComplexGaussian(nn.Module):
    """Gaussian double-kernel, like `FixedGaussian` but with two composants
    introducing the orientation of `emb_i - emb_j` in `adj_ij`
    """

    def __init__(self, sigma, diag=True, norm=False):
        super(FixedComplexGaussian, self).__init__()
        self.sigma = sigma
        self.diag = diag
        self.norm = norm

    def forward(self, emb):
        """takes the exponential of squared distances,
        and directions from `emb_i - emb_j`
        """

        # modulus
        adj = _gaussian(_sqdist(emb), self.sigma)
        if not self.diag:
            adj = _delete_diag(adj)
        if self.norm:
            adj = _stochastich(adj)

        # orientation
        coord = coord = emb[:, 1:3, :]
        size = coord.size()
        coord = coord.unsqueeze(3).expand(size[0], size[1], size[2], size[2])
        diff = coord - coord.transpose(2, 3).contiguous()
        norm_diff = (diff ** 2).sum(1).sqrt()

        zero_div_protection = (norm_diff == 0)
        if isinstance(zero_div_protection, Variable):
            zero_div_protection.detach()
        norm_diff += zero_div_protection.type_as(norm_diff)

        diff /= norm_diff.expand_as(diff)
        adj_r = adj * diff[:, 0, :, :].squeeze(1)
        adj_i = adj * diff[:, 1, :, :].squeeze(1)

        return (adj_r, adj_i)


class Gaussian(nn.Module):
    """Gaussian kernel"""
    def __init__(self, diag=True, norm=False):
        super(Gaussian, self).__init__()
        sigma = Parameter(torch.rand(1) * 0.02 + 0.99)
        self.register_parameter('sigma', sigma)  # Uniform on [0.9, 1.1]
        self.diag = diag
        self.norm = norm

    def forward(self, emb):
        """takes the exponential of squared distances"""

        adj = _gaussian(_sqdist(emb), self.sigma)
        if not self.diag:
            adj = _delete_diag(adj)
        if self.norm:
            adj = _stochastich(adj)

        return adj


class ComplexGaussian(nn.Module):
    """Gaussian double-kernel, like `Gaussian` but with two composants
    introducing the orientation of `emb_i - emb_j` in `adj_ij`
    """

    def __init__(self, diag=True, norm=False):
        super(ComplexGaussian, self).__init__()
        sigma = Parameter(torch.rand(1) * 0.02 + 0.99)
        self.register_parameter('sigma', sigma)  # Uniform on [0.9, 1.1]
        self.diag = diag
        self.norm = norm

    def forward(self, emb):
        """takes the exponential of squared distances,
        and directions from `emb_i - emb_j`
        """

        # modulus
        adj = _gaussian(_sqdist(emb), self.sigma)
        if not self.diag:
            adj = _delete_diag(adj)
        if self.norm:
            adj = _stochastich(adj)

        # orientation
        coord = coord = emb[:, 1:3, :]
        size = coord.size()
        coord = coord.unsqueeze(3).expand(size[0], size[1], size[2], size[2])
        diff = coord - coord.transpose(2, 3).contiguous()
        norm_diff = (diff ** 2).sum(1).sqrt()

        zero_div_protection = (norm_diff == 0)
        if isinstance(zero_div_protection, Variable):
            zero_div_protection.detach()
        norm_diff += zero_div_protection.type_as(norm_diff)

        diff /= norm_diff.expand_as(diff)
        adj_r = adj * diff[:, 0, :, :].squeeze(1)
        adj_i = adj * diff[:, 1, :, :].squeeze(1)

        return (adj_r, adj_i)


class QCDDist(nn.Module):
    """kernel based on 'QCD-Aware Recursive Neural Networks for Jet Physics'"""

    def __init__(self, alpha, radius):
        super(QCDDist, self).__init__()
        self.alpha = alpha
        self.radius = radius

    def forward(self, emb):
        sqdist = _sqdist(emb) / (self.radius ** 2)
        pow_momenta = (2 * self.alpha * emb[:, 4, :].log()).exp()
        min_momenta = _mmin(pow_momenta)
        d_ij = sqdist * min_momenta

        return d_ij


class FixedQCDAware(nn.Module):
    """kernel based on 'QCD-Aware Recursive Neural Networks for Jet Physics'"""

    def __init__(self, alpha, beta, radius, epsilon=1e-7):
        super(FixedQCDAware, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.radius = radius
        self.epsilon = epsilon  # protection against division by 0

    def forward(self, emb):
        sqdist = _sqdist(emb) / (self.radius ** 2)
        pow_momenta = (2 * self.alpha * emb[:, 4, :].log()).exp()
        min_momenta = _mmin(pow_momenta)
        d_ij_alpha = sqdist * min_momenta

        d_min, _ = pow_momenta.min(1)
        d_min = (d_min + self.epsilon).expand_as(d_ij_alpha)
        d_ij_center = (d_ij_alpha - d_min) / d_min
        d_ij_center /= 10000
        d_ij_norm = self.beta * d_ij_center
        w_ij = (-d_ij_norm).exp()

        return w_ij


class QCDAware(nn.Module):
    """kernel based on 'QCD-Aware Recursive Neural Networks for Jet Physics'"""

    def __init__(self, alpha, beta, radius, epsilon=1e-7):
        super(QCDAware, self).__init__()
        self.epsilon = epsilon  # protection against division by 0

        alpha = Parameter(alpha * (torch.rand(1, 1) * 0.02 + 0.99))
        beta = Parameter(beta * (torch.rand(1, 1, 1) * 0.02 + 0.99))
        radius = Parameter(radius * (torch.rand(1, 1, 1) * 0.02 + 0.99))
        self.register_parameter('alpha', alpha)
        self.register_parameter('beta', beta)
        self.register_parameter('radius', radius)

    def forward(self, emb):
        sqdist = _sqdist(emb)
        sqradius = (self.radius ** 2).expand_as(sqdist) + self.epsilon
        # radius.register_hook(_hook_reduce_grad(100))
        sqdist = sqdist / sqradius
        momentum = emb[:, 4, :]
        alpha = self.alpha.expand_as(momentum)
        # alpha.register_hook(_hook_reduce_grad(100))
        pow_momenta = (2 * alpha * momentum.log()).exp()
        min_momenta = _mmin(pow_momenta)
        d_ij_alpha = sqdist * min_momenta

        d_min, _ = pow_momenta.min(1)
        d_min = d_min.unsqueeze(2)
        d_min = (d_min + self.epsilon).expand_as(d_ij_alpha)
        d_ij_center = (d_ij_alpha - d_min) / d_min
        d_ij_center /= 10000
        beta = (self.beta ** 2).expand_as(d_ij_center)
        # beta.register_hook(_hook_reduce_grad(100))
        d_ij_norm = beta * d_ij_center
        w_ij = (-d_ij_norm).exp()

        return w_ij
