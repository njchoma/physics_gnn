from math import sqrt
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import utils.tensor as ts


"""Defines different kernels from the embedding"""

def gaussian(sqdist, sigma):
    var = sigma ** 2
    adj = (-sqdist * var).exp()

    return adj


def _delete_diag(adj):
    nb_node = adj.size()[1]
    diag_mask = ts.make_tensor_as(adj, (nb_node, nb_node))
    diag_mask = ts.variable_as(diag_mask, adj)
    diag_mask = diag_mask.unsqueeze(0).expand_as(adj)
    adj.masked_fill_(diag_mask, 0)

    return adj


def _stochastich(adj):
    deg = adj.sum(2)
    adj /= deg.expand_as(adj)

    return adj


def _renorm(bmatrix):
    nb_node = bmatrix.size()[2]
    if bmatrix.is_cuda:
        eye = torch.cuda.FloatTensor(nb_node, nb_node)
    else:
        eye = torch.FloatTensor(nb_node, nb_node)
    eye = Variable(eye)
    nn.init.eye(eye)
    eye = eye.unsqueeze(0).expand_as(bmatrix)

    bmat_nodiag = bmatrix + (1000 * bmatrix.data.max() + 1) * eye  # change diag before min
    ts.check_for_nan(bmat_nodiag, 'nan in _renorm : bmat_nodiag')
    bmat_min, _ = bmat_nodiag.min(1)
    bmat_min, _ = bmat_min.min(2)
    ts.check_for_nan(bmat_min, 'nan in _renorm : bmat_min')
    bmat_min_x = bmat_min.expand_as(bmatrix)
    zero_div_protec = ((bmat_min < 1e-6).detach().type_as(bmatrix) * 1e-6).expand_as(bmatrix)
    bmat_center = (bmat_nodiag - bmat_min_x) / (bmat_min_x + zero_div_protec)

    ts.check_for_nan(bmat_center, 'NAN in _renorm : bmat_center')
    ts.check_for_inf(bmat_center, 'INF in _renorm : bmat_center', action=print, args=(bmat_center, bmatrix, bmat_min))

    return bmat_center


class FixedGaussian(nn.Module):
    """Gaussian kernel with fixed sigma"""
    def __init__(self, sigma, diag=True, norm=False, periodic=False):
        super(FixedGaussian, self).__init__()
        self.sigma = sigma
        self.diag = diag
        self.norm = norm
        self.sqdist = ts.sqdist_periodic_ if periodic else ts.sqdist_

    def forward(self, emb):
        """takes the exponential of squared distances"""

        adj = gaussian(self.sqdist(emb), self.sigma)
        if not self.diag:
            adj = _delete_diag(adj)
        if self.norm:
            adj = _stochastich(adj)

        return adj


class FixedComplexGaussian(nn.Module):
    """Gaussian double-kernel, like `FixedGaussian` but with two composants
    introducing the orientation of `emb_i - emb_j` in `adj_ij`
    """

    def __init__(self, sigma, diag=True, norm=False, periodic=False):
        super(FixedComplexGaussian, self).__init__()
        self.sigma = sigma
        self.diag = diag
        self.norm = norm
        self.sqdist = ts.sqdist_periodic_ if periodic else ts.sqdist_

    def forward(self, emb):
        """takes the exponential of squared distances,
        and directions from `emb_i - emb_j`
        """

        # modulus
        adj = gaussian(self.sqdist(emb), self.sigma)
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
    def __init__(self, diag=True, norm=False, periodic=False):
        super(Gaussian, self).__init__()
        sigma = Parameter(torch.rand(1) * 0.02 + 0.99)
        self.register_parameter('sigma', sigma)  # Uniform on [0.9, 1.1]
        self.diag = diag
        self.norm = norm
        self.sqdist = ts.sqdist_periodic_ if periodic else ts.sqdist_

    def forward(self, emb):
        """takes the exponential of squared distances"""

        adj = gaussian(self.sqdist(emb), self.sigma)
        if not self.diag:
            adj = _delete_diag(adj)
        if self.norm:
            adj = _stochastich(adj)

        return adj


class ComplexGaussian(nn.Module):
    """Gaussian double-kernel, like `Gaussian` but with two composants
    introducing the orientation of `emb_i - emb_j` in `adj_ij`
    """

    def __init__(self, diag=True, norm=False, periodic=False):
        super(ComplexGaussian, self).__init__()
        sigma = Parameter(torch.rand(1) * 0.02 + 0.99)
        self.register_parameter('sigma', sigma)  # Uniform on [0.9, 1.1]
        self.diag = diag
        self.norm = norm
        self.sqdist = ts.sqdist_periodic_ if periodic else ts.sqdist_

    def forward(self, emb):
        """takes the exponential of squared distances,
        and directions from `emb_i - emb_j`
        """

        # modulus
        adj = gaussian(self.sqdist(emb), self.sigma)
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

    def __init__(self, alpha, radius, periodic=False):
        super(QCDDist, self).__init__()
        self.alpha = alpha
        self.radius = radius
        self.sqdist = ts.sqdist_periodic_ if periodic else ts.sqdist_

    def forward(self, emb):
        sqdist = self.sqdist(emb) / (self.radius ** 2)
        pow_momenta = (2 * self.alpha * emb[:, 4, :].log()).exp()
        min_momenta = ts.sym_min(pow_momenta)
        d_ij = sqdist * min_momenta

        return d_ij


class FixedQCDAware(nn.Module):
    """kernel based on 'QCD-Aware Recursive Neural Networks for Jet Physics'"""

    def __init__(self, alpha, beta, periodic=False, epsilon=1e-7):
        super(FixedQCDAware, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon  # protection against division by 0
        self.sqdist = ts.sqdist_periodic_ if periodic else ts.sqdist_

    def forward(self, emb):
        sqdist = self.sqdist(emb)
        pow_momenta = (2 * self.alpha * emb[:, 4, :].log()).exp()
        min_momenta = ts.sym_min(pow_momenta)
        d_ij_alpha = sqdist * min_momenta

        d_min, _ = pow_momenta.min(1)
        d_min = (d_min + self.epsilon).expand_as(d_ij_alpha)
        d_ij_center = (d_ij_alpha - d_min) / d_min
        d_ij_norm = self.beta * d_ij_center
        w_ij = (-d_ij_norm).exp()

        return w_ij


class QCDAware(nn.Module):
    """kernel based on 'QCD-Aware Recursive Neural Networks for Jet Physics'"""

    def __init__(self, alpha, beta, periodic=False):
        super(QCDAware, self).__init__()

        alpha = Parameter(alpha * (torch.rand(1, 1) * 0.02 + 0.99))
        beta = Parameter(beta * (torch.rand(1, 1, 1) * 0.02 + 0.99))
        self.register_parameter('alpha', alpha)
        self.register_parameter('beta', beta)
        self.alpha.register_hook(ts.HookCheckForNan('NAN in backward alpha', action=print))
        self.beta.register_hook(ts.HookCheckForNan('NAN in backward beta',
                                                   action=print, args=self.beta))

        self.init_momenta_stdev = None
        self.softmax = nn.Softmax()
        self.sqdist = ts.sqdist_periodic_ if periodic else ts.sqdist_

    def forward(self, emb):
        ts.check_for_nan(self.alpha, 'nan in kernel param : alpha', )
        ts.check_for_nan(self.beta, 'nan in kernel param : beta')

        sqdist = self.sqdist(emb)
        ts.check_for_inf(sqdist, 'inf in kernel : sqdist')
        momentum = emb[:, 4, :]

        if self.init_momenta_stdev is None:
            var_m = (momentum ** 2).mean().unsqueeze(1)
            self.init_momenta_stdev = var_m.sqrt()

        momentum /= self.init_momenta_stdev.expand_as(momentum)

        alpha = self.alpha.expand_as(momentum)
        # alpha.register_hook(_hook_reduce_grad(100))
        pow_momenta = (2 * alpha * momentum.log()).exp()
        ts.check_for_nan(pow_momenta, 'NAN in kernel : pow_momenta')
        min_momenta = ts.sym_min(pow_momenta)
        min_momenta.register_hook(ts.HookCheckForNan('NAN in backward min_momenta', action=print))
        d_ij_alpha = sqdist * min_momenta
        d_ij_alpha.register_hook(ts.HookCheckForNan('NAN in backward d_ij_alpha', action=print))
        ts.check_for_nan(d_ij_alpha, 'nan in kernel : d_ij_alpha')

        d_ij_center = _renorm(d_ij_alpha)
        d_ij_center.register_hook(ts.HookCheckForNan('NAN in backward d_ij_center', action=print))
        ts.check_for_nan(d_ij_center, 'nan in kernel : d_ij_center')
        beta = (self.beta ** 2).expand_as(d_ij_center)
        beta.register_hook(ts.HookCheckForNan('NAN in backward beta**2', action=print, args=('d_ij_center : {}'.format(d_ij_center),)))
        # beta.register_hook(_hook_reduce_grad(100))
        d_ij_norm = - beta * d_ij_center
        d_ij_norm.register_hook(ts.HookCheckForNan('NAN in backward d_ij_norm', action=print))
        ts.check_for_nan(d_ij_norm, 'nan in kernel : d_ij_norm')
        w_ij = self._softmax(d_ij_norm)
        # w_ij = d_ij_norm.exp()

        return w_ij

    def _softmax(self, dij):
        batch = dij.size()[0]
        
        dij = torch.unbind(dij, dim=0)
        dij = torch.cat(dij, dim=0)
        
        dij = self.softmax(dij)
        
        dij = torch.chunk(dij, batch, dim=0)
        dij = torch.stack(dij, dim=0)
        
        return dij


class QCDAwareNoNorm(nn.Module):
    """kernel based on 'QCD-Aware Recursive Neural Networks for Jet Physics'"""

    def __init__(self, alpha, beta, periodic=False):
        super(QCDAwareNoNorm, self).__init__()

        alpha = Parameter(alpha * (torch.rand(1, 1) * 0.02 + 0.99))
        beta = Parameter(beta * (torch.rand(1, 1, 1) * 0.02 + 0.99))
        self.register_parameter('alpha', alpha)
        self.register_parameter('beta', beta)

        self.sqdist = ts.sqdist_periodic_ if periodic else ts.sqdist_
        self.softmax = nn.Softmax()

    def forward(self, emb):
        ts.check_for_nan(self.alpha, 'nan in kernel param : alpha')
        ts.check_for_nan(self.beta, 'nan in kernel param : beta')

        sqdist = self.sqdist(emb)
        momentum = emb[:, 4, :] + 1e-10
        alpha = self.alpha.expand_as(momentum)
        # alpha.register_hook(_hook_reduce_grad(100))
        pow_momenta = (2 * alpha * momentum.log()).exp()
        min_momenta = ts.sym_min(pow_momenta)
        d_ij_alpha = sqdist * min_momenta
        ts.check_for_nan(d_ij_alpha, 'nan in kernel : d_ij_alpha')

        beta = (self.beta ** 2).expand_as(d_ij_alpha)
        # beta.register_hook(_hook_reduce_grad(100))
        d_ij_norm = - beta * d_ij_alpha
        ts.check_for_nan(d_ij_norm, 'nan in kernel : d_ij_norm')
        w_ij = self._softmax(d_ij_norm)
        # w_ij = d_ij_norm.exp()

        return w_ij

    def _softmax(self, dij):
        batch = dij.size()[0]
        
        dij = torch.unbind(dij, dim=0)
        dij = torch.cat(dij, dim=0)
        
        dij = self.softmax(dij)
        
        dij = torch.chunk(dij, batch, dim=0)
        dij = torch.stack(dij, dim=0)
        
        return dij


class QCDAwareOld(nn.Module):
    """kernel based on 'QCD-Aware Recursive Neural Networks for Jet Physics'"""

    def __init__(self, alpha, beta, epsilon=1e-5):
        super(QCDAwareOld, self).__init__()
        self.epsilon = epsilon  # protection against division by 0

        alpha = Parameter(alpha * (torch.rand(1, 1) * 0.02 + 0.99))
        beta = Parameter(beta * (torch.rand(1, 1, 1) * 0.02 + 0.99))
        self.register_parameter('alpha', alpha)
        self.register_parameter('beta', beta)

        self.sqdist = ts.sqdist_periodic_ if periodic else ts.sqdist_
        self.softmax = nn.Softmax()

    def forward(self, emb):
        sqdist = self.sqdist(emb)
        momentum = emb[:, 4, :]
        alpha = self.alpha.expand_as(momentum)
        # alpha.register_hook(_hook_reduce_grad(100))
        pow_momenta = (2 * alpha * momentum.log()).exp()
        min_momenta = ts.sym_min(pow_momenta)
        d_ij_alpha = sqdist * min_momenta

        d_ij_center = _renorm(d_ij_alpha)
        d_ij_center = d_ij_alpha
        beta = (self.beta ** 2).expand_as(d_ij_center)
        # beta.register_hook(_hook_reduce_grad(100))
        d_ij_norm = - beta * d_ij_center
        # w_ij = self._softmax(d_ij_norm)
        w_ij = d_ij_norm.exp()

        return w_ij

    def _softmax(self, dij):
        batch = dij.size()[0]
        
        dij = torch.unbind(dij, dim=0)
        dij = torch.cat(dij, dim=0)
        
        dij = self.softmax(dij)
        
        dij = torch.chunk(dij, batch, dim=0)
        dij = torch.stack(dij, dim=0)
        
        return dij