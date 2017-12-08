import torch
from torch.autograd import Variable
from math import pi


def variable_as(tensor1, tensor2):
    """Makes tensor1 a Variable depending on tensor2"""

    if isinstance(tensor2, Variable):
        return Variable(tensor1)
    return tensor1


def cuda_as(tensor1, tensor2):
    """Makes tensor1 cuda depending on tensor2"""

    if tensor2.is_cuda:
        return tensor1.cuda()
    return tensor1


def make_tensor_as(tensor, shape):
    """Initiates a tensor with shape `shape`, that is cuda if `tensor` is too"""

    if tensor.is_cuda:
        return torch.cuda.FloatTensor(*shape)
    return torch.FloatTensor(*shape)


def sym_min(tensor):
    """Given a (batch, n) or (batch, edge_fm, n) tensor `tensor`, returns the tensor of same size
    defined as min(`tensor`, `tensor`^T).
    """
    def _sym_min_no_edge_feature(tensor):
        batch, nb_node = tensor.size()

        tens0 = tensor.unsqueeze(2).expand(batch, nb_node, nb_node)
        tens1 = tens0.transpose(1, 2).contiguous()

        res = torch.stack((tens0, tens1), 3)
        res, _ = res.min(3)
        res = res[:,:,0]#.squeeze(3)
        return res

    def _sym_min_edge_feature(tensor):
        batch, edge_fm, nb_node = tensor.size()

        tens0 = tensor.unsqueeze(3).expand(batch, edge_fm, nb_node, nb_node)
        tens1 = tens0.transpose(2, 3).contiguous()

        res = torch.stack((tens0, tens1), 4)
        res, _ = res.min(4)
        res = res.squeeze(4)
        return res

    if len(tensor.size()) == 2:
        return _sym_min_no_edge_feature(tensor)
    else:
        return _sym_min_edge_feature(tensor)


def sqdist_(emb):
    """Squarred euclidean distance over embedding (phi, eta)"""

    coord = emb[:, 1:3, :]
    batch, _, nb_node = coord.size()
    coord = coord.unsqueeze(3).expand(batch, 2, nb_node, nb_node)
    coord_t = coord.transpose(2, 3)
    diff = coord - coord_t
    sqdist = (diff ** 2).sum(1).squeeze(1)

    return sqdist


def sqdist_periodic_(emb):
    """Squarred euclidean distance over embedding (phi, eta) with
    2pi-periodicity over phi
    """

    sqdist1 = sqdist_(emb)
    emb_pi = torch.cat((emb[:, 0:1, :], (emb[:, 2, :].unsqueeze(1) + pi) % (2 * pi)), 1)
    sqdist2 = sqdist_(emb_pi)
    sqdist = torch.min(sqdist1, sqdist2)

    return sqdist



def spatialnorm(emb):
    """Normalisation layer : each feature map is modified to have
    mean 0 and variance 1.

    input : - emb : Tensor of size (batch, fm, n)
    output : - emb_norm : same as emb, such that each emb[batch, fm, :] has
                mean 0 and variance 1. size (batch, fm, n)
             - avg : Tensor containing the mean of each feature maps from emb.abs
                size (batch, fm, 1)
             - var : Tensor containing the variance of each feature maps from emb
                size (batch, fm, 1)
    """

    avg = emb.mean(2)
    emb_centered = emb - avg.unsqueeze(2).expand_as(emb)

    var = (emb_centered ** 2).mean(2)
    var_protect = (var == 0).type_as(var)
    if isinstance(var, Variable):
        var_protect = var_protect.detach()
    emb_norm = emb_centered / (var.sqrt() + var_protect).unsqueeze(2).expand_as(emb_centered)

    return emb_norm, avg, var


def check_for_nan(tensor, error_message, raise_error=True, action=None, args=None):
    """Checks for NAN values in `tensor`, print `error_message` if there
    are NAN values, and raises ValueError by default
    """

    nb_nan = (tensor != tensor).data.sum()
    if nb_nan > 0:
        print(error_message)
        if action is not None:
            action(*args)
        if raise_error:
            raise ValueError('NAN value in network')


def check_for_inf(tensor, error_message, raise_error=True, action=None, args=None):
    """Checks for +inf values in `tensor`, print `error_message` if there
    are NAN values, and raises ValueError by default
    """

    nb_inf = (tensor == float('+inf')).data.sum()
    if nb_inf > 0:
        print(error_message)
        if action is not None:
            action(*args)
        if raise_error:
            raise ValueError('INF value in network')


class HookCheckForNan:
    """Backward hook, calls `check_for_nan` on gradient during back propagation
    with arguments provided during initialization.
    """

    def __init__(self, error_message, raise_error=True, action=None, args=None):
        try:
           super(HookCheckForNan, self).__init__()
        except:
           print("line 151 in 'tensor.py'")
        self.error_message = error_message
        self.raise_error = raise_error
        self.action = action
        self.args = args

    def __call__(self, grad):
        if self.action is None:
            check_for_nan(grad, self.error_message, self.raise_error)
        elif self.args is None:
            check_for_nan(grad, self.error_message, self.raise_error,
                          action=self.action, args=grad.data)
        else:
            check_for_nan(grad, self.error_message, self.raise_error,
                          action=self.action, args=self.args)

