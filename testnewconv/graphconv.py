import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.init
from torch.nn.functional import relu
from math import sqrt


class GraphOpConv(nn.Module):
    """Performs graph convolution.
    parameters : - in_fm : number of feature maps in input
                 - out_fm : number of feature maps in output
                 - nb_op : number of graph operators besides identity.abs
                        e.g. x -> a.x * b.Wx has one operator : W

    inputs : - ops : concatenated graph operators, those are being applied
                    to x by right side dot product : x.op
                    shape (batch, nb_node, nb_node * nb_op)
             - emb_in : signal embedding. shape (batch, in_fm, nb_node)

    output : - emb_out : new embedding. shape (batch, out_fm, nb_node)
    """

    def __init__(self, in_fm, out_fm, nb_op):
        super(GraphOpConv, self).__init__()
        invsqrt2 = sqrt(2) / 2

        weight = torch.Tensor(1, out_fm, in_fm * (nb_op + 2))
        nn.init.uniform(weight, -invsqrt2, invsqrt2)
        self.register_parameter('weight', Parameter(weight))

        bias = torch.Tensor(1, out_fm, 1)
        nn.init.uniform(bias, -invsqrt2, invsqrt2)
        self.register_parameter('bias', Parameter(bias))

    def forward(self, ops, emb_in):
        """Defines the computation performed at every call.
        Computes graph convolution with graph operators `ops`,
        on embedding `emb_in`
        """

        batch_size, _, nb_node = emb_in.size()

        avg = emb_in.mean().expand_as(emb_in)
        if ops is None:  # permutation invariant kernel
            spread = (emb_in, avg,)
        else:
            spread = torch.bmm(emb_in, ops)
            # split spreading from different operators, concatenate on feature maps
            spread = spread.split(nb_node, 2)
            spread = (emb_in, avg,) + spread  # identity operator and average are default
        spread = torch.cat(spread, 1)

        # apply weights and bias
        weight, bias = self._resized_params(batch_size, nb_node)
        emb_out = torch.bmm(weight, spread)
        emb_out += bias

        return emb_out

    def _resized_params(self, batch_size, nb_node):
        no_bs_weight_shape = self.weight.size()[1:]
        weight = self.weight.expand(batch_size, *no_bs_weight_shape)

        nb_fm = self.bias.size()[1]
        bias = self.bias.expand(batch_size, nb_fm, nb_node)

        return (weight, bias)


class ResGOpConv(nn.Module):
    def __init__(self, in_fm, out_fm, nb_op):
        super(ResGOpConv, self).__init__()
        
        if out_fm % 2 != 0:
            raise ValueError('ResGOpConv requires event number of output feature maps : {}'.format(out_fm))

        half_out_fm = int(out_fm / 2)
        self.gconv_lin = GraphOpConv(in_fm, half_out_fm, nb_op)
        self.gconv_nlin = GraphOpConv(in_fm, half_out_fm, nb_op)

    def forward(self, ops, emb_in):
        linear = self.gconv_lin(ops, emb_in)

        nlinear = self.gconv_nlin(ops, emb_in)
        nlinear = relu(nlinear)

        emb_out = torch.cat((linear, nlinear), 1)
        return emb_out


class ResGOpConv_SN(nn.Module):
    def __init__(self, in_fm, out_fm, nb_op):
        super(ResGOpConv_SN, self).__init__()

        if out_fm % 2 != 0:
            raise ValueError('ResGOpConv requires event number of output feature maps : {}'.format(out_fm))

        half_out_fm = int(out_fm / 2)
        self.gconv_lin = GraphOpConv(in_fm, half_out_fm, nb_op)
        self.gconv_nlin = GraphOpConv(in_fm, half_out_fm, nb_op)
        self.gn_lin = GraphNorm(half_out_fm)
        self.gn_nlin = GraphNorm(half_out_fm)

    def forward(self, ops, emb_in):
        linear = self.gconv_lin(ops, emb_in)
        linear = self.gn_lin(linear)

        nlinear = self.gconv_nlin(ops, emb_in)
        nlinear = self.gn_nlin(nlinear)
        nlinear = relu(nlinear)

        emb_out = torch.cat((linear, nlinear), 1)
        return emb_out


class GraphNorm(nn.Module):
    def __init__(self, nb_fm, epsilon=1e-5):
        super(GraphNorm, self).__init__()
        self.epsilon = epsilon

        invsqrt2 = sqrt(2) / 2
        alpha = torch.Tensor(1, nb_fm, 1)
        beta = torch.Tensor(1, nb_fm, 1)
        nn.init.uniform(alpha, -invsqrt2, invsqrt2)
        nn.init.uniform(beta, -invsqrt2, invsqrt2)

        self.register_parameter('alpha', Parameter(alpha))
        self.register_parameter('beta', Parameter(beta))

    def forward(self, emb):
        avg = emb.mean(2)
        emb_centered = emb - avg.expand_as(emb)

        var = (emb_centered ** 2).mean(2)
        emb_norm = emb_centered / (var.sqrt().expand_as(emb_centered) + self.epsilon)

        emb_renorm = emb_norm * self.alpha.expand_as(emb_norm)
        emb_recenter = emb_renorm * self.beta.expand_as(emb_norm)

        return emb_recenter
