import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import sigmoid
import operators as op
import graphconv as gc


class GCNN(nn.Module):
    def __init__(self, kernel, operators, frst_fm, fmaps, nb_layer):
        super(GCNN, self).__init__()

        self.operators = operators
        self.nb_op = op.count_operators(operators, kernel)

        self.kernel = kernel
        self.fst_gconv = gc.ResGOpConv(frst_fm, fmaps, self.nb_op)
        # self.bns = nn.ModuleList(
        #     [nn.BatchNorm1d(fmaps)
        #      for _ in range(nb_layer - 1)]
        # )
        self.resgconvs = nn.ModuleList(
            [gc.ResGOpConv(fmaps, fmaps, self.nb_op)
             for _ in range(nb_layer - 1)]
        )
        self.fcl = nn.Linear(fmaps, 1)

    def forward(self, emb_in):

        adj = self.kernel(emb_in)
        ops = op.join_operators(adj, self.operators)

        # apply Graph Convs
        emb = self.fst_gconv(ops, emb_in)

        for resgconv in self.resgconvs:
            emb, _, _ = spatialNorm(emb)
            # emb = self.bns[i](emb)
            emb = resgconv(ops, emb)

        # pool from graph
        emb = emb.mean(2).squeeze(2)
        # emb = emb.sum(2).squeeze(2) / 100
        # emb = emb.max(2)[0].squeeze(2)

        # logistic regression
        emb = self.fcl(emb).squeeze(1)
        emb = sigmoid(emb)

        if (emb != emb).data.sum() > 0:
            print('WARNING : NAN')

        return emb


def spatialNorm(emb, epsilon=1e-5):
    avg = emb.mean(2)
    emb_centered = emb - avg.expand_as(emb)

    var = (emb_centered ** 2).mean(2)
    emb_norm = emb_centered / (var.sqrt().expand_as(emb_centered) + epsilon)

    return emb_norm, avg, var
