import logging
import torch
import torch.nn as nn
from torch.nn.functional import sigmoid

from model import operators as op
from model import multi_operators as ops
from model import graphconv as gc
from utils.tensor import spatialnorm, check_for_nan


class GCNNSingleKernel(nn.Module):
    """Simple graph neural network : multiple layers of residual
    graph convolutions, normalization and Relus, followed by a
    logistic regression.
    """

    def __init__(self, kernel, adj_kernels, sparse, frst_fm, fmaps, nb_layer):
        super(GCNNSingleKernel, self).__init__()

        # self.operators = [op.degree, op.adjacency, op.adjacency_transpose]
        self.operators = [op.degree, op.adjacency]
        self.nb_op = len(self.operators)

        self.kernel = kernel
        self.fst_gconv = gc.ResGOpConv(frst_fm, fmaps, self.nb_op)

        self.sparse = sparse
        self.adj_kernels = nn.ModuleList(adj_kernels)

        self.resgconvs = nn.ModuleList(
            [gc.ResGOpConv(fmaps, fmaps, self.nb_op)
             for _ in range(nb_layer - 1)]
        )
        self.instance_norm = nn.InstanceNorm1d(1)
        self.fcl = nn.Linear(fmaps, 1)

    def forward(self, emb_in, plotting=None):
        
        # initiate operator
        adj = self.kernel(emb_in)
        check_for_nan(adj, 'NAN in operators')

        # Plot sample
        if plotting is not None:
          plotting[0].plot_graph(emb_in.cpu().squeeze().t().data.numpy(), adj.cpu().squeeze().data.numpy(),0)

        operators = gc.join_operators(adj, self.operators)

        # apply Graph Conv
        emb = self.fst_gconv(operators, emb_in)

        # set sparsity
        sparse_idx = self.sparse.get_indices(emb[0].transpose(0,1))

        # Apply remaining Graph Convs
        for i, resgconv in enumerate(self.resgconvs):
            # Apply message passing to adjacency matrix
            adj = self.adj_kernels[i](adj, emb, sparse_idx)
            operators = gc.join_operators(adj, self.operators)
            # Plot updated representation
            if plotting is not None:
              plotting[i+1].plot_graph(emb.cpu().squeeze().t().data.numpy(), adj.cpu().squeeze().data.numpy(),i+1)
            # Apply graph convolution
            emb, _, _ = spatialnorm(emb)
            emb = resgconv(operators, emb)

        # collapse graph into a single representation (uncomment for different options)
        emb = emb.mean(2)#.squeeze(2)
        # emb = emb.sum(2)#.squeeze(2)
        # emb = emb.max(2)[0].squeeze(2)
        emb = self.instance_norm(emb.unsqueeze(1)).squeeze(1)
        check_for_nan(emb, 'nan coming from instance_norm')

        # logistic regression
        emb = self.fcl(emb).squeeze(1)
        emb = sigmoid(emb)
        check_for_nan(emb, 'nan coming from logistic regression')

        if (emb != emb).data.sum() > 0:
            print('WARNING : NAN')

        return emb


class GCNNLayerKernel(nn.Module):
    """Same as GCNNSingleKernel, but using multiple kernels at the same time.
    The kernel used is 'Node2Edge'.
    """

    def __init__(self, kernel_fun, input_fm, node_fm, edge_fm, nb_layer, periodic=False):
        super(GCNNLayerKernel, self).__init__()

        self.operators = [ops.degree, ops.adjacency]
        self.nb_op = edge_fm * len(self.operators)  # ops on operators use adjacency

        self.instance_norm_in = nn.InstanceNorm1d(input_fm)
        self.kernel = kernel_fun(edge_fm, periodic)
        self.fst_resgconv = gc.ResGOpConv(input_fm, node_fm, self.nb_op)

        self.resgconvs = nn.ModuleList(
            [gc.ResGOpConv(node_fm, node_fm, self.nb_op)
             for _ in range(nb_layer - 1)]
        )

        self.instance_norm_out = nn.InstanceNorm1d(1)
        self.fcl = nn.Linear(node_fm, 1)

    def forward(self, global_input):

        kernel = self.kernel(global_input)
        if (kernel != kernel).data.sum() > 0:
            print('NAN in kernel')
            assert False
        operators = gc.join_operators(kernel, self.operators)
        emb = self.instance_norm_in(global_input)
        emb = self.fst_resgconv(operators, emb)

        for resgconv in self.resgconvs:
            emb, _, _ = spatialnorm(emb)
            emb = resgconv(operators, emb)

        emb = emb.mean(2,keepdim=True).squeeze(2)
        emb = self.instance_norm_out(emb.unsqueeze(1)).squeeze(1)

        # logistic regression
        emb = self.fcl(emb).squeeze(1)
        emb = sigmoid(emb)

        if (emb != emb).data.sum() > 0:
            print('WARNING : NAN')
        return emb


class GCNNMultiKernel(nn.Module):
    """Same as GCNNLayerKernel, but using a different set of kernels at
    each layer.
    """

    def __init__(self, kernel_fun, input_fm, node_fm, edge_fm, nb_layer, periodic=False):
        super(GCNNMultiKernel, self).__init__()

        self.operators = [ops.degree, ops.adjacency]
        self.nb_op = edge_fm * len(self.operators)  # ops on operators use adjacency

        self.fst_kernel = kernel_fun(edge_fm, periodic)
        self.fst_resgconv = gc.ResGOpConv(input_fm, node_fm, self.nb_op)

        self.kernels = nn.ModuleList(
            [kernel_fun(edge_fm, periodic)
             for _ in range(nb_layer - 1)]
        )
        self.resgconvs = nn.ModuleList(
            [gc.ResGOpConv(node_fm, node_fm, self.nb_op)
             for _ in range(nb_layer - 1)]
        )

        self.instance_norm = nn.InstanceNorm1d(1)
        self.fcl = nn.Linear(node_fm, 1)

    def forward(self, global_input):

        kernel = self.fst_kernel(global_input)
        operators = gc.join_operators(kernel, self.operators)
        emb = self.fst_resgconv(operators, global_input)

        for i, resgconv in enumerate(self.resgconvs):
            emb, _, _ = spatialnorm(emb)
            kernel = self.kernels[i](global_input)
            if (kernel != kernel).data.sum() > 0:
                print('NAN at index {}'.format(i))
                assert False
            operators = gc.join_operators(kernel, self.operators)
            emb = resgconv(operators, emb)

        emb = emb.mean(2,keepdim=True).squeeze(2).unsqueeze(1)
        emb = self.instance_norm(emb).squeeze(1)

        # # logistic regression
        emb = self.fcl(emb).squeeze(1)
        emb = sigmoid(emb)

        if (emb != emb).data.sum() > 0:
            print('WARNING : NAN')
        return emb


class GCNNEdgeFeature(nn.Module):
    """Same as GCNNMultiKernel, but using a fully learnt kernel.
    The kernel used can be 'Node2Edge' or 'GatedNode2Edge'.
    """

    def __init__(self, kernel, input_fm, node_fm, edge_fm, nb_layer):
        super(GCNNEdgeFeature, self).__init__()

        self.operators = [ops.degree, ops.adjacency]
        self.nb_op = edge_fm * len(self.operators)  # ops on operators use adjacency

        self.fst_kernel = kernel(input_fm, edge_fm)
        self.fst_resgconv = gc.ResGOpConv(input_fm, node_fm, self.nb_op)

        self.kernels = nn.ModuleList(
            [kernel(input_fm + node_fm, edge_fm)
             for _ in range(nb_layer - 1)]
        )
        self.resgconvs = nn.ModuleList(
            [gc.ResGOpConv(input_fm + node_fm, node_fm, self.nb_op)
             for _ in range(nb_layer - 1)]
        )

        self.instance_norm = nn.InstanceNorm1d(1)
        self.fcl = nn.Linear(node_fm, 1)

    def forward(self, global_input):

        kernel = self.fst_kernel(global_input)
        operators = gc.join_operators(kernel, self.operators)
        emb = self.fst_resgconv(operators, global_input)

        for i, resgconv in enumerate(self.resgconvs):
            emb, _, _ = spatialnorm(emb)
            emb = torch.cat((emb, global_input), dim=1)  # concat (h, x)
            kernel = self.kernels[i](emb)
            operators = gc.join_operators(kernel, self.operators)
            emb = resgconv(operators, emb)

        emb = emb.mean(2).squeeze(2).unsqueeze(1)
        emb = self.instance_norm(emb).squeeze(1)

        # # logistic regression
        emb = self.fcl(emb).squeeze(1)
        emb = sigmoid(emb)

        if (emb != emb).data.sum() > 0:
            print('WARNING : NAN')
        return emb
