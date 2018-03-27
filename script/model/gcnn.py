import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import sigmoid

from model import operators as op
from model import graphconv as gc
from utils.tensor import spatialnorm, check_for_nan, mean_with_padding, mask_embedding
import utils.tensor as ts


class GCNNSingleKernel(nn.Module):
    """Simple graph neural network : multiple layers of residual
    graph convolutions, normalization and Relus, followed by a
    logistic regression.
    """

    def __init__(self, kernels, combine_kernels, frst_fm, fmaps, nb_layer):
        super(GCNNSingleKernel, self).__init__()

        # self.operators = [op.degree, op.adjacency, op.adjacency_transpose]
        self.operators = [op.degree, op.adjacency]
        self.nb_op = len(self.operators)

        # Kernels
        self.kernels = kernels
        self.combine_kernels = combine_kernels

        # Graph convolutions
        self.fst_gconv = gc.ResGOpConv(frst_fm, fmaps, self.nb_op)
        self.resgconvs = nn.ModuleList(
            [gc.ResGOpConv(fmaps, fmaps, self.nb_op)
            for _ in range(nb_layer - 1)]
            )
        self.instance_norm = nn.InstanceNorm1d(1)

        # Logistic last layer
        self.fcl = nn.Linear(fmaps, 1)

    def forward(self, emb_in, adj_mask, batch_nb_nodes, plotting=None):
        batch_size, fmap, nb_pts  = emb_in.size()
        # Create dummy first adjacency matrix
        adj = Variable(torch.ones(batch_size, nb_pts, nb_pts))
        if emb_in.is_cuda:
          adj = adj.cuda()

        # Get adjacency matrices for layer 1
        adj_matrices = [kernel(
                               adj,
                               emb_in, 
                               layer=0, 
                               mask=adj_mask, 
                               batch_nb_nodes=batch_nb_nodes
                               ) for kernel in self.kernels[0]]
        adj = self.combine_kernels[0](adj, adj_matrices)
        # Mask adjacency matrix if batch size greater than 1
        if batch_size != 1:
          adj = torch.mul(adj, adj_mask)

        # Plot sample
        if plotting is not None:
          plotting.plot_graph(
                              emb_in[0].cpu().t().data.numpy(), 
                              adj[0].cpu().data.numpy(),
                              0
                              )

        # Join operators with adjacency matrix
        operators = gc.join_operators(adj, self.operators)

        # Apply first layer Graph Conv
        emb = self.fst_gconv(operators, emb_in, batch_nb_nodes, adj_mask)

        # Apply remaining Graph Convs
        for i, resgconv in enumerate(self.resgconvs):
            layer_idx = i+1
            # Update adjacency matrices at each layer
            # Note this can be set to perform no update
            adj_matrices = [kernel.update(
                                          adj, 
                                          emb,
                                          layer=layer_idx, 
                                          mask=adj_mask,
                                          batch_nb_nodes=batch_nb_nodes
                                          ) for kernel in self.kernels[layer_idx]]
            adj = self.combine_kernels[layer_idx](adj, adj_matrices)
            # Apply mask only if batch size not 1
            if batch_size != 1:
              adj = torch.mul(adj, adj_mask)
            # Join operators with new adjacency matrix
            operators = gc.join_operators(adj, self.operators)
            # Plot updated representation
            if plotting is not None:
              plotting.plot_graph(
                                  emb[0].cpu().t().data.numpy(), 
                                  adj[0].cpu().data.numpy(),
                                  layer_idx
                                  )
            # Apply graph convolution
            emb, _, _ = spatialnorm(emb, batch_nb_nodes, adj_mask)
            # Mask embedding for numerical stability
            # Otherwise can get nan / inf errors
            emb = mask_embedding(emb, adj_mask)
            emb = resgconv(operators, emb, batch_nb_nodes, adj_mask)

        # collapse graph into a single representation 
        #    (uncomment for different options)
        emb = mask_embedding(emb, adj_mask)
        emb = emb.sum(2)#.squeeze(2)
        # emb = emb.max(2)[0]
        '''
        emb = emb.sum(2)#.squeeze(2)
        # Get mean of emb, accounting for zero padding of batches
        batch_div_for_mean = batch_nb_nodes.unsqueeze(1).repeat(1,emb.size()[1])
        emb = emb/batch_div_for_mean
        '''
        # Normalize for FCL
        emb = self.instance_norm(emb.unsqueeze(1)).squeeze(1)

        # Logistic regression
        emb = self.fcl(emb).squeeze(1)
        emb = sigmoid(emb)

        return emb


