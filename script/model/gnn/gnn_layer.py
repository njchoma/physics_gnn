import logging
import torch
import torch.nn as nn

from model.gnn import graphconv as gc
from model.node_update import get_node_update
from model.utils import graph_operators as graph_ops
from utils.tensor import spatialnorm, mask_embedding

class GNN_Layer(nn.Module):
  def __init__(self,kernels,combine_kernels,fmap_in,fmap_out,layer_nb):
    super(GNN_Layer, self).__init__()

    # Save inputs
    self.kernels = kernels
    self.combine_kernels = combine_kernels
    self.fmap_in = fmap_in
    self.fmap_out = fmap_out
    self.layer_nb = layer_nb

    # Define operators
    self.operators = [graph_ops.degree, graph_ops.adjacency]
    self.nb_op = len(self.operators)

    # Define normalization and convolution
    self.spatial_norm = spatialnorm
    self.convolution = gc.get_convolution_layer(fmap_in, fmap_out, self.nb_op)

    # Define method for updating nodes
    self.node_update = get_node_update(fmap_in, fmap_out)

  def forward(self, emb_in, adj_in, adj_mask, batch_nb_nodes):
    # Update adjacency matrix
    adj_matrices = []
    for kernel in self.kernels:
      if self.layer_nb == 0:
        ker_update = kernel.forward
      else:
        ker_update = kernel.update
      adj_matrices.append(ker_update(
                                    adj_in, 
                                    emb_in,
                                    layer=self.layer_nb, 
                                    mask=adj_mask,
                                    batch_nb_nodes=batch_nb_nodes
                                    ))
    adj = self.combine_kernels(adj_in, adj_matrices)
    # Apply mask only if batch size not 1
    if emb_in.size()[0] != 1:
      adj = torch.mul(adj, adj_mask)

    # Join operators with adjacency matrix
    operators = gc.join_operators(adj, self.operators)

    # Apply spatial norm
    if self.layer_nb != 0:
      emb_update, _, _ = spatialnorm(emb_in, batch_nb_nodes, adj_mask)
    else:
      emb_update = emb_in
    # Mask embedding for numerical stability
    # Otherwise can get nan / inf errors
    emb_update = mask_embedding(emb_update, adj_mask)
    # Apply convolution
    emb_update = self.convolution(operators, emb_update, batch_nb_nodes, adj_mask)

    # Apply node update
    emb = self.node_update(emb_in, emb_update)

    # Return embedding and updated adjacency matrix
    return emb, adj
