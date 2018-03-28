from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
from torch.nn import Parameter

from model.kernels import sparse


def cartesian(tensor):
  '''
  Computes cartesian product.
  If tensor is:
  [[0 1 2],
   [3 4 5]]
  Then a = [[0 1 2],   and b = [[0 1 2],
            [0 1 2],            [3,4,5],
            [3,4,5],            [0 1 2],
            [3 4 5]]            [3 4 5]]
  and output is
  [[0 1 2 0 1 2],
   [0 1 2 3 4 5],
   [3 4 5 0 1 2],
   [3 4 5 3 4 5]]
  '''
  batch, nb_node, fmap = tensor.size()
  a = tensor.repeat(1,1,nb_node).resize(batch, nb_node*nb_node, fmap)
  b = tensor.repeat(1,nb_node,1)
  return torch.cat((a,b),2)

def _batch_unbind(adj):
  adj_unbind = torch.unbind(adj, dim=0)
  return torch.cat(adj_unbind, dim=0)


def _batch_bind(adj_unbind, nb_batch):
  adj = torch.chunk(adj_unbind, nb_batch, dim=0)
  return torch.stack(adj, dim=0)


def get_summed_weights(adj_in):
  # Assume adj_in is shape nb_node x nb_node
  # Calculate sum weights
  # to be used as input of MLP
  sum_weights = adj_in.sum(dim=0)
  self_edges = adj_in.diag()
  sum_weights -= self_edges
  # Normalize
  sum_weights /= adj_in.size()[0]-1
  sum_weights = sum_weights.squeeze()
  return sum_weights
  

def _save_adj_no_save(*args, **kwargs):
  pass


# Abstract adjacency kernel class
class Adj_Kernel(nn.Module):
  def __init__(self,*args,sparse=None,layerwise=False,**kwargs):
    super(Adj_Kernel,self).__init__()
    self.sparse = sparse
    if layerwise:
      self.save_adj = _save_adj_no_save
      self.update = self.forward
    else:
      self.save_adj = self._save_adj

  def _save_adj(self, adj_in):
    self.adj_matrix = adj_in

  def forward(self, *args, **kwargs):
    raise Exception("Must be implemented by child class")
  
  def update(self, *args, **kwargs):
    return self.adj_matrix

class Gaussian_prev(Adj_Kernel):
   def __init__(self,fmaps,sparse=None,sigma=2.0):
      super(Gaussian, self).__init__()
      self.sigma = sigma

   def forward(self, adj_in, emb_in,*args, **kwargs):
      batch, fmap, nb_node = emb_in.size()
      # Normalize input coordinates
      coord = emb_in.div(emb_in.std())
      # Expand and transpose coords
      coord = coord.unsqueeze(3).expand(batch,fmap,nb_node,nb_node)
      coord_t = coord.transpose(2,3)

      # Apply gaussian kernel to adj
      adj = coord-coord_t
      adj = adj**2
      adj = adj.mean(1)
      adj = torch.exp(-adj.div(self.sigma))
      return adj

class DirectedGaussian(Adj_Kernel):
   def __init__(self,fmaps,theta=0.67,sigma=1,sparse=None):
      super(DirectedGaussian, self).__init__()
      self.gauss_ker = Gaussian(sigma)
      theta = torch.FloatTensor([theta])
      self.register_parameter('theta', Parameter(theta))
      print("WARNING: Sparse not yet implemented for DirectedGaussian")

   def forward(self, adj_in, emb_in,idx):
      batch, fmap, nb_node = emb_in.size()
      sum_weights = []
      for i in range(batch):
        sum_weights.append(get_summed_weights(adj_in[0]).unsqueeze(0))
      sum_weights = torch.cat(sum_weights,0)
      # Expand to match adj size
      sum_weights = sum_weights.expand(batch, nb_node, nb_node)
      # Transpose so row is uniform
      sum_weights = sum_weights.transpose(1,2)
      adj = self.gauss_ker(adj_in, emb_in)
      adj = self.theta*adj + (1-self.theta) * sum_weights

      return adj

class MPNNdirected(Adj_Kernel):
   # Inspired by AdjacencyMatrix in Neural Message Passing for Jet Physics
   def __init__(self,fmaps):
      super(MPNNdirected, self).__init__()
      self.fmaps = fmaps
      v = torch.FloatTensor(torch.rand(fmaps))
      b = torch.FloatTensor(torch.rand(1))
      self.register_parameter('v', Parameter(v))
      self.register_parameter('b', Parameter(b))

   def forward(self, adj_in, emb_in):
      batch, fmap, nb_node = emb_in.size()
      # Normalize input coordinates
      coord = emb_in.div(emb_in.std())
      # Expand and transpose coords
      coord = coord.unsqueeze(3).expand(batch,fmap,nb_node,nb_node)
      coord_t = coord.transpose(2,3)

      # Apply sigmoid(v^T(h+h')+b) kernel to adj
      coord = coord+coord_t
      v = self.v.expand(batch,1,fmap)
      b = self.b.expand(batch,nb_node)

      if adj_in.is_cuda:
         adj = torch.cuda.FloatTensor(batch,nb_node,nb_node).zero_()
      else:
         adj = torch.FloatTensor(batch,nb_node,nb_node).zero_()
      adj = Variable(adj)

      for i in range(nb_node):
         to_mult = coord[:,:,:,i]
         adj[:,:,i] = torch.bmm(v,to_mult).squeeze(1)+b

      adj = functional.sigmoid(adj)
      return adj


class MLPdirected(Adj_Kernel):
   def __init__(self,fmaps,nb_hidden, *args,sparse=None, **kwargs):
      super(MLPdirected, self).__init__(sparse, *args, **kwargs)
      self.fmaps = fmaps
      self.layer1 = nn.Linear(2*fmaps, nb_hidden)
      self.layer2 = nn.Linear(nb_hidden,1)

   def forward(self, adj_in, emb_in, layer, *args, **kwargs):
      batch, fmap, nb_node = emb_in.size()
      # Convert out of batches
      emb_cartesian = cartesian(emb_in.transpose(1,2))
      # Apply MLP
      edge_out = self.layer1(_batch_unbind(emb_cartesian))
      edge_out = functional.relu(edge_out)
      edge_out = self.layer2(edge_out).resize(batch*nb_node, nb_node)
      edge_out = functional.sigmoid(edge_out)
      # Return to batch sizes
      adj = _batch_bind(edge_out, batch)
      self.save_adj(adj)
      return adj

class Identity(Adj_Kernel):
  def __init__(self, *args, **kwargs):
    super(Identity, self).__init__()

  def forward(self, adj_in, emb_in,*args, **kwargs):
    batches, features, nodes = emb_in.size()
    ones = torch.ones(nodes)
    if emb_in.is_cuda:
      ones = ones.cuda()
    identity = ones.diag().expand(batches, nodes, nodes)
    identity = Variable(identity)
    self.save_adj(identity)
    return identity

################
# Gaspar Kernels
################
import utils.tensor as ts

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


'''
def _stochastich(adj):
  deg = adj.sum(2)
  adj /= deg.expand_as(adj)

  return adj
'''

def _softmax(dij, _softmax_fct):
  batch = dij.size()[0]

  dij = torch.unbind(dij, dim=0)
  dij = torch.cat(dij, dim=0)

  dij = _softmax_fct(dij)

  dij = torch.chunk(dij, batch, dim=0)
  dij = torch.stack(dij, dim=0)

  return dij

def _softmax_with_padding(adj_in, batch_nb_nodes):
  exp = adj_in.exp()
  summed_exp = exp.sum(2)
  # Remove padded nodes from sum
  padding_correction = adj_in.size()[1]-batch_nb_nodes
  padding_correction = padding_correction.unsqueeze(1).expand_as(summed_exp)
  summed_exp = summed_exp-torch.mul(padding_correction, exp[:,-1])
  # Apply softmax
  return exp / (summed_exp+10**-20).unsqueeze(2).expand_as(exp)


class Gaussian(Adj_Kernel):
    """Gaussian kernel"""
    def __init__(self, *args, diag=True, norm=False, periodic=False, spatial_coords=None, **kwargs):
        super(Gaussian, self).__init__(*args, **kwargs)
        sigma = Parameter(torch.rand(1) * 0.02 + 0.99)
        self.register_parameter('sigma', sigma)  # Uniform on [0.9, 1.1]
        self.diag = diag
        self.sqdist = ts.sqdist_periodic_ if periodic else ts.sqdist_
        self.periodic = periodic
        self.spatial_coords = spatial_coords
        if spatial_coords is not None:
          print("Spatial coords: {}".format(spatial_coords))
        else:
          print("Full gaussian kernel")

    def _apply_norm(self, adj, batch_nb_nodes):
      return adj

    def forward(self, adj_in, emb, *args, batch_nb_nodes = None, **kwargs):
        """takes the exponential of squared distances"""
        batch, fmap, nb_node = emb.size()

        if self.periodic:
          adj = gaussian(self.sqdist(emb), self.sigma)
        else:
          if self.spatial_coords is not None:
            coord = emb[:,self.spatial_coords]
            fmap = len(self.spatial_coords)
          else:
            coord = emb
          # Expand and transpose coords
          coord = coord.unsqueeze(3).expand(batch,fmap,nb_node,nb_node)
          coord_t = coord.transpose(2,3)

          # Apply gaussian kernel to adj
          adj = coord-coord_t
          adj = adj**2
          adj = adj.mean(1)
          adj = torch.exp(-adj.div(self.sigma**2))
          
        if not self.diag:
            adj = _delete_diag(adj)

        adj = self._apply_norm(adj, batch_nb_nodes)
        self.save_adj(adj)
        return adj

class GaussianSoftmax(Gaussian):
  def __init__(self, *args, diag=True, norm=False, periodic=False,spatial_coords=None, **kwargs):
    super(GaussianSoftmax, self).__init__(*args, diag=diag, norm=norm, periodic=periodic,spatial_coords=spatial_coords, **kwargs)

  def _apply_norm(self, adj, batch_nb_nodes):
    return _softmax_with_padding(adj, batch_nb_nodes)

    
