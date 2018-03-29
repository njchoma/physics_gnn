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

class DistMult(Adj_Kernel):
  def __init__(self,fmap,*args,sparse=None,**kwargs):
    super(DistMult, self).__init__(*args,sparse,**kwargs)
    self.matrix = nn.Parameter(torch.zeros(fmap,fmap))

  def forward(self, adj_in, emb_in, layer, *args, mask=None, **kwargs):
    adj = torch.matmul(emb_in.transpose(1,2), torch.matmul(self.matrix, emb_in))
    adj = _softmax_with_padding(adj, mask)
    self._save_adj(adj)
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


def _softmax_with_padding(adj_in, mask):
  S = functional.softmax(adj_in)
  if mask is not None:
    S = S * mask
  E = S.sum(2,keepdim=True) + 10**-10
  return S / E


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

    def forward(self, adj_in, emb, *args, mask = None, **kwargs):
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

        adj = self._apply_norm(adj, mask)
        self.save_adj(adj)
        return adj

class GaussianSoftmax(Gaussian):
  def __init__(self, *args, diag=True, norm=False, periodic=False,spatial_coords=None, **kwargs):
    super(GaussianSoftmax, self).__init__(*args, diag=diag, norm=norm, periodic=periodic,spatial_coords=spatial_coords, **kwargs)

  def _apply_norm(self, adj, mask):
    return _softmax_with_padding(adj, mask)

    
