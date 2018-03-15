from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
from torch.nn import Parameter

from model import sparse


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
  n_repeats, dim2 = tensor.size()[0:2]
  a = tensor.repeat(1,n_repeats)
  a = a.resize(n_repeats*n_repeats,dim2)
  b = tensor.repeat(n_repeats,1)
  return torch.cat((a,b),1)

def _get_adj(tensor,adj_size):
  '''
  If tensor is 4x1 tensor
  [[11], [12], [21], [22]]
  returns 2x2 tensor
  [[11 12],
   [21 22]]
  '''
  return tensor.resize(adj_size, adj_size)


def make_samples(emb_in,sum_weights,idx):
  '''
  Assumes emb_in is of shape nb_nodes x fmap
  '''
  nb_node = emb_in.size()[0]
  sample = cartesian(emb_in)
  weights = sum_weights.resize(nb_node,1).repeat(nb_node,1)
  sample = torch.cat((sample,weights),1)
  return sample

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
  

# Abstract adjacency kernel class
class Adj_Kernel(nn.Module):
  def __init__(self,sparse=None):
    super(Adj_Kernel,self).__init__()
    self.sparse = sparse

  def forward(self, args, **kwargs):
    raise Exception("Must be implemented by child class")
  
  def update(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

# Abstract adjacency kernel class
# which keeps reuses first layer adj matrix
class Adj_Kernel_fixed_update(Adj_Kernel):
  def __init__(self):
    super(Adj_Kernel_fixed_update,self).__init__()

  def save_adj(self, adj_in):
    self.adj_matrix = adj_in

  def update(self, *args, **kwargs):
    return self.adj_matrix


class Gaussian(Adj_Kernel):
   def __init__(self,fmaps,sparse=None,sigma=2.0):
      super(Gaussian, self).__init__()
      self.sigma = sigma

   def forward(self, adj_in, emb_in,idx):
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
   def __init__(self,fmaps,nb_hidden,sparse=None):
      super(MLPdirected, self).__init__(sparse)
      self.fmaps = fmaps
      self.layer1 = nn.Linear(2*fmaps+1, nb_hidden)
      self.layer2 = nn.Linear(nb_hidden,1)

   def forward(self, adj_in, emb_in, idx=None):
      batch, fmap, nb_node = emb_in.size()

      adj_list = []
      # Define an adjacency matrix for every
      # sample in the batch
      for i in range(batch):
        # Create samples from emb_in s.t. an MLP will create
        # edges for every j,k node pair
        sum_weights = get_summed_weights(adj_in[i])
        sample = make_samples(emb_in[i].transpose(0,1),sum_weights,idx)
        # MLP is applied to every j,k vertex pair
        # to calculate new j,k edge weights
        edge_out = self.layer1(sample)
        edge_out = functional.relu(edge_out)
        edge_out = self.layer2(edge_out)
        # Copy output of MLP into adj matrix
        # for every j,k pair as previously defined in sparse class
        adj_list.append(_get_adj(edge_out,nb_node).unsqueeze(0))
      # Apply sigmoid to normalize edge weights
      adj = torch.cat(tuple(adj_list),0)
      adj = functional.sigmoid(adj)
      return adj

class Identity(Adj_Kernel_fixed_update):
  def __init__(self, *args, **kwargs):
    super(Identity, self).__init__()

  def forward(self, adj_in, emb_in,idx):
    batches, features, nodes = emb_in.size()
    ones = torch.ones(nodes)
    if emb_in.is_cuda:
      ones = ones.cuda()
    identity = ones.diag().expand(batches, nodes, nodes)
    identity = Variable(identity)
    self.save_adj(identity)
    return identity
