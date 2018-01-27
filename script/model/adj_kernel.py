from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
from torch.nn import Parameter

class Identity(nn.Module):

   def __init__(self,fmaps):
      super(Identity, self).__init__()

   def forward(self, adj_in, emb_in):
      return adj_in

class Gaussian(nn.Module):
   def __init__(self,fmaps,sigma=2.0):
      super(Gaussian, self).__init__()
      self.sigma = sigma

   def forward(self, adj_in, emb_in):
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

class DirectedGaussian(nn.Module):
   def __init__(self,fmaps,theta=0.67,sigma=1):
      super(DirectedGaussian, self).__init__()
      self.gauss_ker = Gaussian(sigma)
      theta = torch.FloatTensor([theta])
      self.register_parameter('theta', Parameter(theta))

   def forward(self, adj_in, emb_in):
      batch, fmap, nb_node = emb_in.size()
      sum_weights = adj_in.sum(dim=1)
      self_edges = adj_in[0].diag().unsqueeze(0)
      sum_weights -= self_edges
      # Normalize
      sum_weights /= adj_in.size()[1]-1
      # Expand to match adj size
      sum_weights = sum_weights.expand(batch, nb_node, nb_node)
      # Transpose so row is uniform
      sum_weights = sum_weights.transpose(1,2)
      adj = self.gauss_ker(adj_in, emb_in)
      adj = self.theta*adj + (1-self.theta) * sum_weights
      

      return adj

class MPNNdirected(nn.Module):
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

class MLPdirected(nn.Module):
   def __init__(self,fmaps,nb_hidden=8):
      super(MLPdirected, self).__init__()
      self.fmaps = fmaps
      self.layer1 = nn.Linear(2*fmaps+1, nb_hidden)
      self.layer2 = nn.Linear(nb_hidden,8)

   def forward(self, adj_in, emb_in):
      batch, fmap, nb_node = emb_in.size()

      if adj_in.is_cuda:
         adj = torch.cuda.FloatTensor(batch,nb_node,nb_node).zero_()
         net_input = torch.cuda.FloatTensor(nb_node*nb_node,2*fmap+1).zero_()
      else:
         adj = torch.FloatTensor(batch,nb_node,nb_node).zero_()
         net_input = torch.cuda.FloatTensor(nb_node*nb_node,2*fmap+1).zero_()

      adj = Variable(adj)
      net_input = Variable(net_input)
      # Define an adjacency matrix for every
      # sample in the batch
      for i in range(batch):
        # Calculate sum weights
        # to be used as input of MLP
        sum_weights = adj_in.sum(dim=1)
        self_edges = adj_in[i].diag().unsqueeze(0)
        sum_weights -= self_edges
        # Normalize
        sum_weights /= adj_in.size()[1]-1
        sum_weights = sum_weights.squeeze()
        # Create samples from emb_in s.t. an MLP will create
        # edges for every j,k node pair
        sample = emb_in[i]
        for j in range(nb_node):
          idx = j*nb_node
          # Put net inputs to correct shape
          fst_sample = sample[:,j].expand(nb_node,fmap)
          sec_sample = sample.transpose(0,1)
          incoming_edge_weight = sum_weights
          # Place net inputs into input tensor
          net_input[idx:idx+nb_node,:fmap] = fst_sample
          net_input[idx:idx+nb_node,fmap:2*fmap] = sec_sample
          net_input[idx:idx+nb_node,2*fmap] = incoming_edge_weight
        # MLP is applied to every j,k vertex pair
        # to calculate new j,k edge weights
        edge_out = self.layer1(net_input)
        edge_out = functional.relu(edge_out)
        edge_out = self.layer2(edge_out)
        # Copy output of MLP into adj matrix
        # for every j,k pair
        for j in range(nb_node):
          adj[i,j] = edge_out[j*nb_node:nb_node*(j+1),0]
      return adj

