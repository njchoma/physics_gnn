from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

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

class No_sparsity(nn.Module):
  def __init__(self):
    super(No_sparsity,self).__init__()

  def get_indices(self,emb_in):
    return None

  def make_samples(self,emb_in,sum_weights,idx):
    '''
    Assumes emb_in is of shape nb_nodes x fmap
    '''
    self.nb_node = emb_in.size()[0]
    sample = cartesian(emb_in)
    weights = sum_weights.resize(self.nb_node,1).repeat(self.nb_node,1)
    sample = torch.cat((sample,weights),1)
    return sample

  def get_adj(self,edges):
    return _get_adj(edges,self.nb_node)
    

class KNN(nn.Module):
  def __init__(self,nb_sparse):
    super(KNN,self).__init__()
    self.nb_sparse = nb_sparse
    self.neighbors = NearestNeighbors(nb_sparse)

  def get_indices(self,emb_in):
    '''
    Returns indices of nearest neighbors of emb_in
    '''
    self.nb_node = emb_in.size()[0]
    self.k = min(self.nb_node, self.nb_sparse)
    self.neighbors.fit(emb_in.cpu().data.numpy())
    idx = self.neighbors.kneighbors(n_neighbors=self.k-1,return_distance=False)
    return idx

  def make_samples(self,emb_in,sum_weights,idx_in):
    '''
    Assumes emb_in is size nb_nodes x fmap
    '''
    self.nb_node = emb_in.size()[0]
    self.k = idx_in.shape[1]+1
    idx = np.concatenate((idx_in,np.arange(self.nb_node,dtype=int).reshape(self.nb_node,1)),1)
    idx = torch.LongTensor(idx)
    if emb_in.is_cuda:
      idx = idx.cuda()
    self.idx = idx
    all_samples = []
    for i in range(self.nb_node):
      node = emb_in[i:i+1].repeat(self.k,1)
      neighbors = emb_in[idx[i]]
      sample = torch.cat((node,neighbors),1)
      edge_wts = sum_weights[idx[i]].resize(self.k,1)
      sample = torch.cat((sample,edge_wts),1)
      all_samples.append(sample)
    all_samples = torch.cat(all_samples,0)
    return all_samples

  def get_adj(self,edge_out):
    edge_out = edge_out.resize(self.nb_node,self.k)
    if edge_out.is_cuda:
       adj = torch.cuda.FloatTensor(self.nb_node,self.nb_node).zero_()
       line= torch.cuda.FloatTensor(self.nb_node).zero_()
    else:
       adj = torch.FloatTensor(self.nb_node,self.nb_node).zero_()
       line = torch.FloatTensor(self.nb_node).zero_()
    adj = Variable(adj)
    line = Variable(line)

    for i in range(self.nb_node):
      line[self.idx[i]] = edge_out[i]
      adj[i] = line
    return adj
