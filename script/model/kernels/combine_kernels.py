import torch
import torch.nn as nn

def _zero_incoming_adj(adj):
  return adj - adj

class Affine(nn.Module):
  def __init__(self, nb_kernels):
    super(Affine, self).__init__()
    self.nb_kernels = nb_kernels
    params = torch.ones(nb_kernels) / self.nb_kernels
    self.register_parameter('params', nn.Parameter(params))

  def normalize(self, adj):
    return adj

  def forward(self, adj, adj_list):
    # Affine combination
    adj_out = _zero_incoming_adj(adj)
    for i in range(self.nb_kernels):
      adj_out += adj_list[i]*self.params[i]
    
    # Normalize
    adj_out = self.normalize(adj_out)
    return adj_out

class Affine_Normalized(Affine):
  def __init__(self, nb_kernels):
    super(Affine_Normalized, self).__init__(nb_kernels)

  def normalize(self, adj):
    adj += adj.min()
    adj /= adj.max()
    return adj

class Fixed_Balanced(nn.Module):
  def __init__(self, nb_kernels):
    super(Fixed_Balanced, self).__init__()
    self.nb_kernels = nb_kernels

  def forward(self, adj, adj_list):
    adj = _zero_incoming_adj(adj)
    for i in range(len(adj_list)):
      adj += adj_list[i]
    adj /= len(adj_list)
    return adj
