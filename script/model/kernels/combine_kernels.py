import torch
import torch.nn as nn

class Affine(nn.Module):
  def __init__(self, nb_kernels):
    super(Affine, self).__init__()
    self.nb_kernels = nb_kernels
    params = torch.ones(nb_kernels) / self.nb_kernels
    self.register_parameter('params', nn.Parameter(params))

  def normalize(self, adj):
    return adj

  def forward(self, adj_list):
    # Affine combination
    adj_out = adj_list[0]*self.params[0]
    for i in range(1, self.nb_kernels):
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

  def forward(self, adj_list):
    adj_out = adj_list[0]
    for i in range(1, self.nb_kernels):
      adj_out += adj_list[i]
    adj_out /= self.nb_kernels
    return adj_out
