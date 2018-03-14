import torch
import torch.nn as nn

class Affine_Normalized(nn.Module):
  def __init__(self, nb_kernels):
    super(Affine_Normalized, self).__init__()
    self.nb_kernels = nb_kernels
    params = torch.ones(nb_kernels)
    self.register_parameter('params', nn.Parameter(params))

  def forward(self, adj_list):
    # Affine combination
    adj_out = adj_list[0]*self.params[0]
    for i in range(1, self.nb_kernels):
      adj_out += adj_list[i]*self.params[i]
    
    # Normalize
    adj_out += adj_out.min()
    adj_out /= adj_out.max()
    return adj_out
