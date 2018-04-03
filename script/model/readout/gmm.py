import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np

from utils.tensor import spatialnorm

class GMM(nn.Module):
  def __init__(self, fmap, nb_gauss=8):
    super(GMM, self).__init__()
    self.fmap = fmap
    self.nb_gauss = nb_gauss
    # Register parameters
    mu = torch.Tensor(nb_gauss, fmap)
    sigma = torch.Tensor(nb_gauss, fmap)
    nn.init.uniform(mu, -2, 2)
    nn.init.uniform(sigma, 0.0, 0.01)
    self.mu=Parameter(mu)
    self.sigma=Parameter(sigma)
    # Fully connected output
    self.fc = nn.Linear(nb_gauss, 1)
    self.act = nn.Sigmoid()

  def forward(self, emb_in, adj_mask=None, batch_nb_nodes=None):
    batch, nb_node, fmap = emb_in.size()
    # Resize parameters
    mu = self.mu.unsqueeze(0).unsqueeze(1).repeat(batch, nb_node, 1,1)
    sigma = self.sigma.unsqueeze(0).unsqueeze(1).repeat(batch,nb_node,1,1)
    # Apply gaussian kernel
    sqdiff = (emb_in.unsqueeze(2) - mu)**2
    sum_feat = (sqdiff * sigma).sum(3)
    exp = torch.exp(sum_feat * -0.5)
    # Mask batches
    masked = exp * adj_mask[:,0].unsqueeze(2)
    # Sum over all nodes in batch
    gaussians = masked.sum(1)
    # Apply transformation and sigmoid
    out = self.act(self.fc(gaussians))
    return out.squeeze(1)
