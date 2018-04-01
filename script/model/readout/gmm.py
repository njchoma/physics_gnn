import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from utils.tensor import spatialnorm

class GMM(nn.Module):
  def __init__(self, k, fmap, n_step=4, init_var=1.0):
    super(GMM, self).__init__()
    self.k = k
    self.init_var = nn.Parameter(torch.Tensor([init_var]))
    self.n_step = n_step
    self.fc = nn.Linear(fmap, 1)
    self.act = nn.Sigmoid()

  def initialize(self, X, mask=None, batch_nb_nodes=None):
    batch, nb_node, fmap = X.size()
    choose_from = int(max(batch_nb_nodes.min().data[0], self.k))
    idx = np.random.choice(choose_from, self.k, replace=False).tolist()
    mu = X[:,idx] # Initialize means as randomly as points
    return mu

  def expectation(self, X, mu):
    batch, nb_node, fmap = X.size()
    diff = X.unsqueeze(2).repeat(1,1,self.k,1)-mu.unsqueeze(1)
    norm = (diff**2).sum(3) # Sum over the features
    likelihood = torch.exp(-0.5 * norm)+10**-20
    P = likelihood / (likelihood.sum(2, keepdim=True)+10**-40)
    return P
    
  def maximization(self, X, P):
    batch, nb_node, fmap = X.size()
    nb_k = P.sum(1).view(batch, self.k, 1)
    mu = torch.bmm(P.transpose(1,2), X)
    return mu

  def forward(self, emb_in, mask, batch_nb_nodes, *args, **kwargs):
    batch, nb_node, fmap = emb_in.size()
    X, _, _ = spatialnorm(emb_in, batch_nb_nodes, mask)
    mu = self.initialize(X, mask, batch_nb_nodes)
    for i in range(self.n_step):
      P = self.expectation(X, mu)
      mu = self.maximization(X, P)
    out = self.fc(mu).squeeze(2).mean(1)
    out = self.act(out)
    return out
