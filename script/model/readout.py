import logging
import torch
import torch.nn as nn
from torch.nn.functional import sigmoid

import loading.model.model_parameters as param
from utils.tensor import mean_with_padding, mask_embedding

def get_readout():
  # Implement option to choose different pooling
  # For now just use summing
  return Readout_Sum(param.args.nb_feature_maps)

class Readout(nn.Module):
  def __init__(self, fmaps):
    super(Readout, self).__init__()
    self.fmaps = fmaps
    self.norm = nn.InstanceNorm1d(1)
    self.fcl = nn.Linear(fmaps,1)

  def _pooling(self, emb_in, adj_mask, batch_nb_nodes):
    logging.error("Readout must be implemented with child class")
    raise

  def forward(self, emb_in, adj_mask, batch_nb_nodes):
    emb = self._pooling(emb_in, adj_mask, batch_nb_nodes)
    emb = self.norm(emb.unsqueeze(1)).squeeze(1)
    emb = self.fcl(emb).squeeze(1)
    return sigmoid(emb)

class Readout_Sum(Readout):
  def __init__(self,fmaps):
    super(Readout_Sum, self).__init__(fmaps)

  def _pooling(self, emb_in, adj_mask, batch_nb_nodes):
    return mask_embedding(emb_in, adj_mask).sum(2)

class Readout_Max(Readout):
  def __init__(self,fmaps):
    super(Readout_Sum, self).__init__(fmaps)

  def _pooling(self, emb_in, adj_mask, batch_nb_nodes):
    return mask_embedding(emb_in, adj_mask).max(0)


# Implement mean pooling at some point
'''
emb = emb.sum(2)#.squeeze(2)
# Get mean of emb, accounting for zero padding of batches
batch_div_for_mean = batch_nb_nodes.unsqueeze(1).repeat(1,emb.size()[1])
emb = emb/batch_div_for_mean
'''

