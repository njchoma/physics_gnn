import logging
import torch
import torch.nn as nn
from torch.nn.functional import sigmoid

import loading.model.model_parameters as param
from utils.tensor import mean_with_padding, mask_embedding
from model.readout.gmm import GMM

def get_readout():
  # Implement option to choose different pooling
  readout_type = param.args.readout
  fmaps = param.args.nb_feature_maps

  if readout_type == 'Sum':
    readout = Sum(fmaps)
  elif readout_type == 'Mean':
    readout = Mean(fmaps)
  elif readout_type == 'Max':
    readout = Max(fmaps)
  elif readout_type == 'DTNN_Sum':
    readout = DTNN_Sum(fmaps)
  elif readout_type == 'DTNN_Mean':
    readout = DTNN_Mean(fmaps)
  elif readout_type == 'DTNN_Max':
    readout = DTNN_Max(fmaps)
  elif readout_type == 'GMM':
    readout = GMM(fmaps, nb_gauss=8)
  else:
    raise Exception("Readout type {} not recognized".format(readout_type))

  return readout


class Readout(nn.Module):
  def __init__(self, fmaps):
    super(Readout, self).__init__()
    self.fmaps = fmaps
    self.norm = nn.InstanceNorm1d(1)
    self.fcl = nn.Linear(fmaps,1)

  def _pooling(self, emb_in, adj_mask, *args, **kwargs):
    logging.error("Readout must be implemented with child class")
    raise

  def forward(self, emb_in, adj_mask, *args, **kwargs):
    emb = self._pooling(emb_in, adj_mask)
    emb = self.norm(emb.unsqueeze(1)).squeeze(1)
    emb = self.fcl(emb).squeeze(1)
    return sigmoid(emb)

class Mean(Readout):
  def __init__(self,fmaps):
    super(Mean, self).__init__(fmaps)

  def forward(self, emb_in, adj_mask, batch_nb_nodes, *args, **kwargs):
    emb = mean_with_padding(emb_in, batch_nb_nodes, adj_mask)
    emb = self.norm(emb.unsqueeze(1)).squeeze(1)
    emb = self.fcl(emb).squeeze(1)
    return sigmoid(emb)

  def _pooling(self, emb_in, adj_mask):
    pass

class Sum(Readout):
  def __init__(self,fmaps):
    super(Sum, self).__init__(fmaps)

  def _pooling(self, emb_in, adj_mask):
    return mask_embedding(emb_in, adj_mask).sum(1)

class Max(Readout):
  def __init__(self,fmaps):
    super(Sum, self).__init__(fmaps)

  def _pooling(self, emb_in, adj_mask):
    return mask_embedding(emb_in, adj_mask).max(0)


class FCL(nn.Module):
  def __init__(self,fmaps):
    super(FCL, self).__init__()
    self.fcl1 = nn.Linear(fmaps, fmaps)
    self.fcl2 = nn.Linear(fmaps, fmaps)
    self.activation = nn.Tanh()

  def forward(self, emb_in):
    emb = self.fcl1(emb_in)
    emb = self.activation(emb)
    return self.fcl2(emb)

class DTNN(nn.Module):
  def __init__(self,fmaps):
    super(DTNN, self).__init__()
    self.dtnn = FCL(fmaps)
    self.readout = None

  def forward(self, emb_in, adj_mask, batch_nb_nodes, *args, **kwargs):
    emb = self.dtnn(emb_in)
    return self.readout(emb, adj_mask, batch_nb_nodes)

class DTNN_Sum(DTNN):
  def __init__(self, fmaps):
    super(DTNN_Sum, self).__init__(fmaps)
    self.readout = Sum(fmaps)

class DTNN_Max(DTNN):
  def __init__(self, fmaps):
    super(DTNN_Max, self).__init__(fmaps)
    self.readout = Max(fmaps)

class DTNN_Mean(DTNN):
  def __init__(self, fmaps):
    super(DTNN_Mean, self).__init__(fmaps)
    self.readout = Mean(fmaps)
