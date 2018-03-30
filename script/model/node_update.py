import torch.nn as nn
import torch.nn.functional as F

import loading.model.model_parameters as param

def get_node_update(fmap_in,fmap_out):
  node_type = param.args.node_type
  init_args = (fmap_in, fmap_out)

  if node_type == 'Identity':
    node_update = Identity
  elif node_type == 'GRU':
    node_update = GRUUpdate
  else:
    raise Exception("Node type {} not recognized".format(node_type))

  return node_update(*init_args)

class Identity(nn.Module):
  def __init__(self, *args, **kwargs):
    super(Identity, self).__init__()

  def forward(self, emb_in, emb_update, *args, **kwargs):
    return emb_update

class GRUUpdate(nn.Module):
  def __init__(self, fmap_in, fmap_out, *args, **kwargs):
    super(GRUUpdate, self).__init__()
    self.ih = nn.Linear(fmap_in, 3 * fmap_out)
    self.hh = nn.Linear(fmap_out, 3 * fmap_out)

  def forward(self, i, h):
    r_i, z_i, n_i = self.ih(i).chunk(3,-1)
    r_h, z_h, n_h = self.hh(h).chunk(3,-1)

    r = F.sigmoid(r_i+r_h)
    z = F.sigmoid(z_i+z_h)
    n = F.tanh(n_i+r*n_h)

    o = (1-z)*n + z*h

    return o
