import logging

import loading.model.model_parameters as param
from model.gnn.gnn import GNN

def init_network():
  '''
  Initialize and return model depending upon architecture
  '''
  architecture = param.args.architecture
  if architecture == 'gnn':
    logging.warning("Initializing GNN...")
    model = GNN()
  else:
    logging.error("Network type {} not recognized".format(architecture))
    raise
  return model
