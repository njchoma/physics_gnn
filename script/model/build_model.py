import logging

import loading.model.model_parameters as param
from model.gnn.gnn import GNN
from graphics.plot_graph import construct_plot

def init_network():
  '''
  Initialize and return model depending upon architecture
  '''
  architecture = param.args.architecture
  plotting = construct_plot()
  if architecture == 'gnn':
    logging.warning("Initializing GNN...")
    model = GNN(plotting=plotting)
  else:
    logging.error("Network type {} not recognized".format(architecture))
    raise
  return model
