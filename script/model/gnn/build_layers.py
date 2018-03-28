import logging
import torch.nn as nn

import loading.model.model_parameters as param
from model.gnn.gnn_layer import GNN_Layer

def get_layers(kernels, combine_kernels):
  '''
  Builds layers for GNN framework
  '''
  init_layer = GNN_Layer(
                         kernels=kernels[0],
                         combine_kernels=combine_kernels[0],
                         fmap_in=param.args.first_fm,
                         fmap_out=param.args.nb_feature_maps,
                         layer_nb=0
                         )
  layers = [init_layer]
  for i in range(1,param.args.nb_layer):
    layers.append( GNN_Layer(
                            kernels=kernels[i],
                            combine_kernels=combine_kernels[i],
                            fmap_in=param.args.nb_feature_maps,
                            fmap_out=param.args.nb_feature_maps,
                            layer_nb=i
                            ))
  return nn.ModuleList(layers)
