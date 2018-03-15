import torch.nn as nn

import model.kernels.combine_kernels as combine_ker
from loading.model import model_parameters as param

def _get_combine_kernel_class(combine_name):
  if combine_name == 'Affine':
    combine = combine_ker.Affine
  elif combine_name == 'Affine_Normalized':
    combine = combine_ker.Affine_Normalized
  elif combine_name == 'Fixed_Balanced':
    combine = combine_ker.Fixed_Balanced
  else:
    raise Exception("Combine kernel method {} not recognized".format(combine_name))

  return combine
  
def get_combine_kernels():
  nb_kernels = len(param.args.kernels)
  combine = _get_combine_kernel_class(param.args.combine_kernels)
  combine_list = []
  for i in range(param.args.nb_layer):
    combine_list.append(combine(nb_kernels))
  return nn.ModuleList(combine_list)
  
