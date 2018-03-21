import torch.nn as nn

import model.kernels.physics as physics
import model.kernels.general as general
import loading.model.model_parameters as param

  
def _get_kernel_class(kernel_name):
  """
  Determines kernel type.
  Returns uninitiated kernel class and initiation arguments
  """
  loop2pi = param.args.data == 'NERSC'  # in NERSC data, phi is 2pi-periodic
  ker_args = ()
  ker_kwargs = {}
  if kernel_name == 'FCG':
    # kernel = ker.FixedComplexGaussian(args.sigma, periodic=loop2pi)
    raise Exception("kernel not configured")
  elif kernel_name == 'FCG_nodiag':
    # kernel = ker.FixedComplexGaussian(args.sigma, diag=False, periodic=loop2pi)
    raise Exception("kernel not configured")
  elif kernel_name == 'FCG_norm':
    # kernel = ker.FixedComplexGaussian(args.sigma, norm=True, periodic=loop2pi)
    raise Exception("kernel not configured")
  elif kernel_name == 'FCG_nodiag_norm':
    # kernel = ker.FixedComplexGaussian(args.sigma, diag=False, norm=True, periodic=loop2pi)
    raise Exception("kernel not configured")
  elif kernel_name == 'FQCDAware':
    # kernel = ker.FixedQCDAware(0.5, 0.1, periodic=loop2pi)
    raise Exception("kernel not configured")
  elif kernel_name == 'QCDAware':
    kernel = physics.QCDAware
    ker_args += (1.0, 0.7,)
    ker_kwargs['periodic'] = loop2pi
  elif kernel_name == 'QCDAwareMeanNorm':
    kernel = physics.QCDAwareMeanNorm
    ker_args += (1.0, 0.7,)
    ker_kwargs['periodic'] = loop2pi
  elif kernel_name == 'QCDAwareNoNorm':
    kernel =physics.QCDAwareNoNorm
    ker_args += (1.0, 0.7,)
    ker_kwargs['periodic'] = loop2pi
  # Instantiate adjacency kernels
  elif kernel_name == 'Gaussian':
    kernel = general.Gaussian
    ker_kwargs['sigma'] = param.args.sigma
  elif kernel_name == 'GaussianSoftmax':
    kernel = general.GaussianSoftmax
  elif kernel_name == 'DirectedGaussian':
    kernel = general.DirectedGaussian
  elif kernel_name == 'MPNNdirected':
    kernel = general.MPNNdirected
  elif kernel_name == 'MLPdirected':
    kernel = general.MLPdirected
    ker_args += (param.args.nb_MLPadj_hidden,)
    ker_args += (None,)
  elif kernel_name == 'Identity':
    kernel = general.Identity
  else:
    raise ValueError('Unknown kernel : {}'.format(kernel_name))
  return kernel, ker_args, ker_kwargs

def _get_one_kernel(kernel_name):
  ker_options = kernel_name.split('-')
  kernel, ker_args, ker_kwargs  = _get_kernel_class(ker_options[0])
  if 'layerwise' in ker_options:
    ker_kwargs['layerwise'] = True
  kernels = []
  # Initiate first kernel with proper nb feature maps
  init_kernel = kernel(*(param.args.first_fm,)+ker_args,**ker_kwargs)
  kernels.append(init_kernel)
  if 'layerwise' in ker_options:
    # Distinct kernel instance for each layer
    ker_kwargs['layerwise'] = True
    for i in range(1,param.args.nb_layer):
      kernels.append(kernel(*(param.args.nb_feature_maps,)+ker_args, **ker_kwargs))
  else: # Use same kernel instance at all layers
    for i in range(1,param.args.nb_layer):
      kernels.append(init_kernel)
  return kernels

def get_kernels():
  kernel_names = param.args.kernels
  all_kernels = []
  for ker_name in kernel_names:
    all_kernels.append(_get_one_kernel(ker_name))
  # Reformat list for ModuleList
  module_kernels = []
  for i in range(param.args.nb_layer):
    layer_kernels = []
    for ker in all_kernels:
      layer_kernels.append(ker[i])
    module_kernels.append(nn.ModuleList(layer_kernels))
  return nn.ModuleList(module_kernels)
