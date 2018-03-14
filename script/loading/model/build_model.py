import logging
import pickle
from os.path import exists, join

from model import gcnn
from model import sparse
from loading.model import build_kernels
from loading.model import build_combine_kernels

def init_network(args, frst_fm):
  """Reads args and initiates a network accordingly.
  Input should be the output of `read_args`.
  """
  kernels = build_kernels.get_kernels()
  combine_kernels = build_combine_kernels.get_combine_kernels()

  return gcnn.GCNNSingleKernel(
      kernels, combine_kernels, frst_fm, args.nb_feature_maps, args.nb_layer
      )

def make_net_if_not_there(args, frst_fm, savedir):
    """Checks for existing network, initiates one if non existant"""

    model_path = join(savedir, args.name)
    if exists(model_path + '.pkl'):
        with open(model_path + '.pkl', 'rb') as filein:
            net = pickle.load(filein)
        logging.warning('Network recovered from previous training')
    else:
        net = init_network(args, frst_fm)
        logging.warning('Network created')
        with open(model_path + '.csv', 'w') as fileres:
            fileres.write('Learning Rate, Train Loss, Test Loss, Train AUC Score'
                          + ', Test AUC Score, 1/FPR_train, 1/FPR_test, Running Loss\n')
    logging.info('parameters : {}'.format(sum([param.numel() for param in net.parameters()])))

    return net
