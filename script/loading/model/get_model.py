import logging
import pickle
from os.path import exists, join

from model.build_model import init_network

def make_net_if_not_there(args, savedir):
    """
    Checks for existing network, initiates one if non existant
    """
    model_path = join(savedir, args.name)
    if exists(model_path + '.pkl'):
        with open(model_path + '.pkl', 'rb') as filein:
            net = pickle.load(filein)
        logging.warning('Network recovered from previous training')
    else:
        net = init_network()
        logging.warning('Network created')
        with open(model_path + '.csv', 'w') as fileres:
            fileres.write(
                  'Learning Rate, Train Loss, Test Loss, Train AUC Score'
                  + ', Test AUC Score, 1/FPR_train, 1/FPR_test, Running Loss\n'
                  )
    logging.info('parameters : {}'.format(
                              sum([param.numel() for param in net.parameters()])
                              ))
    return net
