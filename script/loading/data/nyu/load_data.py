import pickle
import numpy as np
from random import shuffle

from utils.in_out import print_
import loading.model.model_parameters as param



def load_raw_data(filepath, nb_ex, mode):
    """Loads data from the NYU project, makes it into torch Variables"""

    if mode == 'test':
        with open(filepath, 'rb') as filein:
            try:
                data, label, weights = pickle.load(filein, encoding='latin1')  # python3
            except TypeError:
                data, label, weights = pickle.load(filein)  # python2
    else:
        with open(filepath, 'rb') as filein:
            try:
                data, label = pickle.load(filein, encoding='latin1')  # python3
            except TypeError:
                data, label = pickle.load(filein)  # python2

    idx = [i for i in range(len(data))]
    if param.args.shuffle_while_training or mode=='test':
      shuffle(idx)
    idx = idx[:nb_ex]
    

    data  = [data[i]  for i in idx]
    label = [label[i] for i in idx]
    # data, label = data[idx], label[idx]
    # NOTE: Trying all samples unweighted
    weights = [1.0 for _ in range(min(nb_ex,len(data)))]

     
    data = [np.array(X[:, :6]).transpose() for X in data]  # dump px, py, pz
    return data, label, weights
