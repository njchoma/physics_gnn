import pickle
import numpy as np
from utils.in_out import print_



def load_raw_data(filepath, nb_ex, mode):
    """Loads data from the NYU project, makes it into torch Variables"""

    if mode == 'test':
        with open(filepath, 'rb') as filein:
            try:
                data, label, weights = pickle.load(filein, encoding='latin1')  # python3
            except TypeError:
                data, label, weights = pickle.load(filein)  # python2
        if nb_ex is not None:
            data, label, weights = data[:nb_ex], label[:nb_ex], weights[:nb_ex]
    else:
        with open(filepath, 'rb') as filein:
            try:
                data, label = pickle.load(filein, encoding='latin1')  # python3
            except TypeError:
                data, label = pickle.load(filein)  # python2
        if nb_ex is not None:
            data, label = data[:nb_ex], label[:nb_ex]
    # NOTE: Trying all samples unweighted
    weights = [1.0 for _ in range(nb_ex)]

     
    data = [np.array(X[:, :6]).transpose() for X in data]  # dump px, py, pz
    return data, label, weights
