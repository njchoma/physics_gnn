import pickle
from os.path import exists
import numpy as np


def load_data_nyu(filepath, nb_ex, mode):
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
        weights = None

    data = [X[:, :6] for X in data]  # dump px, py, pz
    return data, label, weights


def load_data_nersc(datafile, nb_ex):
    def _iter_data_nersc(datafile, idx):
        event_name = 'event_{}'.format(idx)
        event = datafile[event_name]

        data = event[()]
        label = event.attrs['label']
        weight = event.attrs['weight']

        return (idx, data, label, weight)

    nb_event = datafile['nbevent']
    nb_event = min(nb_ex, nb_event)
    return (_iter_data_nersc(datafile, idx) for idx in range(nb_event)), nb_event


def get_fixed_param():
    """reads parameters from 'new_net_param.txt',
    creates the file if non-existant
    """

    def _get_fixed_param():
        args = dict()
        for line in open('param.txt', 'r'):
            if line.strip():  # not empty line
                arg_txt = line.split('#')[0]  # remove comment
                arg_name, arg_val = arg_txt.split('=')[:2]
                arg_name, arg_val = arg_name.strip(), arg_val.strip()
                args[arg_name] = arg_val
                if arg_val == '':
                    raise ValueError(
                        "Empty parameter in 'param.txt': {}".format(arg_name))
                print("param {} : '{}'".format(arg_name, arg_val))
        return args

    if exists('param.txt'):
        return _get_fixed_param()
    with open('param.txt', 'w') as paramfile:
        paramfile.write(
            "\ntrainfile =  # path to training data `antikt-kt-train-gcnn.pickle`\n"
            + "testfile =  # path to testing data `antikt-kt-test-gcnn.pickle`\n"
        )
    raise FileNotFoundError("'param.txt' created, missing parameters")
