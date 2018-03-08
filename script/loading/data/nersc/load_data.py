import os
from os.path import exists, join
import numpy as np
from numpy.random import permutation
import h5py as h5
from loading.data.nersc.file2weightfactor import init_weight_factors


def _is_train_file(filename):
    return filename.endswith('02.h5')


def _is_test_file(filename):
    return filename.endswith('01.h5')


def load_raw_data(datadir, nb_ex, train_test):
    """creates a data loader containing (data, label, weight)"""
    if train_test == 'train':
      is_used = _is_train_file
    elif train_test == 'test':
      is_used = _is_test_file

    nb_events_files, all_event_coords_ = all_event_coords(datadir, is_used)
    weight_factors = init_weight_factors(is_used, datadir)

    nb_events_ = min(sum(nb_events_files), nb_ex)
    data_loader = (load_raw_event(datadir, event_coord, weight_factors)
                   for event_coord in permutation(all_event_coords_)[:nb_events_])

    samples_to_load = permutation(all_event_coords_)[:nb_events_]
    X = []
    y = []
    w = []
    for idx in samples_to_load:
      X_, y_, w_ = load_raw_event(datadir, idx, weight_factors)
      X.append(X_)
      y.append(y_)
      w.append(w_)

    return X, y, w


def load_raw_event(datadir, event_coord, weight_factors):
    """loads one event, given the file and the event index"""

    chosen_file, idx = event_coord
    datapath = join(datadir, chosen_file)
    event_name = 'event_{}'.format(idx)
    with h5.File(datapath, 'r') as datafile:
        event = datafile[event_name]
        data_fields = ['clusE', 'clusEta', 'clusPhi', 'clusEM']
        data_fields = [event[field][()] for field in data_fields]

        cluspt = data_fields[0] / np.cosh(data_fields[1])  # clusE / cosh(clusEta)
        data_fields.append(cluspt)

        weight = event['weight'][()]

    data = np.array([clus for clus in data_fields])

    weight_factor = weight_factors[chosen_file]
    weight = float(weight * weight_factor)

    label = int(chosen_file.startswith('GG'))

    return (data, label, weight)


def all_event_coords(datadir, is_used):
    data_files_ = data_files(datadir, is_used)
    nb_events_file = [nb_events(join(datadir, filename)) for filename in data_files_]
    all_coords = [(filename, idx)
                  for i, filename in enumerate(data_files_)
                  for idx in range(nb_events_file[i])]

    return nb_events_file, all_coords


def data_files(datadir, is_used):
    """Lists all relevant files in `datadir`"""

    datafiles = [filename for filename in os.listdir(datadir) if filename.endswith('.h5')]
    datafiles = [filename for filename in datafiles if 'data' not in filename]
    datafiles = [filename for filename in datafiles if is_used(filename)]

    return datafiles


def nb_events(path):
    with h5.File(path) as h5file:
        nb = h5file.attrs['nb_event']
    return nb
