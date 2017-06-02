import os
from math import sqrt
from collections import defaultdict
import pickle
import multiprocessing
import h5py as h5
from utils import print_


def global_stats(is_used, rawdatadir, savedir, stdout=None, reprocess=False):
    """
    calls `distribute_filewise_stats`, and gathers joins every stat into a single
    (mean, stdev) over the whole dataset

    inputs: - `is_used` : is_train(filename) is True is the said file
                    is used.
            - `rawdatadir` : path to raw datadir
            - `savedir` : output saved in this directory
            - `stdout` (optionnal) : redirect prints to designated file
            - `reprocess` (optionnal) : if false, existant files in `savedir`
                    wont be reprocessed

    output: - {'feature': (mean, stdev)}
    """

    print_('calculating mean and standard deviation from `{}`.'.format(rawdatadir), stdout)

    # process each file independently
    distribute_filewise_stats(is_used, rawdatadir, savedir, stdout, reprocess)
    processed_files = os.listdir(savedir)
    processed_files = [filename for filename in processed_files
                       if 'global' not in filename]

    # gather stats
    stats = defaultdict(lambda: [0, 0, 0])  # [sum x, sum x**2, nb]
    for filename in processed_files:
        with open(filename, 'rb') as statfile:
            stats_file = pickle.load(statfile)
        for feature in stats_file.keys():
            stats[feature][0] += stats_file[feature][0]
            stats[feature][1] += stats_file[feature][0]
            stats[feature][2] += stats_file[feature][0]
    stats = dict(stats)

    # change [sum x, sum x**2, length] to (mean x, stdev x)
    for feature in stats.keys():
        if stats[feature][2] == 0:
            raise ValueError('`{}` was detected but 0 constituent contributed'.format(feature))

        mean = float(stats[feature][0]) / stats[feature][2]
        var = (float(stats[feature][1]) / stats[feature][2]) - mean ** 2
        stdev = sqrt(var)
        stats[feature] = (mean, stdev)

    return stats

def distribute_filewise_stats(is_used, rawdatadir, savedir, stdout=None, reprocess=False):
    """
    Reads from files and calculates for each file the sum of features
    and of feature squared :

    output: - saves dict(feature: (sum x, sum x**2, nb constituents))
                    for each file
    """

    # find relevant files
    h5_files = [filename for filename in os.listdir(rawdatadir)
                if filename.endswith('.h5') and is_used(filename)]
    if not reprocess:
        processed_files = os.listdir(savedir)
        processed_files = [os.path.splitext(filename)[0] + '.h5' for filename in processed_files]
        h5_files = [filename for filename in h5_files
                    if filename not in processed_files]

    pool = multiprocessing.Pool(32)
    pool.map(Copier(rawdatadir, savedir, stdout), h5_files)


class Copier:
    """stores arguments shared accross files"""

    def __init__(self, rawdatadir, savedir, stdout):
        self.rawdatadir = rawdatadir
        self.savedir = savedir
        self.stdout = stdout

    def __call__(self, filename):
        filewise_stats(self.rawdatadir, self.savedir, filename, self.stdout)



def filewise_stats(rawdatadir, savedir, filename, stdout=None):
    """same as `distribute_filewise_stats` but for one file"""

    print_('{: <30} : start'.format(filename), stdout)
    stats = defaultdict(lambda: [0, 0, 0])  # [sum x, sum x**2, nb]
    curr_path = os.path.join(rawdatadir, filename)

    with h5.File(curr_path, 'r') as curr_file:
        for event_name in curr_file:
            if event_name.startswith('event'):
                if "jet_length" in curr_file[event_name].attrs:
                    length = curr_file[event_name].attrs["jet_length"]
                else:
                    length = curr_file[event_name]['clusE'].shape[0]  # number of energy peaks

                for feature in curr_file[event_name]:
                    value = curr_file[event_name][feature].value
                    stats[feature][0] += value.sum()
                    stats[feature][1] += (value ** 2).sum()
                    stats[feature][2] += length

    stats = dict(stats)  # change type

    savename = os.path.splitext(filename)[0] + '.pkl'
    with open(os.path.join(savedir, savename), 'wb') as fileout:
        pickle.dump(stats, fileout, pickle.HIGHEST_PROTOCOL)
    print_('{: <30} ... saved : {}'.format(filename, stats), stdout)
