import os
import h5py as h5
from collections import defaultdict
import pickle
from utils import print_
import multiprocessing


def len2namenum(is_used, rawdatadir, savedir, stdout=None, reprocess=False):
    """
    calls `filelen2nums`, and gathers every (filename, len -> num) pair
    to a single len -> (filename, num) mapping.
    inputs: - `is_used` : is_train(filename) is True is the said file
                    is used.
            - `rawdatadir` : path to raw datadir
            - `savedir` : output saved in this directory
            - `stdout` (optionnal) : redirect prints to designated file
            - `reprocess` (optionnal) : if false, existant files in `savedir`
                    wont be reprocessed

    output: - `len2namenum` : mapping from envent length to a list of
                    filename and corresponding event id
    """

    def add_name_to_iter(filename, iterable):
        return [(filename, item) for item in iterable]

    print_('organizing data from `{}`.'.format(rawdatadir), stdout)

    # initiate len2namenum
    len2namenum = defaultdict(list)

    # process each file independently
    filelen2nums(is_used, rawdatadir, savedir, stdout, reprocess)
    processed_files = os.listdir(savedir)

    # gather in len2namenum
    for filename in processed_files:
        with open(os.path.join(savedir, filename), 'rb') as l2nfile:
            len2num = pickle.load(l2nfile, pickle.HIGHEST_PROTOCOL)
        for length, num in len2num.items():
            len2namenum[length].extend(add_name_to_iter(filename, num))

    # change type of len2namenum to regular dict
    len2namenum = dict(len2namenum)
    print_('\nChanged type of len2namenum.', stdout)

    return len2namenum


def filelen2nums(is_used, rawdatadir, savedir, stdout=None, reprocess=False):
    """
    Reads from files and creates for each file a mapping from length to
    event index.

    inputs: - `is_used` : is_train(filename) is True is the said file
                    is used.
            - `rawdatadir` : path to raw datadir
            - `savedir` : output saved in this directory
            - `stdout` (optionnal) : redirect prints to designated file
            - `reprocess` (optionnal) : if false, existant files in `savedir`
                    wont be reprocessed

    output: None
    """

    def _len2num(filename):
        len2num(rawdatadir, savedir, filename, stdout)

    # find relevant files
    h5_files = [filename for filename in os.listdir(rawdatadir)
                if filename.endswith('.h5') and is_used(filename)]
    if not reprocess:
        processed_files = os.listdir(savedir)
        processed_files = [os.path.splitext(filename)[0] + '.h5' for filename in processed_files]
        h5_files = [filename for filename in h5_files
                    if filename not in processed_files]

    # iterate over files
    pool = multiprocessing.Pool(32)
    pool.map(_len2num, h5_files)


def len2num(rawdatadir, savedir, filename, stdout=None):
    print_('{: <30} : start'.format(filename), stdout)
    len2num = defaultdict(list)
    curr_path = os.path.join(rawdatadir, filename)

    with h5.File(curr_path, 'r') as curr_file:
        for event_name in curr_file:
            if event_name.startswith('event'):
                length = curr_file[event_name]['clusE'].shape[0]  # number of energy peaks
                len2num[length].append(event_name)

    len2num = dict(len2num)  # change type
    savename = os.path.splitext(filename)[0] + '.pkl'
    with open(os.path.join(savedir, savename), 'wb') as fileout:
        pickle.dump(len2num, fileout, pickle.HIGHEST_PROTOCOL)
    print_('{: <30} ... saved'.format(filename), stdout)
