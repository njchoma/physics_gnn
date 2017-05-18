import os
import h5py as h5
from collections import defaultdict
from utils import print_


def len2namenum(is_used, rawdatadir, stdout=None):
    """
    inputs: - `is_used` : is_train(filename) is True is the said file
                    is used.
            - `datadir` : path to raw datadir
            - `stdout` (optionnal) : redirect prints to designated file
    output: - `len2namenum` : mapping from envent length to a list of
                    filename and corresponding event id
    """

    print_('organizing data from `{}`.'.format(rawdatadir), stdout)

    # initiate make_len2namenum
    len2namenum = defaultdict(list)

    # find relevant files
    h5_files = (filename for filename in os.listdir(rawdatadir)
                if filename.endswith('.h5') and is_used(filename))

    # iterate over files
    for filename in h5_files:
        print_('{: <30} : start'.format(filename), stdout)
        curr_path = os.path.join(rawdatadir, filename)
        with h5.File(curr_path, 'r') as curr_file:
            for event_name in curr_file:
                if event_name.startswith('event'):
                    length = curr_file[event_name]['clusE'].shape[0]  # number of energy peaks
                    len2namenum[length].append((filename, event_name))
        print_('{: <30} ... done'.format(filename), stdout)

    # change type of len2namenum to regular dict
    len2namenum = dict(len2namenum)
    print_('\nChanged type of len2namenum.', stdout)

    return len2namenum
