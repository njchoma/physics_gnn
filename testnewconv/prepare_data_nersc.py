from os import path, listdir
import h5py as h5
from numpy.random import permutation


def _data_files(datadir):
    datafiles = [filename for filename in listdir(datadir) if filename.endswith('.h5')]
    datafiles = [filename for filename in datafiles if filename != 'data.h5']
    
    return datafiles


def _count_events(datadir, datafiles):
    return sum([len(h5.File(path.join(datadir, datafile), 'r')) for datafile in datafiles])


def _transfer_one_file(datapath, fileout, permutation, label):
    for event in h5.File(datapath, 'r'):
        event_new_idx = permutation.pop()
        ########################
        ### NEED TO DO STUFF ###
        ########################


def main(datadir):
    datafiles = _data_files(datadir)
    nb_events = _count_events(datadir, datafiles)
    permutation = permutation(nb_events)
    with open(path.join(datadir, 'data.h5'), 'w') as fileout:




if __name__ == '__main__':
    main('/home/gaspar/Desktop/dataNERSC')
