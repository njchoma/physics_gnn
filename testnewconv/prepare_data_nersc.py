from os import path, listdir
import h5py as h5
import numpy as np
from numpy.random import permutation


########################################
######## pT = clusE / cosh(Eta) ########
########################################

def _data_files(datadir):
    datafiles = [filename for filename in listdir(datadir) if filename.endswith('.h5')]
    datafiles = [filename for filename in datafiles if 'data' not in filename]

    return datafiles


def _count_events(datadir, datafiles, print_fun):
    def _print_and_len(datafile):
        print_fun('{} : start'.format(datafile))
        try:
            with h5.File(path.join(datadir, datafile), 'r') as h5file:
                length = h5file.attrs['nb_event']
        except KeyError:
            with h5.File(path.join(datadir, datafile), 'r') as h5file:
                length = len(h5file)
            with h5.File(path.join(datadir, datafile), 'a') as h5file:
                h5file.attrs['nb_event'] = length
            print_fun("attribute 'nb_event' added in {}".format(datafile))
        print_fun('{} : done'.format(datafile))
        return length
    return sum([_print_and_len(datafile) for datafile in datafiles])


def _transfer_one_event(filein, fileout, event, new_idx, label):
    event_name = 'event_{}'.format(new_idx)
    shape = filein[event]['clusE'].shape + (5,)
    fileout.create_dataset(event_name, shape=shape)
    fileout[event_name].attrs['label'] = label
    fileout[event_name].attrs['weight'] = filein[event]['weight'][()]

    cluse, cluseta = filein[event]['clusE'][()] / 10000, filein[event]['clusEta'][()]
    cluspt = cluse / np.cosh(cluseta)
    fileout[event_name][:, 0] = cluse
    fileout[event_name][:, 1] = cluseta
    fileout[event_name][:, 2] = filein[event]['clusPhi'][()]
    fileout[event_name][:, 3] = cluspt
    fileout[event_name][:, 4] = filein[event]['clusEM'][()] / 10000


def _transfer_one_file(datapath, files, idx_permutation, label, train_thr):
    filein = h5.File(datapath, 'r')
    for event in filein:
        new_idx = idx_permutation.pop()
        if new_idx < train_thr:
            _transfer_one_event(filein, files['test'], event, new_idx, label)
        else:
            _transfer_one_event(filein, files['train'], event, new_idx - train_thr, label)
    filein.close()


def main(datadir, outputdir, nb_events=None, dest=None):
    def print_(string, dest=None):
        if dest is None:
            print(string)
        else:
            with open(dest, 'a') as fileout:
                fileout.write(string + '\n')

    print_fun = lambda string: print_(string, dest)
    datafiles = _data_files(datadir)
    if nb_events is None:
        nb_events = _count_events(datadir, datafiles, print_fun)
    print_fun('rearanging {} events...'.format(nb_events))
    print_fun('...done\n')
    trainfile = h5.File(path.join(outputdir, 'data_train.h5'), 'w')
    testfile = h5.File(path.join(outputdir, 'data_test.h5'), 'w')

    train_thr = nb_events // 20
    testfile.create_dataset('nbevent', data=(train_thr,))
    trainfile.create_dataset('nbevent', data=(nb_events - train_thr,))
    idx_permutation = list(permutation(nb_events))

    for filein in datafiles:
        print_fun('{} start...'.format(filein))
        datapath = path.join(datadir, filein)
        label = filein.startswith('GG')
        files = {'train': trainfile, 'test': testfile}
        _transfer_one_file(datapath, files, idx_permutation, label, train_thr)
        print_fun('...done\n{} events transfered'.format(nb_events - len(idx_permutation)))


if __name__ == '__main__':
    indir = '/global/homes/r/rochette/lhc_gnn/data/rawdata'
    outdir = '/global/homes/r/rochette/lhc_gnn/data'
    dest = '/global/homes/r/rochette/preprocess.out'
    
    if dest is not None:
        with open(dest, 'w') as fileout:
            pass  # empty file
        
    main(indir, outdir, dest=dest)
    print('DONE')
