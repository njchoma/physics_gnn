from os import path, listdir
import h5py as h5
import numpy as np
from numpy.random import permutation
from file2weightfactor import make_weightfactor_if_not_there, findtype


########################################
######## pT = clusE / cosh(Eta) ########
########################################

def _data_files(datadir, is_used):
    datafiles = [filename for filename in listdir(datadir) if filename.endswith('.h5')]
    datafiles = [filename for filename in datafiles if 'data' not in filename]
    datafiles = [filename for filename in datafiles if is_used(filename)]

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


def _transfer_one_event(filein, fileout, event, new_idx, label, wf_file):
    event_name = 'event_{}'.format(new_idx)
    shape = filein[event]['clusE'].shape + (5,)
    fileout.create_dataset(event_name, shape=shape)
    fileout[event_name].attrs['label'] = label
    fileout[event_name].attrs['weight'] = filein[event]['weight'][()] * wf_file

    cluse, cluseta = filein[event]['clusE'][()] / 10000, filein[event]['clusEta'][()]
    cluspt = cluse / np.cosh(cluseta)

    fileout[event_name][:, 0] = cluse
    fileout[event_name][:, 1] = cluseta
    fileout[event_name][:, 2] = filein[event]['clusPhi'][()]
    fileout[event_name][:, 3] = filein[event]['clusEM'][()] / 10000
    fileout[event_name][:, 4] = cluspt


def _transfer_one_file(datapath, fileout, idx_permutation, label, wf_file):
    with h5.File(datapath, 'r') as filein:
        for event in filein:
            new_idx = idx_permutation.pop()
            _transfer_one_event(filein, fileout, event, new_idx, label, wf_file)


def _create_dataset(datadir, outputpath, is_used, weight_fact):
    def print_(string, dest=None):
        if dest is None:
            print(string)
        else:
            with open(dest, 'a') as fileout:
                fileout.write(string + '\n')

    print_fun = lambda string: print_(string, dest)
    datafiles = _data_files(datadir, is_used)
    nb_events = _count_events(datadir, datafiles, print_fun)
    print_fun('rearanging {} events...'.format(nb_events))
    print_fun('...done\n')
    fileout = h5.File(outputpath, 'w')

    fileout.create_dataset('nbevent', data=(nb_events,))
    idx_permutation = list(permutation(nb_events))

    for filein in datafiles:
        print_fun('{} start...'.format(filein))
        datapath = path.join(datadir, filein)
        label = filein.startswith('GG')
        wf_file = weight_fact[filein]
        _transfer_one_file(datapath, fileout, idx_permutation, label, wf_file)
        print_fun('...done\n{} events transfered'.format(nb_events - len(idx_permutation)))


def main(datadir, outputdir, dest=None):
    is_test = lambda string: string.endswith('_01.h5')
    weight_test = make_weightfactor_if_not_there(datadir, path.join(outputdir, 'wf_test.pkl'), is_test)
    print('weights calculated for testing set')
    _create_dataset(datadir, path.join(outputdir, 'data_test.h5'), is_test, weight_test)

    is_train = lambda string: not is_test(string)
    weight_train = make_weightfactor_if_not_there(datadir, path.join(outputdir, 'wf_train.pkl'), is_train)
    print('weights calculated for training set')
    _create_dataset(datadir, path.join(outputdir, 'data_train.h5'), is_train, weight_train)


if __name__ == '__main__':
    indir = '/data/grochette/data_nersc'
    outdir = '/data/grochette/data_nersc'
    dest = None
    
    # indir = '/home/gaspar/Desktop/rawdataNERSC'
    # outdir = '/home/gaspar/Desktop/dataNERSC'
    # dest = None
    
    if dest is not None:
        with open(dest, 'w') as fileout:
            pass  # empty file
        
    main(indir, outdir, dest=dest)
    print('DONE')
