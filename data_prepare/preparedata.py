from os.path import join
import argparse
import pickle
from utils import makedir_if_not_there, print_
from file2weightfactor import init_weight_factors, NoWeightFactor
from len2namenum import len2namenum
from normalizefeatures import global_stats
from groupdata import group_batchs


"""Recovers data from a `rawdatadir` directory, selects usefull pieces of data
and transfers it to a `datadir` directory according to the following format :

`datadir`/
    |-- train/  # contains everything concerning the training set
    |       |-- weightfactors.pkl
    |       |-- len2namenum.pkl
    |       |-- data.h5  # hdf5 file containing all training batch
    |               |-- nb_events  # total number of events in data.h5
    |               |-- nb_batch   # total number of batch in data.h5
    |               |-- batch0     # one batch of data
    |               |       |-- E       # energy
    |               |       |-- EM      # electro magnetic energy
    |               |       |-- eta     # pseudorapidity
    |               |       |-- phi     # azimuth
    |               |       |-- label   # label
    |               |       |-- weight  # weight
    |               |
    |               |-- batch1
    |               |       |...
    |               |
    |               |...
    |               |
    |               |-- batch`maxbatch`
    |                       |...
    |
    |-- test/  # contains everything concerning the testing set
            |-- weightfactors.pkl
            |-- len2namenum.pkl
            |-- data.h5
                    |...
"""


# Some functions to seperate training data from testing data

def is_test_nersc(filename):
    if not(filename.endswith('.h5')):
        raise ValueError('`{}` is not an hdf5 file.'.format(filename))
    filename = filename.split('.')[0]  # removes `.h5` extension
    fileid = filename.split('_')[-1]
    return (fileid == '01')


def is_train_nersc(filename):
    return not(is_test_nersc(filename))


def is_test_nyu(filename):
    return ('test' in filename)


def is_train_nyu(filename):
    return not(is_test_nyu(filename))


# A function to read `stdin` and recover arguments

def readargs():
    descr = 'Reads data from `rawdata` directory, transfers relevant data ' + \
        'to `data` directory, divided in testing and training sets.'
    parser = argparse.ArgumentParser(description=descr)
    add_arg = parser.add_argument

    add_arg('--project', dest='project', required=True,
            help='type of data to be organized : `nersc` or `nyu`')
    add_arg('-i', '--rawdata', dest='rawdatadir', required=True,
            help='path to unprocessed data')
    add_arg('-o', '--data', dest='datadir', required=True,
            help='path to store processed data')
    add_arg('--batchsize', dest='batchsize', type=int, default=20,
            help='number of events per batch')
    add_arg('--stdout', dest='stdout', default=None,
            help='file to redirect prints')

    add_arg('--dotrain', dest='dotrain', action='store_true',
            help='process training set if true')
    add_arg('--dotest', dest='dotest', action='store_true',
            help='process testing set if true')

    add_arg('--wftrain', dest='wftrain', action='store_true',
            help='run weight factors')
    add_arg('--l2nntrain', dest='l2nntrain', action='store_true',
            help='run len2namenum')
    add_arg('--statstrain', dest='statstrain', action='store_true',
            help='run normalizefeature')
    add_arg('--groupbatchtrain', dest='gbtrain', action='store_true',
            help='run group_batchs')


    add_arg('--wftest', dest='wftest', action='store_true',
            help='same as --wfdone but restricted to testing set')
    add_arg('--l2nntest', dest='l2nntest', action='store_true',
            help='same as --l2nndone but restricted to testing set')
    add_arg('--groupbatchtest', dest='gbtest', action='store_true',
            help='run group_batchs')

    args = parser.parse_args()
    return args


# main function to call all other function with one mode

def prepare_data(datatype, args):
    """preprocess one dataset, either train or test"""

    # get names of datasets to copy
    if args.project == 'nersc':
        datasets = [
            ('clusE', 'E'), ('clusEM', 'EM'),
            ('clusPhi', 'Phi'), ('clusEta', 'Eta')]
        datasetnormalize = ['clusE', 'clusEM']
        attributes = []
        normalize_weights = True
        is_used = is_test_nersc if datatype == 'test' else is_train_nersc  # recognise used files
    elif args.project == 'nyu':
        datasets = [('p', 'p'), ('eta', 'eta'), ('phi', 'phi'), ('E', 'E'),
                    ('pt', 'pt'), ('theta', 'theta')]
        datasetnormalize = []
        attributes = ['jet_phi', 'jet_pt', 'jet_energy', 'jet_eta', 'jet_mass', 'jet_length']
        normalize_weights = False
        is_used = is_test_nyu if datatype == 'test' else is_train_nyu  # recognise used files
    else:
        raise ValueError('--project should be `nersc` or `nyu`')

    print_('\n----- PROCESSING {} DATA -----\n'.format(datatype.upper()), args.stdout)

    # make processed data directory for type `datatype`
    if datatype not in ['train', 'test']:
        raise ValueError(
            'Unknown value :`{}` is neither `train` nor `test`'.format(datatype))
    datadir = join(args.datadir, datatype)
    len2numdir = join(datadir, 'len2num')
    makedir_if_not_there(len2numdir)  # also makes datadir
    statsdir = join(datadir, 'stats')
    makedir_if_not_there(statsdir)

    # weight renormalizers
    if normalize_weights:
        if args.__dict__['wf' + datatype]:
            weight_factors = init_weight_factors(is_used, args.rawdatadir)
            with open(join(datadir, 'weightfactors.pkl'), 'wb') as wffile:
                pickle.dump(weight_factors, wffile, pickle.HIGHEST_PROTOCOL)
            print_(
                '\nSaved `weightfactors.pkl` in `{}`\n'.format(datadir) +
                '\n'.join(
                    '{}: {}'.format(filename, weight_factors[filename])
                    for filename in weight_factors) +
                '\n' + '-' * 30 + '\n',
                args.stdout)
        else:
            print_('`weightfactors.pkl` reused for set `{}`'.format(datatype))
            with open(join(datadir, 'weightfactors.pkl'), 'rb') as wffile:
                weight_factors = pickle.load(wffile)
    else:
        weight_factors = NoWeightFactor()

    # make and save len2namenum dictionary
    if args.__dict__['l2nn' + datatype]:
        l2nn = len2namenum(is_used, args.rawdatadir, len2numdir, args.stdout, reprocess=False)
        with open(join(datadir, 'len2namenum.pkl'), 'wb') as l2nnfile:
            pickle.dump(l2nn, l2nnfile, pickle.HIGHEST_PROTOCOL)
        print_('\nSaved `len2namenum.pkl` in `{}`\n'.format(datadir) +
               '-' * 30 + '\n',
               args.stdout)
    else:
        print_('`len2namenum.pkl` reused for set `{}`'.format(datatype))
        with open(join(datadir, 'len2namenum.pkl'), 'rb') as l2nnfile:
            l2nn = pickle.load(l2nnfile)

    # compute mean and variance for each feature
    if (datatype == 'train') & args.statstrain:
        stats = global_stats(
            is_used, args.rawdatadir, statsdir,
            args.stdout, reprocess=False
        )
        with open(join(datadir, 'stats.pkl'), 'wb') as statsfile:
            pickle.dump(stats, statsfile, pickle.HIGHEST_PROTOCOL)
        print_('\nSaved `stats.pkl` in `{}`\n'.format(datadir) +
               '-' * 30 + '\n',
               args.stdout)
    else:
        print_('`stats.pkl` reused for set `{}`'.format(datatype))
        with open(join(join(args.datadir, 'train'), 'stats.pkl'), 'rb') as statsfile:
            stats = pickle.load(l2nnfile)

    # organise data
    if args.__dict__['gb' + datatype]:
        group_batchs(l2nn,
                     datasets, datasetnormalize, attributes,
                     args.batchsize, weight_factors,
                     args.rawdatadir, datadir)
        print_('\nFinished organizing {} data.\n'.format(datatype) +
               '-' * 30 + '\n',
               args.stdout)


# call `prepare_data` on both the training and testing modes

def main(args):
    if args.dotrain:
        prepare_data('train', args)
    if args.dotest:
        prepare_data('test', args)


if __name__ == '__main__':
    main(readargs())
