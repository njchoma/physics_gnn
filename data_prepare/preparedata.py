from os.path import join
import argparse
from file2weightfactor import init_weight_factors
from len2namenum import len2namenum
from groupdata import group_batchs
from utils import makedir_if_not_there, print_
import pickle


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

def is_test(filename):
    if not(filename.endswith('.h5')):
        raise ValueError('`{}` is not an hdf5 file.'.format(filename))
    filename = filename.split('.')[0]  # removes `.h5` extension
    fileid = filename.split('_')[-1]
    return (fileid == '01')


def is_train(filename):
    return not(is_test(filename))


# A function to read `stdin` and recover arguments

def readargs():
    descr = 'Reads data from `rawdata` directory, transfers relevant data ' + \
        'to `data` directory, divided in testing and training sets.'
    parser = argparse.ArgumentParser(description=descr)
    add_arg = parser.add_argument

    add_arg('-i', '--rawdata', dest='rawdata', required=True,
            help='path to unprocessed data')
    add_arg('-o', '--data', dest='data', required=True,
            help='path to store processed data')
    add_arg('--batchsize', dest='batchsize', type=int, default=None,
            help='number of events per batch')
    add_arg('--stdout', dest='stdout', default=None,
            help='file to redirect prints')

    add_arg('--wfdone', dest='wfdone', action='store_true',
            help='when weightfactors are already done')
    add_arg('--l2nndone', dest='l2nndone', action='store_true',
            help='when len2namenum are already done')

    # same as two last argument for specific datasets

    add_arg('--wftraindone', dest='wftraindone', action='store_true',
            help='same as --wfdone but restricted to training set')
    add_arg('--l2nntraindone', dest='l2nntraindone', action='store_true',
            help='same as --l2nndone but restricted to training set')

    add_arg('--wftestdone', dest='wftestdone', action='store_true',
            help='same as --wfdone but restricted to testing set')
    add_arg('--l2nntestdone', dest='l2nntestdone', action='store_true',
            help='same as --l2nndone but restricted to testing set')

    args = parser.parse_args()
    return args.__dict__


# main function to call all other function with one mode

def prepare_data(rawdatadir, datadir, batchsize, datatype, args_done, stdout=None):
    print_('\n----- PROCESSING {} DATA -----\n'.format(datatype.upper()), stdout)

    # make processed data directory for type `datatype`
    if datatype not in ['train', 'test']:
        raise ValueError(
            'Unknown value :`{}` is neither `train` nor `test`'.format(datatype))
    datadir = join(datadir, datatype)
    makedir_if_not_there(datadir)

    # recognise used files
    is_used = is_test if datatype == 'test' else is_train

    # weight renormalizers
    if args_done['wfdone']:
        print_('`weightfactors.pkl` reused for set `{}`'.format(datatype))
        with open(join(datadir, 'weightfactors.pkl'), 'rb') as wffile:
            weight_factors = pickle.load(wffile, pickle.HIGHEST_PROTOCOL)
    else:
        weight_factors = init_weight_factors(is_used, rawdatadir)
        with open(join(datadir, 'weightfactors.pkl'), 'wb') as wffile:
            pickle.dump(weight_factors, wffile, pickle.HIGHEST_PROTOCOL)
        print_(
            '\nSaved `weightfactors.pkl` in `{}`\n'.format(datadir) +
            '\n'.join(
                '{}: {}'.format(filename, weight_factors[filename])
                for filename in weight_factors) +
            '\n' + '-' * 30 + '\n',
            stdout)

    # make and save len2namenum dictionary
    if args_done['l2nndone']:
        print_('`len2namenum.pkl` reused for set `{}`'.format(datatype))
        with open(join(datadir, 'len2namenum.pkl'), 'rb') as l2nnfile:
            l2nn = pickle.load(l2nnfile, pickle.HIGHEST_PROTOCOL)
    else:
        l2nn = len2namenum(is_used, rawdatadir, stdout)
        with open(join(datadir, 'len2namenum.pkl'), 'wb') as l2nnfile:
            pickle.dump(l2nn, l2nnfile, pickle.HIGHEST_PROTOCOL)
        print_('\nSaved `len2namenum.pkl` in `{}`\n'.format(datadir) +
               '-' * 30 + '\n',
               stdout)

    # organise data
    group_batchs(l2nn, batchsize, weight_factors, rawdatadir, datadir)
    print_('\nFinished organizing {} data.\n'.format(datatype) +
           '-' * 30 + '\n',
           stdout)


def which_done(args, mode):
    return {
        'wfdone': args['wfdone'] or args['wf' + mode + 'done'],
        'l2nndone': args['l2nndone'] or args['l2nn' + mode + 'done']
    }


# call `prepare_data` on both the training and testing modes

def main(args):
    if args['batchsize'] is None:
        args['batchsize'] = 20

    prepare_data(args['rawdata'], args['data'], args['batchsize'],
                 'train', which_done(args, 'train'), args['stdout'])
    prepare_data(args['rawdata'], args['data'], args['batchsize'],
                 'test', which_done(args, 'test'), args['stdout'])


if __name__ == '__main__':
    args = readargs()
    main(args)
