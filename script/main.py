import logging
import os.path as path
import read_args as ra

from run_experiment import train_model
from utils.in_out import make_dir_if_not_there


def main():
    """Reads args, loads specified dataset, and trains model"""

    args = ra.read_args()

    # Set up model directory
    project_root_dir = path.dirname(path.abspath(path.join(__file__, '..')))
    modelsdir = path.join(project_root_dir, 'models' + args.data)
    args.savedir = path.join(modelsdir, args.name)
    make_dir_if_not_there(args.savedir)

    # Set up logging
    if (args.quiet):
      logging_level = logging.WARNING
    else:
      logging_level = logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s',level=logging.DEBUG)
    
    # Dataset-specific operations
    logging.info("Loading data...")
    if args.data == 'NYU':
        from loading.data.nyu.load_data import load_raw_data
        datadir = '/data/grochette/data_nyu/'
        trainfile = 'antikt-kt-train-gcnn.pickle'
        testfile  = 'antikt-kt-test.pickle'
        train_X, train_y, train_w = load_raw_data(datadir+trainfile, args.nbtrain,'train')
        test_X, test_y, test_w    = load_raw_data(datadir+testfile, args.nbtest, 'test')
        args.first_fm = 6
    elif args.data == 'NERSC':
        from loading.data.nersc.load_data import load_raw_data
        datadir = '/data/grochette/data_nersc/'
        train_X, train_y, train_w = load_raw_data(datadir, args.nbtrain, 'train')
        test_X, test_y, test_w    = load_raw_data(datadir, args.nbtest, 'test')
        args.first_fm = 5
    else:
        raise ValueError('--data should be NYU or NERSC')
    logging.info("Data loaded")

    # Update number train, test in case requested more data than available
    args.nbtrain = len(train_X)
    args.nbtest  = len(test_X)

    # Train
    train_model(args, train_X, train_y, train_w, test_X, test_y, test_w)


if __name__ == '__main__':
    main()
