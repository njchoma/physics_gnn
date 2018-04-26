import logging
import os.path as path

from experiment_handler import train_model
from utils.in_out import make_dir_if_not_there
import loading.model.read_args as ra
import loading.model.model_parameters as param


def main():
    """Reads args, loads specified dataset, and trains model"""

    # Parse args
    args = ra.read_args()

    # Set up model directory
    project_root_dir = path.dirname(path.abspath(path.join(__file__, '..')))
    modelsdir = path.join(project_root_dir, 'models' + args.data)
    args.savedir = path.join(modelsdir, args.name)
    make_dir_if_not_there(args.savedir)

    # Set up logging
    logfile = path.join(args.savedir, "log.txt")
    if (args.quiet):
      logging_level = logging.WARNING
    else:
      logging_level = logging.INFO
    logging.basicConfig(filename=logfile,format='%(message)s',level=logging_level)
    logging.getLogger().addHandler(logging.StreamHandler())
    
    # Set global model parameters
    # Restores model parameters if some training has already occurred
    param.init(args)

    # Dataset-specific operations
    logging.info("Loading data...")
    if param.args.data == 'NYU':
        from loading.data.nyu.load_data import load_raw_data
        datadir = '/misc/vlgscratch4/BrunaGroup/data_nyu/'
        trainfile = 'antikt-kt-train-gcnn.pickle'
        testfile  = 'antikt-kt-test.pickle'
        train_X, train_y, train_w = load_raw_data(
                                                  datadir+trainfile, 
                                                  param.args.nbtrain,
                                                  'train'
                                                  )
        test_X,  test_y,  test_w  = load_raw_data(
                                                  datadir+testfile, 
                                                  param.args.nbtest, 
                                                  'test'
                                                  )
        param.args.first_fm = 6
        param.args.spatial_coords = [1,2]
    elif param.args.data == 'NERSC':
        from loading.data.nersc.load_data import load_raw_data
        datadir = '/misc/vlgscratch4/BrunaGroup/data_nersc'
        train_X, train_y, train_w = load_raw_data(
                                                  datadir, 
                                                  param.args.nbtrain, 
                                                  'train'
                                                  )
        test_X,  test_y,  test_w  = load_raw_data(
                                                  datadir, 
                                                  param.args.nbtest, 
                                                  'test'
                                                  )
        param.args.first_fm = 5
        param.args.spatial_coords = [1,2]
    elif param.args.data == 'ICECUBE':
        from loading.data.icecube.load_data import load_raw_data
        # datadir = '/global/homes/n/njchoma/data/njc_data'
        datadir = '/home/nc2201/data/icecube'
        trainfile = 'train.pickle'
        testfile  = 'test.pickle'
        train_X, train_y, train_w = load_raw_data(
                                                  path.join(datadir,trainfile), 
                                                  param.args.nbtrain
                                                  )
        test_X,  test_y,  test_w  = load_raw_data(
                                                  path.join(datadir,testfile), 
                                                  param.args.nbtest
                                                  )
        param.args.first_fm = 6
        param.args.spatial_coords = [0,1,2]
    else:
        raise ValueError('--data should be NYU or NERSC')
    logging.info("Data loaded")

    # Update number train, test in case requested more data than available
    param.args.nbtrain = len(train_X)
    param.args.nbtest  = len(test_X)

    # Train
    try:
      train_model(train_X, train_y, train_w, test_X, test_y, test_w)
    except Exception as e:
      logging.error(e)
      raise e


if __name__ == '__main__':
    main()
