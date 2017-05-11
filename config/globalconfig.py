from config.getdir import getdatadir, getnetdir, getstdout

"""Values by default for parameters. Any parameters given would
overwrite those"""


globalconfig = {
    # paths and stdout
    'datadir': getdatadir(),
    'netdir': getnetdir(),
    'stdout': getstdout(),

    # training parameters
    'epoch': 10,
    'nb_save': 50000,

    # optimizer options
    'optimizer': 'Adam',
    'loss': 'BCE',
    'lr': 0.0002,
    'lr_thr': 0.98,
    'lr_update': 0.90,
    'lr_nbbatch': 20000,

    # printing parameters
    'nbdisplay': 500,
    'nbstep': 5,

    # ploting parameters
    'zoom': [1, 0.1, 0.01, 0.001, 0.0001],
    'possible_stats': ['loss', 'kernel', 'avg1', 'std1', 'avg0', 'std0'],

    # initialisation parameters
    'modeltype': 'GNNModifSpatial',
    'dim': [5] * 1,
    'deg': [1] * 2,
    'modifdeg': [1] * 2,

}
