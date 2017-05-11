from os.path import join
import argparse
from config.hyperparameters import hyperparameters
from config.globalconfig import globalconfig


def readargs(description):
    """Reads from stdin and returns arguments in a dictionary container"""

    parser = argparse.ArgumentParser(description=description)
    add_arg = parser.add_argument

    # training or testing
    add_arg('--mode', dest='mode',
            help="'test', 'train', 'description', 'plot', prepare_data', 'weight_average'")
    add_arg('--datatype', dest='datatype',
            help="'train' or 'test': type of data observed")

    # paths and modelname
    add_arg('--model', dest='model',
            help='model name')
    add_arg('--datadir', dest='datadir',
            help='path to data')
    add_arg('--netdir', dest='netdir',
            help='path to models directory')
    add_arg('--stdout', dest='stdout',
            help='redirects stdout')

    # model initialization parameters
    add_arg('--modeltype', dest='modeltype',
            help='type of architecture')
    add_arg('--batchnorm', dest='batchnorm', action='store_true',
            help='use batch normalisation in layers')

    add_arg('--dim', dest='dim', type=int, nargs='+',
            help='list of dimensions for GNN layers')
    add_arg('--deg', dest='deg', type=int, nargs='+',
            help='list of degrees for GNN layers')
    add_arg('--modifdeg', dest='modifdeg', type=int, nargs='+',
            help='list of degrees for GNN final modification layers')

    add_arg('--nb_layer', dest='nb_layer', type=int,
            help='number of layers in GNN')
    add_arg('--deg_layer', dest='deg_layer', type=int,
            help='degree of each layers in GNN')
    add_arg('--feature_maps', dest='feature_maps', type=int,
            help='number of feature maps for each layer in GNN')
    add_arg('--nb_modiflayer', dest='nb_modiflayer', type=int,
            help='number of modification layers in GNN')
    add_arg('--deg_modiflayer', dest='deg_modiflayer', type=int,
            help='degree of each modification layers in GNN')

    add_arg('--lr', dest='lr', type=float,
            help='initial learning rate')
    add_arg('--logistic_bias', dest='logistic_bias', type=float,
            help='biais applied before logistic regression')

    # training parameters
    add_arg('--epoch', dest='epoch', type=int,
            help='number of epoch for training')
    add_arg('--nb_batch', dest='nb_batch', type=int,
            help='number of batchs for training')
    add_arg('--lr_thr', dest='lr_thr', type=float,
            help='threshold to update learning rate')
    add_arg('--lr_update', dest='lr_update', type=float,
            help='multiplication factor to update learning rate')
    add_arg('--lr_nbbatch', dest='lr_nbbatch', type=int,
            help='time window (in number of batch) to update learning rate')
    add_arg('--weightfunc', dest='weightfunc',
            help='name of function that should be applied to use custom weights')
    add_arg('--nb_save', dest='nb_save', type=int,
            help='number of batchs after which the model is saved')

    add_arg('--optimizer', dest='optimizer',
            help='optimizer method')
    add_arg('--loss', dest='loss',
            help='criterion function')

    # statistics parameters
    add_arg('--nbdisplay', dest='nbdisplay', type=int,
            help='number of batchs to average statistics')
    add_arg('--nbstep', dest='nbstep', type=int,
            help='number of batchs in one step for statistics')

    # graphics parameters
    add_arg('--zoom', dest='zoom', type=float, nargs='+',
            help='list of False Positive rate for zooming on ROC curve')

    # cuda
    add_arg('--cuda', dest='cuda', action='store_true',
            help='run model on GPU')

    # verbosity
    add_arg('--quiet', dest='quiet', action='store_true',
            help='decrease verbosity')

    args = parser.parse_args()
    return(args.__dict__)


class Config:
    def __init__(self, description='python script used for GNN training or testing'):
        self.update(hyperparameters())  # code related parameters
        self.update(globalconfig)
        param = readargs(description)
        self.update(param)

        # some paths that depend on input
        self.netdir = join(self.netdir, self.model)
        self.graphdir = join(self.netdir, 'graphic')
        self.statdir = join(self.netdir, 'stat')

        self.check_mode()
        if self.mode == 'test':
            self.epoch = 1
        elif self.mode == 'train':
            self.datatype = 'train'

        self.init_same_layers_parameters()

    # mode should be either 'test' or 'train'
    def check_mode(self):
        if self.mode not in self.possible_modes:
            raise Exception('Unknown mode : {}\n'.format(self.mode))

    # some attributes are not necessary
    def __getattr__(self, name):
        """Handles default values for some arguments"""

        if name == 'nb_batch':
            return float('+inf')
        elif name in ['zoom', 'stdout', 'weightfunc']:
            return None
        elif name == 'logistic_bias':
            return 0
        else:
            raise AttributeError('Missing parameter : {}\n'.format(name))

    def init_same_layers_parameters(self):
        """simple parameters for network initialisation with same parameters for
        each layer"""
        if hasattr(self, 'nb_layer'):
            self.dim = [self.feature_maps] * (self.nb_layer - 1)
        if hasattr(self, 'deg_layer'):
            self.deg = [self.deg_layer] * self.nb_layer
        if hasattr(self, 'deg_modiflayer'):
            self.modifdeg = [self.deg_modiflayer] * self.nb_modiflayer

    def update(self, dictionnary):
        """adds key -> value from dictionnary if value is not None"""
        for key in dictionnary.keys():
            if dictionnary[key] is not None:
                self.__dict__[key] = dictionnary[key]
