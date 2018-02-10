import argparse
from os.path import exists, join
import pickle
from model import kernel as ker
from model import multi_kernel as mker
from model import adj_kernel as adj_ker
from model import gcnn
from utils.in_out import print_


def read_args():
    """Parses stdin for arguments used for training or network initialisation"""

    parser = argparse.ArgumentParser(description='simple arguments to train GCNN')
    add_arg = parser.add_argument

    add_arg('--name', dest='name', help='network reference')
    add_arg('--data', dest='data', help='project to take data from', default='NYU')

    add_arg('--kernel', dest='kernel', help='name of kernel used', default='FCG')
    add_arg('--sigma', dest='sigma', help='kernel stdev initial value', type=float)

    add_arg('--fm', dest='nb_feature_maps', help='number of feature maps per layer', type=int)
    add_arg('--edge_fm', dest='nb_edge_feature', type=int,
            help='number of edge features for GCNN_EdgeFeature')
    add_arg('--depth', dest='nb_layer', help='number of layers', type=int)

    add_arg('--cuda', dest='cuda', action='store_true')

    add_arg('--nbtrain', dest='nbtrain', help='number of training examples', type=int)
    add_arg('--nbtest', dest='nbtest', help='number of testing examples', type=int)

    add_arg('--nbprint', dest='nbprint', help='print frequency', type=int, default=10000)
    add_arg('--quiet', dest='quiet', help='reduces print', action='store_true')

    add_arg('--lr', dest='lrate', help='learning rate', type=float)
    add_arg('--lrdecay', dest='lrdecay', help='learning rate decay, `lr *= lrdecay` each epoch',
            type=float, default=0.95)
    add_arg('--adj_kernel', dest='adj_kernel', help='name of kernel for updating adjacency matrix',default='Identity')

    args = parser.parse_args()
    return args


def get_fixed_param(data_type, project_root_dir):
    """reads parameters from 'new_net_param.txt',
    creates the file if non-existant
    """

    def _get_fixed_param(param_file):
        args = dict()
        for line in open(param_file, 'r'):
            if line.strip():  # not empty line
                arg_txt = line.split('#')[0]  # remove comment
                arg_name, arg_val = arg_txt.split('=')[:2]
                arg_name, arg_val = arg_name.strip(), arg_val.strip()
                args[arg_name] = arg_val
                if arg_val == '':
                    raise ValueError(
                        "Empty parameter in 'param.txt': {}".format(arg_name))
                print("param {} : '{}'".format(arg_name, arg_val))
        return args

    param_file = 'param' + data_type + '.txt'

    if exists(param_file):
        return _get_fixed_param(param_file)

    default_data_path = join(project_root_dir, 'data' + data_type + '/')
    if data_type == 'NYU':
        default_train = join(default_data_path, 'train_uncropped.pickle')
        default_testtrain = join(default_data_path, 'train_cropped.pickle')
        default_test = join(default_data_path, 'test_cropped.pickle')
    else:
        default_train = join(default_data_path, 'train.h5')
        default_testtrain = join(default_data_path, 'train.h5')
        default_test = join(default_data_path, 'test.h5')

    with open(param_file, 'w') as paramfile:
        paramfile.write(
            "\ntrainfile = {} # path to training data (not cropped)\n".format(default_train)
            + "testtrainfile = {} # path to training data (cropped)\n".format(default_testtrain)
            + "testfile = {} # path to testing data (cropped)\n".format(default_test)
        )


def init_network(args, frst_fm):
    """Reads args and initiates a network accordingly.
    Input should be the output of `read_args`.
    """

    loop2pi = args.data == 'NERSC'  # in NERSC data, phi is 2pi-periodic

    if args.kernel in ['FCG', 'FCG_nodiag', 'FCG_norm', 'FCG_nodiag_norm', 'FQCDAware',
                       'QCDAware', 'QCDAwareMeanNorm', 'QCDAwareNoNorm']:
        if args.kernel == 'FCG':
            kernel = ker.FixedComplexGaussian(args.sigma, periodic=loop2pi)
        elif args.kernel == 'FCG_nodiag':
            kernel = ker.FixedComplexGaussian(args.sigma, diag=False, periodic=loop2pi)
        elif args.kernel == 'FCG_norm':
            kernel = ker.FixedComplexGaussian(args.sigma, norm=True, periodic=loop2pi)
        elif args.kernel == 'FCG_nodiag_norm':
            kernel = ker.FixedComplexGaussian(args.sigma, diag=False, norm=True, periodic=loop2pi)
        elif args.kernel == 'FQCDAware':
            kernel = ker.FixedQCDAware(0.5, 0.1, periodic=loop2pi)
        elif args.kernel == 'QCDAware':
            kernel = ker.QCDAware(1., 0.7, periodic=loop2pi)
        elif args.kernel == 'QCDAwareMeanNorm':
            kernel = ker.QCDAwareMeanNorm(1., 0.7, periodic=loop2pi)
        elif args.kernel == 'QCDAwareNoNorm':
            kernel = ker.QCDAwareNoNorm(1., 0.7, periodic=loop2pi)
        
        adj_kernel_args = (args.nb_feature_maps,)
        if args.adj_kernel == 'Gaussian':
            adj_kernel = adj_ker.Gaussian
        elif args.adj_kernel == 'DirectedGaussian':
            adj_kernel = adj_ker.DirectedGaussian
        elif args.adj_kernel == 'MPNNdirected':
            adj_kernel = adj_ker.MPNNdirected
        elif args.adj_kernel == 'MLPdirected':
            adj_kernel = adj_ker.MLPdirected
        else:
            adj_kernel = adj_ker.Identity

        adj_kernels = [adj_kernel(*adj_kernel_args) for _ in range(args.nb_layer-1)]


        return gcnn.GCNNSingleKernel(
            kernel, adj_kernels, frst_fm, args.nb_feature_maps, args.nb_layer
            )

    elif args.kernel == 'LayerQCDAware':
        kernel = mker.MultiQCDAware
        return gcnn.GCNNLayerKernel(
            kernel, frst_fm, args.nb_feature_maps, args.nb_edge_feature,
            args.nb_layer, periodic=loop2pi
            )

    elif args.kernel == 'MultiQCDAware':
        kernel = mker.MultiQCDAware
        return gcnn.GCNNMultiKernel(
            kernel, frst_fm, args.nb_feature_maps, args.nb_edge_feature,
            args.nb_layer, periodic=loop2pi
            )

    elif args.kernel in ['EdgeFeature', 'GatedEdgeFeature']:
        if args.kernel == 'EdgeFeature':
            kernel = mker.Node2Edge
        elif args.kernel == 'GatedEdgeFeature':
            kernel = mker.GatedNode2Edge

        return gcnn.GCNNEdgeFeature(
            kernel, frst_fm, args.nb_feature_maps, args.nb_edge_feature, args.nb_layer
            )

    else:
        raise ValueError('Unknown kernel : {}'.format(args.kernel))


def make_net_if_not_there(args, frst_fm, savedir):
    """Checks for existing network, initiates one if non existant"""

    model_path = join(savedir, args.name)
    if exists(model_path + '.pkl'):
        with open(model_path + '.pkl', 'rb') as filein:
            net = pickle.load(filein)
        print_('Network recovered from previous training', args.quiet)
    else:
        net = init_network(args, frst_fm)
        print_('Network created', args.quiet)
        with open(model_path + '.csv', 'w') as fileres:
            fileres.write('Learning Rate, Train Loss, Test Loss, Train AUC Score'
                          + ', Test AUC Score, 1/FPR_train, 1/FPR_test, Running Loss\n')
    print_('parameters : {}'.format(sum([param.numel() for param in net.parameters()])), args.quiet)

    return net
