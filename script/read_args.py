import argparse
from os.path import exists, join
import pickle
from model import kernel as ker
from model import multi_kernel as mker
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
    with open(param_file, 'w') as paramfile:
        paramfile.write(
            "\ntrainfile = {} # path to training data (not cropped)\n".format(default_data_path)
            + "testtrainfile = {} # path to training data (cropped)\n".format(default_data_path)
            + "testfile = {} # path to testing data (cropped)\n".format(default_data_path)
        )
    raise FileNotFoundError(
        "\n'" + param_file + "' created, please provide paths to training and testing sets\n"
        )


def init_network(args, frst_fm):
    """Reads args and initiates a network accordingly.
    Input should be the output of `read_args`.
    """

    if args.kernel in ['FCG', 'FCG_nodiag', 'FCG_norm', 'FCG_nodiag_norm', 'FQCDAware', 'QCDAware']:
        if args.kernel == 'FCG':
            kernel = ker.FixedComplexGaussian(args.sigma)
        elif args.kernel == 'FCG_nodiag':
            kernel = ker.FixedComplexGaussian(args.sigma, diag=False)
        elif args.kernel == 'FCG_norm':
            kernel = ker.FixedComplexGaussian(args.sigma, norm=True)
        elif args.kernel == 'FCG_nodiag_norm':
            kernel = ker.FixedComplexGaussian(args.sigma, diag=False, norm=True)
        elif args.kernel == 'FQCDAware':
            kernel = ker.FixedQCDAware(1.0, 0.1)
        elif args.kernel == 'QCDAware':
            kernel = ker.QCDAware(1., 0.7)

        return gcnn.GCNNSingleKernel(
            kernel, frst_fm, args.nb_feature_maps, args.nb_layer
            )

    elif args.kernel == 'MultiQCDAware':
        kernel = mker.MultiQCDAware
        return gcnn.GCNNMultiKernel(
            kernel, frst_fm, args.nb_feature_maps, args.nb_edge_feature, args.nb_layer
            )

    elif args.kernel == 'LayerQCDAware':
        kernel = mker.MultiQCDAware
        return gcnn.GCNNLayerKernel(
            kernel, frst_fm, args.nb_feature_maps, args.nb_edge_feature, args.nb_layer
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
        raise ValueError('Unknown kernel : {}'.format(kernel))


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
