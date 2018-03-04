from os.path import exists, join
import pickle
from model import kernel as ker
from model import multi_kernel as mker
from model import adj_kernel as adj_ker
from model import gcnn
from model import sparse
from utils.in_out import print_

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
        
        # Instantiate sparsity
        sparse_args = ()
        if args.sparse == 'knn':
          sparse_args += (args.nb_sparse,)
          sparse_type = sparse.KNN
        else:
          sparse_type = sparse.No_sparsity

        sparse_instances = [sparse_type(*sparse_args) for _ in range(args.nb_layer)]

        # Instantiate adjacency kernels
        adj_kernel_args = (args.nb_feature_maps,)
        if args.adj_kernel == 'Gaussian':
            adj_kernel = adj_ker.Gaussian
        elif args.adj_kernel == 'DirectedGaussian':
            adj_kernel = adj_ker.DirectedGaussian
        elif args.adj_kernel == 'MPNNdirected':
            adj_kernel = adj_ker.MPNNdirected
        elif args.adj_kernel == 'MLPdirected':
            adj_kernel_args += (args.nb_MLPadj_hidden,)
            adj_kernel = adj_ker.MLPdirected
        else:
            adj_kernel = adj_ker.Identity

        adj_kernels = [adj_kernel(*adj_kernel_args,sparse=sparse_instances[i]) for i in range(args.nb_layer-1)]



        return gcnn.GCNNSingleKernel(
            kernel, adj_kernels, sparse_instances[-1], frst_fm, args.nb_feature_maps, args.nb_layer
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
