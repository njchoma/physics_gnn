import multiprocessing
import main
import argparse


def _read_args_from_string(string):
    parser = argparse.ArgumentParser(description='simple arguments to train GCNN')
    add_arg = parser.add_argument

    add_arg('--name', dest='name', help='network reference')
    add_arg('--kernel', dest='kernel', help='name of kernel used', default='FCG')
    add_arg('--data', dest='data', help='project to take data from', default='NYU')
    add_arg('--fm', dest='nb_feature_maps', help='number of feature maps per layer', type=int)
    add_arg('--edge_fm', dest='nb_edge_feature', type=int,
            help='number of edge features for GCNN_EdgeFeature')
    add_arg('--depth', dest='nb_layer', help='number of layers', type=int)
    add_arg('--sigma', dest='sigma', help='kernel stdev initial value', type=float)
    add_arg('--lr', dest='lrate', help='learning rate', type=float)
    add_arg('--lrdecay', dest='lrdecay', help='learning rate decay, `lr *= lrdecay` each epoch',
            type=float, default=0.95)
    add_arg('--cuda', dest='cuda', action='store_true')
    add_arg('--isdummy', dest='isdummy', action='store_true')
    add_arg('--nbtrain', dest='nbtrain', help='number of training examples', type=int)
    add_arg('--nbtest', dest='nbtest', help='number of testing examples', type=int)
    add_arg('--nbprint', dest='nbprint', help='print frequency', type=int, default=10000)
    add_arg('--quiet', dest='quiet', help='reduces print', action='store_true')

    args = parser.parse_args(string.split(' '))
    return args


def parallel():
    """Reads sets of arguments from args.txt and runs main.main on each
    set in parallel.
    """

    with open('args.txt', 'r') as argfile:
        args = argfile.read()
    args = [arg for arg in args.split('\n')]
    args = [_read_args_from_string(arg) for arg in args]
    for arg in args:
        arg.is_quiet = True  # multiprocess forces quiet

    pool = multiprocessing.Pool(32)
    pool.map(main.main, args)

if __name__ == '__main__':
    parallel()