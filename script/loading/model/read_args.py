import argparse
from os.path import exists, join

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
    add_arg('--sparse', dest='sparse', help='type of sparsity to use when updating adjacency matrix',type=str,default='None')
    add_arg('--nb_sparse', dest='nb_sparse', help='number of non-zero edges associated with each node when updating adjacency matrix',type=int,default=10)
    add_arg('--nb_MLPadj_hidden', dest='nb_MLPadj_hidden', help='number of hidden units associated with each adj_kernel layer when using MLP adj_kernel',type=int,default=8)
    add_arg('--plot', dest='plot', help='type of plotting to perform',type=str,default=None)

    args = parser.parse_args()
    return args


def get_fixed_param(data_type, project_root_dir):
    """reads parameters from 'param.txt',
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


