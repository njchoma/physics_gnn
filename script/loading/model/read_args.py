import argparse
from os.path import exists, join

def read_args():
  """Parses stdin for arguments used for training or network initialisation"""

  parser = argparse.ArgumentParser(description='simple arguments to train GCNN')
  add_arg = parser.add_argument

  add_arg('--name', dest='name', help='network reference')
  add_arg('--data', dest='data', help='project to take data from', default='NYU')
  add_arg('--cuda', dest='cuda', action='store_true')
  add_arg('--nbtrain',dest='nbtrain', help='number training examples', type=int)
  add_arg('--nbtest', dest='nbtest', help='number testing examples', type=int)
  add_arg('--nbprint',dest='nbprint',help='print freq',type=int,default=10000)
  add_arg('--nbepoch',dest='nbepoch',help='Number epochs through train set',type=int,default=100)
  add_arg('--quiet', dest='quiet', help='reduces print', action='store_true')
  add_arg('--plot', dest='plot', help='type of plotting to perform',type=str,default=None)
  add_arg('--save_best_model', dest='save_best_model', help='saves best model based upon test 1/FPR',action='store_true')
  add_arg('--tpr_target', dest='tpr_target', help='Sets TPR score at which 1/FPR is evaluated',type=float,default=0.5)
  add_arg('--no_shuffle', dest='shuffle_while_training',help='Process samples in order of dataset for every epoch',action='store_false')
  add_arg('--nb_batch', dest='nb_batch',help='minibatch size',type=int, default=1)
  add_arg('--sorted_training', dest='sorted_training',help='Group similar-sized samples in training (less 0-padding->faster, but worse gradient estimates)',action='store_true')

  # Kernel-specific
  add_arg('--kernels', dest='kernels', help='List of kernels. Add \'-layerwise\' to kernel name to create one kernel instance per layer. E.g. \'MLPDirected-layerwise\'', default='Gaussian',nargs='+')
  add_arg('--combine_kernels', dest='combine_kernels', help='Method for combining kernels to form single adj matrix', default='Affine_Normalized')
  add_arg('--sigma', dest='sigma', help='kernel stdev initial value', type=float, default = 2.0)
  add_arg('--nb_MLPadj_hidden', dest='nb_MLPadj_hidden', help='number of hidden units associated with each adj_kernel layer when using MLP adj_kernel',type=int,default=8)

  # Model
  add_arg('--architecture',dest='architecture',help='Model architecture',default='gnn')
  add_arg('--fm', dest='nb_feature_maps', help='number of feature maps per layer', type=int)
  add_arg('--edge_fm', dest='nb_edge_feature', type=int,
          help='number of edge features for GCNN_EdgeFeature')
  add_arg('--depth', dest='nb_layer', help='number of layers', type=int)
  add_arg('--sparse', dest='sparse', help='type of sparsity to use when updating adjacency matrix',type=str,default='None')
  add_arg('--nb_sparse', dest='nb_sparse', help='number of non-zero edges associated with each node when updating adjacency matrix',type=int,default=10)
  add_arg('--lr', dest='lrate', help='learning rate', type=float)
  add_arg('--lrdecay', dest='lrdecay', help='learning rate decay, `lr *= lrdecay` each epoch',type=float, default=0.95)
  add_arg('--nb_extra_nodes', dest='nb_extra_nodes',help='Number of nodes of initial value 0 to append to each sample',type=int,default=0)
  add_arg('--readout', dest='readout',help='Type of pooling after GNN layers',default='Sum')
  add_arg('--node_type', dest='node_type',help='Type of nodes to use in gnn layers',default='Identity')
  add_arg('--conv_type', dest='conv_type',help='Type of graph convolution to use in gnn layers',default='ResGNN')


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


