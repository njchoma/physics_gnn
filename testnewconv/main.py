from os.path import exists
import pickle
import argparse
import torch
import torch.nn as nn
from gcnn import GCNN
import gcnn_ef
import kernel as Kernel
import operators as op
import dataload
import model_nyu
import model_nersc


def _read_args():
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

    args = parser.parse_args()
    return args

def main(args):
    """Loads data, recover network then train, test and save network"""

    def _pass_fun(string):
        pass
    
    print_ = _pass_fun if args.quiet else print 

    print_("\nUpdate the spatialnorm to torch.nn.InstanceNorm1d, don't forget to train and eval !")

    param = dataload.get_fixed_param()
    trainfile = param['trainfile']
    testtrainfile = param['testtrainfile']
    testfile = param['testfile']

    if args.data == 'NYU':
        model = model_nyu
        frst_fm = 6
    elif args.data == 'NERSC':
        model = model_nersc
        frst_fm = 5
        print_('check frst_fm, may be wrong...')
    else:
        raise ValueError('Unknown project : {}'.format(args.data))

    operators = [
        op.degree,
        op.adjacency,
    ]
    if args.isdummy:
        operators = []

    if args.kernel == 'FCG':
        kernel = Kernel.FixedComplexGaussian(args.sigma)
    elif args.kernel == 'FCG_nodiag':
        kernel = Kernel.FixedComplexGaussian(args.sigma, diag=False)
    elif args.kernel == 'FCG_norm':
        kernel = Kernel.FixedComplexGaussian(args.sigma, norm=True)
    elif args.kernel == 'FCG_nodiag_norm':
        kernel = Kernel.FixedComplexGaussian(args.sigma, diag=False, norm=True)
    elif args.kernel == 'QCDAware':
        kernel = Kernel.QCDAware(1., 0.7, 1.)
    elif args.kernel == 'FQCDAware':
        kernel = Kernel.FixedQCDAware(1.0, 0.001, 1.)
    elif args.kernel == 'EdgeFeature':
        kernel = gcnn_ef.Node2Edge
        operators = [
            gcnn_ef.degree_multikernel, gcnn_ef.adjacency_multikernel
        ]
    elif args.kernel == 'GatedEdgeFeature':
        kernel = gcnn_ef.GatedNode2Edge
        operators = [
            gcnn_ef.degree_multikernel, gcnn_ef.adjacency_multikernel
        ]
    else:
        raise ValueError('Unknown kernel : {}'.format(kernel))

    model_pkl_path = 'models/' + args.name + '.pkl'
    if exists(model_pkl_path):
        with open(model_pkl_path, 'rb') as filein:
            net = pickle.load(filein)
        print_('Network recovered from previous training')
    else:
        if args.kernel in ['EdgeFeature', 'GatedEdgeFeature']:
            net = gcnn_ef.GCNN_EdgeFeature(kernel, operators, frst_fm, args.nb_feature_maps,
                                   args.nb_edge_feature, args.nb_layer)
        else:
            net = GCNN(kernel, operators, frst_fm, args.nb_feature_maps, args.nb_layer)
        print_('Network created')
        with open('models/' + args.name + '.csv', 'w') as fileres:
            fileres.write('Learning Rate, Train Loss, Test Loss, Train AUC Score, Test AUC score\n')
    print_('parameters : {}'.format(sum([param.numel() for param in net.parameters()])))

    try:
        sigma = net.kernel.sigma
        if isinstance(sigma, nn.Parameter):
            sigma = sigma.data[0]
        print_('sigma : {}'.format(sigma))
    except AttributeError:
        pass

    if args.kernel == 'QCDAware':
        alpha = net.kernel.alpha.squeeze().data[0]
        beta = net.kernel.beta.squeeze().data[0] ** 2
        radius = net.kernel.radius.squeeze().data[0]
        print('alpha : {} -- beta : {} -- radius : {}'.format(alpha, beta, radius))

    if args.cuda:
        net = net.cuda()

    criterion = nn.BCELoss()
    learning_rate = args.lrate
    lr_decay = args.lrdecay

    # score_test, loss_test, fpr50_test = model.test_net(net, testfile, criterion, args)
    # print_(
    #     args.name + ' test : ' +
    #     'AUC {: >.3E} -- loss {: >.3E} -- FPR {: >.3E}'.format(score_test, loss_test, fpr50_test)
    # )
    # assert False

    for epoch in range(50):
        optimizer = torch.optim.Adamax(net.parameters(), lr=learning_rate)
        epoch_loss_avg = model.train_net(net, trainfile, criterion, optimizer, args)
        print_(
            args.name + ' loss epoch {} : {}'.format(epoch + 1, epoch_loss_avg)
        )

        score_train, loss_train, fpr50_train = model.test_net(net, testtrainfile, criterion, args)
        print_(
            args.name + ' epoch {}. train : '.format(epoch + 1) +
            'AUC {: >.3E} -- loss {: >.3E} -- FPR {: >.3E}'.format(score_train, loss_train, fpr50_train)
        )

        score_test, loss_test, fpr50_test = model.test_net(net, testfile, criterion, args)
        print_(
            args.name + ' epoch {}.  test : '.format(epoch + 1) +
            'AUC {: >.3E} -- loss {: >.3E} -- FPR {: >.3E}'.format(score_test, loss_test, fpr50_test)
        )

        with open('models/' + args.name + '.csv', 'a') as fileres:
            fileres.write(
                str(learning_rate) + ','
                + str(loss_train) + ','
                + str(loss_test) + ','
                + str(score_train) + ','
                + str(score_test) + '\n'
            )

        with open('models/' + args.name + '.pkl', 'wb') as fileout:
            pickle.dump(net, fileout)
        print_('saved\n')
        learning_rate *= lr_decay


if __name__ == '__main__':
    args = _read_args()
    main(args)
