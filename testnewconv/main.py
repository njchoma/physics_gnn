import pickle
import argparse
import torch
import torch.nn as nn
from gcnn import GCNN
import kernel as Kernel
import operators as op
import model


def _read_args():
    parser = argparse.ArgumentParser(description='simple arguments to train GCNN')
    add_arg = parser.add_argument

    add_arg('--name', dest='name', help='network reference')
    add_arg('--fm', dest='nb_feature_maps', help='number of feature maps per layer', type=int)
    add_arg('--depth', dest='nb_layer', help='number of layers', type=int)
    add_arg('--sigma', dest='sigma', help='kernel stdev initial value', type=float)
    add_arg('--lr', dest='lrate', help='learning rate', type=float)
    add_arg('--cuda', dest='cuda', action='store_true')
    add_arg('--isdummy', dest='isdummy', action='store_true')
    add_arg('--nbtrain', dest='nbtrain', help='number of training examples', type=int)
    add_arg('--nbtest', dest='nbtest', help='number of testing examples', type=int)
    add_arg('--nbprint', dest='nbprint', help='print frequency', type=int)

    args = parser.parse_args()
    return args

def main():
    """Loads data, recover network then train, test and save network"""

    args = _read_args()
    name = args.name
    nb_feature_maps = args.nb_feature_maps
    nb_layer = args.nb_layer
    cuda = args.cuda
    nb_train = args.nbtrain
    nb_test = args.nbtest
    lrate = args.lrate

    param = model.get_fixed_param()
    trainfile = param['trainfile']
    testfile = param['testfile']
    kernel = param['kernel']

    if kernel == 'FixedComplexGaussian':
        kernel = Kernel.FixedComplexGaussian(args.sigma, diag=True)
    elif kernel == 'FixedComplexGaussianNoDiag':
        kernel = Kernel.FixedComplexGaussian(args.sigma, diag=False)
    elif kernel == 'QCDAware':
        kernel = Kernel.QCDAware(1., 0.001, 1.)
    else:
        raise ValueError('Unknown kernel : {}'.format(kernel))

    operators = [
        op.degree,
        op.adjacency,
    ]
    if args.isdummy:
        operators = []

    try:
        if name == 'test':
            print('name `test` is used for testing, this model not be recovered')
            raise FileNotFoundError('model `test` never recovered')  # do as if no such file exists
        with open('models/' + name + '.pkl', 'rb') as filein:
            net = pickle.load(filein)
        print('Network recovered from previous training')

    except FileNotFoundError:
        print('Network created')
        net = GCNN(kernel, operators, nb_feature_maps, nb_layer)
        with open(name + '.out', 'w') as fileres:
            fileres.write('Loss, Train AUC Score, Test AUC score, Learning Rate\n')
    print('parameters : {}'.format(sum([param.numel() for param in net.parameters()])))
    try:
        sigma = net.kernel.sigma
        if isinstance(sigma, nn.Parameter):
            sigma = sigma.data[0]
        print('sigma : {}'.format(sigma))
    except AttributeError:
        pass

    if cuda:
        net = net.cuda()

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adamax(net.parameters(), lr=lrate)

    for epoch in range(50):
        data, label = model.load_data(trainfile)
        data, label = data[:nb_train], label[:nb_train]

        epoch_loss_avg = model.train_net(net, (data, label), criterion, optimizer, args)
        print(
            name + ' loss epoch {} : {}'.format(epoch + 1, epoch_loss_avg)
        )

        score_train, loss_train = model.test_net(net, trainfile, nb_test, cuda)
        print(
            name + ' epoch {}. train : '.format(epoch + 1) +
            'AUC {: >.3E} -- loss {: >.3E}'.format(score_train, loss_train)
        )

        score_test, loss_test = model.test_net(net, testfile, nb_test, cuda)
        print(
            name + ' epoch {}.  test : '.format(epoch + 1) +
            'AUC {: >.2E} -- loss {: >.2E}'.format(score_test, loss_test)
        )

        with open('models/' + name + '.out', 'a') as fileres:
            fileres.write(
                str(epoch_loss_avg) + ','
                + str(score_train) + ','
                + str(score_test) + ','
                + str(lrate) + '\n'
            )

        with open('models/' + name + '.pkl', 'wb') as fileout:
            pickle.dump(net, fileout)
        print('saved\n')


if __name__ == '__main__':
    main()
