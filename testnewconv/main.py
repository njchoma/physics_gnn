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
    add_arg('--kernel', dest='kernel', help='name of kernel used', default='FCG')
    add_arg('--data', dest='data', help='project to take data from', default='NYU')
    add_arg('--fm', dest='nb_feature_maps', help='number of feature maps per layer', type=int)
    add_arg('--depth', dest='nb_layer', help='number of layers', type=int)
    add_arg('--sigma', dest='sigma', help='kernel stdev initial value', type=float)
    add_arg('--lr', dest='lrate', help='learning rate', type=float)
    add_arg('--cuda', dest='cuda', action='store_true')
    add_arg('--isdummy', dest='isdummy', action='store_true')
    add_arg('--nbtrain', dest='nbtrain', help='number of training examples', type=int)
    add_arg('--nbtest', dest='nbtest', help='number of testing examples', type=int)
    add_arg('--nbprint', dest='nbprint', help='print frequency', type=int, default=10000)

    args = parser.parse_args()
    return args

def main():
    """Loads data, recover network then train, test and save network"""

    print("\nUpdate the spatialnorm to torch.nn.InstanceNorm1d, don't forget to train and eval !")

    args = _read_args()
    param = model.get_fixed_param()
    trainfile = param['trainfile']
    testfile = param['testfile']

    if args.kernel == 'FCG':
        kernel = Kernel.FixedComplexGaussian(args.sigma)
    elif args.kernel == 'FCG_nodiag':
        kernel = Kernel.FixedComplexGaussian(args.sigma, diag=False)
    elif args.kernel == 'FCG_norm':
        kernel = Kernel.FixedComplexGaussian(args.sigma, norm=True)
    elif args.kernel == 'FCG_nodiag_norm':
        kernel = Kernel.FixedComplexGaussian(args.sigma, diag=False, norm=True)
    elif args.kernel == 'FQCDAware':
        kernel = Kernel.FixedQCDAware(1., 0.001, 1.)
    else:
        raise ValueError('Unknown kernel : {}'.format(kernel))

    operators = [
        op.degree,
        op.adjacency,
    ]
    if args.isdummy:
        operators = []

    try:
        if args.name == 'test':
            print('\nWARNING : name `test` is used for testing, this model not be recovered\n')
            raise FileNotFoundError('model `test` never recovered')  # do as if no such file exists
        with open('models/' + args.name + '.pkl', 'rb') as filein:
            net = pickle.load(filein)
        print('Network recovered from previous training')

    except FileNotFoundError:
        print('Network created')
        net = GCNN(kernel, operators, args.nb_feature_maps, args.nb_layer)
        with open('models/' + args.name + '.csv', 'w') as fileres:
            fileres.write('Learning Rate, Train Loss, Test Loss, Train AUC Score, Test AUC score\n')
    print('parameters : {}'.format(sum([param.numel() for param in net.parameters()])))
    try:
        sigma = net.kernel.sigma
        if isinstance(sigma, nn.Parameter):
            sigma = sigma.data[0]
        print('sigma : {}'.format(sigma))
    except AttributeError:
        pass

    if args.cuda:
        net = net.cuda()

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adamax(net.parameters(), lr=args.lrate)

    for epoch in range(50):
        epoch_loss_avg = model.train_net(net, trainfile, criterion, optimizer, args)
        print(
            args.name + ' loss epoch {} : {}'.format(epoch + 1, epoch_loss_avg)
        )

        score_train, loss_train = model.test_net(net, trainfile, criterion, args)
        print(
            args.name + ' epoch {}. train : '.format(epoch + 1) +
            'AUC {: >.3E} -- loss {: >.3E}'.format(score_train, loss_train)
        )

        score_test, loss_test = model.test_net(net, testfile, criterion, args)
        print(
            args.name + ' epoch {}.  test : '.format(epoch + 1) +
            'AUC {: >.2E} -- loss {: >.2E}'.format(score_test, loss_test)
        )

        with open('models/' + args.name + '.csv', 'a') as fileres:
            fileres.write(
                str(args.lrate) + ','
                + str(loss_train) + ','
                + str(loss_test) + ','
                + str(score_train) + ','
                + str(score_test) + '\n'
            )

        with open('models/' + args.name + '.pkl', 'wb') as fileout:
            pickle.dump(net, fileout)
        print('saved\n')


if __name__ == '__main__':
    main()
