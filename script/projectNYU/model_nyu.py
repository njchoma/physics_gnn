from os.path import exists
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from graphics.roccurve import ROCCurve
from utils.in_out import print_


def train_net(net, trainfile, criterion, optimizer, args):
    """Trains net for one epoch using criterion loss and optimizer"""

    data, label, _ = load_data(trainfile, args.nbtrain, 'train')
    print_('training on {} events'.format(len(label)), args.quiet)
    epoch_loss = 0
    step_loss = 0
    net.train()

    for i, ground_truth in enumerate(label):
        optimizer.zero_grad()

        ground_truth = Variable(torch.Tensor([int(ground_truth)]))
        jet = Variable(torch.Tensor(data[i])).t().unsqueeze(0)
        if args.cuda:
            ground_truth = ground_truth.cuda()
            jet = jet.cuda()

        if i == 0:
          out = net(jet)#,mode='plot')
        else:
          out = net(jet)

        loss = criterion(out, ground_truth)
        epoch_loss += loss.data[0]
        step_loss += loss.data[0]

        if (i + 1) % args.nbprint == 0:
            print_('    {} : {}'.format(i + 1, step_loss / args.nbprint), args.quiet)
            step_loss = 0

        loss.backward()
        optimizer.step()
    epoch_loss_avg = epoch_loss / len(label)
    return epoch_loss_avg


def test_net(net, testfile, criterion, args, savedir):
    """Tests the network, returns the ROC AUC and epoch loss"""

    data, label, weight = load_data(testfile, args.nbtest, 'test')
    criterion = nn.BCELoss()
    print_('testing on {} events'.format(len(label)), args.quiet)
    epoch_loss = 0
    roccurve = ROCCurve()
    net.eval()

    for i, ground_truth in enumerate(label):
        ground_truth = Variable(torch.Tensor([int(ground_truth)]))
        jet = Variable(torch.Tensor(data[i])).t().unsqueeze(0)
        if args.cuda:
            ground_truth = ground_truth.cuda()
            jet = jet.cuda()

        out = net(jet)
        loss = criterion(out, ground_truth)
        epoch_loss += loss.data[0]
        roccurve.update(out, ground_truth, weight)

        if (i + 1) % 5000 == 0:
            print_('tested on {}'.format(i + 1), args.quiet)

    score = roccurve.roc_score(False)
    type_ = 'train' if 'train' in testfile else 'test'
    fpr50 = roccurve.plot_roc_curve(args.name, type_, savedir)
    epoch_loss /= len(label)
    return score, epoch_loss, fpr50


def load_data(filepath, nb_ex, mode):
    """Loads data from the NYU project, makes it into torch Variables"""

    if mode == 'test':
        with open(filepath, 'rb') as filein:
            try:
                data, label, weights = pickle.load(filein, encoding='latin1')  # python3
            except TypeError:
                data, label, weights = pickle.load(filein)  # python2
        if nb_ex is not None:
            data, label, weights = data[:nb_ex], label[:nb_ex], weights[:nb_ex]
    else:
        with open(filepath, 'rb') as filein:
            try:
                data, label = pickle.load(filein, encoding='latin1')  # python3
            except TypeError:
                data, label = pickle.load(filein)  # python2
        if nb_ex is not None:
            data, label = data[:nb_ex], label[:nb_ex]
        weights = None

    data = [X[:, :6] for X in data]  # dump px, py, pz
    return data, label, weights
