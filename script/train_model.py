from os.path import exists
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from graphics.roccurve import ROCCurve
from utils.in_out import print_


def train_net(net, X, y, w, criterion, optimizer, args):
    """Trains net for one epoch using criterion loss and optimizer"""

    print_('training on {} events'.format(len(y)), args.quiet)
    epoch_loss = 0
    step_loss = 0
    net.train()

    for i, ground_truth in enumerate(y):
        optimizer.zero_grad()

        ground_truth = Variable(torch.Tensor([int(ground_truth)]))
        jet = Variable(torch.Tensor(X[i])).unsqueeze(0)
        weight = Variable(torch.Tensor([w[i]]))
        if args.cuda:
            ground_truth = ground_truth.cuda()
            jet = jet.cuda()
            weight = weight.cuda()

        if i == 0:
          out = net(jet)#,mode='plot')
        else:
          out = net(jet)

        loss = criterion(out, ground_truth, weight)
        epoch_loss += loss.data[0]
        step_loss += loss.data[0]

        if (i + 1) % args.nbprint == 0:
            print_('    {} : {}'.format(i + 1, step_loss / args.nbprint), args.quiet)
            step_loss = 0

        loss.backward()
        optimizer.step()
    epoch_loss_avg = epoch_loss / len(y)
    return epoch_loss_avg


def test_net(net, X, y, w, criterion, args, savedir, type_):
    """Tests the network, returns the ROC AUC and epoch loss"""

    # criterion = nn.BCELoss()
    print_('testing on {} events'.format(len(y)), args.quiet)
    epoch_loss = 0
    roccurve = ROCCurve()
    net.eval()

    for i, ground_truth in enumerate(y):
        ground_truth = Variable(torch.Tensor([int(ground_truth)]))
        jet = Variable(torch.Tensor(X[i])).unsqueeze(0)
        weight = Variable(torch.Tensor([w[i]]))
        if args.cuda:
            ground_truth = ground_truth.cuda()
            jet = jet.cuda()
            weight = weight.cuda()

        out = net(jet)
        loss = criterion(out, ground_truth, weight)
        epoch_loss += loss.data[0]
        roccurve.update(out, ground_truth, weight)

        if (i + 1) % 5000 == 0:
            print_('tested on {}'.format(i + 1), args.quiet)

    score = roccurve.roc_score(True)
    fpr50 = roccurve.plot_roc_curve(args.name, type_, savedir, zooms=[1., 0.001, 0.0001])
    epoch_loss /= len(y)
    return score, epoch_loss, fpr50

