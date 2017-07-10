from os.path import exists
import torch
import torch.nn as nn
from torch.autograd import Variable
from roccurve import ROCCurve
from dataload import load_data_nyu as load_data


def train_net(net, trainfile, criterion, optimizer, args):
    """Trains net for one epoch using criterion loss and optimizer"""

    data, label, _ = load_data(trainfile, args.nbtrain, 'train')
    print(len(label))
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

        out = net(jet)

        loss = criterion(out, ground_truth)
        epoch_loss += loss.data[0]
        step_loss += loss.data[0]

        if (i + 1) % args.nbprint == 0:
            print('    {} : {}'.format(i + 1, step_loss / args.nbprint))
            step_loss = 0

        loss.backward()
        optimizer.step()
    epoch_loss_avg = epoch_loss / len(label)
    return epoch_loss_avg


def test_net(net, testfile, criterion, args):
    """Tests the network, returns the ROC AUC and epoch loss"""

    data, label, weight = load_data(testfile, args.nbtest, 'test')
    criterion = nn.BCELoss()
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
            print('tested on {}'.format(i + 1))

    score = roccurve.roc_score(False)
    type_ = 'train' if 'train' in testfile else 'test'
    fpr50 = roccurve.plot_roc_curve(args.name, type_, 'models')
    epoch_loss /= len(label)
    return score, epoch_loss, fpr50
