from os.path import exists
import h5py as h5
import torch
import torch.nn as nn
from torch.autograd import Variable
from roccurve import ROCCurve
from dataload import load_data_nersc as load_data


"""NEED TO SEPARATE TRAIN AND TEST"""

def train_net(net, trainfile, criterion, optimizer, args):
    """Trains net for one epoch using criterion loss and optimizer"""

    datafile = h5.File(trainfile, 'r')
    dataloader, nb_event = load_data(datafile, args.nbtrain)
    epoch_loss = 0
    step_loss = 0

    for i, data, ground_truth, _ in dataloader:
        optimizer.zero_grad()

        ground_truth = Variable(torch.Tensor([ground_truth]))
        jet = Variable(torch.Tensor(data)).t().unsqueeze(0)
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
    datafile.close()
    epoch_loss_avg = epoch_loss / nb_event
    return epoch_loss_avg


def test_net(net, testfile, criterion, args):
    """Tests the network, returns the ROC AUC and epoch loss"""

    datafile = h5.File(testfile, 'r')
    dataloader, nb_event = load_data(datafile, args.nbtest)
    criterion = nn.BCELoss()
    epoch_loss = 0
    roccurve = ROCCurve()

    for i, data, ground_truth, weight in dataloader:
        ground_truth = Variable(torch.Tensor([ground_truth]))
        jet = Variable(torch.Tensor(data[i])).t().unsqueeze(0)
        if args.cuda:
            ground_truth = ground_truth.cuda()
            jet = jet.cuda()

        out = net(jet)
        loss = criterion(out, ground_truth)
        epoch_loss += loss.data[0]
        roccurve.update(out, ground_truth, weight)
    datafile.close()

    score = roccurve.roc_score(False)
    type_ = 'train' if 'train' in testfile else 'test'
    roccurve.plot_roc_curve(type_, 'models')
    epoch_loss /= nb_event
    return score, epoch_loss
