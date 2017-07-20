from os.path import exists, join
from numpy.random import choice
import h5py as h5
import torch
from torch.autograd import Variable
from graphics.roccurve import ROCCurve
from utils.in_out import print_


def train_net(net, trainfile, criterion, optimizer, args):
    """Trains net for one epoch using criterion loss and optimizer"""

    datafile = h5.File(trainfile, 'r')
    dataloader, nb_event = load_data(datafile, args.nbtrain, rand=True)
    epoch_loss = 0
    step_loss = 0

    net.train()
    for i, event in enumerate(dataloader):
        data, ground_truth, weight = event
        optimizer.zero_grad()

        ground_truth = Variable(torch.Tensor([int(ground_truth)]))
        jet = Variable(torch.Tensor(data)).t().unsqueeze(0)
        weight = torch.Tensor([float(weight)])
        if args.cuda:
            ground_truth = ground_truth.cuda()
            jet = jet.cuda()
            weight = weight.cuda()

        out = net(jet)

        loss = criterion(out, ground_truth, weight)
        epoch_loss += loss.data[0]
        step_loss += loss.data[0]

        if (i + 1) % args.nbprint == 0:
            print_('    {} : {}'.format(i + 1, step_loss / args.nbprint), args.quiet)
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
    epoch_loss = 0
    roccurve = ROCCurve()

    net.eval()
    for event in dataloader:
        data, ground_truth, weight = event
        ground_truth = Variable(torch.Tensor([int(ground_truth)]))
        jet = Variable(torch.Tensor(data)).t().unsqueeze(0)
        weight = torch.Tensor([float(weight)])
        if args.cuda:
            ground_truth = ground_truth.cuda()
            jet = jet.cuda()
            weight = weight.cuda()

        out = net(jet)
        loss = criterion(out, ground_truth, weight)
        epoch_loss += loss.data[0]
        roccurve.update(out, ground_truth, weight)
    datafile.close()

    score = roccurve.roc_score(False)
    type_ = 'train' if 'train' in testfile else 'test'
    fpr50 = roccurve.plot_roc_curve(args.name, type_, 'models', zooms=[1., 0.001, 0.0001])
    epoch_loss /= nb_event
    return score, epoch_loss, fpr50


def load_data(datafile, nb_ex, rand=False):
    """Loads data from the NYU project, makes it into torch Variables"""

    def _iter_data_nersc(datafile, idx):
        event_name = 'event_{}'.format(idx)
        event = datafile[event_name]

        data = event[()]
        label = event.attrs['label']
        weight = event.attrs['weight']

        return (data, label, weight)

    nb_event = int(datafile['nbevent'][()])
    nb_ex = min(nb_ex, nb_event)

    if rand:
        permutation = choice(nb_event, nb_ex)
        return (_iter_data_nersc(datafile, permutation[idx]) for idx in range(nb_ex)), nb_ex
    return (_iter_data_nersc(datafile, idx) for idx in range(nb_ex)), nb_ex
