import torch
from torch.autograd import Variable
from projectNERSC.load_data import load_raw_data
from graphics.roccurve import ROCCurve
from utils.in_out import print_


def train_net(net, datadir, criterion, optimizer, args):
    """Trains net for one epoch using criterion loss and optimizer"""

    dataloader, nb_event = load_raw_data(datadir, args.nbtrain, _is_train_file)
    epoch_loss = 0
    step_loss = 0

    print_('training on {} events'.format(nb_event), args.quiet)
    net.train()
    for i, event in enumerate(dataloader):
        jet, ground_truth, weight = event
        optimizer.zero_grad()

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
    epoch_loss_avg = epoch_loss / nb_event
    return epoch_loss_avg


def test_net(net, datadir, datatype, criterion, args, savedir):
    """Tests the network, returns the ROC AUC and epoch loss"""

    is_used = _is_train_file if datatype == 'train' else _is_test_file
    dataloader, nb_event = load_raw_data(datadir,args.nbtest, is_used)
    epoch_loss = 0
    roccurve = ROCCurve()

    net.eval()
    for event in dataloader:
        jet, ground_truth, weight = event
        if args.cuda:
            ground_truth = ground_truth.cuda()
            jet = jet.cuda()
            weight = weight.cuda()

        out = net(jet)
        loss = criterion(out, ground_truth, weight)
        epoch_loss += loss.data[0]
        roccurve.update(out, ground_truth, weight)

    score = roccurve.roc_score(False)
    fpr50 = roccurve.plot_roc_curve(args.name, datatype, savedir, zooms=[1., 0.001, 0.0001])
    epoch_loss /= nb_event
    return score, epoch_loss, fpr50


def _is_train_file(filename):
    return filename.endswith('02.h5')


def _is_test_file(filename):
    return filename.endswith('01.h5')


