import os
from os.path import exists, join
import numpy as np
from numpy.random import permutation
import h5py as h5
import torch
from torch.autograd import Variable
from projectNERSC.file2weightfactor import init_weight_factors
from graphics.roccurve import ROCCurve
from utils.in_out import print_


def train_net(net, datadir, criterion, optimizer, args):
    """Trains net for one epoch using criterion loss and optimizer"""

    dataloader, nb_event = load_raw_data(datadir, _is_train_file)
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
    dataloader, nb_event = load_raw_data(datadir, is_used)
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


def load_raw_data(datadir, is_used):
    """creates a data loader containing (data, label, weight)"""

    nb_events_files, all_event_coords_ = all_event_coords(datadir, is_used)
    weight_factors = init_weight_factors(is_used, datadir)

    data_loader = (load_raw_event(datadir, event_coord, weight_factors)
                   for event_coord in permutation(all_event_coords_))
    nb_events_ = sum(nb_events_files)

    return data_loader, nb_events_


def load_raw_event(datadir, event_coord, weight_factors):
    """loads one event, given the file and the event index"""

    chosen_file, idx = event_coord
    datapath = join(datadir, chosen_file)
    event_name = 'event_{}'.format(idx)
    with h5.File(datapath, 'r') as datafile:
        event = datafile[event_name]
        data_fields = ['clusE', 'clusEta', 'clusPhi', 'clusEM']
        data_fields = [event[field][()] for field in data_fields]

        cluspt = data_fields[0] / np.cosh(data_fields[1])  # clusE / cosh(clusEta)
        data_fields.append(cluspt)

        weight = event['weight'][()]

    data = [Variable(torch.Tensor(clus)).unsqueeze(0) for clus in data_fields]
    data = torch.stack(data, 1)

    weight_factor = weight_factors[chosen_file]
    weight = torch.Tensor([float(weight * weight_factor)])

    label = Variable(torch.Tensor([int(chosen_file.startswith('GG'))]))

    return (data, label, weight)


def all_event_coords(datadir, is_used):
    data_files_ = data_files(datadir, is_used)
    nb_events_file = [nb_events(join(datadir, filename)) for filename in data_files_]
    all_coords = [(filename, idx)
                  for i, filename in enumerate(data_files_)
                  for idx in range(nb_events_file[i])]

    return nb_events_file, all_coords


def data_files(datadir, is_used):
    """Lists all relevant files in `datadir`"""

    datafiles = [filename for filename in os.listdir(datadir) if filename.endswith('.h5')]
    datafiles = [filename for filename in datafiles if 'data' not in filename]
    datafiles = [filename for filename in datafiles if is_used(filename)]

    return datafiles


def nb_events(path):
    with h5.File(path) as h5file:
        nb = h5file.attrs['nb_event']
    return nb
