import logging
from os.path import exists
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable

from graphics.roccurve import ROCCurve
from graphics.plot_graph import construct_plot
import loading.model.model_parameters as param


def train_net(net, X, y, w, criterion, optimizer):
    """Trains net for one epoch using criterion loss and optimizer"""

    logging.warning('training on {} events'.format(len(y)))
    epoch_loss = 0
    step_loss = 0
    net.train()

    plots = construct_plot(param.args)

    for i, ground_truth in enumerate(y):
        optimizer.zero_grad()

        ground_truth = Variable(torch.Tensor([int(ground_truth)]))
        jet = Variable(torch.Tensor(X[i])).unsqueeze(0)
        weight = Variable(torch.Tensor([w[i]]))
        if param.args.cuda:
            ground_truth = ground_truth.cuda()
            jet = jet.cuda()
            weight = weight.cuda()

        if i == 2:
          out = net(jet, plots)
        else:
          out = net(jet)
          # out = net(jet, plots)

        loss = criterion(out, ground_truth, weight)
        epoch_loss += loss.data[0]
        step_loss += loss.data[0]

        if (i + 1) % param.args.nbprint == 0:
            logging.info('    {} : {}'.format(i + 1, step_loss / param.args.nbprint))
            step_loss = 0

        loss.backward()
        optimizer.step()
    epoch_loss_avg = epoch_loss / len(y)
    if plots is not None:
      plots.epoch_finished()
    return epoch_loss_avg


def test_net(net, X, y, w, criterion, type_):
    """Tests the network, returns the ROC AUC and epoch loss"""

    # criterion = nn.BCELoss()
    logging.warning('testing on {} events'.format(len(y)))
    epoch_loss = 0
    roccurve = ROCCurve()
    net.eval()

    for i, ground_truth in enumerate(y):
        ground_truth = Variable(torch.Tensor([int(ground_truth)]))
        jet = Variable(torch.Tensor(X[i])).unsqueeze(0)
        weight = Variable(torch.Tensor([w[i]]))
        if param.args.cuda:
            ground_truth = ground_truth.cuda()
            jet = jet.cuda()
            weight = weight.cuda()

        out = net(jet)
        loss = criterion(out, ground_truth, weight)
        epoch_loss += loss.data[0]
        roccurve.update(out, ground_truth, weight)

        if (i + 1) % 5000 == 0:
            logging.info('tested on {}'.format(i + 1))

    score = roccurve.roc_score(True)
    fpr50 = roccurve.plot_roc_curve(param.args.name, type_, param.args.savedir, zooms=[1., 0.001, 0.0001])
    epoch_loss /= len(y)
    return score, epoch_loss, fpr50

