import logging
import time
import numpy as np
from os.path import exists
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable

from graphics.plot_graph import construct_plot
import loading.model.model_parameters as param
import data_ops.batching as batching


def train_net(net, X, y, w, criterion, optimizer):
    """Trains net for one epoch using criterion loss and optimizer"""

    logging.warning('training on {} events'.format(len(y)))
    epoch_loss = 0
    step_loss = 0
    net.train()

    plots = construct_plot(param.args)

    batch_idx = batching.get_sorted_batches(len(y), 
                                     param.args.nb_batch, 
                                     X,
                                     param.args.shuffle_while_training)

    for i, idx in enumerate(batch_idx):
        optimizer.zero_grad()

        # X = [np.random.randint(0, 10, size=(6,3))]
        # X = [np.random.randint(0,3,size=(6,2)), np.random.randint(0, 10, size=(6,3))]
        # Put samples into batches
        batch_X, adj_mask, batch_nb_nodes = batching.pad_batch(
                                              [X[s] for s in idx],
                                              param.args.nb_extra_nodes
                                              )
        batch_y = [int(y[s]) for s in idx]
        batch_w = [w[s] for s in idx]

        # Put into variables
        ground_truth = Variable(torch.Tensor(batch_y))
        jet = Variable(torch.Tensor(batch_X))
        weight = Variable(torch.Tensor(batch_w))
        adj_mask = Variable(torch.Tensor(adj_mask))
        batch_nb_nodes = Variable(torch.Tensor(batch_nb_nodes.tolist()))

        # Put on cuda if necessary
        if param.args.cuda:
            ground_truth = ground_truth.cuda()
            jet = jet.cuda()
            weight = weight.cuda()
            adj_mask = adj_mask.cuda()
            batch_nb_nodes = batch_nb_nodes.cuda()

        # t0 = time.time()
        if i == 2:
          out = net(jet, adj_mask, batch_nb_nodes, plots)
        else:
          out = net(jet, adj_mask, batch_nb_nodes)
          # out = net(jet, plots)
        # print("sample took {:.3e} s".format(time.time()-t0))

        loss = criterion(out, ground_truth, weight)
        epoch_loss += loss.data[0]
        step_loss += loss.data[0]

        # Print info
        if (i + 1) % param.args.nbprint == 0:
          logging.info('  {} : {}'.format(
                                            (i+1)*param.args.nb_batch, 
                                            step_loss / param.args.nbprint)
                                            )
          step_loss = 0
        loss.backward()
        optimizer.step()
    epoch_loss_avg = epoch_loss / len(batch_idx)
    if plots is not None:
      plots.epoch_finished()
    return epoch_loss_avg


def test_net(net, X, y, w, criterion, roccurve):
    """Tests the network, returns the ROC AUC and epoch loss"""

    logging.warning('Testing on {} events'.format(len(y)))
    epoch_loss = 0
    roccurve.reset()
    net.eval()

    # Sort test batches by size which greatly reduces padded zeros
    batch_idx = batching.get_sorted_batches(
                                      len(y), 
                                      param.args.nb_batch, 
                                      X
                                      )
    
    for i, idx in enumerate(batch_idx):
        # Create batches
        batch_X, adj_mask, batch_nb_nodes = batching.pad_batch(
                                                [X[s] for s in idx],
                                                param.args.nb_extra_nodes
                                                )
        batch_y = [int(y[s]) for s in idx]
        batch_w = [w[s] for s in idx]

        # Put Variables in batches
        ground_truth = Variable(torch.Tensor(batch_y))
        jet = Variable(torch.Tensor(batch_X))
        weight = Variable(torch.Tensor(batch_w))
        adj_mask = Variable(torch.Tensor(adj_mask))
        batch_nb_nodes = Variable(torch.Tensor(batch_nb_nodes.tolist()))

        # Put on cuda if necessary
        if param.args.cuda:
            ground_truth = ground_truth.cuda()
            jet = jet.cuda()
            weight = weight.cuda()
            adj_mask = adj_mask.cuda()
            batch_nb_nodes = batch_nb_nodes.cuda()

        out = net(jet, adj_mask, batch_nb_nodes)
        loss = criterion(out, ground_truth, weight)
        epoch_loss += loss.data[0]
        roccurve.update(out.data, ground_truth.data, weight.data)

        if (i + 1) % (5*param.args.nbprint) == 0:
            logging.info('  tested on {}'.format((i+1)*param.args.nb_batch))

    score = roccurve.score_auc()
    fpr50 = roccurve.score_fpr()
    epoch_loss /= len(batch_idx)
    return score, epoch_loss, fpr50, roccurve

