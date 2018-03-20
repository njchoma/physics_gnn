import logging
import pickle
import time
import torch
import torch.nn as nn
import os.path as path

import train_model as model
from loading.model import build_model
import loading.model.model_parameters as param


def train_model(train_X, train_y, train_w, test_X, test_y, test_w):
    """Loads data, recover network then train, test and save network"""


    net = build_model.make_net_if_not_there(param.args, param.args.first_fm, param.args.savedir)

    if param.args.cuda:
        net = net.cuda()
        logging.warning('Working on GPU')
    else:
        net = net.cpu()
        logging.warning('Working on CPU')

    criterion = nn.functional.binary_cross_entropy

    for epoch in range(50):
        t0 = time.time()
        logging.info('learning rate : {}'.format(param.args.lrate))
        optimizer = torch.optim.Adamax(net.parameters(), lr=param.args.lrate)

        epoch_loss_avg = model.train_net(net, train_X, train_y, train_w, criterion, optimizer)
        param.args.lrate *= param.args.lrdecay
        logging.info(
            param.args.name + ' loss epoch {} : {}'.format(epoch + 1, epoch_loss_avg))

        # Model performance on subset of training data
        score_train, loss_train, fpr50_train = model.test_net(net, train_X[:param.args.nbtest], train_y[:param.args.nbtest], train_w[:param.args.nbtest], criterion, "train")
        logging.info(
            param.args.name + ' epoch {}. train : '.format(epoch + 1) +
            'AUC {: >.3E}'.format(score_train)
            + ' -- loss {: >.3E}'.format(loss_train)
            + ' -- FPR {: >.3E}'.format(fpr50_train)
        )

        # Model performance on test data
        score_test, loss_test, fpr50_test = model.test_net(net, test_X, test_y, test_w, criterion, "test")
        logging.info(
            param.args.name + ' epoch {}.  test : '.format(epoch + 1) +
            'AUC {: >.3E}'.format(score_test)
            + ' -- loss {: >.3E}'.format(loss_test)
            + ' -- FPR {: >.3E}'.format(fpr50_test)
        )

        with open(path.join(param.args.savedir, param.args.name + '.csv'), 'a') as fileres:
            fileres.write(
                str(param.args.lrate) + ','
                + str(loss_train) + ','
                + str(loss_test) + ','
                + str(score_train) + ','
                + str(score_test) + ','
                + str(1 / fpr50_train)  + ','
                + str(1 / fpr50_test)  + ','
                + str(epoch_loss_avg) + '\n'
            )

        logging.info("Epoch took {} seconds".format(int(time.time()-t0)))
        try:
          param.save_args()
          with open(path.join(param.args.savedir, param.args.name + '.pkl'), 'wb') as fileout:
            # net -> cpu in case later running without cuda
            pickle.dump(net.cpu(), fileout)
            if (param.args.cuda):
              net.cuda()
          logging.warning('Model saved\n')
        except:
          logging.error("Issue saving model or model parameters")
          exit()
