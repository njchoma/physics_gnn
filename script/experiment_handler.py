import logging
import pickle
import torch
import torch.nn as nn
import os.path as path

import train_model as model
from loading.model import build_model


def train_model(args, train_X, train_y, train_w, test_X, test_y, test_w):
    """Loads data, recover network then train, test and save network"""


    net = build_model.make_net_if_not_there(args, args.first_fm, args.savedir)

    if args.cuda:
        net = net.cuda()
        logging.warning('Working on GPU')
    else:
        net = net.cpu()
        logging.warning('Working on CPU')

    criterion = nn.functional.binary_cross_entropy
    learning_rate = args.lrate
    lr_decay = args.lrdecay

    # score_test, loss_test, fpr50_test = model.test_net(net, testfile, criterion, args)
    # print_(
    #     args.name + ' test : ' +
    #     'AUC {: >.3E} -- loss {: >.3E} -- FPR {: >.3E}'.format(score_test, loss_test, fpr50_test)
    # )
    # assert False

    for epoch in range(50):
        logging.info('learning rate : {}'.format(learning_rate))
        optimizer = torch.optim.Adamax(net.parameters(), lr=learning_rate)

        epoch_loss_avg = model.train_net(net, train_X, train_y, train_w, criterion, optimizer, args)
        logging.info(
            args.name + ' loss epoch {} : {}'.format(epoch + 1, epoch_loss_avg))

        # Model performance on subset of training data
        score_train, loss_train, fpr50_train = model.test_net(net, train_X[:args.nbtest], train_y[:args.nbtest], train_w[:args.nbtest], criterion, args, args.savedir, "train")
        logging.info(
            args.name + ' epoch {}. train : '.format(epoch + 1) +
            'AUC {: >.3E}'.format(score_train)
            + ' -- loss {: >.3E}'.format(loss_train)
            + ' -- FPR {: >.3E}'.format(fpr50_train)
        )

        # Model performance on test data
        score_test, loss_test, fpr50_test = model.test_net(net, test_X, test_y, test_w, criterion, args, args.savedir, "test")
        logging.info(
            args.name + ' epoch {}.  test : '.format(epoch + 1) +
            'AUC {: >.3E}'.format(score_test)
            + ' -- loss {: >.3E}'.format(loss_test)
            + ' -- FPR {: >.3E}'.format(fpr50_test)
        )

        with open(path.join(args.savedir, args.name + '.csv'), 'a') as fileres:
            fileres.write(
                str(learning_rate) + ','
                + str(loss_train) + ','
                + str(loss_test) + ','
                + str(score_train) + ','
                + str(score_test) + ','
                + str(1 / fpr50_train)  + ','
                + str(1 / fpr50_test)  + ','
                + str(epoch_loss_avg) + '\n'
            )

        with open(path.join(args.savedir, args.name + '.pkl'), 'wb') as fileout:
            pickle.dump(net, fileout)
        logging.warning('Model saved\n')
        learning_rate *= lr_decay
