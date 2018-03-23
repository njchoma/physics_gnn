import logging
import pickle
import time
import torch
import torch.nn as nn
import os.path as path

from graphics.roccurve import ROCCurve
import train_model as model
from loading.model import build_model
import loading.model.model_parameters as param


def train_model(train_X, train_y, train_w, test_X, test_y, test_w):
  """Loads data, recover network then train, test and save network"""


  net = build_model.make_net_if_not_there(param.args, param.args.first_fm, param.args.savedir)
  logging.info(net)

  if param.args.cuda:
    net = net.cuda()
    logging.warning('Working on GPU')
  else:
    net = net.cpu()
    logging.warning('Working on CPU')

  criterion = nn.functional.binary_cross_entropy

  # Track best model performance
  param.args.bestInvFpr = 0.0
  param.args.bestAuc = 0.0

  # Set up roc plotting
  zooms = [1., 0.01, 0.001]
  roc_train = ROCCurve("train", zooms=zooms)
  roc_test  = ROCCurve("test", zooms=zooms)

  for epoch in range(50):
    t0 = time.time()
    logging.info('\nLearning rate: {0:.3g}'.format(param.args.lrate))
    optimizer = torch.optim.Adamax(net.parameters(), lr=param.args.lrate)

    epoch_loss_avg = model.train_net(net, train_X, train_y, train_w, criterion, optimizer)
    param.args.lrate *= param.args.lrdecay
    logging.info(param.args.name+' loss epoch {} : {}'.format(epoch+1,epoch_loss_avg))

    # Model performance on subset of training data, test data
    auc_train, loss_train, fpr_train, roc_train = model.test_net(net, train_X[:param.args.nbtest], train_y[:param.args.nbtest], train_w[:param.args.nbtest], criterion, roc_train)
    auc_test, loss_test, fpr_test, roc_test = model.test_net(net, test_X, test_y, test_w, criterion, roc_test)

    # Log eval performance
    print_epoch_info(epoch, "train", loss_train, auc_train, (1/fpr_train))
    print_epoch_info(epoch, " test", loss_test,  auc_test,  (1/fpr_test))

    # Log epoch information
    with open(path.join(param.args.savedir, param.args.name + '.csv'), 'a') as fileres:
      fileres.write(
                    str(param.args.lrate) + ','
                    + str(loss_train) + ','
                    + str(loss_test) + ','
                    + str(auc_train) + ','
                    + str(auc_test) + ','
                    + str(1 / fpr_train)  + ','
                    + str(1 / fpr_test)  + ','
                    + str(epoch_loss_avg) + '\n'
                    )

    # Save model if beat previous best
    beatFpr = ((1/(fpr_test+10**-20)) > param.args.bestInvFpr)
    matchFpr = ((1/(fpr_test+10**-20)) == param.args.bestInvFpr)
    beatAuc = (auc_test > param.args.bestAuc)
    if beatFpr or (matchFpr and beatAuc):
      logging.info("\nBeat previous best. Updating...")
      param.args.bestInvFpr = (1/(fpr_test+10**-20))
      param.args.bestAuc = auc_test
      roc_train.plot_roc_curve()
      roc_test.plot_roc_curve()
      if (param.args.save_best_model):
        save_model(net)

    logging.info("Epoch took {} seconds\n".format(int(time.time()-t0)))


def print_epoch_info(epoch, name, loss, auc, invFpr):
  logging.info(
      param.args.name + ' epoch {}. {}: '.format(epoch + 1, name)
      + 'loss {: >.3E}'.format(loss)
      + ' -- AUC {: >.3E}'.format(auc)
      + ' -- 1/FPR {: >.3E}'.format(invFpr)
  )

def save_model(net):
  try:
    with open(path.join(param.args.savedir, param.args.name + '.pkl'), 'wb') as fileout:
      # net -> cpu in case later running without cuda
      pickle.dump(net.cpu(), fileout)
      if (param.args.cuda):
        net.cuda()
    logging.warning('Model saved')
    param.save_args()
  except:
    logging.error("Issue saving model or model parameters")
    exit()
