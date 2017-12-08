import os.path as path
import pickle
import torch
import torch.nn as nn
import read_args as ra
from projectNERSC import model_raw_nersc as model
from utils.in_out import print_, make_dir_if_not_there


def main_nersc(args, project_root_dir):
    """Loads data, recover network then train, test and save network"""

    frst_fm = 5

    modelsdir = path.join(project_root_dir, 'models' + args.data)
    savedir = path.join(modelsdir, args.name)
    make_dir_if_not_there(savedir)

    datadir = path.join(project_root_dir, 'dataNERSC')

    param = ra.get_fixed_param(args.data, project_root_dir)
    print (param)
    datadir = param['testfile']

    net = ra.make_net_if_not_there(args, frst_fm, savedir)

    if args.cuda:
        net = net.cuda()
        print('Working on GPU')
    else:
        net = net.cpu()
        print('Working on CPU')

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
        print('learning rate : {}'.format(learning_rate))
        optimizer = torch.optim.Adamax(net.parameters(), lr=learning_rate)

        epoch_loss_avg = model.train_net(net, datadir, criterion, optimizer, args)
        print_(
            args.name + ' loss epoch {} : {}'.format(epoch + 1, epoch_loss_avg), args.quiet
        )

        score_train, loss_train, fpr50_train = model.test_net(net, datadir, 'train', criterion, args, savedir)
        print_(
            args.name + ' epoch {}. train : '.format(epoch + 1) +
            'AUC {: >.3E}'.format(score_train)
            + ' -- loss {: >.3E}'.format(loss_train)
            + ' -- FPR {: >.3E}'.format(fpr50_train),
            args.quiet
        )

        score_test, loss_test, fpr50_test = model.test_net(net, datadir, 'test', criterion, args, savedir)
        print_(
            args.name + ' epoch {}.  test : '.format(epoch + 1) +
            'AUC {: >.3E}'.format(score_test)
            + ' -- loss {: >.3E}'.format(loss_test)
            + ' -- FPR {: >.3E}'.format(fpr50_test),
            args.quiet
        )

        with open(path.join(savedir, args.name + '.csv'), 'a') as fileres:
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

        with open(path.join(savedir, args.name + '.pkl'), 'wb') as fileout:
            pickle.dump(net, fileout)
        print_('saved\n', args.quiet)
        learning_rate *= lr_decay
