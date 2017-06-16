from os.path import exists
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from roccurve import ROCCurve


def load_data(filepath, nb_ex, datatype):
    """Loads data at `filepath` and returns (`data`, `label`).
        - `data` : list of events represented by (nb_node, nb_features) numpy arrays
        - `label` : corresponding labels
    """

    if datatype == 'NYU':
        return _load_data_nyu(filepath, nb_ex)
    elif datatype == 'NERSC':
        return _load_data_nersc(filepath, nb_ex)


def _load_data_nyu(filepath, nb_ex):
    data, label = pickle.load(open(filepath, 'rb'), encoding='latin1')
    if nb_ex is not None:
        data, label = data[:nb_ex], label[:nb_ex]
    return data, label


def _load_data_nersc(filepath, nb_ex):
    raise NotImplementedError


def get_fixed_param():
    """reads parameters from 'new_net_param.txt',
    creates the file if non-existant
    """

    def _get_fixed_param():
        args = dict()
        for line in open('param.txt', 'r'):
            if line.strip():  # not empty line
                arg_txt = line.split('#')[0]  # remove comment
                arg_name, arg_val = arg_txt.split('=')[:2]
                arg_name, arg_val = arg_name.strip(), arg_val.strip()
                args[arg_name] = arg_val
                if arg_val == '':
                    raise ValueError(
                        "Empty parameter in 'param.txt': {}".format(arg_name))
                print("param {} : '{}'".format(arg_name, arg_val))
        return args

    if exists('param.txt'):
        return _get_fixed_param()
    with open('param.txt', 'w') as paramfile:
        paramfile.write(
            "\ntrainfile =  # path to training data `antikt-kt-train-gcnn.pickle`\n"
            + "testfile =  # path to testing data `antikt-kt-test-gcnn.pickle`\n"
        )
    raise FileNotFoundError("'param.txt' created, missing parameters")


def train_net(net, trainfile, criterion, optimizer, args):
    """Trains net for one epoch using criterion loss and optimizer"""

    data, label = load_data(trainfile, args.nbtrain, args.data)
    epoch_loss = 0
    step_loss = 0

    for i, ground_truth in enumerate(label):
        optimizer.zero_grad()

        ground_truth = Variable(torch.Tensor([ground_truth]))
        jet = Variable(torch.Tensor(data[i])).t().unsqueeze(0)
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
    epoch_loss_avg = epoch_loss / len(label)
    return epoch_loss_avg


def test_net(net, testfile, criterion, args):
    """Tests the network, returns the ROC AUC and epoch loss"""

    data, label = load_data(testfile, args.nbtest, args.data)
    criterion = nn.BCELoss()
    epoch_loss = 0
    roccurve = ROCCurve()

    for i, ground_truth in enumerate(label):
        ground_truth = Variable(torch.Tensor([ground_truth]))
        jet = Variable(torch.Tensor(data[i])).t().unsqueeze(0)
        if args.cuda:
            ground_truth = ground_truth.cuda()
            jet = jet.cuda()

        out = net(jet)
        loss = criterion(out, ground_truth)
        epoch_loss += loss.data[0]
        roccurve.update(out, ground_truth, [1.])
    score = roccurve.roc_score(False)
    epoch_loss /= len(label)
    return score, epoch_loss
