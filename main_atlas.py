import sys
from os.path import join, abspath, pardir
from utils.files import makefile_if_not_there, print_
from utils.loss_modif import MarginRankingLoss
from torch.nn.functional import binary_cross_entropy
from config.getconfig import Config
from Atlas.getmodel import get_model
from Atlas.optim.optimizer import SGDOptimizer, AdamOptimizer
from graphic.plotstats import plot_statistics


def train_or_test(param, stdout=None):
    # Retrieve of initialize model
    model, ret = get_model(param, stdout=stdout)
    print_(('retrieved' if ret else 'created') + ' model', stdout=stdout)
    model.printdescr(stdout=stdout)

    # Loss function
    if param.loss == 'BCE':
        criterion = binary_cross_entropy
    elif param.loss == 'HingeEmbedding':
        criterion = MarginRankingLoss()

    # Optimizer
    if param.optimizer == 'SGD':
        Optimizer = SGDOptimizer
    elif param.optimizer == 'Adam':
        Optimizer = AdamOptimizer
    optimizer = Optimizer(model, param)

    # Training or Testing
    nb_epoch = param.epoch
    for epoch in range(nb_epoch):
        print_('starting epoch {}'.format(epoch + 1), stdout=stdout)
        model.do_one_epoch(optimizer, criterion)

    print_('Done {}ing.'.format(param.mode), stdout=stdout)


def prepare_data(param, stdout=None):
    raise NotImplementedError("Mode 'prepare_data' hasn't been implemented yet")


def plot(param):
    plot_statistics(param)


def main():
    # get meta parameters
    project_dir = abspath(join(join(__file__, pardir), pardir))
    param = Config(project_dir, description='Train or test GNNs and build ROC curve')

    # printing parameter
    if param.stdout is not None:
        filename = param.model + '.out'
        stdout = join(param.stdout, filename)
        makefile_if_not_there(param.stdout, filename)
    else:
        stdout = None

    # choose action depending on parameters
    action = param.mode
    if action in ['train', 'test']:
        train_or_test(param, stdout)
    elif action == 'plot':
        plot(param)
    elif action == 'description':
        model, _ = get_model(param, stdout=stdout)
        model.print_()
    elif action == 'prepare_data':
        raise NotImplementedError
    elif action == 'weight_average':
        raise NotImplementedError
    elif action == 'setdefault':
        raise NotImplementedError
    else:
        raise Exception('Unknown mode : {}'.format(action))


if __name__ == '__main__':
    main()
    sys.exit(0)
