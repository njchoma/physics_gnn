from torch.nn.functional import binary_cross_entropy
from utils.loss_modif import MarginRankingLoss
from Atlas.optim.optimizer import SGDOptimizer, AdamOptimizer


def loss(loss):
    if loss == 'BCE':
        criterion = binary_cross_entropy
    elif loss == 'HingeEmbedding':
        criterion = MarginRankingLoss()
    return criterion


def optimizer(optimizer):
    if optimizer == 'SGD':
        Optimizer = SGDOptimizer
    elif optimizer == 'Adam':
        Optimizer = AdamOptimizer
    return Optimizer
