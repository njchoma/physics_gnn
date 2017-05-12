import torch.optim as optim
from utils.files import print_


class Optimizer:
    """Encapsules the optimizer and learning rate. Counts the number
    of events seen and updates the learning rate if needed"""

    def __init__(self, model, param):
        """initiates torch.optim.Adam optimizer with learning
        rate `lr`. If the minimum of the loss function hasn't
        decreased in a period of `batch_window` (number of batch),
        the learning rate is multiplied by `updt_mult` < 1."""

        # arguments
        self.model = model
        self.gamma = param.lr_update
        self.thr = param.lr_thr
        self.batch_window = param.lr_nbbatch

        # printing option
        self.verbose = not param.quiet

        # counters
        self.avgloss = float('+inf')
        self.curr_loss = 0.0
        self.batch_seen = 0

        # initiate optimizer
        self.optim()

    def optim(self):
        """Updates optimizer using arguments in `self.model`.
        Stores the optimizer in `self.optimizer`"""
        raise NotImplementedError

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def update(self, batchloss):
        self.curr_loss += batchloss
        self.batch_seen += 1
        if self.batch_seen >= self.batch_window:
            self.new_window_()

    def new_window_(self):
        curr_avgloss = self.curr_loss / self.batch_seen

        # if loss doesn't decrease, change learning rate
        if curr_avgloss > self.avgloss * self.thr:
            self.model.lr *= self.gamma
            self.optim()
            if self.verbose:
                print_('lr update : {: <30} --- average loss : {}'.format(self.model.lr, curr_avgloss), stdout=self.model.statistics.stdout)

        # zero counting parameters
        self.avgloss = curr_avgloss
        self.curr_loss = 0.0
        self.batch_seen = 0


class AdamOptimizer(Optimizer):
    """Optimizer instance using Adam Optimizer"""

    def optim(self):
        self.optimizer = optim.Adam(self.model.net.parameters(), lr=self.model.lr)


class SGDOptimizer(Optimizer):
    """Optimizer instance using Stochastic Gradient Descent"""

    def optim(self):
        self.optimizer = optim.SGD(self.model.net.parameters(), lr=self.model.lr)
